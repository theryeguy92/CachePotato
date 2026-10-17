#define _GNU_SOURCE
#include "weights_v4.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* ---------------------------------------------------------------- little-endian reads */

static uint32_t rd_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_u16(const unsigned char *p) {
    return (uint16_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8));
}
static uint64_t rd_u64(const unsigned char *p) {
    return (uint64_t)rd_u32(p) | ((uint64_t)rd_u32(p + 4) << 32);
}
static float rd_f32(const unsigned char *p) {
    uint32_t u = rd_u32(p); float f; memcpy(&f, &u, 4); return f;
}

/* CRC-32 (IEEE 802.3): poly 0xEDB88320 reflected, init/final xor 0xFFFFFFFF.
   Identical to tools/v3_to_v4.py's region_digest and Python's zlib.crc32, and checked
   against the standard vector crc32("123456789") == 0xCBF43926 in tools/run_gates.sh. */
static uint32_t crc_table[256];
static int crc_ready = 0;
static void crc_init(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc_table[i] = c;
    }
    crc_ready = 1;
}
static uint32_t crc32_bytes(const unsigned char *p, size_t n) {
    if (!crc_ready) crc_init();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) c = crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

/* ---------------------------------------------------------------- tables */

enum { KIND_EMBED = 1, KIND_NORM = 2, KIND_ATTN = 0x10, KIND_DENSE_FFN = 0x11,
       KIND_MOE = 0x20, KIND_MOE_LATENT = 0x21, KIND_SSM = 0x30 };
enum { ROLE_OTHER = 0, ROLE_EMBEDDING = 1, ROLE_NORM = 2, ROLE_ATTN = 3, ROLE_ROUTER = 4,
       ROLE_EXPERT = 5, ROLE_SHARED_EXPERT = 6, ROLE_DENSE_FFN = 7, ROLE_HEAD = 8 };

#define MAX_TENSORS 65536
#define MAX_BLOCKS  4096
#define MAX_REFS    64
#define MAX_PARAMS  32

typedef struct {
    char *name; uint32_t dtype_id; uint8_t rank; uint32_t shape[4];
    uint64_t off, len;      /* off is RELATIVE to off_data (SPEC §5) */
    uint32_t block_id; uint16_t role;
} TRec;
typedef struct { uint8_t kind; uint32_t layer; uint8_t n_refs; uint32_t refs[MAX_REFS];
                 uint8_t n_params; uint32_t params[MAX_PARAMS]; } BRec;
typedef struct { uint64_t off, len; uint32_t lo, hi; uint8_t kind; uint16_t slot;
                 uint32_t mlo, mhi; uint64_t digest; } RRec;
typedef struct { char *name; uint8_t kind, scalar; } ERec;

int v4_read_header(int fd, V4Header *h) {
    unsigned char b[88];
    if (pread(fd, b, sizeof b, 0) != (ssize_t)sizeof b) return -1;
    if (rd_u32(b) != 0x34575043u || rd_u32(b + 4) != 4) return -1;
    if (rd_u32(b + 8) != sizeof b) {
        fprintf(stderr, "v4: header_bytes %u, this loader expects %zu\n", rd_u32(b + 8), sizeof b);
        return -1;
    }
    for (int i = 0; i < 4; i++)
        if (rd_u32(b + 72 + 4 * i) != 0) {
            fprintf(stderr, "v4: reserved word %d nonzero — refusing\n", i);
            return -1;
        }
    h->n_tensors = rd_u32(b + 16);   h->n_blocks = rd_u32(b + 20);
    h->n_regions = rd_u32(b + 24);   h->n_encodings = rd_u32(b + 28);
    h->off_encodings = rd_u64(b + 32); h->off_tensors = rd_u64(b + 40);
    h->off_blocks = rd_u64(b + 48);    h->off_regions = rd_u64(b + 56);
    h->off_data = rd_u64(b + 64);
    return 0;
}

/* ---------------------------------------------------------------- loader */

int model_load_v4(Model *m, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror("open"); return -1; }
    struct stat st;
    if (fstat(fd, &st) < 0) { perror("fstat"); close(fd); return -1; }
    size_t fsz = (size_t)st.st_size;

    V4Header h;
    if (v4_read_header(fd, &h) != 0) { close(fd); return -1; }
    if (h.n_tensors > MAX_TENSORS || h.n_blocks > MAX_BLOCKS) {
        fprintf(stderr, "v4: implausible table sizes (%u tensors, %u blocks)\n",
                h.n_tensors, h.n_blocks);
        close(fd); return -1;
    }
    if (!(h.off_encodings <= h.off_tensors && h.off_tensors <= h.off_blocks &&
          h.off_blocks <= h.off_regions && h.off_regions <= h.off_data && h.off_data <= fsz)) {
        fprintf(stderr, "v4: table offsets not ordered or past EOF\n"); close(fd); return -1;
    }

    unsigned char *img = malloc(fsz);
    if (!img) { fprintf(stderr, "v4: oom for %zu bytes\n", fsz); close(fd); return -1; }
    if (pread(fd, img, fsz, 0) != (ssize_t)fsz) {
        fprintf(stderr, "v4: short read\n"); free(img); close(fd); return -1;
    }

    int rc = -1;
    TRec *tens = NULL; BRec *blks = NULL; RRec *regs = NULL; ERec *encs = NULL;
    float *layout = NULL;
    uint32_t n_enc = h.n_encodings;

    /* ---- encoding table */
    encs = calloc(n_enc ? n_enc : 1, sizeof *encs);
    if (!encs) goto done;
    {
        size_t o = h.off_encodings;
        for (uint32_t i = 0; i < n_enc; i++) {
            if (o + 13 > fsz) goto done;
            ERec *e = &encs[i];
            o += 4; e->kind = img[o++]; e->scalar = img[o++]; o += 4 + 3;
            uint16_t n = rd_u16(img + o); o += 2;
            if (o + n > fsz) goto done;
            e->name = strndup((const char *)img + o, n); o += n;
            if (!e->name) goto done;
        }
        if (o != h.off_tensors) {
            fprintf(stderr, "v4: encoding table ends at %zu, header says %llu\n", o,
                    (unsigned long long)h.off_tensors);
            goto done;
        }
    }
    for (uint32_t i = 0; i < n_enc; i++)
        if (encs[i].kind != 0 || encs[i].scalar != 0) {
            fprintf(stderr, "v4: encoding \"%s\" (kind %u scalar %u) is not raw f32 — this "
                            "loader implements raw f32 only and refuses rather than "
                            "approximating\n", encs[i].name, encs[i].kind, encs[i].scalar);
            goto done;
        }

    /* ---- tensor table */
    tens = calloc(h.n_tensors ? h.n_tensors : 1, sizeof *tens);
    if (!tens) goto done;
    {
        size_t o = h.off_tensors;
        uint32_t cnt = rd_u32(img + o); o += 4;
        if (cnt != h.n_tensors) { fprintf(stderr, "v4: tensor count %u != header %u\n", cnt, h.n_tensors); goto done; }
        for (uint32_t i = 0; i < h.n_tensors; i++) {
            TRec *t = &tens[i];
            uint16_t n = rd_u16(img + o); o += 2;
            if (o + n > fsz) goto done;
            t->name = strndup((const char *)img + o, n); o += n;
            if (!t->name) goto done;
            t->dtype_id = rd_u32(img + o); o += 4;
            t->rank = img[o++];
            if (t->rank < 1 || t->rank > 4) { fprintf(stderr, "v4: %s: rank %u\n", t->name, t->rank); goto done; }
            for (int d = 0; d < t->rank; d++) { t->shape[d] = rd_u32(img + o); o += 4; }
            t->off = rd_u64(img + o); o += 8;
            t->len = rd_u64(img + o); o += 8;
            t->block_id = rd_u32(img + o); o += 4;
            t->role = rd_u16(img + o); o += 2;
            if (t->dtype_id >= n_enc) { fprintf(stderr, "v4: %s: dtype %u out of range\n", t->name, t->dtype_id); goto done; }
            if (t->off % 64) { fprintf(stderr, "v4: %s: offset not 64-aligned\n", t->name); goto done; }
            if (h.off_data + t->off + t->len > fsz) {
                fprintf(stderr, "v4: %s: payload past EOF\n", t->name); goto done;
            }
        }
        if (o != h.off_blocks) {
            fprintf(stderr, "v4: tensor table ends at %zu, header says %llu\n", o,
                    (unsigned long long)h.off_blocks);
            goto done;
        }
    }

    /* ---- overlap check (simple insertion sort: tables are 10^2-10^3 entries) */
    {
        long *ord = malloc((size_t)h.n_tensors * sizeof *ord);
        if (!ord) goto done;
        for (uint32_t i = 0; i < h.n_tensors; i++) ord[i] = i;
        for (uint32_t i = 1; i < h.n_tensors; i++) {
            long key = ord[i]; uint32_t j = i;
            while (j > 0 && tens[ord[j - 1]].off > tens[key].off) { ord[j] = ord[j - 1]; j--; }
            ord[j] = key;
        }
        for (uint32_t i = 1; i < h.n_tensors; i++) {
            const TRec *a = &tens[ord[i - 1]], *b = &tens[ord[i]];
            if (a->off + a->len > b->off) {
                fprintf(stderr, "v4: tensors %s and %s overlap\n", a->name, b->name);
                free(ord); goto done;
            }
        }
        free(ord);
    }

    /* ---- block table */
    blks = calloc(h.n_blocks ? h.n_blocks : 1, sizeof *blks);
    if (!blks) goto done;
    {
        size_t o = h.off_blocks;
        uint32_t cnt = rd_u32(img + o); o += 4;
        if (cnt != h.n_blocks) { fprintf(stderr, "v4: block count %u != header %u\n", cnt, h.n_blocks); goto done; }
        for (uint32_t i = 0; i < h.n_blocks; i++) {
            BRec *b = &blks[i];
            b->kind = img[o++]; b->layer = rd_u32(img + o); o += 4;
            b->n_refs = img[o++];
            if (b->n_refs > MAX_REFS) { fprintf(stderr, "v4: block %u: %u refs\n", i, b->n_refs); goto done; }
            for (int r = 0; r < b->n_refs; r++) {
                b->refs[r] = rd_u32(img + o); o += 4;
                if (b->refs[r] >= h.n_tensors) { fprintf(stderr, "v4: block %u ref out of range\n", i); goto done; }
            }
            b->n_params = img[o++];
            if (b->n_params > MAX_PARAMS) { fprintf(stderr, "v4: block %u: %u params\n", i, b->n_params); goto done; }
            for (int p = 0; p < b->n_params; p++) { b->params[p] = rd_u32(img + o); o += 4; }
        }
        if (o != h.off_regions) {
            fprintf(stderr, "v4: block table ends at %zu, header says %llu\n", o,
                    (unsigned long long)h.off_regions);
            goto done;
        }
    }

    /* ---- region table + digests */
    regs = calloc(h.n_regions ? h.n_regions : 1, sizeof *regs);
    if (!regs) goto done;
    {
        size_t o = h.off_regions;
        uint32_t cnt = rd_u32(img + o); o += 4;
        if (cnt != h.n_regions) { fprintf(stderr, "v4: region count %u != header %u\n", cnt, h.n_regions); goto done; }
        for (uint32_t i = 0; i < h.n_regions; i++) {
            RRec *r = &regs[i];
            r->off = rd_u64(img + o); o += 8;
            r->len = rd_u64(img + o); o += 8;
            r->lo = rd_u32(img + o); o += 4;
            r->hi = rd_u32(img + o); o += 4;
            r->kind = img[o++];
            r->slot = rd_u16(img + o); o += 2;
            r->mlo = rd_u32(img + o); o += 4;
            r->mhi = rd_u32(img + o); o += 4;
            r->digest = rd_u64(img + o); o += 8;
            if (r->hi <= r->lo || r->hi > h.n_tensors) { fprintf(stderr, "v4: region %u range invalid\n", i); goto done; }
            if (h.off_data + r->off + r->len > fsz) { fprintf(stderr, "v4: region %u past EOF\n", i); goto done; }
        }
        if (o > h.off_data) { fprintf(stderr, "v4: region table past off_data\n"); goto done; }
    }
    for (uint32_t i = 0; i < h.n_regions; i++) {
        const RRec *r = &regs[i];
        uint32_t got = crc32_bytes(img + h.off_data + r->off, (size_t)r->len);
        if ((uint64_t)got != r->digest) {
            fprintf(stderr, "v4: region %u (%s): digest mismatch — corrupt or misaligned "
                            "(stored %08x, computed %08x)\n",
                    i, tens[r->lo].name, (uint32_t)r->digest, got);
            goto done;
        }
    }
    {
        char *cov = calloc(h.n_tensors, 1);
        if (!cov) goto done;
        for (uint32_t i = 0; i < h.n_regions; i++)
            for (uint32_t t = regs[i].lo; t < regs[i].hi; t++) cov[t] = 1;
        for (uint32_t i = 0; i < h.n_tensors; i++)
            if (!cov[i]) {
                fprintf(stderr, "v4: tensor %s is not covered by any region — it could not be "
                                "paged and would read through the page cache\n", tens[i].name);
                free(cov); goto done;
            }
        free(cov);
    }

    /* ---- model description from the block table */
    {
        int32_t vocab = 0, d = 0, L = -1, ctx = 0, ff = 0, heads = 0;
        int32_t n_exp = 0, topk = 0, n_moe = 0, n_dense = 0;
        float lam = 0.0f;
        int moe_from = -1;
        /* L comes from FFN blocks only: a trailing norm block carries layer == L (it runs
           after the last layer) and counting it gave L=13 for a 12-layer model. */
        for (uint32_t i = 0; i < h.n_blocks; i++) {
            const BRec *b = &blks[i];
            if ((b->kind == KIND_MOE || b->kind == KIND_DENSE_FFN || b->kind == KIND_ATTN) &&
                (int)b->layer + 1 > L) L = (int)b->layer + 1;
        }
        for (uint32_t i = 0; i < h.n_blocks; i++) {
            const BRec *b = &blks[i];
            switch (b->kind) {
            case KIND_EMBED:
                if (b->n_params < 3) { fprintf(stderr, "v4: embedding block malformed\n"); goto done; }
                vocab = (int32_t)b->params[0]; d = (int32_t)b->params[1]; ctx = (int32_t)b->params[2];
                break;
            case KIND_ATTN:
                if (b->n_refs < 8 || b->n_params < 3) { fprintf(stderr, "v4: attention block malformed\n"); goto done; }
                heads = (int32_t)b->params[0];
                if ((int32_t)b->params[1] != heads || (int32_t)b->params[2] != 0) {
                    fprintf(stderr, "v4: attention needs n_kv_heads == n_heads and "
                                    "head_dim == d_model/n_heads (got %u, %u)\n",
                            b->params[1], b->params[2]);
                    goto done;
                }
                break;
            case KIND_DENSE_FFN:
                if (b->n_params < 2) { fprintf(stderr, "v4: dense_ffn block malformed\n"); goto done; }
                if ((int32_t)b->params[0] != 0) {
                    fprintf(stderr, "v4: dense_ffn activation %u unsupported (gelu_tanh only)\n",
                            b->params[0]);
                    goto done;
                }
                ff = (int32_t)b->params[1]; n_dense++;
                /* params[3] = dense_packed: weights present but never executed (the model's
                   own last-layer deviation). Such a layer must NOT create a MoE slot. */
                break;
            case KIND_MOE:
                if (b->n_params < 6) { fprintf(stderr, "v4: moe block malformed\n"); goto done; }
                n_exp = (int32_t)b->params[0]; topk = (int32_t)b->params[1];
                if ((int32_t)b->params[2] != 0) {
                    fprintf(stderr, "v4: moe scoring %u unsupported (softmax only)\n", b->params[2]);
                    goto done;
                }
                if ((int32_t)b->params[4] != 0) {
                    fprintf(stderr, "v4: moe latent_size %u != 0 — latent MoE is specified but "
                                    "not implemented by this engine; refusing\n", b->params[4]);
                    goto done;
                }
                {
                    unsigned char lb[4];
                    lb[0] = (unsigned char)(b->params[3] & 0xFF);
                    lb[1] = (unsigned char)((b->params[3] >> 8) & 0xFF);
                    lb[2] = (unsigned char)((b->params[3] >> 16) & 0xFF);
                    lb[3] = (unsigned char)((b->params[3] >> 24) & 0xFF);
                    lam = rd_f32(lb);
                }
                n_moe++;
                if (moe_from < 0 || (int)b->layer < moe_from) moe_from = (int)b->layer;
                break;
            case KIND_NORM: break;
            case KIND_MOE_LATENT:
                fprintf(stderr, "v4: block kind moe_latent (0x21) is specified but not "
                                "implemented by this engine; refusing\n"); goto done;
            case KIND_SSM:
                fprintf(stderr, "v4: block kind ssm (0x30) is specified but not implemented by "
                                "this engine; refusing\n"); goto done;
            default:
                fprintf(stderr, "v4: unknown block kind 0x%02x at index %u — refusing\n", b->kind, i);
                goto done;
            }
        }
        if (vocab <= 0 || d <= 0 || L <= 0 || ctx <= 0 || ff <= 0 || heads <= 0 || n_moe == 0) {
            fprintf(stderr, "v4: incomplete model description (vocab %d d %d L %d ctx %d ff %d "
                            "H %d moe %d)\n", vocab, d, L, ctx, ff, heads, n_moe);
            goto done;
        }
        for (int l = 0; l < L; l++) {
            int is_moe = -1;
            for (uint32_t i = 0; i < h.n_blocks; i++)
                if ((int)blks[i].layer == l) {
                    if (blks[i].kind == KIND_MOE || blks[i].kind == KIND_MOE_LATENT) is_moe = 1;
                    else if (blks[i].kind == KIND_DENSE_FFN) is_moe = 0;
                }
            if (is_moe < 0) { fprintf(stderr, "v4: layer %d has no FFN block\n", l); goto done; }
            /* expected: dense before moe_from (or pack-dense anywhere), MoE elsewhere. The
               engine's predicate is moe_from <= l < n_layer-1, and moe_from was taken as the
               first MoE block, so a packed-dense layer at a later position must be excluded
               from the MoE run rather than treated as interleaving. */
            int packed = 0;
            for (uint32_t i = 0; i < h.n_blocks; i++)
                if ((int)blks[i].layer == l && blks[i].kind == KIND_DENSE_FFN &&
                    blks[i].n_params >= 4 && blks[i].params[3] != 0) packed = 1;
            int expect = (!packed && l >= moe_from && l < L - 1);
            if (is_moe != expect) {
                fprintf(stderr, "v4: layer %d is %s but the engine's forward pass expects %s "
                                "(moe_from %d). Interleaved MoE/dense stacks need the "
                                "block-table forward pass, which does not exist yet — "
                                "refusing rather than mis-loading.\n",
                        l, is_moe ? "MoE" : "dense", expect ? "MoE" : "dense", moe_from);
                goto done;
            }
        }

        m->is_v3 = 0;
        m->hdr.magic = 0x34575043u; m->hdr.version = 4;
        m->hdr.vocab = vocab; m->hdr.d_model = d; m->hdr.n_heads = heads;
        m->hdr.d_ff = ff; m->hdr.n_layer = L; m->hdr.n_ctx = ctx;
        m->h3.lam = lam; m->h3.n_exp = n_exp; m->h3.topk = topk;
        m->h3.moe_from = moe_from; m->h3.n_moe_layers = n_moe; m->h3.n_dense_layers = n_dense;

        /* ---- build the engine's strided layout in one buffer.
           The forward pass indexes per-layer arrays by stride (layer l of Wq at Wq + l*d*d).
           In v3 that held because the writer emits one tensor per (type, all layers). A v4
           file makes no such promise, so rather than assume it, copy each tensor into its
           slot in engine layout. --shuffle exists precisely to prove this is done by table
           lookup and not by position. */
        long v = vocab, dd = d, ffl = ff, Ll = L;
        long nmoe = n_moe, ndense = n_dense, N = n_exp;
        long exp_floats = 0;
        for (uint32_t i = 0; i < h.n_tensors; i++) {
            int s, e;
            if (tens[i].role == ROLE_EXPERT && sscanf(tens[i].name, "experts.%d.%d", &s, &e) == 2) {
                exp_floats = (long)(tens[i].len / 4);
                break;
            }
        }
        if (exp_floats <= 0) { fprintf(stderr, "v4: no routed expert tensors\n"); goto done; }
        long exp_hid = (exp_floats - dd) / (2L * dd);
        if (exp_hid <= 0) { fprintf(stderr, "v4: cannot derive exp_hid from %ld floats\n", exp_floats); goto done; }

        long need = v * dd * 2            /* token_emb, pos_emb share the (v,d) and (ctx,d) */
                  + 4 * Ll * dd           /* ln1_g,b, ln2_g,b */
                  + Ll * (4 * dd * dd + 4 * dd)
                  + ndense * ((dd * ffl + ffl) + (ffl * dd + dd))
                  + nmoe * (dd * N + N)
                  + nmoe * (2 * (dd * dd + dd))
                  + nmoe * N * exp_floats
                  + 2 * dd;
        layout = calloc((size_t)need + 16, sizeof(float));
        if (!layout) { fprintf(stderr, "v4: oom for %ld-float layout buffer\n", need); goto done; }

        long o = 0;
        float *p_token = layout + o; o += v * dd;
        float *p_pos   = layout + o; o += ctx * dd;
        float *p_ln1g  = layout + o; o += Ll * dd;
        float *p_ln1b  = layout + o; o += Ll * dd;
        float *p_Wq    = layout + o; o += Ll * dd * dd;
        float *p_bq    = layout + o; o += Ll * dd;
        float *p_Wk    = layout + o; o += Ll * dd * dd;
        float *p_bk    = layout + o; o += Ll * dd;
        float *p_Wv    = layout + o; o += Ll * dd * dd;
        float *p_bv    = layout + o; o += Ll * dd;
        float *p_Wo    = layout + o; o += Ll * dd * dd;
        float *p_bo    = layout + o; o += Ll * dd;
        float *p_ln2g  = layout + o; o += Ll * dd;
        float *p_ln2b  = layout + o; o += Ll * dd;
        float *p_W1    = layout + o; o += ndense * dd * ffl;
        float *p_b1    = layout + o; o += ndense * ffl;
        float *p_W2    = layout + o; o += ndense * ffl * dd;
        float *p_b2    = layout + o; o += ndense * dd;
        float *p_Wr    = layout + o; o += nmoe * dd * N;
        float *p_br    = layout + o; o += nmoe * N;
        float *p_W1s   = layout + o; o += nmoe * dd * dd;
        float *p_b1s   = layout + o; o += nmoe * dd;
        float *p_W2s   = layout + o; o += nmoe * dd * dd;
        float *p_b2s   = layout + o; o += nmoe * dd;
        float *p_exp   = layout + o; o += nmoe * N * exp_floats;
        float *p_lnfg  = layout + o; o += dd;
        float *p_lnfb  = layout + o;
        if (o + dd > need + 16) { fprintf(stderr, "v4: layout arithmetic overflow\n"); goto done; }

        /* Per-tensor copy with an explicit size check and an EXPLICIT stride per tensor
           type. An earlier version used one stride (d*d) for every layer-indexed tensor and
           guarded the offset with `li > 0`, which (a) gave layer 0 no offset at all and
           (b) is wrong for biases and norms whose per-layer stride is d, not d*d. Both
           errors are silent garbage, so both are now structural: every branch names its
           stride, and the offset applies for li >= 0. */
        int bad = 0;
        for (uint32_t i = 0; i < h.n_tensors && !bad; i++) {
            const TRec *t = &tens[i];
            float *src = (float *)(img + h.off_data + t->off);
            long cnt = (long)(t->len / 4);
            long exp = 0, stride = 0, index = 0;
            float *base = NULL;
            int s2 = -1, e2 = -1;
            const char *nm = t->name;

            if (t->role == ROLE_EXPERT) {
                if (sscanf(nm, "experts.%d.%d", &s2, &e2) != 2) continue;
                base = p_exp; stride = exp_floats; index = (long)s2 * N + e2; exp = exp_floats;
            } else if (!strcmp(nm, "token_emb")) { base = p_token; stride = 0; exp = v * dd; }
            else if (!strcmp(nm, "pos_emb"))      { base = p_pos;   stride = 0; exp = ctx * dd; }
            else if (!strcmp(nm, "lnf_g"))        { base = p_lnfg;  stride = 0; exp = dd; }
            else if (!strcmp(nm, "lnf_b"))        { base = p_lnfb;  stride = 0; exp = dd; }
            else {
                struct { const char *n; float *b; long s; } tab[] = {
                    {"ln1_g", p_ln1g, dd},  {"ln1_b", p_ln1b, dd},
                    {"ln2_g", p_ln2g, dd},  {"ln2_b", p_ln2b, dd},
                    {"Wq", p_Wq, dd*dd},    {"bq", p_bq, dd},
                    {"Wk", p_Wk, dd*dd},    {"bk", p_bk, dd},
                    {"Wv", p_Wv, dd*dd},    {"bv", p_bv, dd},
                    {"Wo", p_Wo, dd*dd},    {"bo", p_bo, dd},
                };
                int matched = 0;
                for (size_t k = 0; k < sizeof tab / sizeof tab[0]; k++)
                    if (!strcmp(nm, tab[k].n)) { base = tab[k].b; stride = tab[k].s; matched = 1; break; }
                if (matched) {
                    /* Layer-indexed per-tensor arrays are (L, ...): shape[0] is the number of
                       LAYERS, and the array starts at layer 0, so the payload is copied once
                       with index 0 and the rest of the layers follow contiguously. (Dense and
                       MoE arrays below are (n_dense, ...) / (n_moe, ...) and shape[0] is a real
                       index into the group.) Treating both the same made the loader reject
                       ln1_g with "layer 12 out of range" for a 12-layer model. */
                    if (t->rank < 2) { fprintf(stderr, "v4: %s needs rank >= 2\n", nm); bad = 1; break; }
                    if ((long)t->shape[0] != Ll) {
                        fprintf(stderr, "v4: %s declares %u layers, the model has %ld\n",
                                nm, t->shape[0], Ll);
                        bad = 1; break;
                    }
                    index = 0;
                    exp = stride * Ll;
                } else if (!strcmp(nm, "W1") || !strcmp(nm, "b1") ||
                           !strcmp(nm, "W2") || !strcmp(nm, "b2")) {
                    static const char *dn[] = {"W1","b1","W2","b2"};
                    (void)dn;
                    base = (!strcmp(nm,"W1")) ? p_W1 : (!strcmp(nm,"b1")) ? p_b1 :
                           (!strcmp(nm,"W2")) ? p_W2 : p_b2;
                    stride = (!strcmp(nm,"W1")) ? dd*ffl : (!strcmp(nm,"b1")) ? ffl :
                             (!strcmp(nm,"W2")) ? ffl*dd : dd;
                    /* (n_dense, ...): shape[0] is the count; the payload is the whole group. */
                    if ((long)t->shape[0] != ndense) {
                        fprintf(stderr, "v4: %s declares %u dense layers, the model has %ld\n",
                                nm, t->shape[0], ndense);
                        bad = 1; break;
                    }
                    index = 0;
                    exp = stride * ndense;
                } else if (!strcmp(nm,"Wr") || !strcmp(nm,"br") || !strcmp(nm,"W1s") ||
                           !strcmp(nm,"b1s") || !strcmp(nm,"W2s") || !strcmp(nm,"b2s")) {
                    base = (!strcmp(nm,"Wr"))  ? p_Wr  : (!strcmp(nm,"br"))  ? p_br  :
                           (!strcmp(nm,"W1s")) ? p_W1s : (!strcmp(nm,"b1s")) ? p_b1s :
                           (!strcmp(nm,"W2s")) ? p_W2s : p_b2s;
                    stride = (!strcmp(nm,"Wr"))  ? dd*N  : (!strcmp(nm,"br"))  ? N :
                             (!strcmp(nm,"W1s")) ? dd*dd : (!strcmp(nm,"b1s")) ? dd :
                             (!strcmp(nm,"W2s")) ? dd*dd : dd;
                    /* (n_moe, ...): likewise, shape[0] is the count. */
                    if ((long)t->shape[0] != nmoe) {
                        fprintf(stderr, "v4: %s declares %u moe layers, the model has %ld\n",
                                nm, t->shape[0], nmoe);
                        bad = 1; break;
                    }
                    index = 0;
                    exp = stride * nmoe;
                } else {
                    continue;                 /* regioned, but the forward pass ignores it */
                }
            }
            if (cnt != exp) {
                fprintf(stderr, "v4: %s has %ld floats, engine layout expects %ld — the file "
                                "and the forward pass disagree on shape\n", nm, cnt, exp);
                bad = 1; break;
            }
            memcpy(base + index * stride, src, (size_t)cnt * sizeof(float));
        }
        if (bad) goto done;

        Weights *w = &m->w;
        memset(w, 0, sizeof *w);
        w->token_emb = p_token; w->pos_emb = p_pos;
        w->ln1_g = p_ln1g; w->ln1_b = p_ln1b; w->ln2_g = p_ln2g; w->ln2_b = p_ln2b;
        w->Wq = p_Wq; w->bq = p_bq; w->Wk = p_Wk; w->bk = p_bk;
        w->Wv = p_Wv; w->bv = p_bv; w->Wo = p_Wo; w->bo = p_bo;
        w->W1 = p_W1; w->b1 = p_b1; w->W2 = p_W2; w->b2 = p_b2;
        w->Wr = p_Wr; w->br = p_br; w->W1s = p_W1s; w->b1s = p_b1s;
        w->W2s = p_W2s; w->b2s = p_b2s; w->experts = p_exp;
        w->lnf_g = p_lnfg; w->lnf_b = p_lnfb;
        w->is_moe = 1; w->d_model = (int)dd; w->n_moe = (int)nmoe; w->n_exp = (int)N;
        w->exp_hid = (int)exp_hid; w->topk = topk; w->moe_from = moe_from;
        w->moe_slot = malloc((size_t)Ll * sizeof(int));
        w->dense_slot = malloc((size_t)Ll * sizeof(int));
        if (!w->moe_slot || !w->dense_slot) { fprintf(stderr, "v4: oom for layer maps\n"); goto done; }
        {
            int nm = 0, nd = 0;
            for (long l = 0; l < Ll; l++) {
                if ((int)l >= moe_from && l < Ll - 1) { w->moe_slot[l] = nm++; w->dense_slot[l] = -1; }
                else { w->moe_slot[l] = -1; w->dense_slot[l] = nd++; }
            }
        }
        if (w->topk != 2) {
            fprintf(stderr, "v4: topk %d — the engine implements top-2 only\n", w->topk);
            goto done;
        }
        m->v4_scratch = layout;
        layout = NULL;
    }

    m->file_size = fsz;
    m->data = (float *)img;
    m->fd = fd;
    printf("v4: loaded %s — %u tensors, %u blocks, %u regions; vocab %d d %d L %d; "
           "%d MoE slots x %d experts top-%d lam %.2f; regions CRC-verified\n",
           path, h.n_tensors, h.n_blocks, h.n_regions, m->hdr.vocab, m->hdr.d_model,
           m->hdr.n_layer, m->h3.n_moe_layers, m->h3.n_exp, m->h3.topk, m->h3.lam);
    rc = 0;

done:
    free(layout);
    if (tens) { for (uint32_t i = 0; i < h.n_tensors; i++) free(tens[i].name); free(tens); }
    if (blks) free(blks);
    if (regs) free(regs);
    if (encs) { for (uint32_t i = 0; i < n_enc; i++) free(encs[i].name); free(encs); }
    if (rc != 0) { free(img); close(fd); }
    return rc;
}
