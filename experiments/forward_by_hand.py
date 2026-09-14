"""forward_by_hand.py — a transformer forward pass with numbers small enough
to check by hand. Phase 0, Level 1.

Config (deliberately tiny):
    vocab   = 8    tokens total, so logits fit on one line
    d_model = 4    residual stream width
    n_heads = 2    -> head_dim = 2
    d_ff    = 8    FFN hidden width (4x is standard; we use 2x, doesn't matter)
    n_layer = 1    a single transformer block — the pattern repeats for more

Everything is float32 numpy, fixed seed, so your output must match exactly.
Run:  python3 forward_by_hand.py

LEARNING OBJECTIVE (Phase 0 gate preparation)
    Implement every stage of a transformer forward pass from formulas,
    verify each numerically, then map every stage to its lines in
    llama2.c's forward() — such that run.c reads as "this exercise,
    in C, plus five named deltas":
      1. LayerNorm (here)      -> RMSNorm (no mean subtraction)
      2. no positional enc     -> RoPE (rotates q,k by position)
      3. all tokens at once    -> one token/step + KV cache
      4. FFN W1->ReLU->W2      -> SwiGLU: silu(x@W1)*(x@W3) @ W2
      5. explicit causal mask  -> causality free from KV cache

    run.c section map (read after finishing this file):
      Config struct            -> the file header = our config below
      memory_map_weights()     -> our W dict, as one flat mmap'd blob
      forward()                -> steps 0..N below
      RunState struct          -> the activation buffers between steps
"""

import numpy as np

rng = np.random.default_rng(42)

VOCAB, D, H, D_FF = 8, 4, 2, 8
HEAD_DIM = D // H

def rand_weight(*shape):
    """Small random weights — large values make softmax saturate and
    hand-checking painful."""
    return rng.standard_normal(shape, dtype=np.float32) * 0.2

# ---- the model's entire parameter set -------------------------------------
# This dict IS the weight file. In Phase 1 this becomes bytes we mmap.
W = {
    "token_emb": rand_weight(VOCAB, D),   # (8,4) one row per token
    "Wq": rand_weight(D, D),              # (4,4) per block: queries
    "Wk": rand_weight(D, D),              # (4,4) keys
    "Wv": rand_weight(D, D),              # (4,4) values
    "Wo": rand_weight(D, D),              # (4,4) attention output proj
    "W1": rand_weight(D, D_FF),           # (4,8) FFN up
    "W2": rand_weight(D_FF, D),           # (8,4) FFN down
    "ln1_g": np.ones(D, dtype=np.float32),   # layernorm gains (bias=0, kept
    "ln2_g": np.ones(D, dtype=np.float32),   # out for simplicity)
    "lnf_g": np.ones(D, dtype=np.float32),
    "unemb": rand_weight(VOCAB, D),       # (8,4) logits = x @ unemb.T
}

# ---- the input -------------------------------------------------------------
tokens = [5, 1, 7]          # a 3-token "sentence"
T = len(tokens)

# ---- step 0: embedding -----------------------------------------------------
# Embedding is not a matmul. It's a LOOKUP: row `t` of token_emb for token t.
# x becomes (T, D) = (3, 4) — the residual stream, one row per position.
x = W["token_emb"][tokens]                    # (3,4)

np.set_printoptions(precision=4, suppress=True)
print("step 0 — embedding, x:", x.shape)
print(x)

# ===========================================================================
# EXERCISE 1 — layernorm. You write this function.
#
# For each ROW v (one token's 4-dim vector):
#     mean = v.mean()
#     var  = v.var()          # population variance, ddof=0
#     out  = (v - mean) / sqrt(var + eps) * gain
# eps = 1e-5. No bias term (llama2.c style). Apply row-wise over x.
#
# Sanity check afterwards: every row of your output has mean ~0, std ~1
# (times the gain, which is 1.0 here). Verify with:
#     print(x_norm.mean(axis=1), x_norm.std(axis=1))
# ===========================================================================

# For layernorm, we need ot write logic for the mean, var and out.

# For the mean, we average out reach row. So, v will be x and shaped like (3,4): 3 tokens with 4 values each.
# The reason why we do this is because each token gets normalized by its own stats, never mixed.
# So we have axis = -1, keepdims=True ensures that the result has a shape like (3,1).
# Average each token's 4 values, ket as a columns so it subjtracts row-wise.

#For Variance, as the name implies, averages the square distance from the rows mean 


def layernorm(v, gain, eps=1e-5):
    mean = v.mean(axis=-1, keepdims=True)
    var = v.var(axis=-1, keepdims=True)
    return (v-mean) / np.sqrt(var + eps) * gain
    # YOUR CODE HERE — compute from v and gain only, then return the result.
    # v can be a single row (4,) OR the whole matrix (3,4); write it so both
    # work. Hint: v.mean(axis=-1, keepdims=True) normalizes each row
    # independently and keeps broadcasting happy.
    #raise NotImplementedError("you write this — ~4 lines")

x_norm = layernorm(x, W["ln1_g"])
print("step 1 — layernorm:", x_norm.shape)
print(x_norm)
print("row means:", x_norm.mean(axis=1), "row stds:", x_norm.std(axis=1))

# ===========================================================================
# EXERCISE 2 — QKV projections + head split. You write this.
#
# A linear projection is one matmul: out = x_norm @ W
#     x_norm: (3,4)   W: (4,4)   out: (3,4)
#     out[i,j] = sum_k x_norm[i,k] * W[k,j]
# i.e. each output feature of a token is a learned weighted combination of
# ALL 4 of that token's input features. Tokens never mix here — projections
# are per-token. (Mixing across tokens is attention's job, next exercise.)
#
# Three separate projections, same input:
#   q = x_norm @ Wq    queries — "what am I looking for?"
#   k = x_norm @ Wk    keys    — "what do I advertise?"
#   v = x_norm @ Wv    values  — "what do I pass on if attended to?"
#
# Then split each (3,4) into H=2 heads of HEAD_DIM=2:
#   q.reshape(T, H, HEAD_DIM)   -> (3,2,2)
# Head 0 sees dims 0-1, head 1 sees dims 2-3. Each head will learn to track
# a different kind of relationship between tokens.
#
# Checkpoint: shapes are (3,2,2), and hand-verify ONE number:
#   q[0,0] must equal the dot product x_norm[0] @ Wq's first column:
#     print(q[0,0], x_norm[0] @ W["Wq"][:,0])
# (If those don't match, your matmul orientation is transposed.)
# ===========================================================================

q = x_norm @ W["Wq"]
k = x_norm @ W["Wk"]
v = x_norm @ W["Wv"]
q_heads = q.reshape(T, H, HEAD_DIM)
k_heads = k.reshape(T, H, HEAD_DIM)
v_heads = v.reshape(T, H, HEAD_DIM)
print("step 2 — q_heads:", q_heads.shape)
print(q_heads)
print("hand check:", q_heads[0,0], "vs", (x_norm[0] @ W["Wq"][:,0]))

# ---- bonus, no exercise: run.c uses RMSNorm instead of LayerNorm ----------
# Same idea, but no mean subtraction — it scales by the root-mean-square of
# the values themselves. This is run.c's rmsnorm(), almost verbatim:
def rmsnorm(v, gain, eps=1e-5):
    ms = (v ** 2).mean(axis=-1, keepdims=True)
    return v / np.sqrt(ms + eps) * gain

# ===========================================================================
# EXERCISE 3 — attention scores, causal mask, softmax. You write this.
#
# For head h:  scores = q_h @ k_h.T / sqrt(HEAD_DIM)
#   q_h, k_h: (3,2)   scores: (3,3)
#   scores[i,j] = how much token i cares about token j, in this head.
#
# THE SCALE: dividing by sqrt(HEAD_DIM) keeps dot products from growing
# with dimension; unscaled, softmax saturates to one-hot and gradients die
# (matters at d=128, harmless-but-correct at d=2 — do it anyway).
#
# CAUSAL MASK: token i may not attend to j > i — the future doesn't exist
# at generation time. Set those scores to -inf BEFORE softmax:
#   mask = np.triu(np.ones((T, T), dtype=bool), k=1)   # j > i
#   scores = np.where(mask, -np.inf, scores)
#
# SOFTMAX per row, the numerically stable way (subtract the row max first —
# this is exactly run.c's softmax()):
#   m = r.max(); e = np.exp(r - m); w = e / e.sum()
# exp(-inf) = 0, so masked (future) positions get exactly weight 0.
#
# Loop over the 2 heads — clearer than vectorizing, and mirrors the C.
#
# Checkpoint: att has shape (2,3,3); every row sums to 1; every upper
# triangle is exactly 0.
# ===========================================================================


att = np.zeros((H, T, T), dtype=np.float32)
for h in range(H):
    q_h = q_heads[:, h, :]
    k_h = k_heads[:, h, :]

    scores = q_h @ k_h.T / np.sqrt(HEAD_DIM)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(mask, -np.inf, scores)
    m = scores.max(axis=1, keepdims=True)
    e = np.exp(scores - m)
    att[h] = e /e.sum(axis=-1, keepdims=True)

print("step 3 — attention weights:", att.shape)
print(att)
print("row sums:\n", att.sum(axis=-1))

# ===========================================================================
# EXERCISE 4 — weighted values, output projection, residual. You write this.
#
# Per head:  out_h = att[h] @ v_heads[:, h, :]      (3,3)@(3,2) = (3,2)
# Token i's new vector = attention-weighted blend of the VALUE vectors of
# the tokens it looked at. THIS is where information crosses between tokens.
#
# Glue the heads back together, side by side:  (3,2)+(3,2) -> (3,4)
#   np.concatenate([out_0, out_1], axis=1)   # or (T,H,HEAD_DIM).reshape(T,D)
#
# Output projection mixes the heads' findings back to residual width:
#   proj = attn_out @ Wo                          (3,4)@(4,4) = (3,4)
#
# RESIDUAL CONNECTION:  x = x + proj
# Add to the ORIGINAL x from step 0 — NOT x_norm. The residual stream is a
# shared channel: each sub-block READS from it (through a norm) and WRITES
# back by addition. Nothing is ever overwritten, so early information
# survives to the end (and gradients survive during training).
#
# Checkpoint: token 0 attends 100% to itself in every head (one-hot row),
# so attn_out[0, :2] must equal v_heads[0, 0]:
#   print(attn_out[0, :HEAD_DIM], v_heads[0, 0])
# ===========================================================================

# Per head
out_0 = att[0] @ v_heads[:, 0 , :]
out_1 = att[1] @ v_heads[:, 1, :]

attn_out = np.concatenate([out_0, out_1], axis=1)
x = x + attn_out @ W["Wo"]

print("attnt out shape:", attn_out.shape)
print(attn_out)
print(x)

# ===========================================================================
# EXERCISE 5 — the FFN (feed-forward network). You write this.
#
# Attention = COMMUNICATION (tokens mix). FFN = COMPUTATION (each token
# processes its own vector alone). The FFN is ~2/3 of a real transformer's
# parameters — and in Phase 4, THIS is the piece MoE replaces with experts.
#
# Structure (per token, no mixing):
#   x2 = layernorm(x, W["ln2_g"])      # norm the UPDATED stream (post-attn!)
#   h  = x2 @ W1                        # expand:  (3,4)@(4,8) = (3,8)
#   h  = relu(h) = np.maximum(0, h)     # nonlinearity — without it the two
#                                       # matmuls would collapse into one
#   x  = x + h @ W2                     # contract: (3,8)@(8,4) = (3,4), then
#                                       # the second residual add
# Why expand-then-contract? The wide hidden layer gives each token room to
# form richer features before compressing back to the stream width.
# (Delta #4: GPT-2 uses GELU, run.c uses SwiGLU w/ 3 matrices — ReLU here
# keeps the numbers hand-checkable.)
#
# Checkpoints:
#   1. h has shape (3,8) and NO negative values (ReLU zeroed them)
#   2. count the zeros:  print("dead activations:", (h == 0).sum(), "of", h.size)
#      — typically ~half. That per-token sparsity is a toy echo of the
#      structured sparsity MoE exploits. Remember it for Phase 4.
# ===========================================================================

x2 = layernorm(x, W["ln2_g"])
h = x2 @ W["W1"]
h = np.maximum(0, h)
x = x + h @ W["W2"]

print("h:", h.shape)
print(h)
print("dead activations:", (h == 0).sum(), "of", h.size)
print("final x:", x.shape)
print(x)

# ===========================================================================
# EXERCISE 6 — final norm, logits, next token. You write this. (The payoff.)
#
# After the last block, the stream gets one final norm, then is projected
# to VOCAB width — the LOGITS, one raw score per possible next token:
#   xf     = layernorm(x, W["lnf_g"])     # (3,4)
#   logits = xf @ W["unemb"].T            # (3,4)@(4,8) = (3,8)
#
# Only the LAST row matters for prediction: token 7's row scores what
# comes next. (The other rows' logits are used in TRAINING, compared
# against the actual next tokens — that's the loss function. Phase 5.)
#
#   probs      = softmax(logits[-1])      # stable softmax, you own this now
#   next_token = np.argmax(probs)         # greedy decoding: take the max
# (Real systems often SAMPLE from probs instead — argmax makes every run
# identical. Temperature/top-k are variations on sampling. Phase 1 REPL
# will do both.)
#
# Checkpoints:
#   1. logits shape (3,8), probs sum to 1.0
#   2. print(next_token, "with prob", probs[next_token]) — the model's
#      first-ever prediction (of random weights — expect garbage, the
#      MECHANICS are what you're verifying)
#
# CAPSTONE (optional, ~5 lines): append next_token to `tokens`, re-run the
# whole forward pass, pick again. That loop — forward, pick, append,
# repeat — is autoregressive generation. In C, over mmap'd weights, with a
# tokenizer on the front: that IS Phase 1's REPL.
# ===========================================================================

xf = layernorm(x, W["lnf_g"])
logits = xf @ W["unemb"].T

def softmax(r):
    m = r.max()
    e = np.exp(r - m)
    return e / e.sum()

probs = softmax(logits[-1])
next_token = np.argmax(probs)

print("logits (last row):", logits[-1])
print("probs:", probs)
print(next_token, "with prob", probs[next_token])