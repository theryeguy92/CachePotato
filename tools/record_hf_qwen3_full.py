#!/usr/bin/env python3
"""record_hf_qwen3_full.py — the FULL 48-layer baseline for Qwen3-30B-A3B. Run on a big-RAM box.

Records, from the published checkpoint:
  * logits for a fixed set of prompts (argmax agreement is the Tier-C quality target);
  * per-layer intermediates for ONE short prompt (the block-by-block debugging target);
  * the ROUTING TRACE — top-8 selections and weights, every layer, every token. This is the
    analogue of gpt2-moe's `sel`: the artifact a Tier-B layout must leave unchanged, and the
    strongest single check this port will have.

MEMORY. The weights at bf16 are 61 GB; at fp32 they would be 122 GB. This loads bf16 and therefore
computes in bf16 — which is the honest baseline for a machine that cannot hold the fp32 model. The
intermediates for a 5-token prompt are a few MB because T is small; it is the WEIGHTS that decide
which machine can run this, not the recording.

RECORD ONCE. If a gate fails later, fix the code — do not regenerate this file to make a test pass.
The script runs the model twice and refuses to write unless both runs agree bit for bit.
"""
import argparse, json, os, sys
# Must precede any CUDA initialisation. The weight-loading path allocates many short-lived
# tensors while stacking per-expert checkpoint tensors into the runtime's fused layout, which is
# exactly the pattern that fragments the default allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np
import torch

INTERMEDIATE_SITES = ("input_layernorm", "self_attn.q_proj", "self_attn.k_proj",
                      "self_attn.v_proj", "self_attn.q_norm", "self_attn.k_norm",
                      "self_attn", "post_attention_layernorm", "mlp.gate", "mlp")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1929, the stock market",
    "Explain why the sky is blue in one sentence.",
]

def capture(model, tok, prompt, layers, want_intermediates):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    rec, sites = {}, []
    handles = []
    def hook(name):
        def fn(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                rec[name] = o.detach().float().to("cpu").numpy().copy()
        return fn
    if want_intermediates:
        for i in range(layers):
            for sub in INTERMEDIATE_SITES:
                mod = model.model.layers[i]
                ok = True
                for part in sub.split("."):
                    if not hasattr(mod, part): ok = False; break
                    mod = getattr(mod, part)
                if not ok:
                    continue
                handles.append(mod.register_forward_hook(hook(f"L{i}.{sub}")))
                sites.append(f"L{i}.{sub}")
    with torch.no_grad():
        out = model(ids)
    for h in handles:
        h.remove()
    logits = out.logits.float().to("cpu").numpy()
    # routing trace derived from the router logits with Qwen3's documented rule
    if want_intermediates:
        for i in range(layers):
            key = f"L{i}.mlp.gate"
            if key not in rec:
                continue
            lg = torch.from_numpy(rec[key]).float()
            p = torch.softmax(lg, dim=-1)
            tv, ti = torch.topk(p, 8, dim=-1)
            rec[f"L{i}.top8"] = ti.numpy().astype(np.int64)
            rec[f"L{i}.top8_w"] = (tv / tv.sum(-1, keepdim=True)).numpy().astype(np.float32)
    # ids live on the model's device (required once the model is split), so bring them back
    return ids[0].detach().cpu().numpy().astype(np.int64), logits, rec, sites

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="weights/qwen3-30b-a3b")
    ap.add_argument("--out", default="data/traces/qwen3_30b_a3b_full.npz")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--intermediate-prompt", default=PROMPTS[0])
    ap.add_argument("--device-map", choices=["auto", "cpu", "spread", "cuda"],
                    default="auto", help="auto decides from the measured RAM/VRAM")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(args.model)

    # PLACEMENT. bf16 is 61.1 GB. A box with 61 GB of RAM has 58 GB available, so the model
    # cannot live in RAM — but it does not need to: 48 GB of VRAM plus RAM holds it, if it is
    # SPREAD. Deciding this from the actual numbers rather than trying and reading the OOM.
    idx = json.load(open(Path(args.model) / "model.safetensors.index.json"))
    need = idx["metadata"]["total_size"]
    # MemAvailable, NOT free pages. SC_AVPHYS_PAGES is MemFree, which ignores reclaimable page
    # cache: a box that has just read 57 GB of weights reports ~21 GB free while 58 GB is
    # available, and the script refused a model that fits easily. On Linux, page cache IS memory
    # you can have back.
    def avail_ram():
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        except OSError:
            pass
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    avail = avail_ram()
    vram = 0
    if torch.cuda.is_available():
        # SUM EVERY DEVICE. mem_get_info() defaults to the CURRENT device, so a two-GPU box was
        # measured as if it had one — 25 GB instead of 48 GB — and the script refused a model that
        # actually fits. Reported per device so the number can be checked against nvidia-smi.
        for d in range(torch.cuda.device_count()):
            free_b, total_b = torch.cuda.mem_get_info(d)
            vram += free_b
            print(f"  cuda:{d} {torch.cuda.get_device_name(d)}: "
                  f"free {free_b/1e9:.1f} / {total_b/1e9:.1f} GB", flush=True)
    print(f"weights {need/1e9:.1f} GB | RAM available {avail/1e9:.1f} GB | "
          f"VRAM free {vram/1e9:.1f} GB", flush=True)
    place = args.device_map
    if place == "auto":
        place = "cpu" if need < avail * 0.92 else "spread"
    if place in ("spread", "cuda") and not torch.cuda.is_available():
        raise SystemExit(f"model needs {need/1e9:.1f} GB but only {avail/1e9:.1f} GB RAM is "
                         "available and there is no GPU — cannot place it without changing "
                         "what the baseline is (quantizing), which needs a decision, not a retry")
    if place == "spread" and need > avail + vram * 0.9:
        raise SystemExit(f"cannot fit {need/1e9:.1f} GB in {avail/1e9:.1f} GB RAM + "
                         f"{vram/1e9:.1f} GB VRAM")
    print(f"placing: {place}", flush=True)
    extra = {}
    if place in ("spread", "cuda"):
        # device_map="auto" fills every GPU to the last byte, and then the loader needs somewhere
        # to put its TRANSIENTS: transformers 5.x stacks the checkpoint's per-expert `gate_proj`
        # and `up_proj` into the runtime's fused `gate_up_proj`, on-device. With ~180 MiB free that
        # is an OOM half-way through layer 37 — which reads like "the model doesn't fit" but is
        # really "the loader had no scratch space". Reserve headroom per device explicitly.
        max_mem = {}
        if place == "spread":
            for d in range(torch.cuda.device_count()):
                free_b, _ = torch.cuda.mem_get_info(d)
                gib = free_b / 2**30
                max_mem[d] = f"{max(1.0, gib - 3.0):.0f}GiB"      # 3 GiB scratch per card
            max_mem["cpu"] = f"{max(1.0, avail / 2**30 - 8.0):.0f}GiB"
            print(f"max_memory: {max_mem}", flush=True)
        extra["device_map"] = "auto" if place == "spread" else "cuda:0"
        if max_mem:
            extra["max_memory"] = max_mem
    # transformers renamed the argument (torch_dtype -> dtype); accept either so the box does
    # not have to match the T480's version. Silently falling back to fp32 here would need 122 GB
    # and fail with an OOM that looks like a hardware limit rather than a keyword mismatch.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True, **extra)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True, **extra)
    if next(model.parameters()).dtype != torch.bfloat16:
        raise SystemExit(f"weights came back as {next(model.parameters()).dtype}, not bfloat16 — "
                         "refusing to record a baseline at the wrong precision")
    model.eval()
    L = args.layers or model.config.num_hidden_layers
    print(f"loaded: {L} layers, dtype {next(model.parameters()).dtype}", flush=True)

    # pass 1: intermediates + routing trace for one short prompt
    ids, logits, rec, sites = capture(model, tok, args.intermediate_prompt, L, True)
    print(f"  intermediates: {len(sites)} sites, {len(rec)} arrays", flush=True)
    trace = {f"L{i}.top8": rec[f"L{i}.top8"] for i in range(L) if f"L{i}.top8" in rec}
    if len(trace) != L:
        raise SystemExit(f"routing trace incomplete: {len(trace)}/{L} layers")

    # pass 2: determinism. Same prompt, same weights — must agree bit for bit.
    ids2, logits2, rec2, _ = capture(model, tok, args.intermediate_prompt, L, True)
    if not (np.array_equal(ids, ids2) and np.array_equal(logits, logits2)):
        raise SystemExit("NOT DETERMINISTIC: two runs of the same prompt differ. "
                         "A baseline that is not reproducible is not a baseline.")
    print("  determinism: two runs bit-identical", flush=True)

    # all prompts: logits only
    ids_by_prompt, logits_by_prompt = {"p0": ids}, {"p0": logits}
    for k, p in enumerate(PROMPTS[1:], 1):
        i2, l2, _, _ = capture(model, tok, p, L, False)
        ids_by_prompt[f"p{k}"] = i2; logits_by_prompt[f"p{k}"] = l2
        print(f"  prompt p{k}: {len(i2)} tokens, argmax {int(l2[0, -1].argmax())}", flush=True)

    payload = dict(rec)
    payload["logits_p0"] = logits
    payload["input_ids_p0"] = ids
    for k in ids_by_prompt:
        payload[f"input_ids_{k}"] = ids_by_prompt[k]
        payload[f"logits_{k}"] = logits_by_prompt[k]
    payload["prompts"] = np.array(json.dumps(PROMPTS))
    payload["meta"] = np.array(json.dumps({
        "model": "Qwen/Qwen3-30B-A3B", "dtype": "bfloat16", "layers": L,
        "intermediate_prompt": args.intermediate_prompt,
        "note": "baseline: logits, per-layer intermediates, routing trace. Do not regenerate."}))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB, {len(payload)} arrays)")
    print("B3 COMPLETE")
    return 0

if __name__ == "__main__":
    sys.exit(main())
