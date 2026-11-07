"""
train/verifier_eval.py
=======================
Evaluate the trained MemeAgent Decision Verifier on the test split.

Runs inference on every example in test.jsonl, parses the predicted verdict JSON,
and computes:
  - Per-label accuracy
  - Confusion matrix
  - Safety recall (critical: did we catch all BLOCK + HONEST_BLOCK?)
  - Sample outputs for qualitative inspection

Usage
-----
python train/verifier_eval.py \
    --adapter train_outputs/verifier_v0 \
    --test-data train_outputs/verifier_v0/datasets/test.jsonl

# Limit to first N examples
python train/verifier_eval.py --adapter train_outputs/verifier_v0 --limit 20

# Show full model output for each example
python train/verifier_eval.py --adapter train_outputs/verifier_v0 --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verifier_train import VerifierCollator


LABELS = ["PASS", "FLAG", "BLOCK", "HONEST_BLOCK"]
UNSAFE_LABELS = {"BLOCK", "HONEST_BLOCK"}


# ── Inference helpers ─────────────────────────────────────────────────────────

def load_model(adapter_dir: str):
    import torch
    from transformers import AutoProcessor
    from peft import PeftModel

    adapter_path = ROOT / adapter_dir
    print(f"[eval] Loading processor from {adapter_path}")
    processor = AutoProcessor.from_pretrained(str(adapter_path), trust_remote_code=True)

    from model_registry import resolve_model_spec, load_processor_and_model
    spec = resolve_model_spec("qwen3_vl_2b")

    print(f"[eval] Loading base model + adapter...")
    _, base_model = load_processor_and_model(
        spec,
        load_in_4bit=torch.cuda.is_available(),
        device_map="auto",
        gradient_checkpointing=False,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    print(f"[eval] Model ready on {next(model.parameters()).device}")
    return processor, model


def generate_verdict(
    model, processor, example: dict, max_new_tokens: int = 200
) -> str:
    """Run one inference step. Returns raw model output string."""
    import torch
    from PIL import Image

    collator = VerifierCollator.__new__(VerifierCollator)
    collator.processor  = processor
    collator.max_length = 2048
    collator._image_cache = {}
    collator._response_start_ids = None  # not needed for generation

    _MAX_PX = 336 * 336
    if hasattr(processor, "image_processor"):
        ip = processor.image_processor
        for attr in ("max_pixels", "max_image_size"):
            if hasattr(ip, attr):
                setattr(ip, attr, _MAX_PX)

    # Build prompt-only messages (no target)
    messages = [
        {"role": "system",    "content": [{"type": "text", "text": collator.SYSTEM_PROMPT}]},
        {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": example["prompt"]}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    img = Image.open(example["image_path"]).convert("RGB")
    if max(img.width, img.height) > 336:
        img.thumbnail((336, 336), Image.LANCZOS)

    inputs = processor(
        text=[text], images=[img],
        return_tensors="pt", padding=False, truncation=False,
    )
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_label(raw: str) -> str | None:
    """Extract safety_label from model output JSON string."""
    try:
        data = json.loads(raw)
        return data.get("safety_label")
    except json.JSONDecodeError:
        pass
    # Fallback: regex search
    m = re.search(r'"safety_label"\s*:\s*"([^"]+)"', raw)
    return m.group(1) if m else None


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
    adapter_dir: str,
    test_data_path: str,
    limit: int | None,
    verbose: bool,
) -> None:
    processor, model = load_model(adapter_dir)

    test_path = ROOT / test_data_path
    with open(test_path, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f if l.strip()]
    if limit:
        examples = examples[:limit]

    print(f"\n[eval] Running inference on {len(examples)} examples...\n")

    true_labels:  list[str] = []
    pred_labels:  list[str] = []
    raw_outputs:  list[str] = []
    parse_fails:  int = 0

    for i, ex in enumerate(examples):
        true_label = ex["safety_label"]
        raw = generate_verdict(model, processor, ex)
        pred = parse_label(raw)

        if pred not in LABELS:
            parse_fails += 1
            pred = "PARSE_FAIL"

        true_labels.append(true_label)
        pred_labels.append(pred)
        raw_outputs.append(raw)

        marker = "✓" if pred == true_label else "✗"
        print(f"  [{i+1:3d}/{len(examples)}] true={true_label:12s} pred={pred:12s} {marker}  "
              f"| model={ex.get('model_source','?'):15s} tier={ex.get('meme_tier','?')}")

        if verbose:
            print(f"         OUTPUT: {raw[:300]}")
            print()

    # ── Metrics ───────────────────────────────────────────────────────────────
    n = len(examples)
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct / n

    # Per-label precision / recall
    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)
    for t, p in zip(true_labels, pred_labels):
        if t == p:
            label_tp[t] += 1
        else:
            label_fn[t] += 1
            if p != "PARSE_FAIL":
                label_fp[p] += 1

    # Safety recall: did we catch all unsafe (BLOCK + HONEST_BLOCK)?
    unsafe_true = [l in UNSAFE_LABELS for l in true_labels]
    unsafe_pred = [l in UNSAFE_LABELS for l in pred_labels]
    safety_tp = sum(t and p for t, p in zip(unsafe_true, unsafe_pred))
    safety_fn = sum(t and not p for t, p in zip(unsafe_true, unsafe_pred))
    safety_recall = safety_tp / max(1, safety_tp + safety_fn)

    # Confusion matrix
    conf: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(true_labels, pred_labels):
        conf[t][p] += 1

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VERIFIER EVALUATION — {Path(adapter_dir).name}")
    print(f"{'='*60}")
    print(f"  Examples    : {n}")
    print(f"  Overall Acc : {accuracy:.1%}  ({correct}/{n})")
    print(f"  Safety Recall (BLOCK+HONEST_BLOCK): {safety_recall:.1%}  "
          f"({safety_tp}/{safety_tp+safety_fn} unsafe examples caught)")
    print(f"  JSON parse fails: {parse_fails}")

    print(f"\n  Per-label breakdown:")
    print(f"  {'Label':15s} {'TP':>4} {'FP':>4} {'FN':>4}  {'Prec':>6}  {'Rec':>6}")
    print(f"  {'-'*52}")
    for label in LABELS:
        tp = label_tp[label]
        fp = label_fp[label]
        fn = label_fn[label]
        prec = tp / max(1, tp + fp)
        rec  = tp / max(1, tp + fn)
        print(f"  {label:15s} {tp:>4} {fp:>4} {fn:>4}  {prec:>6.1%}  {rec:>6.1%}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    all_preds = sorted({p for p in pred_labels})
    row_label = "TRUE \\ PRED"
    header = f"  {row_label:15s}" + "".join(f"{p:>14s}" for p in all_preds)
    print(header)
    for t in LABELS:
        row = f"  {t:15s}" + "".join(f"{conf[t][p]:>14d}" for p in all_preds)
        print(row)

    # ── Sample outputs ────────────────────────────────────────────────────────
    print(f"\n  Sample outputs (first 3 HONEST_BLOCK examples):")
    shown = 0
    for i, (ex, raw, pred, true) in enumerate(zip(examples, raw_outputs, pred_labels, true_labels)):
        if true == "HONEST_BLOCK" and shown < 3:
            print(f"\n  [{i+1}] Instance: {ex.get('instance')} | Model: {ex.get('model_source')}")
            print(f"       True: {true} | Pred: {pred}")
            print(f"       Output: {raw[:400]}")
            shown += 1

    print(f"\n{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate MemeAgent Decision Verifier")
    p.add_argument("--adapter",    default="train_outputs/verifier_v0",
                   help="Path to trained adapter directory (relative to project root)")
    p.add_argument("--test-data",  default="train_outputs/verifier_v0/datasets/test.jsonl",
                   help="Path to test JSONL (relative to project root)")
    p.add_argument("--limit",  type=int, default=None,
                   help="Limit to first N examples")
    p.add_argument("--verbose", action="store_true",
                   help="Print full model output for each example")
    args = p.parse_args()

    evaluate(
        adapter_dir=args.adapter,
        test_data_path=args.test_data,
        limit=args.limit,
        verbose=args.verbose,
    )
