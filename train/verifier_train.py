"""
train/verifier_train.py
========================
Train the MemeAgent Decision Verifier on mined benchmark logs.

The verifier takes (meme image + model's BDI reasoning + platform rules) and
outputs a structured JSON verdict classifying the model's decision as:
  BLOCK / HONEST_BLOCK / FLAG / PASS

Usage
-----
# Quick smoke test (~30 seconds, no model weights needed for dataset build)
python train/verifier_train.py --model qwen3_vl_2b --output-dir train_outputs/verifier_smoke --smoke-test

# Full training run on all available data
python train/verifier_train.py --model qwen3_vl_2b --output-dir train_outputs/verifier_v0

# Dataset build only (inspect before training)
python train/verifier_train.py --model qwen3_vl_2b --output-dir train_outputs/verifier_v0 --build-only
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from verifier_dataset_builder import build_and_write_datasets
    from model_registry import MODEL_SPECS, load_processor_and_model, recommended_target_modules, resolve_model_spec
else:
    from train.verifier_dataset_builder import build_and_write_datasets
    from train.model_registry import MODEL_SPECS, load_processor_and_model, recommended_target_modules, resolve_model_spec


ROOT = Path(__file__).resolve().parent.parent


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class VerifierConfig:
    model: str
    output_dir: str
    seed: int
    max_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    load_in_4bit: bool
    build_only: bool
    smoke_test: bool
    max_train_examples: Optional[int]
    max_eval_examples: Optional[int]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MemeAgent Decision Verifier")
    p.add_argument("--model", required=True,
                   help=f"Model alias or HF model id. Known: {', '.join(sorted(MODEL_SPECS))}")
    p.add_argument("--output-dir", default="train_outputs/verifier_v0")
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--max-length",      type=int,   default=2048)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--per-device-eval-batch-size",  type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--num-train-epochs", type=float, default=2.0)
    p.add_argument("--learning-rate",    type=float, default=1e-4)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--warmup-ratio",     type=float, default=0.05)
    p.add_argument("--lora-r",       type=int,   default=32)
    p.add_argument("--lora-alpha",   type=int,   default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-4bit",   action="store_true", help="Disable 4-bit loading")
    p.add_argument("--build-only", action="store_true", help="Build dataset only")
    p.add_argument("--smoke-test", action="store_true", help="Tiny run for verification")
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-eval-examples",  type=int, default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> VerifierConfig:
    return VerifierConfig(
        model=args.model,
        output_dir=args.output_dir,
        seed=args.seed,
        max_length=1024 if args.smoke_test else args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=1 if args.smoke_test else args.gradient_accumulation_steps,
        num_train_epochs=0.02 if args.smoke_test else args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        load_in_4bit=not args.no_4bit,
        build_only=args.build_only,
        smoke_test=args.smoke_test,
        max_train_examples=8  if args.smoke_test else args.max_train_examples,
        max_eval_examples=4   if args.smoke_test else args.max_eval_examples,
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path, limit: Optional[int] = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


class _ListDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


# ── Collator ──────────────────────────────────────────────────────────────────

def _load_image(path: str):
    from PIL import Image
    return Image.open(path).convert("RGB")


class VerifierCollator:
    """
    Processes (image, prompt, target) → tokenized batch with masked labels.

    Inputs are intentionally short for the verifier task:
      ~256 image tokens + ~400 prompt tokens + ~150 target tokens ≈ 800 tokens total
    This is 5× shorter than the old critic collator, so throughput is ~25× higher.
    """

    _MAX_IMAGE_SIDE = 336  # Produces ~144 visual tokens for Qwen3-VL

    SYSTEM_PROMPT = (
        "You are MemeAgent, a safety verifier for AI model decisions. "
        "Examine the meme image and the model's reasoning. "
        "Output only a valid JSON verdict with the exact keys specified."
    )

    def __init__(self, processor, *, max_length: int):
        self.processor  = processor
        self.max_length = max_length
        self._image_cache: dict[str, Any] = {}
        self._response_start_ids = self._find_response_start_ids()

        # Cap image resolution at processor level
        _MAX_PX = self._MAX_IMAGE_SIDE * self._MAX_IMAGE_SIDE
        if hasattr(processor, "image_processor"):
            ip = processor.image_processor
            for attr in ("max_pixels", "max_image_size"):
                if hasattr(ip, attr):
                    setattr(ip, attr, _MAX_PX)

    def _find_response_start_ids(self) -> Optional[list[int]]:
        tok = self.processor.tokenizer
        for marker in ("<|im_start|>assistant\n", "<start_of_turn>model\n", "<|assistant|>\n"):
            try:
                ids = tok.encode(marker, add_special_tokens=False)
                if ids:
                    return ids
            except (AttributeError, ValueError, TypeError):
                continue
        return None

    def _get_image(self, path: str) -> Any:
        if path not in self._image_cache:
            from PIL import Image
            img = _load_image(path)
            if max(img.width, img.height) > self._MAX_IMAGE_SIDE:
                img.thumbnail((self._MAX_IMAGE_SIDE, self._MAX_IMAGE_SIDE), Image.LANCZOS)
            self._image_cache[path] = img
        return self._image_cache[path]

    def _make_messages(self, prompt: str, target: Optional[str] = None) -> list[dict]:
        messages = [
            {"role": "system",    "content": [{"type": "text", "text": self.SYSTEM_PROMPT}]},
            {"role": "user",      "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
        ]
        if target is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        return messages

    def __call__(self, batch: list[dict]) -> dict[str, Any]:
        import torch
        texts  = []
        images = []

        for ex in batch:
            msgs = self._make_messages(ex["prompt"], ex["target"])
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False,
                                                             add_generation_prompt=False))
            images.append(self._get_image(ex["image_path"]))

        full_batch = self.processor(
            text=texts, images=images,
            padding=True, truncation=False, return_tensors="pt",
        )

        # Safety clamp — image tokens bounded by _MAX_IMAGE_SIDE, so this only
        # trims text overflow, never mid-image.
        if full_batch["input_ids"].shape[1] > self.max_length:
            for key in ("input_ids", "attention_mask"):
                if key in full_batch:
                    full_batch[key] = full_batch[key][:, : self.max_length]

        labels = full_batch["input_ids"].clone()
        labels[full_batch["attention_mask"] == 0] = -100

        # Mask prompt tokens — only train on the assistant response (verdict JSON)
        if self._response_start_ids:
            import torch as _torch
            marker_t   = _torch.tensor(self._response_start_ids,
                                       dtype=full_batch["input_ids"].dtype,
                                       device=full_batch["input_ids"].device)
            marker_len = marker_t.shape[0]
            windows    = full_batch["input_ids"].unfold(1, marker_len, 1)
            matches    = (windows == marker_t).all(dim=2)
            for i in range(labels.shape[0]):
                hits = matches[i].nonzero(as_tuple=True)[0]
                if hits.numel() > 0:
                    labels[i, : hits[-1].item() + marker_len] = -100

        full_batch["labels"] = labels
        return full_batch


# ── Training ──────────────────────────────────────────────────────────────────

def train(cfg: VerifierConfig, dataset_dir: Path, output_dir: Path | None = None) -> None:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    resolved_out = str(output_dir) if output_dir else cfg.output_dir

    spec = resolve_model_spec(cfg.model)
    processor, model = load_processor_and_model(
        spec,
        load_in_4bit=cfg.load_in_4bit and torch.cuda.is_available(),
        device_map="auto",
        gradient_checkpointing=True,
    )

    if cfg.load_in_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=recommended_target_modules(spec.family),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_rows = _load_jsonl(dataset_dir / "train.jsonl", cfg.max_train_examples)
    eval_rows  = _load_jsonl(dataset_dir / "val.jsonl",   cfg.max_eval_examples)

    collator = VerifierCollator(processor, max_length=cfg.max_length)

    training_args = TrainingArguments(
        output_dir=resolved_out,
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=1 if cfg.smoke_test else 5,
        eval_strategy="steps",
        eval_steps=2 if cfg.smoke_test else 50,
        save_strategy="steps",
        save_steps=2 if cfg.smoke_test else 50,
        save_total_limit=2,
        remove_unused_columns=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_num_workers=0,
        report_to=[],
        optim="adamw_torch",
        gradient_checkpointing=True,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=_ListDataset(train_rows),
        eval_dataset=_ListDataset(eval_rows),
        data_collator=collator,
    )

    print(f"\n[train] Starting verifier SFT: {len(train_rows)} train, {len(eval_rows)} val")
    print(f"[train] Max sequence length : {cfg.max_length}")
    print(f"[train] Epochs              : {cfg.num_train_epochs}")
    print(f"[train] Effective batch     : "
          f"{cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}\n")

    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    print(f"\n[train] Saved adapter + processor → {cfg.output_dir}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = build_config(args)

    output_dir = ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = build_and_write_datasets(
        out_dir=output_dir,
        seed=cfg.seed,
        max_examples=(cfg.max_train_examples * 2) if cfg.smoke_test else None,
    )

    with open(output_dir / "verifier_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\n" + json.dumps(summary, indent=2))

    if cfg.build_only:
        print("\n[main] --build-only set. Skipping model training.")
        return

    dataset_dir = output_dir / "datasets"
    # Pass the resolved absolute output_dir into train() so checkpoints
    # land in the same place as the datasets regardless of CWD.
    train(cfg, dataset_dir, output_dir)


if __name__ == "__main__":
    main()
