# MemeAgent training

MemeAgent is a decision verifier fine-tuned on the simulation's logs. Given a meme image, the
agent's BDI reasoning and the platform rules, it outputs a structured verdict:
`PASS`, `FLAG`, `BLOCK` or `HONEST_BLOCK`.

The base model is Qwen3-VL (alias `qwen3_vl_2b`), trained with QLoRA (4-bit, LoRA r=32, alpha=64).
Training needs a GPU and the requirements in `train/requirements.txt`. Weights are not distributed.

## Files

| File | Purpose |
|---|---|
| `verifier_dataset_builder.py` | builds the dataset from run logs in `results/` and images in `memes/` |
| `verifier_train.py` | trains the adapter |
| `verifier_eval.py` | evaluates an adapter on the test split |
| `rule_library.py` | the platform rules shown to the verifier |
| `model_registry.py` | supported base models |

## Usage

Run from the repository root, after producing run logs with the benchmark.

```bash
# inspect the dataset only
python train/verifier_train.py --model qwen3_vl_2b --output-dir train_outputs/verifier_v0 --build-only

# train
python train/verifier_train.py --model qwen3_vl_2b --output-dir train_outputs/verifier_v0

# evaluate
python train/verifier_eval.py --adapter train_outputs/verifier_v0 \
    --test-data train_outputs/verifier_v0/datasets/test.jsonl
```

Use `--no-4bit` where bitsandbytes is unavailable, and `--smoke-test` for a tiny verification run.
