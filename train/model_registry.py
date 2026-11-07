from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    family: str
    default_dtype: str = "bfloat16"
    recommended_load_in_4bit: bool = True
    notes: str = ""


# Gemma model cards point to the non-instruction variants, but the HF examples
# for chat-style multimodal use load the `-it` checkpoints. We keep the CLI
# override path open for raw IDs, but default the aliases to the instruct
# versions because this repo is training a structured critic/policy model.
MODEL_SPECS: Dict[str, ModelSpec] = {
    "gemma4_e2b": ModelSpec(
        alias="gemma4_e2b",
        model_id="google/gemma-4-E2B-it",
        family="gemma4",
        notes="Default alias for MemeAgent on Gemma 4 E2B.",
    ),
    "gemma4_e4b": ModelSpec(
        alias="gemma4_e4b",
        model_id="google/gemma-4-E4B-it",
        family="gemma4",
        notes="Default alias for MemeAgent on Gemma 4 E4B.",
    ),
    "qwen3_vl_2b": ModelSpec(
        alias="qwen3_vl_2b",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        family="qwen3_vl",
        notes="Compact Qwen3-VL baseline.",
    ),
    "qwen3_vl_4b": ModelSpec(
        alias="qwen3_vl_4b",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        family="qwen3_vl",
        notes="Stronger Qwen3-VL critic backbone.",
    ),
}


def infer_family(model_id: str) -> str:
    lowered = model_id.lower()
    if "qwen3-vl" in lowered:
        return "qwen3_vl"
    if "gemma-4" in lowered:
        return "gemma4"
    raise ValueError(
        f"Could not infer model family from {model_id!r}. "
        "Use a known alias or add support in train/model_registry.py."
    )


def resolve_model_spec(model_name_or_id: str) -> ModelSpec:
    if model_name_or_id in MODEL_SPECS:
        return MODEL_SPECS[model_name_or_id]
    return ModelSpec(
        alias=model_name_or_id.replace("/", "_").replace("-", "_"),
        model_id=model_name_or_id,
        family=infer_family(model_name_or_id),
    )


def recommended_target_modules(model_family: str) -> list[str]:
    # Shared conservative list that works for both Gemma/Qwen transformer blocks.
    common = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if model_family in {"gemma4", "qwen3_vl"}:
        return common
    return common


def load_processor_and_model(
    spec: ModelSpec,
    *,
    load_in_4bit: bool = False,
    device_map: str = "auto",
    gradient_checkpointing: bool = True,
    torch_dtype_override: Optional[str] = None,
):
    import torch
    from transformers import AutoProcessor

    dtype_name = torch_dtype_override or spec.default_dtype
    dtype = getattr(torch, dtype_name)

    quantization_config = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes/quantization support is unavailable. "
                "Install the training requirements on the RTX machine."
            ) from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    processor = AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=True)

    if spec.family == "qwen3_vl":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Your installed transformers build does not expose "
                "Qwen3VLForConditionalGeneration. Upgrade transformers first."
            ) from exc
        model_cls = Qwen3VLForConditionalGeneration
    elif spec.family == "gemma4":
        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForCausalLM
            model_cls = AutoModelForCausalLM
    else:
        raise ValueError(f"Unsupported family: {spec.family}")

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = dtype

    model = model_cls.from_pretrained(spec.model_id, **model_kwargs)
    model.config.use_cache = False

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    return processor, model

