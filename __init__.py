import hashlib
import gc
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image


_MODEL_CACHE: Dict[Tuple[str, str], Tuple[object, object]] = {}
_LOCK_DIR = Path(__file__).with_name("prompt_locks")
_DEFAULT_NEGATIVE = (
    "change the source, change colors, relight the scene, alter the subject, "
    "alter the background, sharp reflection, strong reflection, invented details, "
    "duplicate body, new objects, black blob, dirty stain, hard shadow, "
    "overpainted area, color shift, flicker, AI artifacts"
)


def _tensor_to_pil(image):
    if image.ndim == 4:
        image = image[0]
    array = (image.detach().cpu().clamp(0, 1).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def _load_model(model_id: str, dtype: str):
    key = (model_id, dtype)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForMultimodalLM as AutoModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelClass

    torch_dtype = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelClass.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()
    _MODEL_CACHE[key] = (processor, model)
    return processor, model


def _release_gemma_vram():
    _MODEL_CACHE.clear()
    gc.collect()

    try:
        import comfy.model_management

        comfy.model_management.soft_empty_cache(force=True)
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def _generate_text(image, instruction, model_id, max_new_tokens, temperature, dtype):
    processor, model = _load_model(model_id, dtype)
    pil_image = _tensor_to_pil(image)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": instruction},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    kwargs = {"max_new_tokens": max_new_tokens}
    if temperature <= 0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = temperature

    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)

    response = processor.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=False)
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(response)
        response = parsed[-1] if isinstance(parsed, (list, tuple)) else parsed
    return str(response).strip()


def _lock_path(unique_id):
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(unique_id or "default"))
    return _LOCK_DIR / f"{safe}.json"


def _read_lock(unique_id):
    path = _lock_path(unique_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    positive = str(data.get("positive_prompt", "")).strip()
    negative = str(data.get("negative_prompt", "")).strip()
    if not positive:
        return None
    return positive, negative or _DEFAULT_NEGATIVE


def _write_lock(unique_id, positive, negative):
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    _lock_path(unique_id).write_text(
        json.dumps(
            {"positive_prompt": positive.strip(), "negative_prompt": negative.strip()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _parse_prompt_pair(text):
    text = str(text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            positive = str(data.get("positive_prompt") or data.get("positive") or "").strip()
            negative = str(data.get("negative_prompt") or data.get("negative") or "").strip()
            if positive:
                return positive, negative or _DEFAULT_NEGATIVE
        except json.JSONDecodeError:
            pass

    match = re.search(r"(?is)positive\s*:?\s*(.+?)(?:\n\s*negative\s*:?\s*|\Z)(.*)", text)
    if match:
        positive = match.group(1).strip(" \n\r\t-")
        negative = match.group(2).strip(" \n\r\t-") or _DEFAULT_NEGATIVE
        return positive, negative
    return text, _DEFAULT_NEGATIVE


class Gemma4ReferencePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Describe this reference image as a concise cinematic "
                            "Wan/VACE video generation prompt. Focus on subject, "
                            "materials, lighting, camera, environment, and motion-ready "
                            "visual details. Output one English paragraph only."
                        ),
                    },
                ),
                "model_id": (
                    ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it"],
                    {"default": "google/gemma-4-E2B-it"},
                ),
                "max_new_tokens": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "unload_after_generate": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "caption"
    CATEGORY = "Gemma4"
    DESCRIPTION = "Gemma 4 image+text to text prompt generator for reference-image prompt routing."

    def caption(self, image, instruction, model_id, max_new_tokens, temperature, dtype, unload_after_generate):
        try:
            return (_generate_text(image, instruction, model_id, max_new_tokens, temperature, dtype),)
        finally:
            if unload_after_generate:
                _release_gemma_vram()

    @classmethod
    def IS_CHANGED(cls, image, instruction, model_id, max_new_tokens, temperature, dtype, unload_after_generate):
        h = hashlib.sha1()
        h.update(str(instruction).encode("utf-8"))
        h.update(str(model_id).encode("utf-8"))
        h.update(str(max_new_tokens).encode("utf-8"))
        h.update(str(temperature).encode("utf-8"))
        h.update(str(dtype).encode("utf-8"))
        h.update(str(unload_after_generate).encode("utf-8"))
        return h.hexdigest()


class Gemma4ReferencePromptPairLock:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "positive_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Describe this reference image as a concise Wan/VACE VFX positive prompt. "
                            "Preserve the original source subject and composition. Focus on visible "
                            "materials, lighting, floor contact, reflection, shadow softness, camera, "
                            "and environment. Keep it one practical English prompt."
                        ),
                    },
                ),
                "negative_instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": (
                            "Write a stable Wan/VACE negative prompt for this same reference. "
                            "List what must not change: source identity, colors, background, body shape, "
                            "new objects, hard shadows, flicker, color shift, dirty blobs, invented detail, "
                            "overpainted areas, and AI artifacts."
                        ),
                    },
                ),
                "lock_prompt": ("BOOLEAN", {"default": False}),
                "refresh_lock": ("BOOLEAN", {"default": False}),
                "model_id": (
                    ["google/gemma-4-E2B-it", "google/gemma-4-E4B-it"],
                    {"default": "google/gemma-4-E2B-it"},
                ),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "dtype": (["auto", "bf16", "fp16", "fp32"], {"default": "auto"}),
                "unload_after_generate": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "caption_pair"
    CATEGORY = "Gemma4"
    DESCRIPTION = "Gemma 4 reference-image positive/negative prompt pair with persistent lock."

    def caption_pair(
        self,
        image,
        positive_instruction,
        negative_instruction,
        lock_prompt,
        refresh_lock,
        model_id,
        max_new_tokens,
        temperature,
        dtype,
        unload_after_generate,
        unique_id=None,
    ):
        if lock_prompt and not refresh_lock:
            cached = _read_lock(unique_id)
            if cached:
                return cached

        instruction = (
            f"{positive_instruction}\n\n"
            f"{negative_instruction}\n\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            "{\"positive_prompt\":\"...\", \"negative_prompt\":\"...\"}"
        )
        try:
            text = _generate_text(image, instruction, model_id, max_new_tokens, temperature, dtype)
            positive, negative = _parse_prompt_pair(text)
            if lock_prompt or refresh_lock:
                _write_lock(unique_id, positive, negative)
            return positive, negative
        finally:
            if unload_after_generate:
                _release_gemma_vram()

    @classmethod
    def IS_CHANGED(
        cls,
        image,
        positive_instruction,
        negative_instruction,
        lock_prompt,
        refresh_lock,
        model_id,
        max_new_tokens,
        temperature,
        dtype,
        unload_after_generate,
        unique_id=None,
    ):
        if refresh_lock:
            return float("nan")
        if lock_prompt:
            cached = _read_lock(unique_id)
            if cached:
                return hashlib.sha1(("locked:" + "\n".join(cached)).encode("utf-8")).hexdigest()

        h = hashlib.sha1()
        for value in (
            positive_instruction,
            negative_instruction,
            model_id,
            max_new_tokens,
            temperature,
            dtype,
            lock_prompt,
            unload_after_generate,
        ):
            h.update(str(value).encode("utf-8"))
        return h.hexdigest()


NODE_CLASS_MAPPINGS = {
    "Gemma4ReferencePrompt": Gemma4ReferencePrompt,
    "Gemma4ReferencePromptPairLock": Gemma4ReferencePromptPairLock,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemma4ReferencePrompt": "Gemma 4 Reference Prompt",
    "Gemma4ReferencePromptPairLock": "Gemma 4 Reference Prompt Pair Lock",
}
