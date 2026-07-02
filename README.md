# ComfyUI-Gemma4-ReferencePrompt

Gemma 4 reference-image prompt nodes for ComfyUI.

Nodes:

- `Gemma4ReferencePrompt`
- `Gemma4ReferencePromptPairLock`

The pair-lock node creates positive and negative prompts from one reference image and can reuse the locked result across repeated shots.

## Install

With ComfyUI Manager:

1. Open Manager.
2. Use "Install via Git URL".
3. Paste:

```text
https://github.com/MaikiOS/ComfyUI-Gemma4-ReferencePrompt
```

Manual install:

```powershell
cd ComfyUI/custom_nodes
git clone https://github.com/MaikiOS/ComfyUI-Gemma4-ReferencePrompt
python -m pip install -r ComfyUI-Gemma4-ReferencePrompt/requirements.txt
```

Then fully restart ComfyUI.

## Model

The node uses HuggingFace Transformers model id:

```text
google/gemma-4-E2B-it
```

It is not placed in `ComfyUI/models`. Transformers downloads it to the HuggingFace cache, usually:

```text
C:\Users\<USER>\.cache\huggingface\hub\models--google--gemma-4-E2B-it
```

If HuggingFace asks for access/token:

```powershell
huggingface-cli login
```

## Requirements

Gemma 4 needs `transformers` with Gemma4 classes. If ComfyUI errors with `Unrecognized processing class`, update:

```powershell
python -m pip install --upgrade "transformers>=5.5.0" accelerate
```

## Outputs

- `Gemma4ReferencePrompt` returns one text prompt.
- `Gemma4ReferencePromptPairLock` returns `positive_prompt` and `negative_prompt`.
- Lock mode stores generated prompts in `prompt_locks/` inside this node folder.

## Notes

This pack depends on a large HuggingFace multimodal model. The first run can take time while the model downloads and loads into VRAM/RAM.
