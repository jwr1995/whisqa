# scripts/

Auxiliary scripts for maintaining the WhiSQA model checkpoints.
These are **not** part of the installable Python package.

## strip_whisper_weights.py

Strips the frozen Whisper encoder weights from the full training checkpoints
before uploading them to HuggingFace Hub for distribution.

**Why?** The full checkpoints saved during training include the entire
`openai/whisper-small` encoder (~240 MB, frozen). Since these weights are
already available on HuggingFace Hub and are downloaded automatically at
inference time, bundling them in the distribution checkpoint is wasteful.
Stripping reduces each file from ~350 MB to a few MB.

**Workflow:**

1. Pull the full checkpoints via git-lfs:
   ```
   git lfs pull
   ```

2. Strip each checkpoint:
   ```
   python scripts/strip_whisper_weights.py \
       --input  checkpoints/single_head_model.pt \
       --output checkpoints/single_head_stripped.pt

   python scripts/strip_whisper_weights.py \
       --input  checkpoints/multi_head_model.pt \
       --output checkpoints/multi_head_stripped.pt
   ```

3. Upload the stripped files to the `leto19/whisqa` HuggingFace Hub repo
   as `single_head_model.pt` and `multi_head_model.pt`:
   ```
   pip install huggingface_hub
   huggingface-cli upload leto19/whisqa checkpoints/single_head_stripped.pt single_head_model.pt
   huggingface-cli upload leto19/whisqa checkpoints/multi_head_stripped.pt  multi_head_model.pt
   ```

The stripped checkpoints are loaded with `strict=False` in the package so
missing Whisper encoder keys are silently ignored.
