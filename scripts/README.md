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

2. Strip each checkpoint (outputs go to `whisqa/_models/` by default):
   ```
   python scripts/strip_whisper_weights.py --input checkpoints/single_head_model.pt
   python scripts/strip_whisper_weights.py --input checkpoints/multi_head_model.pt
   ```

3. Commit the stripped files — they are a few MB so regular git is fine,
   no LFS needed:
   ```
   git add whisqa/_models/single_head_model.pt whisqa/_models/multi_head_model.pt
   git commit -m "Add stripped head-only checkpoints"
   ```

The stripped checkpoints are bundled into the wheel as package data and
loaded at inference time with `strict=False` so missing Whisper encoder
keys are silently ignored.
