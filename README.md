# whisqa: a PyPI package for the WhiSQA Non-Intrusive Speech Quality Prediction Model

[![PyPI](https://img.shields.io/pypi/v/whisqa)](https://pypi.org/project/whisqa/)

This project is a fork of the original WhiSQA project repackaged for PyPI.

WhiSQA is a non-intrusive MOS (ITU-T P.835) speech quality predictor.

Original repo: https://github.com/leto19/WhiSQA

## Improvements over the original

- **`pip install whisqa`** — fully installable package; no manual cloning or dependency wrangling.
- **Bundled weights** — the frozen Whisper encoder weights (~350 MB per checkpoint) are stripped down to head-only files of 11 MB and 13 MB that ship inside the wheel. The [strip script](scripts/strip_whisper_weights.py) is kept in the repo for full reproducibility.
- **Flexible input** — `predict()` accepts file paths, NumPy arrays, and Torch tensors, making it easy to integrate into existing pipelines without writing audio to disk.
- **Auto-resampling** — any sample rate is accepted; audio is resampled to 16 kHz automatically with a suppressible warning.
- **CLI** — score a file from the terminal with `whisqa speech.wav`.
- **Clean API** — `predict()` and `load_model()` with type hints, docstrings, and 48 unit tests that run without model weights or network access.

## Install

```bash
pip install whisqa
```

> **First-run note:** `load_model()` downloads `openai/whisper-small` (~240 MB) from HuggingFace Hub and caches it in `~/.cache/huggingface/`. The WhiSQA head weights ship inside the package and require no download.

## Usage

```python
import whisqa

# File path — WAV or any format supported by libsndfile
whisqa.predict("speech.wav")
# → {'mos': 3.82}

# NumPy array
import numpy as np
whisqa.predict(np.zeros(16000, dtype=np.float32), sample_rate=16000)

# Torch tensor — drop straight in from your existing pipeline
import torch
whisqa.predict(torch.zeros(1, 16000), sample_rate=16000)

# All five P.835 dimensions
whisqa.predict("speech.wav", model_type="multi")
# → {'mos': 3.82, 'noisiness': 4.10, 'coloration': 3.55, 'discontinuity': 4.20, 'loudness': 3.90}

# Efficient repeated inference — load once, score many
model = whisqa.load_model("single")
for f in my_files:
    print(whisqa.predict(f, model=model))
```

```bash
# CLI
whisqa speech.wav
whisqa speech.wav --model multi
whisqa speech.wav --no-warn-resample
```

## Citation

[![arXiv](https://img.shields.io/badge/arXiv-2508.02210-b31b1b.svg)](https://arxiv.org/abs/2508.02210)

If you use WhiSQA in your work, please cite the original authors:

```bibtex
@inproceedings{close2025whisqa,
  title     = {{WhiSQA}: Non-Intrusive Speech Quality Prediction Using {Whisper} Encoder Features},
  author    = {Close, George and Hong, Kris Y. and Hain, Thomas and Goetze, Stefan},
  booktitle = {Speech and Computer -- 27th International Conference, {SPECOM} 2025,
               Szeged, Hungary, October 13--15, 2025, Proceedings, Part {I}},
  editor    = {Karpov, Alexey and Gosztolya, G{\'a}bor},
  series    = {Lecture Notes in Computer Science},
  volume    = {16187},
  pages     = {39--51},
  publisher = {Springer},
  year      = {2025},
  doi       = {10.1007/978-3-032-07956-5_3},
}
```
