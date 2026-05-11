!THIS PROJECT IS STILL WIP - numerical stability is not yet there!

# whisqa: a PyPi package for the WHiSQA Non-Intrusive Speech Quality Prediction Model

This project is a fork of the original WhiSQA project repackaged for pypi.

WhiSQA uses the frozen encoder of OpenAI's Whisper-small model as a feature extractor, adds a lightweight trainable transformer head, and predicts ITU-T P.835 speech quality dimensions without requiring a clean reference signal.

Original repo: https://github.com/leto19/WhiSQA

## Install

```bash
pip install whisqa
```

## Usage

```python
import whisqa

# Single MOS score
whisqa.predict("speech.wav")
# → {'mos': 3.82}

# Full P.835 dimensions
whisqa.predict("speech.wav", model_type="multi")
# → {'mos': 3.82, 'noisiness': 4.10, 'coloration': 3.55, 'discontinuity': 4.20, 'loudness': 3.90}

# Efficient repeated inference
model = whisqa.load_model("single")
for f in my_files:
    print(whisqa.predict(f, model=model))
```

```bash
# CLI
whisqa speech.wav
whisqa speech.wav --model multi
```

Input must be mono WAV. Any sample rate is accepted; audio is resampled to 16 kHz automatically.

> **First-run note:** `load_model()` downloads `openai/whisper-small` (~240 MB) from HuggingFace Hub and caches it in `~/.cache/huggingface/`. The WhiSQA head weights ship inside the package and require no download.

## Results

![Results](results.png)

## Citation

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
[![arXiv](https://img.shields.io/badge/arXiv-2508.02210-b31b1b.svg)](https://arxiv.org/abs/2508.02210)
