from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO_ID = "leto19/whisqa"

_FILENAMES = {
    "single": "single_head_model.pt",
    "multi": "multi_head_model.pt",
}


def get_checkpoint_path(model_type: str) -> Path:
    """
    Return a local path to the head-only checkpoint for `model_type`,
    downloading from HuggingFace Hub on first use.

    Cached automatically by huggingface_hub in ~/.cache/huggingface/.
    """
    if model_type not in _FILENAMES:
        raise ValueError(
            f"Unknown model type {model_type!r}. Choose from: {list(_FILENAMES)}"
        )
    return Path(hf_hub_download(repo_id=HF_REPO_ID, filename=_FILENAMES[model_type]))
