from models.whisper_ni_predictors import SingleHeadPredictor, MultiHeadPredictor
import sys
import argparse
import torch
import soundfile as sf
import torchaudio.functional

DIMENSIONS = ["MOS", "Noisiness", "Coloration", "Discontinuity", "Loudness"]


def get_score(audio_file: str, model_type: str = "single") -> torch.Tensor:
    """
    Predict speech quality score(s) for a mono 16 kHz WAV file.

    Args:
        audio_file:  Path to the audio file. Must be mono; any sample rate is
                     accepted (resampled to 16 kHz automatically).
        model_type:  'single' → scalar MOS.
                     'multi'  → [MOS, Noisiness, Coloration, Discontinuity, Loudness].

    Returns:
        torch.Tensor: shape (1,) for 'single', (5,) for 'multi'. Values in [0, 1].
    """
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    if model_type == "single":
        model = SingleHeadPredictor()
        model.load_state_dict(
            torch.load("checkpoints/single_head_model.pt", map_location=device)
        )
    elif model_type == "multi":
        model = MultiHeadPredictor()
        model.load_state_dict(
            torch.load("checkpoints/multi_head_model.pt", map_location=device)
        )
    else:
        raise ValueError(f"Unknown model_type {model_type!r}. Choose 'single' or 'multi'.")

    model.eval()
    model.to(device)

    data, sample_rate = sf.read(audio_file, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)   # (channels, samples)

    if waveform.shape[0] != 1:
        raise ValueError(f"Audio must be mono (1 channel), got {waveform.shape[0]}.")

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    waveform = waveform.to(device)
    with torch.no_grad():
        score = model(waveform)

    if model_type == "multi":
        score = score.squeeze(0)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict speech quality scores for a WAV file.")
    parser.add_argument("audio_file")
    parser.add_argument(
        "--model_type",
        choices=["single", "multi"],
        default="single",
        help="'single' for MOS only, 'multi' for MOS + 4 P.835 dimensions.",
    )
    args = parser.parse_args()

    score = get_score(args.audio_file, args.model_type)

    if args.model_type == "single":
        print(f"MOS: {score.item() * 5:.4f}")
    else:
        for name, val in zip(DIMENSIONS, score):
            print(f"{name}: {val.item() * 5:.4f}")

    sys.exit(0)
