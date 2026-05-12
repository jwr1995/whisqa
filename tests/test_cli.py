"""
Tests for the ``whisqa`` CLI entry point.
"""
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import torch

from whisqa.cli import main


def _run_cli(args: list, mock_scores: dict):
    """Invoke main() with patched predict() and return (exit_code, output)."""
    runner_args = args

    with patch("whisqa.predict", return_value=mock_scores) as mock_predict:
        # Simulate argparse by calling main() via the Click test runner pattern.
        import sys
        from io import StringIO

        old_argv = sys.argv
        old_stdout = sys.stdout
        sys.argv = ["whisqa"] + runner_args
        sys.stdout = captured = StringIO()
        exit_code = 0
        try:
            main()
        except SystemExit as e:
            exit_code = e.code or 0
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout

        output = captured.getvalue()
        return exit_code, output, mock_predict


class TestCLIOutput:
    def test_single_prints_mos(self, wav_16k):
        _, output, _ = _run_cli([str(wav_16k)], {"mos": 3.82})
        assert "Mos: 3.82" in output

    def test_multi_prints_all_dimensions(self, wav_16k):
        scores = {
            "mos": 3.82,
            "noisiness": 4.10,
            "coloration": 3.55,
            "discontinuity": 4.20,
            "loudness": 3.90,
        }
        _, output, _ = _run_cli([str(wav_16k), "--model", "multi"], scores)
        for key in scores:
            assert key.capitalize() in output

    def test_exit_code_zero_on_success(self, wav_16k):
        code, _, _ = _run_cli([str(wav_16k)], {"mos": 3.0})
        assert code == 0


class TestCLIErrorHandling:
    def test_missing_file_exits_nonzero(self, tmp_path):
        missing = str(tmp_path / "nope.wav")
        with patch("whisqa.predict", side_effect=FileNotFoundError("not found")):
            import sys
            old_argv, sys.argv = sys.argv, ["whisqa", missing]
            try:
                main()
            except SystemExit as e:
                assert e.code != 0
            finally:
                sys.argv = old_argv

    def test_value_error_exits_nonzero(self, wav_stereo):
        with patch("whisqa.predict", side_effect=ValueError("must be mono")):
            import sys
            old_argv, sys.argv = sys.argv, ["whisqa", str(wav_stereo)]
            try:
                main()
            except SystemExit as e:
                assert e.code != 0
            finally:
                sys.argv = old_argv


class TestCLIDirectoryMode:
    def _make_audio_dir(self, tmp_path, n=3):
        for i in range(n):
            sf.write(str(tmp_path / f"file{i}.wav"), np.zeros(16000, dtype=np.float32), 16000)
        return tmp_path

    def _run_dir(self, args):
        mock_results = [{"file": "/a/b.wav", "mos": 3.5}]
        old_argv, sys.argv = sys.argv, ["whisqa"] + args
        old_stdout, sys.stdout = sys.stdout, StringIO()
        exit_code = 0
        with patch("whisqa.WhiSQA") as MockScorer:
            MockScorer.return_value.predict_dir.return_value = mock_results
            try:
                main()
            except SystemExit as e:
                exit_code = e.code or 0
            finally:
                output = sys.stdout.getvalue()
                sys.argv = old_argv
                sys.stdout = old_stdout
        return exit_code, output, MockScorer

    def test_directory_arg_uses_whisqa_class(self, tmp_path):
        self._make_audio_dir(tmp_path)
        code, _, MockScorer = self._run_dir([str(tmp_path)])
        MockScorer.assert_called_once()

    def test_directory_prints_filenames(self, tmp_path):
        self._make_audio_dir(tmp_path)
        _, output, _ = self._run_dir([str(tmp_path)])
        assert "b.wav" in output

    def test_directory_exit_zero_on_success(self, tmp_path):
        self._make_audio_dir(tmp_path)
        code, _, _ = self._run_dir([str(tmp_path)])
        assert code == 0

    def test_directory_csv_output(self, tmp_path):
        self._make_audio_dir(tmp_path)
        csv_path = tmp_path / "out.csv"
        mock_results = [{"file": "/a/b.wav", "mos": 3.5}]
        old_argv, sys.argv = sys.argv, ["whisqa", str(tmp_path), "--output", str(csv_path)]
        with patch("whisqa.WhiSQA") as MockScorer:
            MockScorer.return_value.predict_dir.return_value = mock_results
            try:
                main()
            except SystemExit:
                pass
            finally:
                sys.argv = old_argv
        assert csv_path.exists()
        assert "mos" in csv_path.read_text()

    def test_no_recursive_flag_passed_through(self, tmp_path):
        self._make_audio_dir(tmp_path)
        _, _, MockScorer = self._run_dir([str(tmp_path), "--no-recursive"])
        MockScorer.return_value.predict_dir.assert_called_once()
        _, kwargs = MockScorer.return_value.predict_dir.call_args
        assert kwargs.get("recursive") is False


class TestCLIFlags:
    def test_no_warn_resample_flag_passed_through(self, wav_16k):
        _, _, mock_predict = _run_cli(
            [str(wav_16k), "--no-warn-resample"],
            {"mos": 3.0},
        )
        _, kwargs = mock_predict.call_args
        assert kwargs.get("warn_resample") is False

    def test_warn_resample_true_by_default(self, wav_16k):
        _, _, mock_predict = _run_cli([str(wav_16k)], {"mos": 3.0})
        _, kwargs = mock_predict.call_args
        assert kwargs.get("warn_resample") is True
