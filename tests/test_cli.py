"""
Tests for the ``whisqa`` CLI entry point.
"""
from unittest.mock import patch

import pytest
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
