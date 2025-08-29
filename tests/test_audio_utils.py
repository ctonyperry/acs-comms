"""Tests for audio utility functions."""

import wave
from unittest.mock import patch

import pytest

from src.acs_bridge.audio.utils import ensure_16k_mono_wav


class TestEnsure16kMonoWav:
    """Test cases for ensure_16k_mono_wav function."""

    def test_correct_format_passthrough(self, test_wav_file):
        """Test that correctly formatted WAV files are passed through unchanged."""
        result = ensure_16k_mono_wav(test_wav_file)
        assert result == str(test_wav_file)

    def test_string_path_input(self, test_wav_file):
        """Test that string paths work correctly."""
        result = ensure_16k_mono_wav(str(test_wav_file))
        assert result == str(test_wav_file)

    def test_incorrect_format_requires_ffmpeg(self, temp_dir):
        """Test that incorrect format files require ffmpeg conversion."""
        # Create a WAV file with wrong format (8kHz)
        wrong_wav = temp_dir / "wrong.wav"
        with wave.open(str(wrong_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)  # Wrong sample rate
            wf.writeframes(b"\x00\x00" * 8000)

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
                ensure_16k_mono_wav(wrong_wav)

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_ffmpeg_conversion(self, mock_run, mock_which, temp_dir):
        """Test ffmpeg conversion is called correctly."""
        mock_which.return_value = "/usr/bin/ffmpeg"

        # Create a WAV file with wrong format
        wrong_wav = temp_dir / "wrong.wav"
        with wave.open(str(wrong_wav), "wb") as wf:
            wf.setnchannels(2)  # Stereo
            wf.setsampwidth(2)
            wf.setframerate(44100)  # Wrong sample rate
            wf.writeframes(b"\x00\x00\x00\x00" * 44100)

        result = ensure_16k_mono_wav(wrong_wav)

        # Check that ffmpeg was called
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "ffmpeg" in call_args
        assert "-ac" in call_args
        assert "1" in call_args  # mono
        assert "-ar" in call_args
        assert "16000" in call_args  # 16kHz

        # Check result path
        expected_path = str(wrong_wav).replace(".wav", "_16k_mono.wav")
        assert result == expected_path

    def test_invalid_wav_file(self, temp_dir):
        """Test handling of invalid WAV files."""
        # Create a non-WAV file
        invalid_file = temp_dir / "invalid.wav"
        invalid_file.write_text("not a wav file")

        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
                ensure_16k_mono_wav(invalid_file)
