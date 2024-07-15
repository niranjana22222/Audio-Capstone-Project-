"""
Functions for getting audio samples from: 
- music file (e.g. mp3 file)
- a microphone
"""

from pathlib import Path
from typing import Tuple, Union

import defaults
import librosa
import numpy as np
from microphone import record_audio


def get_digital_recording(time: float) -> Tuple[np.ndarray, int]:
    """
    Get the digital samples and sampling rate of a microphone's recording.

    Parameters
    ----------
    time : float
        Time, in seconds to record from the mic.

    Returns
    -------
    Tuple[numpy.ndarray, int]
        The digital samples (mono: shape-(N,)) from the recording and
        the sampling rate used.
    """
    frames, sample_rate = record_audio(time)
    digital_data = np.hstack([np.frombuffer(i, np.int16) for i in frames])
    return digital_data, sample_rate


def load_song_file(
    path_to_song_file: Union[Path, str], sampling_rate: int = defaults.SAMPLING_RATE
) -> Tuple[np.ndarray, int]:
    """
    Load the digital samples from an audio file (e.g. a .mp3 file).

    Parameters
    ----------
    path_to_song_file: Path | str
        The path to the audio file

    sampling_rate: int,  optional (global default provided)
        The sampling rate used

    Returns
    -------
    Tuple[numpy.ndarray, int]
        The digital samples (mono: shape-(N,)) from the loaded audio file and
        the sampling rate used.
    """
    digital, fs = librosa.load(str(path_to_song_file), sr=sampling_rate, mono=True)
    return digital, fs
