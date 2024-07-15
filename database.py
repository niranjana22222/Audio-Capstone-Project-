"""
Our "database" simply consists of two parts 

1) fp_lookup: A default-dict that stores:
    - key: (freq_m, freq_n, (t_n - t_m))
    - value: [(song-ID, t_m), ....]

    Note that all frequencies and times are actually integer-valued
    indices associated without our spectrogram's bins. We don't store
    the actual time/frequency values in seconds/Hz.
    See https://rsokl.github.io/CogWeb/Audio/audio_features.html for more details

2) song_id_to_name: A dictionary that stores song-ID to song-name

Basic functions for working with our song database.
- create / save / load db
- add a song (its fingerprints & name) to the db
- record audio for a mic and return its best match from the db
"""

import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, NamedTuple, Tuple, Union
from typing_extensions import TypeAlias

import numpy as np

SongId: TypeAlias = Any

# This is simply a named-tuple that includes type-annotations
# for its fields
class Database(NamedTuple):
    # (f1, f2, dt_12) -> [(song-ID, t1), ...]
    fp_lookup: DefaultDict[Tuple[int, int, int], List[Tuple[SongId, int]]]
    song_id_to_name: Dict[SongId, str]


def create_database() -> Database:
    return Database(fp_lookup=defaultdict(list), song_id_to_name=dict())


def save_db(db: Database, path: Union[str, Path]):
    with open(path, "wb") as f:
        # convert named tuple to "vanilla" tuple before saving
        pickle.dump(tuple(db), f)


def load_db(path: Union[str, Path]) -> Database:
    with open(path, "rb") as f:
        # convert vanilla tuple to named tuple upon loading
        return Database(*pickle.load(f))


def add_song_to_database(
    song_samples: np.ndarray, sample_rate: int, song_name: str, db: Database
) -> Database:
    """Convert a song's digital samples to a spectrogram and extract its
    'fingerprints', and store them in the song database.

    Parameters
    ----------
    song_samples : np.ndarray, shape-(N,)
        The digital samples

    sample_rate : int
        The sampling rate used to produce `song_samples`

    song_name : str

    db : Database

    Returns
    -------
    db : Database"""
    from fingerprints import digital_to_spec, local_peaks, peaks_to_fingerprints

    # This doesn't check to see if the song already exists in our database
    # If it does, then we will simply have multiple song-IDs pointing to
    # the same song name

    # Generate song-ID simply based off how many other songs are in the DB
    # (not safe if we ever delete a song from the database!)
    new_song_id = len(db.song_id_to_name)

    log_spec, threshold_amp = digital_to_spec(song_samples, sample_rate)
    time_freq_peak_locations = local_peaks(log_spec, threshold_amp)

    for f1_f2_dt, t1 in peaks_to_fingerprints(time_freq_peak_locations):
        db.fp_lookup[f1_f2_dt].append((new_song_id, t1))

    db.song_id_to_name[new_song_id] = song_name

    return db


def fingerprints_to_matches(
    sample_fingerprints: Iterable[Tuple[Tuple[int, int, int], int]],
    db: Database,
) -> Iterable[Tuple[SongId, int]]:
    """Generates database matches from all of a sample's fingerprints.

    Parameters
    ----------
    sample_fingerprints : Iterable[Tuple[Tuple[int, int, int], int]]
        ((f_{n}, f_{n+j}, dt), t_{n})
        The frequency value of peak n and peak n+j, along with the time at which peak n occurred.

    db : Database
        Our database

    Yields
    ------
    Tuple[song_ID, dt]
        A song ID that had a matching peak-pair signature, and the time offset between
        when the signature occurred in the song versus the sample.

    See Also
    --------
    fingerprints.peaks_to_fingerprints"""
    for f1_f2_dt, t_sample in sample_fingerprints:
        match = db.fp_lookup.get(f1_f2_dt)
        if match is not None:
            for s_id, t_song in match:
                yield s_id, t_song - t_sample


def match_recording(listen_time: float, db: Database) -> Tuple[Union[str, None], int]:
    """Records from a microphone and return the name of the matched song and the
    number of matches from the database (i.e. the number of consistent (song-ID, dt)
    occurences found in the top-match).

    Parameters
    ----------
    listen_time : float

    db : Database

    Returns
    -------
    (song-name, num-matches)
        If no match, then song-name is `None`.
    """
    from audio import get_digital_recording
    from fingerprints import digital_to_spec, local_peaks, peaks_to_fingerprints

    samples, sampling_rate = get_digital_recording(listen_time)
    log_spec, threshold_amp = digital_to_spec(samples, sampling_rate)
    time_freq_peak_locations = local_peaks(log_spec, threshold_amp)
    fingerprints = peaks_to_fingerprints(time_freq_peak_locations)
    matches = fingerprints_to_matches(fingerprints, db=db)

    # find most consistent match
    cntr = Counter(matches)  # counts (song-ID, dt) occurrences
    if not cntr:
        # no matches!
        return None, 0

    # We could modify this to return the top-k matches to analyze our
    # results -- e.g., how close was the 2nd-best match?
    (top_song_id, dt), num_matches = cntr.most_common(1)[0]
    return db.song_id_to_name[top_song_id], num_matches
