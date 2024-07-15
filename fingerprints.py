"""
Functions involving getting fingerprints from audio samples. These include
- Generating (log-scaled) spectrogram
- Finding 2D peaks
- Converting peaks to fingerprints
"""

from typing import Iterable, List, Sequence, Tuple

import defaults
import matplotlib.mlab as mlab
import numpy as np
from numba import njit
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure


def digital_to_spec(
    audio_samples: np.ndarray, fs: float, frac_cut: float = defaults.MIN_FRAC_AMP_CUTOFF
) -> Tuple[np.ndarray, float]:
    """Produces a log-scaled spectrogram and a cut-off intensity to yield the
    specified fraction of data.

    Parameters
    ----------
    audio_samples : numpy.ndarray, shape=(Ts, )
        The digital samples of the audio-signal.

    fs : float
        The sample-frequency used to create the digital signal.

    frac_cut : float
        The fractional portion of intensities for which the cutoff is selected.
        E.g. frac_cut=0.8 will produce a cutoff intensity such that the bottom 80%
        of intensities are excluded.

    Returns
    -------
    Union[Tuple[numpy.ndarray, float]]
        The log-scaled spectrogram and the spectrogram element that partitions the
        bottom `frac_cut` elements in the spectrogram from the top elements.
    """
    assert 0.0 <= frac_cut <= 1.0

    S, _, _ = mlab.specgram(
        audio_samples,
        NFFT=defaults.NFFT,
        Fs=fs,
        window=mlab.window_hanning,
        noverlap=int(defaults.NFFT / 2),
    )

    # log-scaled Fourier amplitudes have a much more gradual distribution
    # for audio data.
    #
    # We need to clip the spectrogram before taking the log so that its
    # smallest value does not fall below `1e-20` because log(0) is undefined
    np.clip(S, a_min=1e-20, a_max=None, out=S)  # clip in-place
    np.log(S, out=S)  # take the log in-place

    # Compute percentile-based threshold amplitude; this is greatly optimized by
    # leveraging the apt numpy.partition function.
    cutoff_index = int(frac_cut * S.size)
    cutoff = np.partition(S.ravel(), cutoff_index)[cutoff_index]

    return S, cutoff


@njit
def _peaks(
    data_2d: np.ndarray, row_deltas: np.ndarray, col_deltas: np.ndarray, amp_min: float
) -> List[Tuple[int, int]]:
    """
    A Numba-optimized 2-D peak-finding algorithm.

    Parameters
    ----------
    data_2d : numpy.ndarray, shape-(H, W)
        The 2D array of data in which local peaks will be detected.

    row_deltas : numpy.ndarray, shape-(N,)
        The row-index offsets used to traverse the local neighborhood,
        relative to some element at location (row, col).

        E.g. (row + row_deltas[0], col + col_deltas[0]) will move to the
        first spot in the local neighborhood

    col_deltas : numpy.ndarray, shape-(N,)
        The col-index offsets used to traverse the local neighborhood,
        relative to some element at location (row, col).

    amp_min : float
        All amplitudes at and below this value are excluded from being local
        peaks.

    Returns
    -------
    List[Tuple[int, int]]
        (col, row) index pair for each local peak location.
    """
    peaks = []  # stores the (row, col) locations of all the local peaks

    # Iterate over the 2-D data in col-major order
    # we want to see if there is a local peak located at
    # row=r, col=c
    for c, r in np.ndindex(*data_2d.shape[::-1]):
        if data_2d[r, c] <= amp_min:
            # The amplitude falls beneath the minimum threshold
            # thus this can't be a peak.
            continue

        # Iterating over the neighborhood centered on (r, c)
        # dr: displacement from r
        # dc: discplacement from c
        for dr, dc in zip(row_deltas, col_deltas):
            if dr == 0 and dc == 0:
                # This would compare (r, c) with itself.. skip!
                continue

            if not (0 <= r + dr < data_2d.shape[0]):
                # neighbor falls outside of boundary
                continue

            if not (0 <= c + dc < data_2d.shape[1]):
                # neighbor falls outside of boundary
                continue

            if data_2d[r, c] < data_2d[r + dr, c + dc]:
                # One of the amplitudes within the neighborhood
                # is larger, thus data_2d[r, c] cannot be a peak
                break
        else:
            # if we did not break from the for-loop then (r, c) is a peak
            peaks.append((c, r))  # <- note col, row ordering
    return peaks


def local_peaks(
    log_spectrogram: np.ndarray,
    amp_min: float,
    p_nn: int = defaults.LOCAL_PEAK_NN_RADIUS,
) -> List[Tuple[int, int]]:
    """
    Defines a local neighborhood and finds the local peaks
    in the spectrogram, which must be larger than the
    specified `amp_min`.

    Parameters
    ----------
    log_spectrogram : numpy.ndarray, shape=(n_freq, n_time)
        Log-scaled spectrogram. Columns are the periodograms of
        successive segments of a frequency-time spectrum.

    amp_min : float
        Amplitude threshold applied to local maxima

    p_nn : int
        The neighborhood radius used for determining if a spectrogram value
        is a local peak. Specified in spectrogram cells.

    Returns
    -------
    List[Tuple[int, int]]
        Time and frequency index-values of the local peaks in spectrogram.
        Sorted by ascending frequency and then time.

    Notes
    -----
    The local peaks are returned in column-major order for the spectrogram.
    That is, the peaks are ordered by time. Thus, we look for nearest
    neighbors of increasing frequencies at the same times, and then move to
    the next time bin.
    """

    # generating a neighborhood from a basic binary structure
    # (2D array of True/False values)
    struct = generate_binary_structure(2, 1)
    # Growing the array of True/False values by iterating the binary structure
    neighborhood = np.asarray(iterate_structure(struct, p_nn))

    # Our neighborhood must have odd-values height & width so
    # that the center pixel of the neighborhood is unambiguous
    assert neighborhood.shape[0] % 2 == 1
    assert neighborhood.shape[1] % 2 == 1

    # the row and column indices where our neighborhood contains `True`
    rows, cols = np.where(neighborhood)

    # center neighborhood indices around center of neighborhood
    rows -= neighborhood.shape[0] // 2
    cols -= neighborhood.shape[1] // 2

    # Extract peaks; encoded in terms of time and freq bin indices.
    # dt and df are always the same size for the spectrogram that is produced,
    # so the bin indices consistently map to the same physical units:
    # t_n = n*dt, f_m = m*df (m and n are integer indices)
    # Thus we can codify our peaks with integer bin indices, (n, m) instead of their
    # physical (t, f) coordinates. This makes storage and compression of peak
    # locations much simpler.
    time_freq_peak_locations = _peaks(log_spectrogram, rows, cols, amp_min=amp_min)

    return time_freq_peak_locations


def peaks_to_fingerprints(
    peaks: Sequence[Tuple[int, int]], fan_value: int = defaults.FINGERPRINT_FANOUT
) -> Iterable[Tuple[Tuple[int, int, int], int]]:
    """Given the time-frequency locations of spectrogram peaks, generates
    'fingerprint' features.

    Parameters
    ----------
    peaks : Sequence[Tuple[int, int]]
        A sequence of time-frequency pairs

    fan_value : int, optional (uses global default)
        Given a peak, `fan_value` indicates the number of subsequent peaks
        to be used to form fingerprint features.

    Yields
    ------
    Tuple[Tuple[int, int, int], int]
        ((f_{n}, f_{n+j}, t_{n+j} - t_{n}), t_{n})
        The frequency value of peak n, peak n+j, their time-offset, along with the
        time at which peak n occurred.

    Notes
    -----
    Because this ends in a Yields statment, this function
    actually acts as a generator. I.e. you have to iterate
    over it to consume it.

    E.g.

    >>> peaks_to_fingerprints(peaks, 15)
    <generator>

    >>> list(peaks_to_fingerprints(peaks, 15))
    [((2, 3, 5), 10), ((12, 4, 5), 16), ....]

    Using a generator is nice because we might be producing
    a lot of fingerprints, and this enables use to consume/count
    them as we product them, instead of having to store them in
    memory all at once."""

    assert 1 <= fan_value
    for n, (t1, f1) in enumerate(peaks):
        for t2, f2 in peaks[n + 1 : n + fan_value + 1]:
            yield (f1, f2, t2 - t1), t1
