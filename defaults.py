"""
Stores the globally-used default configuration values associated with sampling and fingerprinting the audio files
"""

# The target sampling rate used to read in an audio file
SAMPLING_RATE: int = 44100

# The fractional portion of intensities for which the cutoff is selected.
# E.g. frac_cut=0.8 will produce a cutoff intensity such that the bottom 80%
# of intensities are excluded.
MIN_FRAC_AMP_CUTOFF: float = 0.77

# The neighborhood radius used for determining if a spectrogram value
# is a local peak. Specified in spectrogram cells.
LOCAL_PEAK_NN_RADIUS: int = 20

# Given a spectrogram peak, indicates the maximum number of subsequent peaks to
# be used to form fingerprint features.
FINGERPRINT_FANOUT: int = 15

# The number of data points used in each block for the FFT.
NFFT: int = 4096
