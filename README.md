# Mini Audio Project

This is a minimal implementation of the Audio Project. It is designed to be easy to skim and to understand.
There are some bells and whistles sacrificed here. E.g. you cannot easily delete a song from a database, the code 
won't stop you from adding the same code twice, etc.

This is not an installable package. You must change your working directory to `AudioProject/mini_implementation/` to 
use this.

## Outline

It is recommended that you read through this project in the following order:

### `audio.py`

Functions for getting audio samples from: 
- music file (e.g. mp3 file)
- a microphone


### `fingerprints.py`

Functions involving getting fingerprints from audio samples. These include
- Generating (log-scaled) spectrogram
- Finding 2D peaks
- Converting peaks to fingerprints

### `database.py`

Our "database" simply consists of a [named-tuple](https://www.pythonlikeyoumeanit.com/Module2_EssentialsOfPython/DataStructures_III_Sets_and_More.html#Named-Tuple) with two elements: 

1) `fp_lookup`: A [default-dict](https://www.pythonlikeyoumeanit.com/Module2_EssentialsOfPython/DataStructures_III_Sets_and_More.html#Default-Dictionary) that stores:
    - key: `(freq_m, freq_n, (t_n - t_m))`
    - value: `[(song-ID, t_m), ....]`

    Note that all frequencies and times are actually integer-valued
    indices associated without our spectrogram's bins. We don't store
    the actual time/frequency values in seconds/Hz.
    See https://rsokl.github.io/CogWeb/Audio/audio_features.html for more details

2) `song_id_to_name`: A dictionary that stores song-ID to song-name

Basic functions for working with our song database.
- create / save / load db
- add a song (its fingerprints & name) to the db
- record audio for a mic and return its best match from the db


### `defaults.py`

Stores the globally-used default configuration values associated with sampling and fingerprinting the audio files

## `Mini_Project_Demo.ipynb`

This uses the above components to perform the essential song-matching work flow:

- Create a song database
- Add a few songs to the database
- Save the database
- Load the database
- Listen to a microphone and query the database for the best match


## Other Resourcees

To learn more about how this project works, read:
- https://rsokl.github.io/CogWeb/Audio/audio_features.html
- https://rsokl.github.io/CogWeb/Audio/Exercises/PeakFinding.html
- https://rsokl.github.io/CogWeb/Audio/capstone_summary.html
