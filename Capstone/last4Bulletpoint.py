import numpy as np
import random

# keys = freq tuples; values = list of tuples in the format (song, time)
database = {}

# add to database function
# function also check if there are overlaping song
def add_to_db(fingerprint, songID):
    songInDataBase = True
    for songList in database.values():
        for song in songList:
            if song[0] == songID:
                return "Song is already in database."
            else:
                songInDataBase = False
    if songInDataBase == False:
        for pair in fingerprint:
            if pair[0:3] not in database.keys():
                database[pair[0:3]] = []
            time = pair[3]
            database[pair[0:3]].append((songID, time))

# Query function to find match song in database for an unknown sample
def query(unknown_fingerprint):
    possible_songs = []
    # drow = row in database keys (one peak pair)
    for drow in database.keys():
        # urow = row in unknown_fingerprint (one peak pair)
        for urow in unknown_fingerprint:
            if drow == urow[0:3]:
                for song in database[drow]:
                    possible_songs.append(song[0])
    possible_songs = np.array(possible_songs)
    unique, counts = np.unique(possible_songs, return_counts = True)
    match_song = unique[np.argmax(counts)]
    return match_song

#Delete_song function    
def delete_song(songID):
    for key, songList in database.items():
        for song in songList:
            if songID == song[0]:
                songList.remove(song)

def random_clip(array_of_audio):
    samples = []
    total_sample = len(array_of_audio)
    for x in range(3):
        start = random.randint(0,total_sample - 41000)
        samples.append(array_of_audio[start:start + 41001])
    return samples

