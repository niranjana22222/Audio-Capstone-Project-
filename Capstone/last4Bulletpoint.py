import sqlite3
import numpy as np
import unittest
import random

# keys = freq tuples; values = list of tuples in the format (song, time)
database = {(123, 256, 5) : [("songOne", 5), ("songTwo", 25), ("songThree", 30), ("songFour", 23)], 
            (356, 100, 4) : [("songTwo", 30), ("songFour", 53)], 
            (210, 160, 8): [("songThree", 46)],
            (300, 210, 7): [("songOne", 3)],
            (234, 123, 6) : [("songOne", 51), ("songTwo", 25), ("songThree", 33), ("songFour", 13)], 
            
            }


# implement time after fingerprint function
#                  peak  com_p T_d T
test_fingerprint = [(123, 256, 5, 5), (356, 100, 4, 36), (200, 100, 8, 52)]
# add to database function
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

print("Test add to db function: ")
print(" ")
print("Origingal", database)
print(" ")
print(add_to_db(test_fingerprint, "songFive"))
print(" ")
print("Added to db", database)

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

print("Test query function")
print(" ")
print(query(test_fingerprint))
print(" ")

#Delete_song function    
def delete_song(songID):
    for key, songList in database.items():
        for song in songList:
            if songID == song[0]:
                songList.remove(song)
        
print("db before delete", database)
print("Test delete function: ") 
print(delete_song("songFive"))
print("db: ", database)


def random_clip(array_of_audio):
    samples = []
    total_sample = len(array_of_audio)
    for x in range(3):
        start = random.randint(0,total_sample - 41000)
        samples.append(array_of_audio[start:start + 41001])
    return samples


# clip = random_clip(audio_samples)
# print(f"Random clip shape: {clip.shape}")

    


# class TestAudioDatabase(unittest.TestCase):
#     def test_add_song(self):
#         # Test cases for adding songs
#         pass

#     def test_match_song(self):
#         # Test cases for matching songs
#         pass
