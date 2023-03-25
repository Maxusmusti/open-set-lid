# flac files. 1 txt file per flac

import glob, os, librosa
import shutil


path = '../ES/'
for f in os.listdir(path):
    os.rename(path + f, '../SPA/'+f)

path = '../spanishchilean/'
for f in os.listdir(path):
    os.rename(path + f, '../SPA/'+f)
    
path = '../spanishcolombian/'
for f in os.listdir(path):
    os.rename(path + f, '../SPA/'+f)

path = '../spanishperuvian/'
for f in os.listdir(path):
    os.rename(path + f, '../SPA/'+f)

path = '../spanishpuertorican/'
for f in os.listdir(path):
    os.rename(path + f, '../SPA/'+f)


"""
Measure total time
"""
time = 0
os.chdir('../SPA')
wavs = sorted(glob.glob("*.wav"))
for f in wavs: # go through all files that are wav files
    time += librosa.get_duration(filename=f)
print(time)

