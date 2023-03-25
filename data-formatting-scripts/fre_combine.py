# flac files. 1 txt file per flac

import glob, os, librosa
import shutil

"""
Move from FR to FRE
"""
# path = '../FR/'
# for f in os.listdir(path):
#     os.rename(path + f, '../FRE/'+f)


"""
Move from African_Accented_French to FRE
"""
# path = '../African_Accented_French/'
# for f in os.listdir(path):
#     os.rename(path + f, '../FRE/'+f)


"""
Measure total time
"""
time = 0
os.chdir('../FRE')
wavs = sorted(glob.glob("*.wav"))
for f in wavs: # go through all files that are wav files
    time += librosa.get_duration(filename=f)
print(time)

