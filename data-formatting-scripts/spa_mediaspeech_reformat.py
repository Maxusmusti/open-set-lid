# flac files. 1 txt file per flac

import glob, os, librosa

"""
Create copies of flac files but as wav
"""
path = '../ES/'
os.chdir(path)
flacs = glob.glob("*.flac")
for f in flacs: # go through all files that are flac files
    os.system('ffmpeg -i ' + path + f + ' ' + path + f.rsplit( ".", 1 )[ 0 ] + '.wav') # use ffmpeg to convert it to wav


"""
Deleting the flac files since we made all of the wav files
"""
for f in glob.glob("*.flac"): # go through all files that are flac files
    os.remove(path + f) # delete the old flac file
print("num flacs:", len(glob.glob("*.flac")))
print("num wavs:", len(glob.glob("*.wav")))


"""
Trim to 18500 seconds
"""
time = 0
wavs = sorted(glob.glob("*.wav"))
for f in wavs: # go through all files that are wav files
    if time >= 18500:
        without_extension = os.path.splitext(f)[0]
        os.remove(path+without_extension+".wav")
        os.remove(path+without_extension+".txt")
    else:
        time += librosa.get_duration(filename=path+f)
print(time)




"""
Renaming the wav + txt file pairs
"""
i = 0
wavs = sorted(glob.glob("*.wav"))
for f in wavs: # go through all files that are wav files
    filename = ''.join(os.path.splitext(f)[:-1])
    os.rename(path + filename + '.wav', path + 'spa_mediaspeech_u_u-' + f'{i:05}' + '.wav') # rename wav
    os.rename(path + filename + '.txt', path + 'spa_mediaspeech_u_u-' + f'{i:05}' + '.txt') # rename txt
    i+=1
print("num wavs:", len(glob.glob("*.wav")))


"""
Measuring total time of wavs
"""
time = 0
wavs = sorted(glob.glob("*.wav"))
for f in wavs: # go through all files that are wav files
    time += librosa.get_duration(filename=f)
print(time)
