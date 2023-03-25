# dev/EZR/
# test/COL/
# train/1CH, train/1CO, train/1JN, ...
# Each of those folders has 1 flac, 1 text per audio

import glob, os, librosa

"""
Trim the data down to 10 hours
"""
# path = '../LIN/'
# time = 0
# deleted = 0
# for split in os.listdir(path):
#     for book in os.listdir(path + split + '/'):
#         for file in os.listdir(path + split + '/' + book + '/'):
#             full_file_path = path + split + '/' + book + '/' + file
#             if full_file_path.endswith(".flac"):
#                 if time >= 36000:
#                     without_extension = ''.join(os.path.splitext(full_file_path)[:-1])
#                     os.remove(without_extension + ".flac")
#                     os.remove(without_extension + ".txt")
#                     deleted += 1
#                 else:
#                     time += librosa.get_duration(filename=full_file_path)
# print(time)
# print("delted:", deleted)

"""
Flatten the file structure
"""
# path = '../LIN/'
# for split in os.listdir(path):
#     for book in os.listdir(path + split + '/'):
#         for file in os.listdir(path + split + '/' + book + '/'):
#             if file.endswith(".flac"):
#                 full_file_path = path + split + '/' + book + '/' + file
#                 without_extension = ''.join(os.path.splitext(file)[:-1])
#                 os.rename(path + split + '/' + book + '/' + without_extension + '.flac', path + without_extension + '.flac')
#                 os.rename(path + split + '/' + book + '/' + without_extension + '.txt', path + without_extension + '.txt')


"""
Create copies of flac files but as wav
"""
path = '../LIN/'
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
Renaming the wav + txt file pairs
"""
i = 0
wavs = sorted(glob.glob("*.wav"))
for f in wavs: # go through all files that are wav files
    filename = ''.join(os.path.splitext(f)[:-1])
    os.rename(path + filename + '.wav', path + 'lin_bibletts_m_1-' + f'{i:05}' + '.wav') # rename wav
    os.rename(path + filename + '.txt', path + 'lin_bibletts_m_1-' + f'{i:05}' + '.txt') # rename txt
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

# Duration is already at 36000 so we're done