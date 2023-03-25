# ass structure. it's so ass
# Run each section by itself

import glob, os, librosa
import shutil
import random

"""
flattening the file structure for wavs
"""
# path = '../African_Accented_French/speech'
# files = []
# for a in os.listdir(path):
#     if os.path.isdir(path + '/' + a):
#         for b in os.listdir(path + '/' + a):
#             if os.path.isdir(path + '/' + a + '/' + b):
#                 for c in os.listdir(path + '/' + a + '/' + b):
#                     if os.path.isdir(path + '/' + a + '/' + b + '/' + c):
#                         for d in os.listdir(path + '/' + a + '/' + b + '/' + c):
#                             if os.path.isdir(path + '/' + a + '/' + b + '/' + c + '/' + d):
#                                 for e in os.listdir(path + '/' + a + '/' + b + '/' + c + '/' + d):
#                                     if os.path.isdir(path + '/' + a + '/' + b + '/' + c + '/' + d + '/' + e):
#                                         print("need to go deeper")
#                                     else:
#                                         files.append((e, path + '/' + a + '/' + b + '/' + c + '/' + d + '/' + e))
#                             else:
#                                 files.append((d, path + '/' + a + '/' + b + '/' + c + '/' + d))
#                     else:
#                         files.append((c, path + '/' + a + '/' + b + '/' + c))
#             else:
#                 files.append((b, path + '/' + a + '/' + b))
#     else:
#         files.append((a, path + '/' + a))

# print(len(files), len(set(files)))
# for file in files:
#     os.rename(file[1], '../African_Accented_French/' + file[0])


"""
Flatten transcript files
"""
# files = []
# path = '../African_Accented_French/transcripts'
# for a in os.listdir(path):
#     if os.path.isdir(path + '/' + a):
#         for b in os.listdir(path + '/' + a):
#             if os.path.isdir(path + '/' + a + '/' + b):
#                 for c in os.listdir(path + '/' + a + '/' + b):
#                     if os.path.isdir(path + '/' + a + '/' + b + '/' + c):
#                         for d in os.listdir(path + '/' + a + '/' + b + '/' + c):
#                             if os.path.isdir(path + '/' + a + '/' + b + '/' + c + '/' + d):
#                                 for e in os.listdir(path + '/' + a + '/' + b + '/' + c + '/' + d):
#                                     if os.path.isdir(path + '/' + a + '/' + b + '/' + c + '/' + d + '/' + e):
#                                         print("need to go deeper")
#                                     else:
#                                         files.append((e, path + '/' + a + '/' + b + '/' + c + '/' + d + '/' + e))
#                             else:
#                                 files.append((d, path + '/' + a + '/' + b + '/' + c + '/' + d))
#                     else:
#                         files.append((c, path + '/' + a + '/' + b + '/' + c))
#             else:
#                 files.append((b, path + '/' + a + '/' + b))
#     else:
#         files.append((a, path + '/' + a))

# transcript_i = 0
# for file in files:
#     os.rename(file[1], '../African_Accented_French/' + str(transcript_i) + '.txt')
#     transcript_i+=1

"""
Remove empty folders
"""
# shutil.rmtree('../African_Accented_French/speech')
# shutil.rmtree('../African_Accented_French/transcripts')
# os.rename('../African_Accented_French/niger_wav_file_name_transcript.tsv', '../African_Accented_French/niger_wav_file_name_transcript.txt')


"""
Trim the data down to 5 hours
"""
# path = '../African_Accented_French/'
# time = 0
# os.chdir(path)
# wavs = glob.glob("*.wav")
# random.shuffle(wavs)
# for f in wavs: # go through all files that are wav files
#     if time >= 18000:
#         os.remove(path+f)
#     else:
#         time += librosa.get_duration(filename=path+f)
# print(time)

"""
Combining texts into one
"""
# path = '../African_Accented_French/'
# texts = []
# os.chdir(path)
# txts = glob.glob("*.txt")
# for f in txts:
#     texts.append(path + f)

# with open(path + '_transcript.txt', 'w') as outfile:
#     for fname in texts:
#         with open(fname) as infile:
#             for line in infile:
#                 outfile.write(line)
#         os.remove(path + fname)

"""
One txt per wav
"""
# transcripts = {}
# path = '../African_Accented_French/'
# with open(path + '_transcript.txt', 'r') as txt:
#     for line in txt:
#         filepath = line.split(' ')[0]
#         filename = filepath.split('/')[-1]

#         text = ' '.join(line.split(' ')[1:])

#         if filename in transcripts and transcripts[filename] != text:
#             print(filename, "same file name, different transcrit found")
#         transcripts[filename] = ' '.join(line.split(' ')[1:])


# os.chdir('../African_Accented_French/')
# wavs = glob.glob("*.wav")
# missing = 0
# for wav in wavs:
#     if wav not in transcripts:
#         if os.path.splitext(wav)[0] not in transcripts:
#             missing += 1
# print("missing transcripts", missing, "out of", len(wavs))


# os.chdir('../African_Accented_French/')
# wavs = glob.glob("*.wav")
# for wav in wavs:
#     with open(path + ''.join(os.path.splitext(wav)[:-1]) + '.txt', 'w') as txt:
#         if wav in transcripts:
#             txt.write(transcripts[wav])
#         else:
#             if os.path.splitext(wav)[0] in transcripts:
#                 txt.write(transcripts[os.path.splitext(wav)[0]])
#             else:
#                 txt.write("missing transcript")

# os.remove(path + '_transcript.txt')




"""
Renaming the wav + txt file pairs
"""

# sexindex = {'a':1, 'c':2, 'g':3, 'n':4}


# i = 0
# path = '../African_Accented_French/'
# os.chdir(path)
# wavs = glob.glob("*.wav")
# for f in wavs:
#     filename = ''.join(os.path.splitext(f)[:-1])
#     os.rename(path + filename + '.wav', path + 'fre_aafrench_u_' + str(1) + '-' + f'{i:05}' + '.wav') # rename wav
#     os.rename(path + filename + '.txt', path + 'fre_aafrench_u_' + str(1) + '-' + f'{i:05}' + '.txt') # rename txt
#     i+=1
# print("num wavs:", len(glob.glob("*.wav")))

"""
Measuring total time of wavs
"""
# time = 0
# os.chdir('../African_Accented_French/')
# wavs = sorted(glob.glob("*.wav"))
# for f in wavs: # go through all files that are wav files
#     time += librosa.get_duration(filename=f)
# print(time)
