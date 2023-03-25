# deleted the test folder because train is enough hours

import os, glob, shutil, random, librosa

path = '../KOR/'

# # Flatten
# for a in os.listdir(path):
#     if os.path.isdir(path + a):
#         for b in os.listdir(path + a + '/'):
#             if os.path.isdir(path + a + '/' + b):
#                 for c in os.listdir(path + a + '/' + b):
#                     if os.path.isdir(path + a + '/' + b + '/' + c):
#                         for d in os.listdir(path + a + '/' + b + '/' + c):
#                             if os.path.isdir(path + a + '/' + b + '/' + c + '/' + d):
#                                 print(d)
#                             else:
#                                 os.rename(path + a + '/' + b + '/' + c + '/' + d, path + d)

# # Delete train_data_01
# for a in os.listdir(path):
#     if os.path.isdir(path + a):
#         shutil.rmtree(path + a)



# Combine transcripts into one

# print(len(os.listdir(path))) #num files in KOR

# texts = []
# os.chdir(path)
# txts = glob.glob("*.txt")
# for f in txts:
#     texts.append(path + f)
# print(len(texts)) #num texts

# with open(path + '_transcript.txt', 'w') as outfile:
#     for fname in texts:
#         with open(fname) as infile:
#             for line in infile:
#                 outfile.write(line)
#         os.remove(path + fname)
# print(len(os.listdir(path))) #num files in KOR






# # Create copies of flac files but as wav
# os.chdir(path)
# flacs = glob.glob("*.flac")
# for f in flacs:
#     os.system('ffmpeg -i ' + path + f + ' ' + path + f.rsplit( ".", 1 )[ 0 ] + '.wav') # use ffmpeg to convert it to wav




# # Deleting the flac files since we made all of the wav files
# os.chdir(path)
# print("num flacs:", len(glob.glob("*.flac")))
# print("num wavs:", len(glob.glob("*.wav")))
# for f in glob.glob("*.flac"): # go through all files that are flac files
#     os.remove(path + f) # delete the old flac file
# print("num flacs:", len(glob.glob("*.flac")))
# print("num wavs:", len(glob.glob("*.wav")))




# # one text per wav
# transcripts = {}
# with open(path + '_transcript.txt', 'r') as txt:
#     for line in txt:
#         filename = line.split(' ')[0]
#         text = ' '.join(line.split(' ')[1:])

#         if filename in transcripts and transcripts[filename] != text:
#             print(filename, "same file name, different transcrit found")
#         transcripts[filename] = text

# os.chdir(path)
# wavs = glob.glob("*.wav")
# missing = 0
# for wav in wavs:
#     if wav not in transcripts:
#         if os.path.splitext(wav)[0] not in transcripts:
#             missing += 1
# print("missing transcripts", missing, "out of", len(wavs))

# os.chdir(path)
# wavs = glob.glob("*.wav")
# for wav in wavs:
#     with open(path + os.path.splitext(wav)[0] + '.txt', 'w') as txt:
#         txt.write(transcripts[os.path.splitext(wav)[0]])
# os.remove(path + '_transcript.txt')






# trim to 36000
# os.chdir(path)
# time = 0
# wavs = sorted(glob.glob("*.wav"))
# random.shuffle(wavs)

# for f in wavs: # go through all files that are wav files
#     if time >= 36000:
#         without_extension = os.path.splitext(f)[0]
#         os.remove(path+without_extension+".wav")
#         os.remove(path+without_extension+".txt")
#     else:
#         time += librosa.get_duration(filename=path+f)
# print(time)




# # sex dict creation
# sexdict = {}
# m = 1
# f = 1
# with open(path + 'AUDIO_INFO', 'r') as file:
#     for line in file:
#         splitline = line.split('|')
#         mf = splitline[2]
#         sexdict[splitline[0]] = (mf, m if mf == 'm' else f)
#         if mf == 'm':
#             m += 1
#         else:
#             f += 1
# os.remove(path + 'AUDIO_INFO')



# # Rename wav text pairs
# os.chdir(path)
# i = 0
# wavs = sorted(glob.glob("*.wav"))
# for f in wavs: # go through all files that are wav files
#     filename = ''.join(os.path.splitext(f)[:-1])


#     speakerid = ''.join([x for x in filename][:3])
#     sex = sexdict[speakerid]

#     os.rename(path + filename + '.wav', path + 'kor_zeroth_' + sex[0] + '_' + str(sex[1]) + '-' + f'{i:05}' + '.wav') # rename wav
#     os.rename(path + filename + '.txt', path + 'kor_zeroth_' + sex[0] + '_' + str(sex[1]) + '-' + f'{i:05}' + '.txt') # rename txt
#     i+=1
# print("num wavs:", len(glob.glob("*.wav")))



# # total time
# time = 0
# os.chdir(path)
# wavs = sorted(glob.glob("*.wav"))
# for f in wavs: # go through all files that are wav files
#     time += librosa.get_duration(filename=f)
# print(time)

