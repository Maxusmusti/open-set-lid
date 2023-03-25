# data -> wavs
# trans -> text files
# hand deleted the rest of the other files

import os, glob, shutil, random, librosa

path = '../UIG/'

# # Flatten
# for a in os.listdir(path):
#     if os.path.isdir(path + a):
#         for b in os.listdir(path + a + '/'):
#             os.rename(path + a + '/' + b, path + b)





# # Combine transcripts into one

# texts = []
# os.chdir(path)
# wavs = glob.glob("*.wav")
# for f in os.listdir(path):
#     if f not in wavs:
#         texts.append(path + f)
# print(texts) 

# with open(path + '_transcript.txt', 'w') as outfile:
#     for fname in texts:
#         with open(fname) as infile:
#             for line in infile:
#                 outfile.write(line)
#         os.remove(path + fname)






# # one text per wav
# transcripts = {}
# with open(path + '_transcript.txt', 'r') as txt:
#     for line in txt:
#         filename = line.split(' ')[0].split('/')[-1]
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
#         txt.write(transcripts[wav])
# os.remove(path + '_transcript.txt')







# # trim to 36000
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







# # rename wav and txt pairs
# sexdict = {'F': 'f', 'M': 'm'}

# os.chdir(path)
# i = 0
# wavs = sorted(glob.glob("*.wav"))
# for f in wavs: # go through all files that are wav files
#     filename = ''.join(os.path.splitext(f)[:-1])

#     sex = sexdict[''.join([x for x in filename][:1])]

#     os.rename(path + filename + '.wav', path + 'uig_thuyg_' + sex + '_u-' + f'{i:05}' + '.wav') # rename wav
#     os.rename(path + filename + '.txt', path + 'uig_thuyg_' + sex + '_u-' + f'{i:05}' + '.txt') # rename txt
#     i+=1
# print("num wavs:", len(glob.glob("*.wav")))




# # total time
# time = 0
# os.chdir(path)
# wavs = sorted(glob.glob("*.wav"))
# for f in wavs: # go through all files that are wav files
#     time += librosa.get_duration(filename=f)
# print(time)

