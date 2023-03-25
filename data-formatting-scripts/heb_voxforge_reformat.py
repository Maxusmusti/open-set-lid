# a bunch of tar files

import os, glob, shutil, librosa
import random


path = '../HEB/'

# # extract tar files
# for i in os.listdir(path):
#     os.system('tar zxvf ' + path + i + ' -C ' + path)
#     os.remove(path + i)


# # remove LICENSE
# for i in os.listdir(path):
#     if os.path.isdir(path + i):
#         os.remove(path + i + '/LICENSE')


# # flatten
# for i in os.listdir(path):
#     if os.path.isdir(path + i):
#         for j in os.listdir(path + i):
#             if os.path.isdir(path + i + '/' + j):
#                 for k in os.listdir(path + i + '/' + j):
#                     os.rename(path + i + '/' + j + '/' + k, path + i + '/' + k)
#             shutil.rmtree(path + i + '/' + j)


# # remove everything except wav and prommts-original
# for i in os.listdir(path):
#     if os.path.isdir(path + i):
#         os.remove(path + i + '/README')
#         os.remove(path + i + '/PROMPTS')
#         os.remove(path + i + '/GPL_license.txt')


# # rename to _transcript.txt
# for i in os.listdir(path):
#     if os.path.isdir(path + i):
#         os.rename(path + i + '/prompts-original', path + i + '/_transcript.txt')



# # one text per wav
# for folder_tar in os.listdir(path):
#     transcripts = {}
#     with open(path + folder_tar + '/_transcript.txt', 'r') as txt:
#         for line in txt:
#             filename = line.split(' ')[0]
#             text = ' '.join(line.split(' ')[1:])

#             if filename in transcripts and transcripts[filename] != text:
#                 print(filename, "same file name, different transcrit found")
#             transcripts[filename] = text

#     wavs = glob.glob(path + folder_tar + "/*.wav")
#     missing = 0
#     for wav in wavs:
#         if wav not in transcripts:
#             if os.path.splitext(wav)[0].split('/')[-1] not in transcripts:
#                 missing += 1
#     print("missing transcripts", missing, "out of", len(wavs))

#     wavs = glob.glob(path + folder_tar + "/*.wav")
#     for wav in wavs:
#         with open(path + folder_tar + '/' + os.path.splitext(wav)[0].split('/')[-1] + '.txt', 'w') as txt:
#             txt.write(transcripts[os.path.splitext(wav)[0].split('/')[-1]])
#     os.remove(path + folder_tar + '/_transcript.txt')





# #  trim to 36000
# os.chdir(path)
# time = 0
# for folder_tar in os.listdir(path):
#     wavs = glob.glob(path + folder_tar + "/*.wav")
#     random.shuffle(wavs)

#     for f in wavs: # go through all files that are wav files
#         if time >= 36000:
#             without_extension = os.path.splitext(f)[0]
#             os.remove(without_extension+".wav")
#             os.remove(without_extension+".txt")
#         else:
#             time += librosa.get_duration(filename=f)
#     print(time)


# # Rename wav text pairs
# i = 0
# for folder_tar in os.listdir(path):
#     wavs = sorted(glob.glob(path + folder_tar+ "/*.wav"))
#     for f in wavs: # go through all files that are wav files
#         filename = (''.join(os.path.splitext(f)[:-1])).split('/')[-1]
#         os.rename(
#             path + folder_tar + '/' + filename + '.wav', 
#             path + folder_tar + '/' + 'heb_voxforge_' + 'u' + '_' + 'u' + '-' + f'{i:05}' + '.wav'
#         ) 
#         os.rename(
#             path + folder_tar + '/' + filename + '.txt', 
#             path + folder_tar + '/' + 'heb_voxforge_' + 'u' + '_' + 'u' + '-' + f'{i:05}' + '.txt'
#         )
#         i+=1



# # flatten
# for i in os.listdir(path):
#     for j in os.listdir(path + i):
#         os.rename(path + i + '/' + j, path + j )
#     shutil.rmtree(path + i)

