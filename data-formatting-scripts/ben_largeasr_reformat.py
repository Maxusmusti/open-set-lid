import os, shutil, random, glob, librosa

path = '../BEN/'



# # flatten
# for i in os.listdir(path):
#     if os.path.isdir(path + i):
#         for j in os.listdir(path + i):
#             for k in os.listdir(path + i + '/' + j):
#                 if os.path.isdir(path + i + '/' + j + '/' + k):
#                     for l in os.listdir(path + i + '/' + j + '/' + k):
#                         for s in os.listdir(path + i + '/' + j + '/' + k + '/' + l):
#                             os.rename(path + i + '/' + j + '/' + k + '/' + l + '/' + s, path + s)
#                             pass
#                     print(path + i + '/' + j + '/' + k + '/' + l)

#                 else:
#                     os.remove(path + i + '/' + j + '/' + k)
#             print(path + i + '/' + j + '/' + k)
#         print(path + i + '/' + j)






# # rename tsv to _transcript.txt
# os.rename(path + 'utt_spk_text.tsv', path + '_transcript.txt')






# #  trim to 36000
# os.chdir(path)
# time = 0
# wavs = glob.glob(path + "*.flac")
# random.shuffle(wavs)
# print(wavs)
# for f in wavs: 
#     if time >= 36000:
#         os.remove(f)
#     else:
#         time += librosa.get_duration(filename=f)
# print(time)





# # make wavs
# os.chdir(path)
# flacs = glob.glob("*.flac")
# for f in flacs: # go through all files that are flac files
#     os.system('ffmpeg -i ' + path + f + ' ' + path + f.rsplit( ".", 1 )[ 0 ] + '.wav') # use ffmpeg to convert it to wav


# # delete flacs
# for f in glob.glob("*.flac"): # go through all files that are flac files
#     os.remove(path + f) # delete the old flac file
# print("num flacs:", len(glob.glob("*.flac")))
# print("num wavs:", len(glob.glob("*.wav")))








# one text per wav
transcripts = {}
with open(path + '_transcript.txt', 'r') as txt:
    for line in txt:
        filename = line.split()[0] + '.wav'
        text = ' '.join(line.split()[2:])
        if filename in transcripts and transcripts[filename] != text:
            print(filename, "same file name, different transcrit found")
        transcripts[filename] = text

wavs = glob.glob(path + "*.wav")
missing = 0
for wav in wavs:
    if wav.split('/')[-1] not in transcripts:
        missing += 1
print("missing transcripts", missing, "out of", len(wavs))

wavs = glob.glob(path + "*.wav")
for wav in wavs:
    with open(path + os.path.splitext(wav)[0].split('/')[-1] + '.txt', 'w') as txt:
        txt.write(transcripts[os.path.splitext(wav)[0].split('/')[-1] + '.wav'])
os.remove(path + '_transcript.txt')






# Rename wav text pairs
i = 0
wavs = sorted(glob.glob(path + "/*.wav"))
for f in wavs: # go through all files that are wav files
    filename = (''.join(os.path.splitext(f)[:-1])).split('/')[-1]
    os.rename(
        path +  '/' + filename + '.wav', 
        path  + '/' + 'ben_largeasr_' + 'u' + '_' + 'u' + '-' + f'{i:05}' + '.wav'
    ) 
    os.rename(
        path  + '/' + filename + '.txt', 
        path  + '/' + 'ben_largeasr_' + 'u' + '_' + 'u' + '-' + f'{i:05}' + '.txt'
    )
    i+=1

