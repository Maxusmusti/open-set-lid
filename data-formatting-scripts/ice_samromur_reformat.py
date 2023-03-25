import os, random, shutil, librosa, glob

path = '../ICE/'




# # flatten and trim to 36000 (only need dev because it's enough)
# time = 0
# for i in os.listdir(path + 'dev/'):
#     for j in glob.glob(path + 'dev/' + i + '/*.flac'):
#         if time >= 36000:
#             os.remove(j)
#         else:
#             time += librosa.get_duration(filename=j)
# print(time)



# # remove test and train
# shutil.rmtree(path +'test')
# shutil.rmtree(path +'train')



# # flatten
# for i in os.listdir(path + 'dev/'):
#     os.rename(path + 'dev/' + i, path + i)
# for i in os.listdir(path):
#     if os.path.isdir(path + i):
#         for j in os.listdir(path +  i):
#             os.rename(path + i + '/' + j, path + j)
#         shutil.rmtree(path + i)



# # rename to _transcript.txt
# os.rename(path + 'metadata.tsv', path + '_transcript.txt')



# # one text per wav
# transcripts = {}
# with open(path + '_transcript.txt', 'r') as txt:
#     for line in txt:
#         filename = line.split('\t')[2]
#         text = line.split('\t')[3]
#         if filename in transcripts and transcripts[filename] != text:
#             print(filename, "same file name, different transcrit found")
#         transcripts[filename] = text

# flacs = glob.glob(path + "*.flac")
# missing = 0
# for flac in flacs:
#     if flac.split('/')[-1] not in transcripts:
#         missing += 1
# print("missing transcripts", missing, "out of", len(flacs))

# flacs = glob.glob(path + "*.flac")
# for flac in flacs:
#     with open(path + os.path.splitext(flac)[0].split('/')[-1] + '.txt', 'w') as txt:
#         txt.write(transcripts[os.path.splitext(flac)[0].split('/')[-1] + '.flac'])
# os.remove(path + '_transcript.txt')







# # make wavs
# os.chdir(path)
# flacs = glob.glob("*.flac")
# for f in flacs: # go through all files that are flac files
#     os.system('ffmpeg -i ' + path + f + ' ' + path + f.rsplit( ".", 1 )[ 0 ] + '.wav') # use ffmpeg to convert it to wav


# # delete flacs
# for f in glob.glob("*.flac"): # go through all files that are flac files
#     os.remove(path + f) # delete the old flac file


# # Rename wav text pairs
# i = 0
# wavs = sorted(glob.glob(path + "/*.wav"))
# for f in wavs: # go through all files that are wav files
#     filename = (''.join(os.path.splitext(f)[:-1])).split('/')[-1]
#     os.rename(
#         path +  '/' + filename + '.wav', 
#         path  + '/' + 'ice_samromur_' + 'u' + '_' + 'u' + '-' + f'{i:05}' + '.wav'
#     ) 
#     os.rename(
#         path  + '/' + filename + '.txt', 
#         path  + '/' + 'ice_samromur_' + 'u' + '_' + 'u' + '-' + f'{i:05}' + '.txt'
#     )
#     i+=1
