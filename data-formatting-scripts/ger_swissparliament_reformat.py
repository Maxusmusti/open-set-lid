import os, random, shutil, librosa, glob

path = '../GER/'





# # Combine transcripts into one
# texts = []
# os.chdir(path)
# tsvs = glob.glob("*.tsv")
# for f in tsvs:
#     texts.append(path + f)
# print(texts) 

# with open(path + '_transcript.txt', 'w') as outfile:
#     for fname in texts:
#         with open(fname) as infile:
#             for line in infile:
#                 outfile.write(line)
#         os.remove(path + fname)






# #  trim to 36000
# time = 0
# wavs = glob.glob(path + "clips/*.flac")
# random.shuffle(wavs)
# for f in wavs:
#     if time >= 36000:
#         os.remove(f)
#     else:
#         time += librosa.get_duration(filename=f)
# print(time)




# # flatten
# for i in os.listdir(path + 'clips/'):
#     os.rename(path + 'clips/' + i, path + i)
# shutil.rmtree(path + 'clips/')







# # one text per wav
# transcripts = {}
# with open(path + '_transcript.txt', 'r') as txt:
#     for line in txt:
#         filename = line.split('\t')[1]
#         text = line.split('\t')[2]
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
#         try:
#             txt.write(transcripts[os.path.splitext(flac)[0].split('/')[-1] + '.flac'])
#         except:
#             txt.write('missing transcript')
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
#         path  + '/' + 'ger_swissparliament_' + 'u' + '_' + 'u' + '-' + f'{i:05}' + '.wav'
#     ) 
#     os.rename(
#         path  + '/' + filename + '.txt', 
#         path  + '/' + 'ger_swissparliament_' + 'u' + '_' + 'u' + '-' + f'{i:05}' + '.txt'
#     )
#     i+=1
