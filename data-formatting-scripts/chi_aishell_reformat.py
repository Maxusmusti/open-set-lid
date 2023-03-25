# CHI/data_aishell
# transcript/aisheel_transcript_v0.8.txt
# wav/tar fiels
import glob
import os,shutil,random, librosa


path = '../CHI/'



# # flatten structure
# for i in os.listdir(path):
#     for j in os.listdir(path + i):
#         os.rename(path + i + '/' + j, path + j)
# shutil.rmtree(path + 'data_aishell')



# flatten again
# for i in os.listdir(path):
#     for j in os.listdir(path + i):
#         os.rename(path + i + '/' + j, path + j)
# shutil.rmtree(path + 'transcript')
# shutil.rmtree(path + 'wav')



# # rename transcript to _transcript.txt
# os.rename(path + 'aishell_transcript_v0.8.txt', path + '_transcript.txt')



# # untar
# for tar in glob.glob(path + '*.tar.gz'):
#     os.system('tar zxvf ' + tar + ' -C ' + path)




# # remove tars
# for tar in glob.glob(path + '*.tar.gz'):
#     os.remove(tar)




# # flatten again 
# # Run it twice!!!!!
# for i in os.listdir(path):
#     if os.path.isdir(path + i):
#         for j in os.listdir(path + i + '/'):
#             os.rename(path + i + '/' + j, path + j)




# # remove empty folders
# for i in os.listdir(path):
#     if os.path.isdir(path + i):
#         shutil.rmtree(path + i)





# # trim to 36000
# time = 0
# wavs = sorted(glob.glob(path + "*.wav"))
# random.shuffle(wavs)

# for f in wavs: 
#     if time >= 36000:
#         os.remove(f)
#     else:
#         time += librosa.get_duration(filename=path+f)
# print(time)







# # one text per wav
# transcripts = {}
# with open(path + '_transcript.txt', 'r') as txt:
#     for line in txt:
#         filename = line.split()[0]
#         text = ' '.join(line.split()[1:])
#         if filename in transcripts and transcripts[filename] != text:
#             print(filename, "same file name, different transcrit found")
#         transcripts[filename] = text

# wavs = glob.glob(path + "*.wav")
# missing = 0
# for wav in wavs:
#     key = wav.split('/')[-1].split('.')[0]
#     if key not in transcripts:
#         missing += 1
# print("missing transcripts", missing, "out of", len(wavs))


# wavs = glob.glob(path + "*.wav")
# for wav in wavs:
#     key = wav.split('/')[-1].split('.')[0]
#     with open(path + key + '.txt', 'w') as txt:
#         try:
#             txt.write(transcripts[key])
#         except:
#             txt.write("missing transcript")
# os.remove(path + '_transcript.txt')







# # rename wav and txt pairs
# os.chdir(path)
# i = 0
# wavs = sorted(glob.glob("*.wav"))
# for f in wavs:
#     filename = ''.join(os.path.splitext(f)[:-1])
#     os.rename(path + filename + '.wav', path + 'chi_aitree_u_u-' + f'{i:05}' + '.wav') # rename wav
#     os.rename(path + filename + '.txt', path + 'chi_aitree_u_u-' + f'{i:05}' + '.txt') # rename txt
#     i+=1
# print("num wavs:", len(glob.glob("*.wav")))


