# made a new folder called spanishchilean
# 2 folders each has wavs
# 2 tsv

import glob, os, librosa, shutil, random
path = '../BUR/'


#only wavs in sub directories

for f in os.listdir(path):
    if os.path.isdir(path + f):
        for file in os.listdir(path + f):
            if os.path.splitext(file)[1] != '.wav':
                os.remove(path + f + '/' + file)




# flatten
for f in os.listdir(path):
    if os.path.isdir(path+f):
        for wav in os.listdir(path+f):
            os.rename(path+f+'/'+wav, path+wav)
        shutil.rmtree(path+f)

#count wavs
os.chdir(path)
wavs = glob.glob("*.wav")
print(len(wavs))


#create one text
texts = []
os.chdir(path)
txts = glob.glob("*.tsv")
for f in txts:
    texts.append(path + f)


with open(path + '_transcript.txt', 'w') as outfile:
    for fname in texts:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)
        os.remove(path + fname)


# one text per wav
transcripts = {}
with open(path + '_transcript.txt', 'r') as txt:
    for line in txt:
        filename = line.split('\t')[0]
        text = line.split('\t')[1]

        if filename in transcripts and transcripts[filename] != text:
            print(filename, "same file name, different transcrit found")
        transcripts[filename] = text

os.chdir(path)
wavs = glob.glob("*.wav")
missing = 0
for wav in wavs:
    if wav not in transcripts:
        if os.path.splitext(wav)[0] not in transcripts:
            missing += 1
print("missing transcripts", missing, "out of", len(wavs))


os.chdir(path)
wavs = glob.glob("*.wav")
for wav in wavs:
    with open(path + os.path.splitext(wav)[0] + '.txt', 'w') as txt:
        txt.write(transcripts[os.path.splitext(wav)[0]])
os.remove(path + '_transcript.txt')



# trim to 36000
os.chdir(path)
time = 0
wavs = sorted(glob.glob("*.wav"))
random.shuffle(wavs)

for f in wavs: # go through all files that are wav files
    if time >= 36000:
        without_extension = os.path.splitext(f)[0]
        os.remove(path+without_extension+".wav")
        os.remove(path+without_extension+".txt")
    else:
        time += librosa.get_duration(filename=path+f)
print(time)


# rename wav and txt pairs
sexdict = {'bur': 'f', 'bum': 'm'}

os.chdir(path)
i = 0
wavs = sorted(glob.glob("*.wav"))
for f in wavs: # go through all files that are wav files
    filename = ''.join(os.path.splitext(f)[:-1])

    sex = sexdict[''.join([x for x in filename][:3])]

    os.rename(path + filename + '.wav', path + 'bur_burcs_' + sex + '_u-' + f'{i:05}' + '.wav') # rename wav
    os.rename(path + filename + '.txt', path + 'bur_burcs_' + sex + '_u-' + f'{i:05}' + '.txt') # rename txt
    i+=1
print("num wavs:", len(glob.glob("*.wav")))


# total time
time = 0
os.chdir(path)
wavs = sorted(glob.glob("*.wav"))
for f in wavs: # go through all files that are wav files
    time += librosa.get_duration(filename=f)
print(time)
