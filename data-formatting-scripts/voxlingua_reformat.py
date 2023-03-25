import os,glob,random,librosa
import shutil


# extract by hand into open-set-lid with 3 letter folder name, e.g. open-set-lid/TIB


def main():
    reformat('PUS')
    reformat('SWE')
    reformat('THA')
    reformat('URD')

def reformat(lang):
    print(lang)
    path = '../' + lang + '/'
    os.chdir(path)



    # flatten NOR/no/wavs
    for x in os.listdir(path):
        if os.path.isdir(x):
            assert(len(os.listdir(path + x)) == len(glob.glob(path + x + '/*.wav')))
            print(len(os.listdir(path + x)), len(glob.glob(path + x + '/*.wav')))
            wavs = os.listdir(path + x)
            for wav in wavs:
                os.rename(path + x + '/' + wav, path + wav)
            shutil.rmtree(path + x)



    #  trim to 36000
    os.chdir(path)
    time = 0
    wavs = glob.glob(path + "*.wav")
    # random.shuffle(wavs)
    for f in wavs: 
        if time >= 36000:
            os.remove(f)
        else:
            time += librosa.get_duration(filename=f)
    print(time)

    
    # rename wavs
    i = 0
    os.chdir(path)
    wavs = sorted(glob.glob("*.wav"))
    for f in wavs:
        filename = ''.join(os.path.splitext(f)[:-1])
        # print(filename)
        os.rename(path + filename + '.wav',path + lang.lower() + '_voxlingua_u_u-' + f'{i:05}' + '.wav') # rename wav
        i+=1



    # 1 text per wav
    os.chdir(path)
    wavs = sorted(glob.glob("*.wav"))
    for f in wavs:
        filename = ''.join(os.path.splitext(f)[:-1])
        with open(path + filename + '.txt','w') as transcript:
            transcript.write("missing transcript")




if __name__ == "__main__":
    main()
