import os
import json
import librosa
import random

def extract_and_move():
    os.mkdir("reformat")
    for dir in ["dev", "train", "test"]:
        with open(f"{dir}/manifest.json", 'r') as alltexts:
            for line in alltexts:
                #print(line)
                if line != "":
                    entry = json.loads(line)
                    #print(entry["audio_filepath"])
                    audio = entry["audio_filepath"]
                    name = audio.split('/')[-1]
                    os.rename(f"{dir}/{audio}", f"reformat/{name}")
                    text = entry["text"]
                    text_name = name.replace('.wav', '.txt')
                    with open(f"reformat/{text_name}", 'w') as newtxt:
                        newtxt.write(text)

def rename_files():
    for filename in os.listdir("reformat"):

        if filename[-3:] == "txt":
            midname = filename.replace('.txt', '')
            ending = ".txt"
        elif filename[-3:] == "wav":
            midname = filename.replace('.wav', '')
            ending = ".wav"
        else:
            continue
        info = midname.split('_')

        newname = 'rus_ruls_u_'
        for i in range(len(info) - 1):
            newname += info[i]
            if i < len(info) - 2:
                newname += "-"
        newname += "_0" + info[-1]
        newname += ending
        #print(newname)
        os.rename(f"reformat/{filename}", f"reformat/{newname}")

def trim_down():
    files = os.listdir("reformat")
    duration = 0.0
    for filename in files:
        if filename.endswith('wav'):
            duration += librosa.get_duration(filename=f"reformat/{filename}")
    print(duration)

    while duration > 36000:
        ind = random.randint(0, len(files) - 1)
        to_rem = files[ind]
        if to_rem.endswith('txt'):
            to_rem_txt = to_rem
            to_rem_wav = to_rem.replace(".txt", ".wav")
        elif to_rem.endswith('wav'):
            to_rem_wav = to_rem
            to_rem_txt = to_rem.replace(".wav", ".txt")
        duration -= librosa.get_duration(filename=f"reformat/{to_rem_wav}")
        os.remove(f"reformat/{to_rem_wav}")
        os.remove(f"reformat/{to_rem_txt}")
        files = os.listdir("reformat")

def main():
    extract_and_move()
    rename_files()
    trim_down()

if __name__ == "__main__":
    main()
