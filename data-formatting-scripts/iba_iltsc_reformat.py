import os

def extract_text():
    os.mkdir("data/reformat")
    for subdir in ["dev", "train"]:
        with open(f"data/{subdir}/text", 'r') as alltexts:
            for line in alltexts:
                data = line.split(' ')
                txt_name = "data/reformat/" + data[0] + ".txt"
                txt = ""
                for word in data:
                    if word != "" and not (word.startswith("ibf") or word.startswith("ibf")):
                        txt += word
                        txt += " "
                with open(txt_name, 'w') as newtxt:
                    newtxt.write(txt.rstrip())

def move_wavs():
    for dir in os.listdir("data/wav"):
        for file in os.listdir(f"data/wav/{dir}"):
            os.rename(f"data/wav/{dir}/{file}", f"data/reformat/{file}")

def rename_files():
    for filename in os.listdir("data/reformat"):
        if filename == "text.txt":
            continue

        if filename[-3:] == "txt":
            midname = filename.replace('.txt', '')
            ending = ".txt"
        elif filename[-3:] == "wav":
            midname = filename.replace('.wav', '')
            ending = ".wav"
        else:
            continue
        info = midname.split('_')

        newname = 'iba_iltsc_'
        if info[0].endswith("f"):
            newname += "f_"
        elif info[0].endswith("m"):
            newname += "m_"
        else:
            print("ERROR: SOMETHING IS VERY WRONG")
            break
        newname += info[1]
        newname += "_"
        newname += "00" + info[2]
        newname += ending
        #print(newname)
        os.rename(f"data/reformat/{filename}", f"data/reformat/{newname}")

def main():
    extract_text()
    move_wavs()
    rename_files()

if __name__ == "__main__":
    main()
