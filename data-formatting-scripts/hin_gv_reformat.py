import os
import subprocess

def extract_and_move():
    os.mkdir("reformat")
    with open("text", 'r') as text:
        for line in text:
            data = line.split(" ")
            file = data[0]
            text = ' '.join(data[1:])
            with open(f"reformat/{file}.txt", 'w') as new_txt:
                new_txt.write(text)
            os.rename(f"Audio/{file}.mp3", f"reformat/{file}.mp3")

def rename_reformat_files():
    sex = {}
    counters = {"m": 1, "u": 1, "f": 1}
    with open("utt2labels", 'r') as labels:
        for line in labels:
            data = line.split('\t')
            if data[0] == "Uttids":
                continue
            if data[3] == "Female":
                sex[data[0]] = "f"
            elif data[3] == "Male":
                sex[data[0]] = "m"
            else:
                sex[data[0]] = "u"

    with open("utt2labels", 'r') as labels:
        for line in labels:
            data = line.split('\t')
            if data[0] == "Uttids":
                continue
            midname = data[0]
            print(midname)
            for filename in [f"{midname}.txt", f"{midname}.mp3"]:
                if filename[-3:] == "txt":
                    ending = ".txt"
                elif filename[-3:] == "mp3":
                    newname = filename.replace("mp3", 'wav')
                    subprocess.run(["ffmpeg", "-i", f"reformat/{filename}", f"reformat/{newname}", "-loglevel", "error", "-stats"])
                    os.remove(f"reformat/{filename}")
                    filename = newname
                    ending = ".wav"
                else:
                    continue

                newname = 'hin_gv_'
                newname += sex[midname]
                newname += "_u_" + (5 - len(str(counters[sex[midname]]))) * "0" + str(counters[sex[midname]])
                newname += ending
                print(newname)
                os.rename(f"reformat/{filename}", f"reformat/{newname}")
            counters[sex[midname]] += 1

def main():
    extract_and_move()
    rename_reformat_files()

if __name__ == "__main__":
    main()
