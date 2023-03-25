import os

def extract_text():
    with open("text.txt", 'r') as alltexts:
        for line in alltexts:
            name, txt = line.split('\t')
            txt_name = name.replace('.wav', '.txt')
            with open(txt_name, 'w') as newtxt:
                newtxt.write(txt)

def rename_files():
    for filename in os.listdir():
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

        newname = 'eng_freest_'
        if info[0].startswith("m"):
            newname += "male_"
        elif info[0].startswith("f"):
            newname += "female_"
        newname += info[2][-1]
        newname += "_"
        newname += info[3]
        newname += ending
        #print(newname)
        os.rename(filename, newname)

def main():
    #print(os.listdir())
    extract_text()
    rename_files()

if __name__ == "__main__":
    main()
