import os

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

        newname = 'kas_klc_u_'
        for piece in info:
            if not piece.isnumeric():
                newname += piece
            else:
                break
        newname += "_00001"
        newname += ending
        #print(newname)
        os.rename(filename, newname)

def main():
    #print(os.listdir())
    rename_files()

if __name__ == "__main__":
    main()
