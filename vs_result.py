import os, shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import re, glob, click, datetime


@click.command()
@click.option("--path", type=click.STRING, default="./archive", help="棋譜フォルダがあるディレクトリ。デフォルトはarchive。")
@click.option("--num", type=click.INT, default=-1, help="棋譜フォルダの中の数字")
def main(path, num):
    path_list = []
    if num == -1:
        folder_list = sorted(glob.glob(os.path.join("./", path, "*")))
        for folder in folder_list:
            path_list += sorted(glob.glob(os.path.join("./", folder, "*.sgf")))
    else:
        path_list = sorted(glob.glob(os.path.join("./", path, str(num), "*.sgf")))

    if path_list == []:
        print("error: 棋譜がないです")
        return

    print(path_list)######

    model1 = ""
    model2 = ""

    with open(path_list[0]) as f:
        sgf = f.read()
        model1 = re.search(r"PB\[(.*?)\]", sgf).group(1)
        model2 = re.search(r"PW\[(.*?)\]", sgf).group(1)

    result = []
    for tmp_path in path_list:
        with open(tmp_path) as f:
            sgf = f.read()

            #(;FF[4]GM[1]SZ[9]
            # AP[TantamaGo]PB[model/sl-model_default.bin]PW[model/sl-model20240711.bin]RE[W+88.0]KM[7.0];B[ha]C[82 A9:2.243

            # "B" or "W" or "0" (<-draw) ==
            win = re.search(r"RE\[([BW0])[.+\d]*?\]", sgf).group(1)

            is_black_model1 = 1 if re.search(r"PB\[(.*?)\]", sgf).group(1) == model1 else 0

            if win == "0":
                result.append(0.5)
            elif win == "B":
                result.append(is_black_model1)
            else:
                result.append(1 - is_black_model1)




    text = f"""
games: {len(result)}
model1: {model1}, win: ({result.count(1)} + 0.5*{result.count(0.5)})/{len(result)} ({result.count(1) / len(result) * 100:.2f}%)
model2: {model2}, win: ({result.count(0)} + 0.5*{result.count(0.5)})/{len(result)} ({result.count(0) / len(result) * 100:.2f}%)
"""

    print(text)

    if num == -1:
        with open(os.path.join("./", path, f"0000result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), mode="w") as f:
            f.write(text)
    else:
        with open(os.path.join("./", path, str(num), f"0000result_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"), mode="w") as f:
            f.write(text)

if __name__ == "__main__":
    main()