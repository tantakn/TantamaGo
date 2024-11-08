"""対局結果の補正処理。
"""
# -*- coding:utf-8 -*-
from concurrent.futures import ThreadPoolExecutor
import glob
import os
import math
import subprocess
from typing import NoReturn
import click

import logging
mylog = logging.getLogger("mylog")
#mylog.setLevel(logging.DEBUG)#####
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("🐕️%(asctime)s [🐾%(levelname)s🐾] %(pathname)s %(lineno)d %(funcName)s🐈️ %(message)s🦉", datefmt="%y%m%d_%H%M%S"))
mylog.addHandler(handler)

WORKER_THREAD = 4

cnt = 0#####


def get_gnugo_judgment(filename: str, is_japanese_rule: bool) -> str:
    """GNUGoの判定結果の文字列を取得する。

    Args:
        filename (str): 読み込むSGFファイルパス。
        is_japanese_rule (bool): 日本ルール化否かのフラグ。

    Returns:
        str: GNUGoの判定結果の文字列。
    """
    exec_commands = [
        f"loadsgf {filename}",
        "final_score",
    ]

    gnugo_command = [
        "./gnugo-3.8/interface/gnugo",
        "--mode",
        "gtp",
        "--level",
        "10"
    ]

    if is_japanese_rule:
        gnugo_command.append("--japanese-rule")
    else:
        gnugo_command.append("--chinese-rule")

    with subprocess.Popen(gnugo_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8') as process:

        process.stdin.write("\n".join(exec_commands))
        process.stdin.flush()
        process.stdout.flush()
        process.stdin.close()

        response = []
        for line in process.stdout.readlines():
            text = line.rstrip('\n')
            if text:
                response.append(text)

    result = ' '.join(response)

    # 0 : empty line
    # 1 : color
    # 2 : final score
    responses = result.split('= ')

    return responses[2]


def adjust_by_gnugo_judgment(filename: str, pri: bool) -> NoReturn:
    """_summary_

    Args:
        filename (str): _description_
    """
    with open(filename, encoding="utf-8") as in_file:
        sgf = in_file.read()

    if "+R" in sgf:
        return

    current_result = sgf.split('RE[')[1].split(']')[0]

    result = get_gnugo_judgment(filename, False)

    current_result_string = "RE[" + current_result + "]"
    adjust_result_string = "RE[" + result + "]"

    if pri:
        print(f"\rget_final_status  {filename}", end="", flush=True)#####

    global cnt#####
    if current_result_string != adjust_result_string:#####
        cnt += 1#####
        mylog.debug(f"\nAdjust result: {filename} {current_result_string} -> {adjust_result_string}")

    adjusted_sgf = sgf.replace(current_result_string, adjust_result_string)

    with open(filename, encoding="utf-8", mode="w") as out_file:
        out_file.write(adjusted_sgf)

def judgment_worker(kifu_list: str, print: bool) -> NoReturn:
    """_summary_

    Args:
        kifu_list (str): _description_
    """
    for filename in kifu_list:
        adjust_by_gnugo_judgment(filename, print)


@click.command()
@click.option('--kifu-dir', type=click.STRING, default='archive', help='')
@click.option('--pri', '--print', type=click.BOOL, default=True, help='')
def adjust_result(kifu_dir: str, pri: bool) -> NoReturn:
    """_summary_

    Args:
        kifu_dir (str): _description_
    """
    kifu_dir_index_list = [int(os.path.split(dir_path)[-1]) for dir_path in glob.glob(os.path.join(kifu_dir, '*'))]
    newest_index = max(kifu_dir_index_list)

    sgf_file_list = sorted(glob.glob(os.path.join(kifu_dir, str(newest_index), '*')))

    split_size = math.ceil(len(sgf_file_list) / WORKER_THREAD)
    split_file_lists = [sgf_file_list[idx:idx+split_size] \
        for idx in range(0, len(sgf_file_list), split_size)]

    executor = ThreadPoolExecutor(max_workers=WORKER_THREAD)
    futures = []
    for file_list in split_file_lists:
        future = executor.submit(judgment_worker, file_list, print)
        futures.append(future)

    for future in futures:
        future.result()

    if pri:
        global cnt#####
        print(f"\nchange : {cnt}")#####


if __name__ == "__main__":
    adjust_result() # pylint: disable=E1120
