import resource
import subprocess
import psutil
import datetime
import torch
import numpy as np
from concurrent.futures import ProcessPoolExecutor

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


print("mem: ", psutil.virtual_memory())

# メモリ使用量を制限（単位：バイト）
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
print(f"🐾resource.getrlimit(resource.RLIMIT_AS): {resource.getrlimit(resource.RLIMIT_AS)}")

resource.setrlimit(resource.RLIMIT_AS, (1073741824 * 4, hard))  # 1GBに制限

print(f"🐾resource.getrlimit(resource.RLIMIT_AS): {resource.getrlimit(resource.RLIMIT_AS)}")

def tmp_load_data_set(npz_path, rank):
    print(f"🐾resource.getrlimit(resource.RLIMIT_AS)0: {resource.getrlimit(resource.RLIMIT_AS)}")
    try:
        def check_memory_usage():
            if not psutil.virtual_memory().percent < 90:
                print(f"memory usage is too high. mem_use: {psutil.virtual_memory().percent}% [{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}]")
                assert True

        check_memory_usage()

        data = np.load(npz_path)

        check_memory_usage()

        plane_data = data["input"]

        print("mem0: ", psutil.virtual_memory())

    except Exception as e:
        print(f"error: {npz_path}, {e}")
        return None
    return plane_data

# plane_data0 = tmp_load_data_set("../data/sl_data_0.npz", 1)

# print("mem0: ", psutil.virtual_memory())

# plane_data1 = tmp_load_data_set("../data/sl_data_1.npz", 1)

# print("mem1: ", psutil.virtual_memory())

# # サブプロセスを生成してメモリ制限を確認
# def run_subprocess():
#     subprocess.run(["python3", "-c", "import resource; print(resource.getrlimit(resource.RLIMIT_AS))"])

# run_subprocess()



with ProcessPoolExecutor(1) as executor:
    futures = [executor.submit(tmp_load_data_set, "../data/sl_data_0.npz", 1)]
    # futures = [executor.submit(tmp_load_data_set, "../data/sl_data_0.npz", 1) for file_list in file_indice]

    # この .result() は結果を出力するのが目的ではなく、正常終了の確認が目的。
    # 多分、executor.shutdown(wait=True) でも良い。
    # for future in futures:
    #     future.result()
    executor.shutdown(wait=True)