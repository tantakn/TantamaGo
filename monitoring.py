import os
import time, datetime, psutil, subprocess
import torch


def display_train_monitoring_worker(use_gpu: bool, repeat: bool = True, interval:int = 300, pid: int=-1, msg: str="") -> None:
    """ハードの使用率を表示する。

    Args:
        use_gpu (bool): GPU使用フラグ。
        repeat (bool): 繰り返し表示フラグ。
        interval (int): 表示間隔（秒）。デフォルトは300秒。
        msg (str): 追加メッセージ。
        pid (int): プロセスID。os.getpid()
    """

    def fire(percentage: float) -> str:
        if percentage > 99:
            return "🔥🔥🔥"
        elif percentage > 50:
            return "🔥🔥"
        elif percentage > 10:
            return "🔥"
        else:
            return ""

    def gpu_fire(text: str) -> str:
            mem = float(text.split(" %, ")[1].split(" MiB, ")[0])
            return "🔥" if mem > 30 else ""

    def disp(waittime: float) -> None:
        text = ""

        text += f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] monitoring"

        cpu_use = psutil.cpu_percent(interval=waittime)
        text += f"\ncpu: {cpu_use}% {psutil.cpu_percent(interval=waittime, percpu=True)} {fire(cpu_use)}"

        # if pid != -1:
        #     parent = psutil.Process(os.getpid())

        #     # 親プロセスと子プロセスのリストを取得
        #     processes = [parent] + parent.children(recursive=True)
    
        #     try:
        #         cup_usages = [process.cpu_percent(interval=1) for process in processes]
        #         memory_usages = [process.memory_info().rss / 1024 / 1024 for process in processes]
        #     except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        #         print(e)

        #     # total_cpu_usage = 0.0
        #     # total_memory_usage = 0
            
        #     # # 各プロセスのCPU使用率とメモリ使用量を集計
        #     # for process in processes:
        #     #     try:
        #     #         total_cpu_usage += process.cpu_percent(interval=waittime)
        #     #         total_memory_usage += process.memory_info().rss
        #     #     except (psutil.NoSuchProcess, psutil.AccessDenied):
        #     #         continue

        #     text += f"\ncpu(process): {cup_usages}%"

        #     # text += "\n"
        #     # for process in processes:
        #     #     try:
        #     #         text += f"(c: {process.cpu_percent(interval=waittime)})"
        #     #         text += f"(m: {process.memory_info().rss / 1024 / 1024}MB)"
        #     #     except (psutil.NoSuchProcess, psutil.AccessDenied):
        #     #         continue
        #     # text += "\n"

        mem_use = psutil.virtual_memory().percent
        text += f"\nmem: {mem_use}% {fire(mem_use)}"
        # soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        # mem_use = psutil.virtual_memory().percent
        # text += f"\nmem: {mem_use}% (soft: {soft / (1024 ** 3):.2f} GB, hard: {hard / (1024 ** 3):.2f} GB) {fire(mem_use)}"

        # if pid != -1:
        #     text += f"\nmem(process): {memory_usages}"

        if use_gpu:
            result_subprocess = subprocess.run(['nvidia-smi --query-gpu=name,index,utilization.gpu,memory.used,power.draw --format=csv'], capture_output=True, text=True, shell=True)

            gpu_text = result_subprocess.stdout
            gpu_text = gpu_text.split(", power.draw [W]\n")[1]

            for i in range(torch.cuda.device_count()):
                tmp_text = gpu_text.split("\n")[i]
                text += f"\n{tmp_text} {gpu_fire(tmp_text)}"


        # text += f"\nresource.RUSAGE_SELF: {resource.getrusage(resource.RUSAGE_SELF)},\nresource.RUSAGE_CHILDREN: {resource.getrusage(resource.RUSAGE_CHILDREN)},\nresource.RUSAGE_THREAD: {resource.getrusage(resource.RUSAGE_THREAD)}"###############


        print(text + "\n" + msg)


    start_time = time.time()

    time.sleep(0.1)

    disp(0.1)

    if repeat:
        time.sleep(10 if interval > 10 else interval)

        disp(None)

    while repeat:
        time.sleep(interval)

        disp(None)


if __name__ == '__main__':
    display_train_monitoring_worker(True, False)