import os
import time, datetime, psutil, subprocess
import torch


def display_train_monitoring_worker(use_gpu: bool, repeat: bool = True, interval:int = 300, msg: str="") -> None:
    """ハードの使用率を表示する。

    Args:
        use_gpu (bool): GPU使用フラグ。
        repeat (bool): 繰り返し表示フラグ。
        interval (int): 表示間隔（秒）。デフォルトは300秒。
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

        mem_use = psutil.virtual_memory().percent
        text += f"\nmem: {mem_use}% {fire(mem_use)}"
        # assert mem_use < 90, f"memory usage is too high."

        if use_gpu:
            result_subprocess = subprocess.run(['nvidia-smi --query-gpu=name,index,utilization.gpu,memory.used,power.draw --format=csv'], capture_output=True, text=True, shell=True)

            gpu_text = result_subprocess.stdout
            gpu_text = gpu_text.split(", power.draw [W]\n")[1]

            for i in range(torch.cuda.device_count()):
                tmp_text = gpu_text.split("\n")[i]
                text += f"\n{tmp_text} {gpu_fire(tmp_text)}"

        print(text)


    start_time = time.time()

    time.sleep(0.1)

    disp(0.1)

    if repeat:
        time.sleep(10 if interval > 10 else interval)

        disp(1)

    while repeat:
        time.sleep(interval)

        disp(1)


if __name__ == '__main__':
    display_train_monitoring_worker(True, False)