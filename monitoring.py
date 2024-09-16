import os
import time, datetime, psutil, subprocess


def display_train_monitoring_worker(use_gpu: bool, repeat: bool = True, interval:int = 300) -> None:
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

        if use_gpu:
            result_subprocess = subprocess.run(['nvidia-smi --query-gpu=name,index,utilization.gpu,memory.used,power.draw --format=csv'], capture_output=True, text=True, shell=True)

            gpu_text = result_subprocess.stdout
            gpu_text = gpu_text.split(", power.draw [W]\n")[1]
            gpu_text0 = gpu_text.split("\n")[0]
            gpu_text1 = gpu_text.split("\n")[1]

            text += f"\n{gpu_text0} {gpu_fire(gpu_text0)}"
            text += f"\n{gpu_text1} {gpu_fire(gpu_text1)}"

        print(text)



    start_time = time.time()

    time.sleep(0.1)

    disp(0.1)

    if repeat:
        time.sleep(10)

        disp(1)

    while repeat:
        time.sleep(interval)

        disp(1)


if __name__ == '__main__':
    display_train_monitoring_worker(True, False)