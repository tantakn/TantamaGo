import os
import time, datetime, psutil, subprocess


def display_train_monitoring_worker(use_gpu: bool, repeat: bool = True) -> None:
    """ハードの使用率を表示する。

    Args:
        use_gpu (bool): GPU使用フラグ。
        repeat (bool): 繰り返し表示フラグ。
    """

    def disp(waittime: float) -> None:
        print(f"monitoring [datetime: {datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}]")
        print(f"cpu: {psutil.cpu_percent(interval=waittime)}% {psutil.cpu_percent(interval=1, percpu=True)}")
        print(f"mem: {psutil.virtual_memory().percent}%")

        if use_gpu:
            result_subprocess = subprocess.run(['nvidia-smi --query-gpu=name,index,utilization.gpu,memory.used,power.draw --format=csv'], capture_output=True, text=True, shell=True)
            print(result_subprocess.stdout)

    start_time = time.time()

    time.sleep(0.1)

    disp(0.1)

    while repeat:
        time.sleep(60)

        disp(1)


if __name__ == '__main__':
    display_train_monitoring_worker(True, False)