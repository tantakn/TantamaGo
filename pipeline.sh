set -eu

dt=$(date '+%Y%m%d_%H%M%S')

for i in `seq 0 10` ; do
    python3 selfplay_main.py --save-dir archive --model model/rl-model.bin --use-gpu True --net DualNet_256_24
    python3 get_final_status.py
    python3 train.py --rl True --kifu-dir archive --use-gpu True --rl-num $i --rl-datetime $dt --net DualNet_256_24
done
