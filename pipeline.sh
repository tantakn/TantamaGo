set -eu

dt=$(date '+%Y%m%d_%H%M%S')

for i in `seq 0 100` ; do
    python3 selfplay_main.py --save-dir archive --model model_def/sl-model_DualNet_256_24_semeai.bin --use-gpu True --net DualNet_256_24_semeai
    python3 get_final_status.py
    python3 train.py --rl True --kifu-dir archive --use-gpu True --rl-num $i --rl-datetime $dt --net DualNet_256_24_semeai
done
