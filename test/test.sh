set -eu

dt=$(date '+%Y%m%d_%H%M%S')

# echo -e "\naaa" > ./test.txt

echo -e "\naaa\n" | tee -a ./test.txt


# python3 -u ./get_final_status.py 2>&1 | tee -a ./test/test.txt

# for i in `seq 0 100` ; do
#     echo 
#     python3 -u selfplay_main.py --save-dir archive --model model_def/sl-model_DualNet_256_24_semeai.bin --use-gpu True --net DualNet_256_24_semeai 2>&1 | tee -a ./zzlog/${dt}rl.txt
#     python3 -u get_final_status.py --print False 2>&1 | tee -a ./zzlog/${dt}rl.txt
#     python3 -u train.py --rl True --kifu-dir archive --use-gpu True --rl-num $i --rl-datetime $dt --net DualNet_256_24_semeai 2>&1 | tee -a ./zzlog/${dt}rl.txt
# done