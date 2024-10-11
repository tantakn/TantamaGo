# 240606
```
C:\code\igo\TamaGo-main>C:/Users/〇〇/AppData/Local/Programs/Python/Python312/python.exe main.py --model model\sl-model.bin
Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.
                 It can be downloaded at https://aka.ms/vs/16/release/vc_redist.x64.exe
Traceback (most recent call last):
  File "C:\code\igo\TamaGo-main\main.py", line 7, in <module>
    from gtp.client import GtpClient
  File "C:\code\igo\TamaGo-main\gtp\client.py", line 15, in <module>
    from gtp.gogui import GoguiAnalyzeCommand, display_policy_distribution, \
  File "C:\code\igo\TamaGo-main\gtp\gogui.py", line 5, in <module>
    import torch
  File "C:\Users\〇〇\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\__init__.py", line 143, in <module>
    raise err
OSError: [WinError 126] 指定されたモジュールが見つかりません。 Error loading "C:\Users\〇〇\AppData\Local\Programs\Python\Python312\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```

ここで
https://learn.microsoft.com/ja-jp/cpp/windows/latest-supported-vc-redist?view=msvc-170#latest-microsoft-visual-c-redistributable-version
からx64のをインストール

```
C:\code\igo\TamaGo-main>C:/Users/〇〇/AppData/Local/Programs/Python/Python312/python.exe main.py --model model\sl-model.bin
name =tamago
= TamaGo

name
= TamaGo

version
= 0.9.2

showboard
Move : 1
Prisoner(Black) : 0
Prisoner(White) : 0
    A B C D E F G H J
  +-------------------+
 9| + + + + + + + + + |
 8| + + + + + + + + + |
 7| + + + + + + + + + |
 6| + + + + + + + + + |
 5| + + + + + + + + + |
 4| + + + + + + + + + |
 3| + + + + + + + + + |
 2| + + + + + + + + + |
 1| + + + + + + + + + |
  +-------------------+

=
```

pythonを一度アンインストールしてから再インストール（パスを通すにチェックいれる）

```
PS C:\code> python --version
Python 3.12.3
```

```
PS C:\code\igo\TamaGo-main> python main.py --model model\sl-model.bin
name
= TamaGo
```

# 240607
modelファイルにsl-model.binをダウンロードしていれて"python main.py --model model\sl-model.bin"
"--size 5"だと動かない。多分sl-model.binが対応してない。
```
PS C:\code\igo\TantamaGo> python main.py --model model\sl-model.bin
showboard
Move : 1
Prisoner(Black) : 0
Prisoner(White) : 0
    A B C D E F G H J
  +-------------------+
 9| + + + + + + + + + |
 8| + + + + + + + + + |
 7| + + + + + + + + + |
 6| + + + + + + + + + |
 5| + + + + + + + + + |
 4| + + + + + + + + + |
 3| + + + + + + + + + |
 2| + + + + + + + + + |
 1| + + + + + + + + + |
  +-------------------+

=

genmove b
raw_value=0.4992
pos=D6, visits=   39, policy=0.0597, value=0.4894, raw_value=0.5112, pv=D6,F4,E4,E3,D3,E5,D4,E6,C7,D5,C5,D7
pos=E6, visits=   51, policy=0.0646, value=0.5008, raw_value=0.5432, pv=E6,E4,C5,G5,G6,C4,B4,H6,F5,D5,C3
pos=F6, visits=   30, policy=0.0605, value=0.4803, raw_value=0.5104, pv=F6,D4,E4,E3,F3,D3,D7,G6,G5,F7
pos=D5, visits=   34, policy=0.0638, value=0.4833, raw_value=0.5608, pv=D5,F5,E7,E3,D3,F7,F8,D2,E4,E6,G7
pos=E5, visits=  302, policy=0.3238, value=0.5072, raw_value=0.5207, pv=E5,G5,G6,F6,F5,G7,H6,G4,F3,F7,G3,D6,H7,H8,J8,G8,F4,C4,C7,C6,D7
pos=F5, visits=  370, policy=0.0638, value=0.5362, raw_value=0.5565, pv=F5,D5,E3,E7,F7,D3,D6,E6,E5,C6,D4,D7,C4,F6,G6,G7,H7,F8,H5
pos=D4, visits=   24, policy=0.0597, value=0.4660, raw_value=0.4915, pv=D4,F6,E6,E7,D7,F7,F3,C4,C5,D3
pos=E4, visits=   52, policy=0.0641, value=0.5020, raw_value=0.5543, pv=E4,E6,C5,G5,G4,C6,B6,H4,F5,D5,C7,G3
pos=F4, visits=   31, policy=0.0600, value=0.4823, raw_value=0.4994, pv=F4,D6,D5,C5,C4,C6,G6,F3,E3,G4
8.72 seconds, 106.99 visits/s
= F5

showboard
Move : 2
Prisoner(Black) : 0
Prisoner(White) : 0
    A B C D E F G H J
  +-------------------+
 9| + + + + + + + + + |
 8| + + + + + + + + + |
 7| + + + + + + + + + |
 6| + + + + + + + + + |
 5| + + + + + @ + + + |
 4| + + + + + + + + + |
 3| + + + + + + + + + |
 2| + + + + + + + + + |
 1| + + + + + + + + + |
  +-------------------+

=
```

# 240704
source env/bin/activate の後ろに .csh　がつくらしい

# 20240707
```learning_param.py
"""学習用の各種ハイパーパラメータの設定。
"""

# 教師あり学習実行時の学習率
SL_LEARNING_RATE = 0.01

# 強化学習実行時の学習率
RL_LEARNING_RATE = 0.01

# ミニバッチサイズ
BATCH_SIZE = 256

# 学習器のモーメンタムパラメータ
MOMENTUM=0.9

# L2正則化の重み
WEIGHT_DECAY = 1e-4

EPOCHS = 1
# EPOCHS = 15

# 学習率を変更するエポック数と変更後の学習率
LEARNING_SCHEDULE = {
    "learning_rate": {
        1: 0.001,
        2: 0.0001,
        3: 0.00001,
    }
}
# LEARNING_SCHEDULE = {
#     "learning_rate": {
#         5: 0.001,
#         8: 0.0001,
#         10: 0.00001,
#     }
# }

# npzファイル1つに格納するデータの個数
DATA_SET_SIZE = BATCH_SIZE * 1000
# DATA_SET_SIZE = BATCH_SIZE * 4000

# Policyのlossに対するValueのlossの重み比率
SL_VALUE_WEIGHT = 0.02

# Policyのlossに対するValueのlossの重み比率
RL_VALUE_WEIGHT = 1.0

# 自己対戦時の探索回数
SELF_PLAY_VISITS = 16

# 自己対戦実行ワーカ数
NUM_SELF_PLAY_WORKERS = 4

# 1回の学習ごとに生成する棋譜の数
NUM_SELF_PLAY_GAMES = 10000
```
をやってできた f"sl_data_{i}.npz" の rep(i, 10) 以外消してtrain.pyしたら自分よりちょっと弱いAIできた。

# 20240714

igo/TantamaGo/model/sl-model202407144.bin

が自分より少し弱い

# 20240715

gitおかしくなった

# 20240814

branchTest

# 20240909
```shell
(env) (base) 〇〇@g17:~/igo/TantamaGo$ python3 selfplay_main.py --process 8 --num-data 200 --model model/rl-model_default.bin --model2 model/sl-m
odel_20240906_193431_e:03.bin
Self play visits : 16
Generating   107/  200 games (6417.5251 games/hour).
🐋cpu: 50.4% [0.0, 0.0, 100.0, 100.0, 0.0, 100.0, 99.0, 100.0, 100.0, 100.0, 2.9, 0.0, 100.0, 1.0, 0.0, 2.0]
🐋mem: 34.4%
Mon Sep  9 17:54:41 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.119.03   Driver Version: 450.119.03   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  TITAN RTX           On   | 00000000:03:00.0 Off |                  N/A |
| 40%   39C    P2    59W / 280W |   8603MiB / 24219MiB |     26%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  TITAN RTX           On   | 00000000:04:00.0 Off |                  N/A |
| 41%   29C    P8    11W / 280W |      9MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1722      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      1885      G   /usr/bin/gnome-shell                4MiB |
|    0   N/A  N/A     18969      C   python3                          1073MiB |
|    0   N/A  N/A     18970      C   python3                          1073MiB |
|    0   N/A  N/A     18971      C   python3                          1073MiB |
|    0   N/A  N/A     18972      C   python3                          1073MiB |
|    0   N/A  N/A     18973      C   python3                          1073MiB |
|    0   N/A  N/A     18974      C   python3                          1073MiB |
|    0   N/A  N/A     18975      C   python3                          1073MiB |
|    0   N/A  N/A     18976      C   python3                          1073MiB |
|    1   N/A  N/A      1722      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+

Generating   199/  200 games (5863.3893 games/hour).
123.712792 seconds, 5819.931683 games/hour
(env) (base) 〇〇@g17:~/igo/TantamaGo$ python3 selfplay_main.py --process 16 --num-data 200 --model model/rl-model_default.bin --model2 model/sl-
model_20240906_193431_e:03.bin
Self play visits : 16
Generating    87/  200 games (5214.2371 games/hour).
🐋cpu: 94.3% [100.0, 13.1, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
🐋mem: 61.9%
Mon Sep  9 17:57:07 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.119.03   Driver Version: 450.119.03   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  TITAN RTX           On   | 00000000:03:00.0 Off |                  N/A |
| 41%   43C    P2    65W / 280W |  17189MiB / 24219MiB |     39%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  TITAN RTX           On   | 00000000:04:00.0 Off |                  N/A |
| 40%   29C    P8    11W / 280W |      9MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1722      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      1885      G   /usr/bin/gnome-shell                4MiB |
|    0   N/A  N/A     19379      C   python3                          1073MiB |
|    0   N/A  N/A     19380      C   python3                          1073MiB |
|    0   N/A  N/A     19381      C   python3                          1073MiB |
|    0   N/A  N/A     19382      C   python3                          1073MiB |
|    0   N/A  N/A     19383      C   python3                          1073MiB |
|    0   N/A  N/A     19384      C   python3                          1073MiB |
|    0   N/A  N/A     19385      C   python3                          1073MiB |
|    0   N/A  N/A     19386      C   python3                          1073MiB |
|    0   N/A  N/A     19387      C   python3                          1073MiB |
|    0   N/A  N/A     19388      C   python3                          1073MiB |
|    0   N/A  N/A     19389      C   python3                          1073MiB |
|    0   N/A  N/A     19390      C   python3                          1073MiB |
|    0   N/A  N/A     19391      C   python3                          1073MiB |
|    0   N/A  N/A     19392      C   python3                          1073MiB |
|    0   N/A  N/A     19393      C   python3                          1073MiB |
|    0   N/A  N/A     19394      C   python3                          1073MiB |
|    1   N/A  N/A      1722      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+

Generating   199/  200 games (5862.0509 games/hour).
123.289611 seconds, 5839.908135 games/hour
(env) (base) 〇〇@g17:~/igo/TantamaGo$ python3 selfplay_main.py --process 12 --num-data 200 --model model/rl-model_default.bin --model2 model/sl-
model_20240906_193431_e:03.bin
Self play visits : 16
Generating    90/  200 games (5392.9567 games/hour).
🐋cpu: 75.0% [100.0, 100.0, 100.0, 98.0, 100.0, 100.0, 100.0, 6.7, 100.0, 3.0, 2.0, 3.0, 100.0, 100.0, 98.0, 100.0]
🐋mem: 48.2%
Mon Sep  9 17:59:59 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.119.03   Driver Version: 450.119.03   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  TITAN RTX           On   | 00000000:03:00.0 Off |                  N/A |
| 41%   42C    P2    59W / 280W |  12897MiB / 24219MiB |     33%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  TITAN RTX           On   | 00000000:04:00.0 Off |                  N/A |
| 41%   29C    P8    11W / 280W |      9MiB / 24220MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1722      G   /usr/lib/xorg/Xorg                  9MiB |
|    0   N/A  N/A      1885      G   /usr/bin/gnome-shell                4MiB |
|    0   N/A  N/A     19928      C   python3                          1073MiB |
|    0   N/A  N/A     19929      C   python3                          1073MiB |
|    0   N/A  N/A     19930      C   python3                          1073MiB |
|    0   N/A  N/A     19931      C   python3                          1073MiB |
|    0   N/A  N/A     19932      C   python3                          1073MiB |
|    0   N/A  N/A     19933      C   python3                          1073MiB |
|    0   N/A  N/A     19934      C   python3                          1073MiB |
|    0   N/A  N/A     19935      C   python3                          1073MiB |
|    0   N/A  N/A     19936      C   python3                          1073MiB |
|    0   N/A  N/A     19937      C   python3                          1073MiB |
|    0   N/A  N/A     19938      C   python3                          1073MiB |
|    0   N/A  N/A     19939      C   python3                          1073MiB |
|    1   N/A  N/A      1722      G   /usr/lib/xorg/Xorg                  4MiB |
+-----------------------------------------------------------------------------+

Generating   199/  200 games (5861.3577 games/hour).
123.629591 seconds, 5823.848435 games/hour
```


# 20241011
http://www.yss-aya.com/bbs/patio.cgi?read=50