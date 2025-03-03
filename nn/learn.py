"""深層学習の実装。
"""
import glob
import os
import time
import torch
from nn.network import DualNet, DualNet_128_12, DualNet_256_24, DualNet_semeai, DualNet_256_24_semeai
from nn.loss import calculate_policy_loss, calculate_value_loss, \
    calculate_policy_kld_loss
from nn.utility import get_torch_device, print_learning_process, \
    print_evaluation_information, save_model, load_data_set, \
    split_train_test_set, choose_network

from learning_param import SL_LEARNING_RATE, RL_LEARNING_RATE, \
    MOMENTUM, WEIGHT_DECAY, SL_VALUE_WEIGHT, RL_VALUE_WEIGHT, \
    LEARNING_SCHEDULE, BATCH_SIZE, EPOCHS

import datetime###########
dt_now = datetime.datetime.now()############

import copy##########

import numpy as np

import psutil, sys


def train_on_cpu(program_dir: str, board_size: int, batch_size: \
    int, epochs: int) -> None: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
        epochs (int): 実行する最大エポック数。
    """
    print(f"🐾train_on_cpu {dt_now}")###########

    batch_size = 1#######################


    # 学習データと検証用データの分割
    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "sl_data_*.npz")))

    train_data_set, test_data_set = split_train_test_set(data_set, 1)#################
    # train_data_set, test_data_set = split_train_test_set(data_set, 0.8)

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=False)
    
    dual_net = DualNet(device=device, board_size=board_size)

    dual_net.to(device)
    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=SL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)

    current_lr = SL_LEARNING_RATE

    for epoch in range(epochs):
        print(f"🐾epoch: {epoch}")############
        for data_index, train_data_path in enumerate(train_data_set):
            plane_data, policy_data, value_data = load_data_set(train_data_path)
            train_loss = {
                "loss": 0.0,
                "policy": 0.0,
                "value": 0.0,
            }
            iteration = 0
            dual_net.train()
            epoch_time = time.time()
            for i in range(0, len(value_data) - batch_size + 1, batch_size):
                print(f"🐾バッチ: {i}")############
                plane = torch.tensor(plane_data[i:i+batch_size])
                policy = torch.tensor(policy_data[i:i+batch_size])
                value = torch.tensor(value_data[i:i+batch_size])

                print(plane)
                print(policy)
                print(value)

                policy_predict, value_predict = dual_net.forward_for_sl(plane)

                policy_loss = calculate_policy_loss(policy_predict, policy)
                value_loss = calculate_value_loss(value_predict, value)

                dual_net.zero_grad()

                loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss["loss"] += loss.item()
                train_loss["policy"] += policy_loss.mean().item()
                train_loss["value"] += value_loss.mean().item()
                iteration += 1
                
                print("!!!dual_net\n", dual_net)
                print("!!!dual_net.state_dict().keys()\n", dual_net.state_dict().keys())############
                print("!!!dual_net.state_dict()\n", dual_net.state_dict())
                print("!!!list(dual_net.parameters()\n", list(dual_net.parameters()))
                return
            
            print_learning_process(train_loss, epoch, data_index, iteration, epoch_time)

        test_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        test_iteration = 0
        testing_time = time.time()
        for data_index, test_data_path in enumerate(test_data_set):
            dual_net.eval()
            plane_data, policy_data, value_data = load_data_set(test_data_path)
            with torch.no_grad():
                for i in range(0, len(value_data) - batch_size + 1, batch_size):
                    plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                    policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
                    value = torch.tensor(value_data[i:i+batch_size]).to(device)

                    policy_predict, value_predict = dual_net.forward_for_sl(plane)

                    policy_loss = calculate_policy_loss(policy_predict, policy)
                    value_loss = calculate_value_loss(value_predict, value)

                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                    test_loss["loss"] += loss.item()
                    test_loss["policy"] += policy_loss.mean().item()
                    test_loss["value"] += value_loss.mean().item()
                    test_iteration += 1


        print_evaluation_information(test_loss, epoch, test_iteration, testing_time)

        if epoch in LEARNING_SCHEDULE["learning_rate"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

    save_model(dual_net, os.path.join("model", "sl-model.bin"))























def train_on_gpu(program_dir: str, board_size: int, batch_size: int, \
    epochs: int, network_name: str, npz_dir: str = "data") -> None: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
        epochs (int): 実行する最大エポック数。
    """

    print(f"🐾train_on_gpu {dt_now}")###########
    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] device")#############
    print("torch.cuda.current_device: ", torch.cuda.current_device())
    print("torch.cuda.device_count: ", torch.cuda.device_count())
    print("torch.cuda.get_device_name(0): ", torch.cuda.get_device_name(0))
    if torch.cuda.device_count() > 1:##########
        print("torch.cuda.get_device_name(1): ", torch.cuda.get_device_name(1))
    print("torch.cuda.get_device_capability(0): ", torch.cuda.get_device_capability(0))
    if torch.cuda.device_count() > 1:##########
        print("torch.cuda.get_device_capability(1): ", torch.cuda.get_device_capability(1))
    print("torch.cuda.get_arch_list(): ", torch.cuda.get_arch_list())



    # 学習データと検証用データの分割
    data_set = sorted(glob.glob(os.path.join(program_dir, npz_dir, "sl_data_*.npz")))
    """sl_data_*.npz のファイルパスのリスト。"""
    train_data_set, test_data_set = split_train_test_set(data_set, 0.8)
    """sl_data_*.npz のファイルパスのリストを学習データと検証用データの分割したもの。"""

    # 学習処理を行うデバイスの設定
    device = torch.device("cuda")
    # device = get_torch_device(use_gpu=True)######################

    if network_name == "DualNet":
        dual_net = DualNet(device=device, board_size=board_size)
        """DualNetのインスタンス。多分、ここにニューラルネットワークのパラメタとか入ってる。"""
    elif network_name == "DualNet_128_12":
        dual_net = DualNet_128_12(device=device, board_size=board_size)
    elif network_name == "DualNet_256_24":
        dual_net = DualNet_256_24(device=device, board_size=board_size)
    elif network_name == "DualNet_semeai":
        dual_net = DualNet_semeai(device=device, board_size=board_size)
    elif network_name == "DualNet_256_24_semeai":
        dual_net = DualNet_256_24_semeai(device=device, board_size=board_size)
    else:
        print(f"👺network_name: {network_name} is not defined.")
        raise(f"network_name is not defined.")

    if torch.cuda.device_count() > 1:##########ここTrueで作ったので対局しようとするとFailed to load model/sl-model_2024のエラー出る
        dual_net = torch.nn.DataParallel(dual_net)

    dual_net.to(device)
    print(f"🐾device: ", device)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=SL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)

    scaler = torch.cuda.amp.GradScaler()
    """勾配消失とかいうのを防ぐやつらしい"""

    current_lr = SL_LEARNING_RATE
    """学習率"""

    # if device == 'cuda':###########
    #     dual_net = torch.nn.DataParallel(dual_net) # make parallel
    #     torch.backends.cuda.nn.benchmark = True

    for epoch in range(epochs):

        # npz ループ
        for data_index, train_data_path in enumerate(train_data_set):
            plane_data, policy_data, value_data = load_data_set(train_data_path)

            train_loss = {
                "loss": 0.0,
                "policy": 0.0,
                "value": 0.0,
            }

            iteration = 0

            # モデルを訓練モードにする。バッチ正規化とかドロップアウトとか訓練時と推論時で挙動が違うものが訓練モードになる。。
            dual_net.train()

            # バッチループ。epoch_time ってあるけど多分バッチの時間を計測してる。
            epoch_time = time.time()
            for i in range(0, len(value_data) - batch_size + 1, batch_size):
                with torch.cuda.amp.autocast(enabled=True):
                    plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                    """盤面データのミニバッチのリスト。81*6*batch_size のテンソル。"""
                    policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
                    value = torch.tensor(value_data[i:i+batch_size]).to(device)

                    policy_predict, value_predict = dual_net(plane, pram="sl")
                    # if torch.cuda.device_count() > 1:
                    #     policy_predict, value_predict = dual_net(plane)##################
                    #     # policy_predict, value_predict = dual_net.module.forward_for_sl(plane)
                    # else:
                    #     policy_predict, value_predict = dual_net.forward_for_sl(plane)
                    # # policy_predict, value_predict = dual_net.forward_for_sl(plane)###################def

                    # モデルの勾配を初期化
                    # たぶん、ミニバッチ学習で使うためにミニバッチ内の勾配を記録していて、前のミニバッチの勾配が残っているので、それを初期化している。
                    # ？with の上に移動する？
                    dual_net.zero_grad()

                    # ロスの計算
                    policy_loss = calculate_policy_loss(policy_predict, policy)
                    value_loss = calculate_value_loss(value_predict, value)

                    # 多分、policy_loss のが重要だから、value_loss に微小量の重みをかけてる 
                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                scaler.scale(loss).backward()
                # 重みの更新をしてる？
                scaler.step(optimizer)
                scaler.update()

                train_loss["loss"] += loss.item()
                train_loss["policy"] += policy_loss.mean().item()
                train_loss["value"] += value_loss.mean().item()
                iteration += 1

            print_learning_process(train_loss, epoch, data_index, iteration, epoch_time)

        test_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        test_iteration = 0
        testing_time = time.time()
        for data_index, test_data_path in enumerate(test_data_set):
            dual_net.eval()
            plane_data, policy_data, value_data = load_data_set(test_data_path)
            with torch.no_grad():
                for i in range(0, len(value_data) - batch_size + 1, batch_size):
                    plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                    policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
                    value = torch.tensor(value_data[i:i+batch_size]).to(device)

                    policy_predict, value_predict = dual_net(plane, pram="sl")
                    # if torch.cuda.device_count() > 1:
                    #     policy_predict, value_predict = dual_net(plane)##################
                    #     # policy_predict, value_predict = dual_net.module.forward_for_sl(plane)
                    # else:
                    #     policy_predict, value_predict = dual_net.forward_for_sl(plane)
                    # # policy_predict, value_predict = dual_net.forward_for_sl(plane)############def

                    policy_loss = calculate_policy_loss(policy_predict, policy)
                    value_loss = calculate_value_loss(value_predict, value)

                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                    test_loss["loss"] += loss.item()
                    test_loss["policy"] += policy_loss.mean().item()
                    test_loss["value"] += value_loss.mean().item()
                    test_iteration += 1


        print_evaluation_information(test_loss, epoch, test_iteration, testing_time)

        # 学習率を変更する
        if epoch in LEARNING_SCHEDULE["learning_rate"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

        # たぶん、save_model すると変更が入るので、ディープコピーを作ってそれを保存する。
        dual_net_copy = copy.deepcopy(dual_net)######
        # モデルがDataParallelでラップされているかどうかを確認
        if isinstance(dual_net_copy, torch.nn.DataParallel):
            state_dict = dual_net_copy.module.state_dict()
        else:
            state_dict = dual_net_copy.state_dict()

        # モデルの保存
        torch.save(state_dict, os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep{epoch:0>2}.bin"))
        # torch.save(dual_net_copy.to("cpu").module.state_dict(), os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep:{epoch:0>2}.bin"))
        # save_model(dual_net_copy, os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep:{epoch:0>2}.bin"))######epoch毎に保存

    # save_model(dual_net, os.path.join("model", "sl-model.bin"))






def train_on_gpu_ddp_worker(rank, world, train_npz_paths, test_npz_paths, program_dir: str, board_size: int, batch_size: int, epochs: int, network_name: str, npz_dir: str = "data", checkpoint_dir: str = None) -> None: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
        epochs (int): 実行する最大エポック数。
    """

    assert batch_size % world == 0
    batch_size = batch_size // world

    torch.distributed.init_process_group("nccl", rank = rank, world_size = world)


    if network_name == "DualNet":
        dual_net = DualNet(device=rank, board_size=board_size)
        """DualNetのインスタンス。多分、ここにニューラルネットワークのパラメタとか入ってる。"""
    elif network_name == "DualNet_128_12":
        dual_net = DualNet_128_12(device=rank, board_size=board_size)
    elif network_name == "DualNet_256_24":
        dual_net = DualNet_256_24(device=rank, board_size=board_size)
    elif network_name == "DualNet_semeai":
        dual_net = DualNet_semeai(device=rank, board_size=board_size)
    elif network_name == "DualNet_256_24_semeai":
        dual_net = DualNet_256_24_semeai(device=rank, board_size=board_size)
    else:
        print(f"👺network_name: {network_name} is not defined.")
        raise(f"network_name is not defined.")
    
    if checkpoint_dir is not None:
        checkpoint = torch.load(checkpoint_dir)
        state_dict = checkpoint['model_state_dict']
        
        # DataParallelで保存されたモデルのstate_dictを修正
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:] # module.を取り除く
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        
        dual_net.load_state_dict(new_state_dict)
        policy_loss = checkpoint['policy_loss']
        value_loss = checkpoint['value_loss']

    print(f"🐾device: ", rank)#############
    dual_net = dual_net.to(rank)
    dual_net = torch.nn.parallel.DistributedDataParallel(dual_net, device_ids = [rank], output_device = rank, find_unused_parameters=False)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=SL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)

    if checkpoint_dir is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scaler = torch.cuda.amp.GradScaler()
    """勾配消失とかいうのを防ぐやつらしい"""

    current_lr = SL_LEARNING_RATE
    """学習率"""


    def tmp_load_data_set(npz_path, rank):
        def check_memory_usage():
            if not psutil.virtual_memory().percent < 90:
                print(f"memory usage is too high. mem_use: {psutil.virtual_memory().percent}% [{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}]")
                assert True

        check_memory_usage()

        data = np.load(npz_path)

        check_memory_usage()

        plane_data = data["input"]
        policy_data = data["policy"].astype(np.float32)
        value_data = data["value"].astype(np.int64)

        check_memory_usage()

        plane_data = torch.tensor(plane_data)
        policy_data = torch.tensor(policy_data)
        value_data = torch.tensor(value_data)

        return plane_data, policy_data, value_data


    for epoch in range(epochs):

        npz_cnt = 0

        # npz ループ
        for data_index, train_npz_path in enumerate(train_npz_paths):

            plane_data, policy_data, value_data = tmp_load_data_set(train_npz_path, rank)

            train_dataset = torch.utils.data.TensorDataset(plane_data, policy_data, value_data)

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = world, rank = rank, shuffle = True)

            # samplerの設定とshuffleをFalseにすることを忘れない
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False, pin_memory=True, num_workers = 2, sampler = train_sampler)


            train_sampler.set_epoch(epoch)


            train_loss = {
                "loss": 0.0,
                "policy": 0.0,
                "value": 0.0,
            }

            iteration = 0

            dual_net.train()

            epoch_time = time.time()
            for train_batch in train_loader:
                # if iteration % 1000 == 0:##############
                #         print(f"\r{train_npz_path}, iteration: {iteration}/{len(train_loader)}, rank: {rank}", end="")##########
                #         sys.stdout.flush()
                with torch.cuda.amp.autocast(enabled=True):

                    plane, policy, value = train_batch
                    plane = plane.to(rank, non_blocking=True)
                    policy = policy.to(rank, non_blocking=True)
                    value = value.to(rank, non_blocking=True)

                    policy_predict, value_predict = dual_net(plane, pram="sl")

                    dual_net.zero_grad()

                    # ロスの計算
                    policy_loss = calculate_policy_loss(policy_predict, policy)
                    value_loss = calculate_value_loss(value_predict, value)

                    # 多分、policy_loss のが重要だから、value_loss に微小量の重みをかけてる 
                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                scaler.scale(loss).backward()
                # 重みの更新をしてる？
                scaler.step(optimizer)
                scaler.update()

                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(policy_loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(value_loss, op=torch.distributed.ReduceOp.AVG)

                train_loss["loss"] += loss.item()
                train_loss["policy"] += policy_loss.mean().item()
                train_loss["value"] += value_loss.mean().item()
                iteration += 1


            if int(rank) == 0:
                print_learning_process(train_loss, epoch, data_index, iteration, epoch_time)
                npz_cnt += 1
                if npz_cnt % 10 == 0:
                    dual_net_copy = copy.deepcopy(dual_net)######
                    torch.save(dual_net_copy.to("cpu").module.state_dict(), os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_{npz_cnt:0>3}.bin"))
                    # save_model(dual_net_copy, os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep:{epoch:0>2}.bin"))######epoch毎に保存

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': dual_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'policy_loss': policy_loss,
                        'value_loss': value_loss,
                        }, os.path.join("model", f"checkpoint_{dt_now.strftime('%Y%m%d_%H%M%S')}_{npz_cnt:0>3}.bin"))


        test_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        test_iteration = 0
        testing_time = time.time()
        for data_index, test_npz_path in enumerate(test_npz_paths):

            plane_data, policy_data, value_data = tmp_load_data_set(test_npz_path, rank)

            test_dataset = torch.utils.data.TensorDataset(plane_data, policy_data, value_data)

            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas = world, rank = rank, shuffle = False)

            test_sampler.set_epoch(epoch)

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, pin_memory=True, num_workers = 2, sampler = test_sampler)

            torch.distributed.barrier()

            dual_net.eval()
            with torch.no_grad():
                for test_batch in test_loader:
                    # if test_iteration % 1000 == 0:##############
                    #     print(f"\r{test_npz_path}, test_iteration: {test_iteration}/{len(test_loader)}, rank: {rank}", end="")##########
                    #     sys.stdout.flush()
                    with torch.cuda.amp.autocast(enabled=True):
                        plane, policy, value = test_batch
                        plane = plane.to(rank, non_blocking=True)
                        policy = policy.to(rank, non_blocking=True)
                        value = value.to(rank, non_blocking=True)

                        policy_predict, value_predict = dual_net(plane, pram="sl")

                        policy_loss = calculate_policy_loss(policy_predict, policy)
                        value_loss = calculate_value_loss(value_predict, value)

                        loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(policy_loss, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(value_loss, op=torch.distributed.ReduceOp.AVG)

                    test_loss["loss"] += loss.item()
                    test_loss["policy"] += policy_loss.mean().item()
                    test_loss["value"] += value_loss.mean().item()
                    test_iteration += 1

        if int(rank) == 0:
            print_evaluation_information(test_loss, epoch, test_iteration, testing_time)

        torch.distributed.barrier()
        # 学習率を変更する
        if epoch in LEARNING_SCHEDULE["learning_rate"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

        if int(rank) == 0:
            # たぶん、save_model すると変更が入るので、ディープコピーを作ってそれを保存する。
            dual_net_copy = copy.deepcopy(dual_net)######
            torch.save(dual_net_copy.to("cpu").module.state_dict(), os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep{epoch:0>2}.bin"))
            # save_model(dual_net_copy, os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep:{epoch:0>2}.bin"))######epoch毎に保存

            torch.save({
                'epoch': epoch,
                'model_state_dict': dual_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                }, os.path.join("model", f"checkpoint_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep{epoch:0>2}.bin"))

    # save_model(dual_net, os.path.join("model", "sl-model.bin"))

    torch.distributed.destroy_process_group()




def train_on_gpu_ddp(program_dir: str, board_size: int, batch_size: int, epochs: int, network_name: str, npz_dir: str = "data", chckpoint_dir: str = None) -> None: # pylint: disable=R0914,R0915

    print(f"🐾train_on_gpu_ddp {dt_now}")###########
    print(f"    [{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] device")#############
    print("    torch.cuda.current_device: ", torch.cuda.current_device())
    print("    torch.cuda.device_count: ", torch.cuda.device_count())
    print("    torch.cuda.get_device_name(0): ", torch.cuda.get_device_name(0))
    for i in range(torch.cuda.device_count()):
        print(f"    torch.cuda.get_device_name({i}): ", torch.cuda.get_device_name(i))
    # if torch.cuda.device_count() > 1:##########
    #     print("    torch.cuda.get_device_name(1): ", torch.cuda.get_device_name(1))
    for i in range(torch.cuda.device_count()):
        print(f"    torch.cuda.get_device_capability({i}): ", torch.cuda.get_device_capability(i))
    # print("    torch.cuda.get_device_capability(0): ", torch.cuda.get_device_capability(0))
    # if torch.cuda.device_count() > 1:##########
        # print("    torch.cuda.get_device_capability(1): ", torch.cuda.get_device_capability(1))
    print("    torch.cuda.get_arch_list(): ", torch.cuda.get_arch_list())

    # train_dataset, val_dataset = getMyDataset()

    # 学習データと検証用データの分割
    data_set = sorted(glob.glob(os.path.join(program_dir, npz_dir, "sl_data_*.npz")))
    """sl_data_*.npz のファイルパスのリスト。"""
    train_data_set, test_data_set = split_train_test_set(data_set, 0.8)
    """sl_data_*.npz のファイルパスのリストを学習データと検証用データの分割したもの。"""

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '50000'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    torch.multiprocessing.spawn(train_on_gpu_ddp_worker, args=(torch.cuda.device_count(), train_data_set, test_data_set, program_dir, board_size, BATCH_SIZE, EPOCHS, network_name, npz_dir, chckpoint_dir), nprocs = torch.cuda.device_count(), join = True)




def train_on_gpu_ddp_worker2(rank, world, train_npz_paths, test_npz_paths, program_dir: str, board_size: int, batch_size: int, epochs: int, network_name: str, npz_dir: str = "data", checkpoint_dir: str = None) -> None: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
        epochs (int): 実行する最大エポック数。
    """

    assert batch_size % world == 0
    batch_size = batch_size // world

    torch.distributed.init_process_group("nccl", rank = rank, world_size = world)


    if network_name == "DualNet":
        dual_net = DualNet(device=rank, board_size=board_size)
        """DualNetのインスタンス。多分、ここにニューラルネットワークのパラメタとか入ってる。"""
    elif network_name == "DualNet_128_12":
        dual_net = DualNet_128_12(device=rank, board_size=board_size)
    elif network_name == "DualNet_256_24":
        dual_net = DualNet_256_24(device=rank, board_size=board_size)
    elif network_name == "DualNet_semeai":
        dual_net = DualNet_semeai(device=rank, board_size=board_size)
    elif network_name == "DualNet_256_24_semeai":
        dual_net = DualNet_256_24_semeai(device=rank, board_size=board_size)
    else:
        print(f"👺network_name: {network_name} is not defined.")
        raise(f"network_name is not defined.")
    
    if checkpoint_dir is not None:
        checkpoint = torch.load(checkpoint_dir)
        state_dict = checkpoint['model_state_dict']
        
        # DataParallelで保存されたモデルのstate_dictを修正
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:] # module.を取り除く
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v
        
        dual_net.load_state_dict(new_state_dict)
        policy_loss = checkpoint['policy_loss']
        value_loss = checkpoint['value_loss']

    print(f"🐾device: ", rank)#############
    dual_net = dual_net.to(rank)
    dual_net = torch.nn.parallel.DistributedDataParallel(dual_net, device_ids = [rank], output_device = rank, find_unused_parameters=False)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=SL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)

    if checkpoint_dir is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scaler = torch.cuda.amp.GradScaler()
    """勾配消失とかいうのを防ぐやつらしい"""

    current_lr = SL_LEARNING_RATE
    """学習率"""


    def tmp_load_data_set(npz_path, rank):
        def check_memory_usage():
            if not psutil.virtual_memory().percent < 90:
                print(f"memory usage is too high. mem_use: {psutil.virtual_memory().percent}% [{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}]")
                assert True

        check_memory_usage()

        data = np.load(npz_path)

        check_memory_usage()

        plane_data = data["input"]
        policy_data = data["policy"].astype(np.float32)
        value_data = data["value"].astype(np.int64)

        check_memory_usage()

        plane_data = torch.tensor(plane_data)
        policy_data = torch.tensor(policy_data)
        value_data = torch.tensor(value_data)

        return plane_data, policy_data, value_data


    for epoch in range(epochs):

        npz_cnt = 0

        # npz ループ
        while True:
        # for data_index, train_npz_path in enumerate(train_npz_paths):

            sleep_cnt = 0
            sleep_time = time.time()
            while not os.path.isfile(os.path.join(program_dir, npz_dir, f"sl_data_{npz_cnt}.npz")):
                time.sleep(1)
                sleep_cnt += 1
                if time.time() - sleep_time > 3600:
                    break

            train_npz_path = os.path.join(program_dir, npz_dir, f"sl_data_{npz_cnt}.npz")

            plane_data, policy_data, value_data = tmp_load_data_set(train_npz_path, rank)

            train_dataset = torch.utils.data.TensorDataset(plane_data, policy_data, value_data)

            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = world, rank = rank, shuffle = True)

            # samplerの設定とshuffleをFalseにすることを忘れない
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False, pin_memory=True, num_workers = 2, sampler = train_sampler)


            train_sampler.set_epoch(epoch)


            train_loss = {
                "loss": 0.0,
                "policy": 0.0,
                "value": 0.0,
            }

            iteration = 0

            dual_net.train()

            epoch_time = time.time()
            for train_batch in train_loader:
                # if iteration % 1000 == 0:##############
                #         print(f"\r{train_npz_path}, iteration: {iteration}/{len(train_loader)}, rank: {rank}", end="")##########
                #         sys.stdout.flush()
                with torch.cuda.amp.autocast(enabled=True):

                    plane, policy, value = train_batch
                    plane = plane.to(rank, non_blocking=True)
                    policy = policy.to(rank, non_blocking=True)
                    value = value.to(rank, non_blocking=True)

                    policy_predict, value_predict = dual_net(plane, pram="sl")

                    dual_net.zero_grad()

                    # ロスの計算
                    policy_loss = calculate_policy_loss(policy_predict, policy)
                    value_loss = calculate_value_loss(value_predict, value)

                    # 多分、policy_loss のが重要だから、value_loss に微小量の重みをかけてる 
                    loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                scaler.scale(loss).backward()
                # 重みの更新をしてる？
                scaler.step(optimizer)
                scaler.update()

                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(policy_loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(value_loss, op=torch.distributed.ReduceOp.AVG)

                train_loss["loss"] += loss.item()
                train_loss["policy"] += policy_loss.mean().item()
                train_loss["value"] += value_loss.mean().item()
                iteration += 1


            if int(rank) == 0:
                print_learning_process(train_loss, epoch, f"sl_data_{npz_cnt}.npz", iteration, epoch_time)
                npz_cnt += 1
                if npz_cnt % 10 == 0:
                    dual_net_copy = copy.deepcopy(dual_net)######
                    torch.save(dual_net_copy.to("cpu").module.state_dict(), os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_{npz_cnt:0>3}.bin"))
                    # save_model(dual_net_copy, os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep:{epoch:0>2}.bin"))######epoch毎に保存

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': dual_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'policy_loss': policy_loss,
                        'value_loss': value_loss,
                        }, os.path.join("model", f"checkpoint_{dt_now.strftime('%Y%m%d_%H%M%S')}_{npz_cnt:0>3}.bin"))
            
            
            npz_cnt += 1


        test_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        test_iteration = 0
        testing_time = time.time()
        for data_index, test_npz_path in enumerate(test_npz_paths):

            plane_data, policy_data, value_data = tmp_load_data_set(test_npz_path, rank)

            test_dataset = torch.utils.data.TensorDataset(plane_data, policy_data, value_data)

            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas = world, rank = rank, shuffle = False)

            test_sampler.set_epoch(epoch)

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, pin_memory=True, num_workers = 2, sampler = test_sampler)

            torch.distributed.barrier()

            dual_net.eval()
            with torch.no_grad():
                for test_batch in test_loader:
                    # if test_iteration % 1000 == 0:##############
                    #     print(f"\r{test_npz_path}, test_iteration: {test_iteration}/{len(test_loader)}, rank: {rank}", end="")##########
                    #     sys.stdout.flush()
                    with torch.cuda.amp.autocast(enabled=True):
                        plane, policy, value = test_batch
                        plane = plane.to(rank, non_blocking=True)
                        policy = policy.to(rank, non_blocking=True)
                        value = value.to(rank, non_blocking=True)

                        policy_predict, value_predict = dual_net(plane, pram="sl")

                        policy_loss = calculate_policy_loss(policy_predict, policy)
                        value_loss = calculate_value_loss(value_predict, value)

                        loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

                    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(policy_loss, op=torch.distributed.ReduceOp.AVG)
                    torch.distributed.all_reduce(value_loss, op=torch.distributed.ReduceOp.AVG)

                    test_loss["loss"] += loss.item()
                    test_loss["policy"] += policy_loss.mean().item()
                    test_loss["value"] += value_loss.mean().item()
                    test_iteration += 1

        if int(rank) == 0:
            print_evaluation_information(test_loss, epoch, test_iteration, testing_time)

        torch.distributed.barrier()
        # 学習率を変更する
        if epoch in LEARNING_SCHEDULE["learning_rate"]:
            previous_lr = current_lr
            for group in optimizer.param_groups:
                group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
            current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
            print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

        if int(rank) == 0:
            # たぶん、save_model すると変更が入るので、ディープコピーを作ってそれを保存する。
            dual_net_copy = copy.deepcopy(dual_net)######
            torch.save(dual_net_copy.to("cpu").module.state_dict(), os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep{epoch:0>2}.bin"))
            # save_model(dual_net_copy, os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep:{epoch:0>2}.bin"))######epoch毎に保存

            torch.save({
                'epoch': epoch,
                'model_state_dict': dual_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                }, os.path.join("model", f"checkpoint_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep{epoch:0>2}.bin"))

    # save_model(dual_net, os.path.join("model", "sl-model.bin"))

    torch.distributed.destroy_process_group()





def train_on_gpu_ddp2(program_dir: str, board_size: int, batch_size: int, epochs: int, network_name: str, npz_dir: str = "data", chckpoint_dir: str = None) -> None: # pylint: disable=R0914,R0915

    print(f"🐾train_on_gpu_ddp {dt_now}")###########
    print(f"    [{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] device")#############
    print("    torch.cuda.current_device: ", torch.cuda.current_device())
    print("    torch.cuda.device_count: ", torch.cuda.device_count())
    print("    torch.cuda.get_device_name(0): ", torch.cuda.get_device_name(0))
    for i in range(torch.cuda.device_count()):
        print(f"    torch.cuda.get_device_name({i}): ", torch.cuda.get_device_name(i))
    # if torch.cuda.device_count() > 1:##########
    #     print("    torch.cuda.get_device_name(1): ", torch.cuda.get_device_name(1))
    for i in range(torch.cuda.device_count()):
        print(f"    torch.cuda.get_device_capability({i}): ", torch.cuda.get_device_capability(i))
    # print("    torch.cuda.get_device_capability(0): ", torch.cuda.get_device_capability(0))
    # if torch.cuda.device_count() > 1:##########
        # print("    torch.cuda.get_device_capability(1): ", torch.cuda.get_device_capability(1))
    print("    torch.cuda.get_arch_list(): ", torch.cuda.get_arch_list())

    # train_dataset, val_dataset = getMyDataset()

    # 学習データと検証用データの分割
    data_set = sorted(glob.glob(os.path.join(program_dir, npz_dir, "sl_data_*.npz")))
    """sl_data_*.npz のファイルパスのリスト。"""
    train_data_set, test_data_set = split_train_test_set(data_set, 0.8)
    """sl_data_*.npz のファイルパスのリストを学習データと検証用データの分割したもの。"""

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '50000'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    torch.multiprocessing.spawn(train_on_gpu_ddp_worker2, args=(torch.cuda.device_count(), train_data_set, test_data_set, program_dir, board_size, BATCH_SIZE, EPOCHS, network_name, npz_dir, chckpoint_dir), nprocs = torch.cuda.device_count(), join = True)











# def train_on_gpu(program_dir: str, board_size: int, batch_size: int, \
#     epochs: int, network_name: str, npz_dir: str = "data") -> None: # pylint: disable=R0914,R0915
#     """教師あり学習を実行し、学習したモデルを保存する。

#     Args:
#         program_dir (str): プログラムのワーキングディレクトリ。
#         board_size (int): 碁盤の大きさ。
#         batch_size (int): ミニバッチサイズ。
#         epochs (int): 実行する最大エポック数。
#     """

#     print(f"🐾train_on_gpu {dt_now}")###########
#     print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] device")#############
#     print("torch.cuda.current_device: ", torch.cuda.current_device())
#     print("torch.cuda.device_count: ", torch.cuda.device_count())
#     print("torch.cuda.get_device_name(0): ", torch.cuda.get_device_name(0))
#     if torch.cuda.device_count() > 1:##########
#         print("torch.cuda.get_device_name(1): ", torch.cuda.get_device_name(1))
#     print("torch.cuda.get_device_capability(0): ", torch.cuda.get_device_capability(0))
#     if torch.cuda.device_count() > 1:##########
#         print("torch.cuda.get_device_capability(1): ", torch.cuda.get_device_capability(1))
#     print("torch.cuda.get_arch_list(): ", torch.cuda.get_arch_list())

#     # 学習データと検証用データの分割
#     data_set = sorted(glob.glob(os.path.join(program_dir, npz_dir, "sl_data_*.npz")))
#     train_data_set, test_data_set = split_train_test_set(data_set, 0.8)

#     # 学習処理を行うデバイスの設定
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if network_name == "DualNet":
#         dual_net = DualNet(device=device, board_size=board_size)
#     elif network_name == "DualNet_128_12":
#         dual_net = DualNet_128_12(device=device, board_size=board_size)
#     elif network_name == "DualNet_256_24":
#         dual_net = DualNet_256_24(device=device, board_size=board_size)
#     else:
#         print(f"👺network_name: {network_name} is not defined.")
#         raise ValueError(f"network_name: {network_name} is not defined.")

#     if torch.cuda.device_count() > 1:
#         dual_net = torch.nn.DataParallel(dual_net)

#     dual_net.to(device)
#     print(f"🐾device: ", device)

#     optimizer = torch.optim.SGD(dual_net.parameters(),
#                                 lr=SL_LEARNING_RATE,
#                                 momentum=MOMENTUM,
#                                 weight_decay=WEIGHT_DECAY,
#                                 nesterov=True)

#     scaler = torch.cuda.amp.GradScaler()

#     current_lr = SL_LEARNING_RATE

#     for epoch in range(epochs):
#         for data_index, train_data_path in enumerate(train_data_set):
#             plane_data, policy_data, value_data = load_data_set(train_data_path)

#             train_loss = {
#                 "loss": 0.0,
#                 "policy": 0.0,
#                 "value": 0.0,
#             }

#             iteration = 0
#             dual_net.train()
#             epoch_time = time.time()
#             for i in range(0, len(value_data) - batch_size + 1, batch_size):
#                 with torch.cuda.amp.autocast(enabled=True):
#                     plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
#                     policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
#                     value = torch.tensor(value_data[i:i+batch_size]).to(device)

#                     policy_predict, value_predict = dual_net(plane)

#                     dual_net.zero_grad()

#                     policy_loss = calculate_policy_loss(policy_predict, policy)
#                     value_loss = calculate_value_loss(value_predict, value)

#                     loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()

#                 train_loss["loss"] += loss.item()
#                 train_loss["policy"] += policy_loss.mean().item()
#                 train_loss["value"] += value_loss.mean().item()
#                 iteration += 1

#             print_learning_process(train_loss, epoch, data_index, iteration, epoch_time)

#         test_loss = {
#             "loss": 0.0,
#             "policy": 0.0,
#             "value": 0.0,
#         }
#         test_iteration = 0
#         testing_time = time.time()
#         for data_index, test_data_path in enumerate(test_data_set):
#             dual_net.eval()
#             plane_data, policy_data, value_data = load_data_set(test_data_path)
#             with torch.no_grad():
#                 for i in range(0, len(value_data) - batch_size + 1, batch_size):
#                     plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
#                     policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
#                     value = torch.tensor(value_data[i:i+batch_size]).to(device)

#                     policy_predict, value_predict = dual_net(plane)

#                     policy_loss = calculate_policy_loss(policy_predict, policy)
#                     value_loss = calculate_value_loss(value_predict, value)

#                     loss = (policy_loss + SL_VALUE_WEIGHT * value_loss).mean()

#                     test_loss["loss"] += loss.item()
#                     test_loss["policy"] += policy_loss.mean().item()
#                     test_loss["value"] += value_loss.mean().item()
#                     test_iteration += 1

#         print_evaluation_information(test_loss, epoch, test_iteration, testing_time)

#         if epoch in LEARNING_SCHEDULE["learning_rate"]:
#             previous_lr = current_lr
#             for group in optimizer.param_groups:
#                 group["lr"] = LEARNING_SCHEDULE["learning_rate"][epoch]
#             current_lr = LEARNING_SCHEDULE["learning_rate"][epoch]
#             print(f"Epoch {epoch}, learning rate has changed {previous_lr} -> {current_lr}")

#         dual_net_copy = copy.deepcopy(dual_net)
#         save_model(dual_net_copy, os.path.join("model", f"sl-model_{dt_now.strftime('%Y%m%d_%H%M%S')}_Ep:{epoch:0>2}.bin"))























def train_with_gumbel_alphazero_on_cpu(program_dir: str, board_size: int, \
    batch_size: int) -> None: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。CPUで実行。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
    """
    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "rl_data_*.npz")))

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=False)

    dual_net = DualNet(device=device, board_size=board_size)

    dual_net.to(device)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=RL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)
    num_trained_batches = 0

    model_file_path = os.path.join(program_dir, "model", "rl-model.bin")
    if os.path.exists(model_file_path):
        print(f"load {model_file_path}")
        dual_net.load_state_dict(torch.load(model_file_path))

    state_file_path = os.path.join(program_dir, "model", "rl-state.ckpt")
    if os.path.exists(state_file_path):
        print(f"load {state_file_path}")
        checkpoint = torch.load(state_file_path, map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        num_trained_batches = checkpoint["num_trained_batches"]
        for group in optimizer.param_groups:
            group["lr"] = RL_LEARNING_RATE
        print(f"num_trained_batches : {num_trained_batches}")

    for data_index, train_data_path in enumerate(data_set):
        plane_data, policy_data, value_data = load_data_set(train_data_path)
        train_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        iteration = 0
        dual_net.train()
        epoch_time = time.time()
        for i in range(0, len(value_data) - batch_size + 1, batch_size):
            plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
            policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
            value = torch.tensor(value_data[i:i+batch_size]).to(device)

            policy_predict, value_predict = dual_net.forward(plane)

            dual_net.zero_grad()
            policy_loss = calculate_policy_kld_loss(policy_predict, policy)
            value_loss = calculate_value_loss(value_predict, value)

            loss = (policy_loss + RL_VALUE_WEIGHT * value_loss).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_trained_batches += 1

            train_loss["loss"] += loss.item()
            train_loss["policy"] += policy_loss.mean().item()
            train_loss["value"] += value_loss.mean().item()
            iteration += 1

        print_learning_process(train_loss, 0, data_index, iteration, epoch_time)

    save_model(dual_net, model_file_path)

    state = {
        "num_trained_batches": num_trained_batches,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(state, state_file_path)





def train_with_gumbel_alphazero_on_gpu(program_dir: str, board_size: int, \
    batch_size: int, rl_num: int, rl_datetime: str, network_name: str="DualNet") -> None: # pylint: disable=R0914,R0915
    """教師あり学習を実行し、学習したモデルを保存する。GPUで実行。

    Args:
        program_dir (str): プログラムのワーキングディレクトリ。
        board_size (int): 碁盤の大きさ。
        batch_size (int): ミニバッチサイズ。
    """

    print(f"🐾train_with_gumbel_alphazero_on_gpu {dt_now}")###########

    data_set = sorted(glob.glob(os.path.join(program_dir, "data", "rl_data_*.npz")))

    # 学習処理を行うデバイスの設定
    device = get_torch_device(use_gpu=True)

    if network_name == "DualNet":
        dual_net = DualNet(device=device, board_size=board_size)
        """DualNetのインスタンス。多分、ここにニューラルネットワークのパラメタとか入ってる。"""
    elif network_name == "DualNet_128_12":
        dual_net = DualNet_128_12(device=device, board_size=board_size)
    elif network_name == "DualNet_256_24":
        dual_net = DualNet_256_24(device=device, board_size=board_size)
    elif network_name == "DualNet_256_24_semeai":
        dual_net = DualNet_256_24_semeai(device=device, board_size=board_size)
    else:
        print(f"👺network_name: {network_name} is not defined.")
        raise(f"network_name is not defined.")#############
    # dual_net = DualNet(device=device, board_size=board_size)

    dual_net.to(device)

    optimizer = torch.optim.SGD(dual_net.parameters(),
                                lr=RL_LEARNING_RATE,
                                momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY,
                                nesterov=True)

    scaler = torch.cuda.amp.GradScaler()

    num_trained_batches = 0

    model_file_path = os.path.join(program_dir, "model", "rl-model.bin")
    if os.path.exists(model_file_path):
        print(f"load {model_file_path}")
        dual_net.load_state_dict(torch.load(model_file_path))

    state_file_path = os.path.join(program_dir, "model", "rl-state.ckpt")
    if os.path.exists(state_file_path):
        print(f"load {state_file_path}")
        checkpoint = torch.load(state_file_path, map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        num_trained_batches = checkpoint["num_trained_batches"]
        print(f"num_trained_batches : {num_trained_batches}")

    for data_index, train_data_path in enumerate(data_set):
        print(f"🐾train_rl_gpu, idx: {data_index}")###########
        plane_data, policy_data, value_data = load_data_set(train_data_path)
        train_loss = {
            "loss": 0.0,
            "policy": 0.0,
            "value": 0.0,
        }
        iteration = 0
        dual_net.train()
        epoch_time = time.time()
        for i in range(0, len(value_data) - batch_size + 1, batch_size):
            with torch.cuda.amp.autocast(enabled=True):
                plane = torch.tensor(plane_data[i:i+batch_size]).to(device)
                policy = torch.tensor(policy_data[i:i+batch_size]).to(device)
                value = torch.tensor(value_data[i:i+batch_size]).to(device)

                policy_predict, value_predict = dual_net.forward(plane)

                dual_net.zero_grad()
                policy_loss = calculate_policy_kld_loss(policy_predict, policy)
                value_loss = calculate_value_loss(value_predict, value)

                loss = (policy_loss + RL_VALUE_WEIGHT * value_loss).mean()

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                num_trained_batches += 1

            train_loss["loss"] += loss.item()
            train_loss["policy"] += policy_loss.mean().item()
            train_loss["value"] += value_loss.mean().item()
            iteration += 1

        print_learning_process(train_loss, 0, data_index, iteration, epoch_time)

    save_model(dual_net, os.path.join(program_dir, "model", f"rl-model_{rl_datetime}_{rl_num}.bin"))
    save_model(dual_net, model_file_path)

    state = {
        "num_trained_batches": num_trained_batches,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict()
    }
    torch.save(state, os.path.join(program_dir, "model", f"rl-state_{rl_datetime}_{rl_num}.ckpt"))
    torch.save(state, state_file_path)
