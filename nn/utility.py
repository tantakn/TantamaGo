"""深層学習に関するユーティリティ。
"""
from typing import NoReturn, Dict, List, Tuple
import time
import datetime
import torch
import numpy as np

from common.print_console import print_err
from nn.network import DualNet, DualNet_128_12, DualNet_256_24, DualNet_semeai, DualNet_256_24_semeai


def get_torch_device(use_gpu: bool, gpu_num: int = -1) -> torch.device:
    """torch.deviceを取得する。

    Args:
        use_gpu (bool): GPU使用フラグ。

    Returns:
        torch.device: デバイス情報。
    """

    if use_gpu:
        if gpu_num == -1:
            gpu_num = 0

        torch.cuda.set_device(gpu_num)
        return torch.device("cuda")
    return torch.device("cpu")


def _calculate_losses(loss: Dict[str, float], iteration: int) \
    -> Tuple[float, float, float]:
    """各種損失関数値を算出する。

    Args:
        loss (Dict[str, float]): 損失関数値の情報。
        iteration (int): イテレーション数。

    Returns:
        Tuple[float, float, float]: Total loss, Policy loss, Value loss。
    """
    return loss["loss"] / iteration, loss["policy"] / iteration, \
        loss["value"] / iteration



def print_learning_process(loss_data: Dict[str, float], epoch: int, index: int, \
    iteration: int, start_time: float) -> NoReturn:
    """学習経過情報を表示する。

    Args:
        loss_data (Dict[str]): 損失関数値の情報。
        epoch (int): 学習エポック数。
        index (int): データセットインデックス。
        iteration (int): バッチサイズの学習イテレーション数。
        start_time (float): 学習開始時間。
    """
    loss, policy_loss, value_loss = _calculate_losses(loss_data, iteration)
    training_time = time.time() - start_time

    print_err(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] learn")################
    print_err(f"epoch {epoch}, data-{index} : loss = {loss:6f}, time = {training_time:.1f} [s].")
    print_err(f"\tpolicy loss : {policy_loss:6f}")
    print_err(f"\tvalue loss  : {value_loss:6f}")


def print_evaluation_information(loss_data: Dict[str, float], epoch: int, \
    iteration: int, start_time: float) -> None:
    """テストデータの評価情報を表示する。

    Args:
        loss_data (Dict[str, float]): 損失関数値の情報。
        epoch (int): 学習エポック数。
        iteration (int): テストイテレーション数。
        start_time (float): 評価開始時間。
    """
    loss, policy_loss, value_loss = _calculate_losses(loss_data, iteration)
    testing_time = time.time() - start_time

    print_err(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] test")################
    print_err(f"Test {epoch} : loss = {loss:6f}, time = {testing_time:3f} [s].")
    print_err(f"\tpolicy loss : {policy_loss:6f}")
    print_err(f"\tvalue loss  : {value_loss:6f}")


def save_model(network: torch.nn.Module, path: str) -> NoReturn:
    """ニューラルネットワークのパラメータを保存する。

    Args:
        network (torch.nnModel): ニューラルネットワークのモデル。
        path (str): パラメータファイルパス。
    """
    torch.save(network.to("cpu").state_dict(), path)


def load_data_set(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """学習データセットを読み込む。シャッフルもする。

    Args:
        path (str): データセットのファイルパス。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 入力データ、Policy、Value。
    """
    data = np.load(path)

    # それぞれの関係を保ったままシャッフルして返す
    perm = np.random.permutation(len(data["value"]))
    return data["input"][perm], \
        data["policy"][perm].astype(np.float32), \
        data["value"][perm].astype(np.int64)


def split_train_test_set(file_list: List[str], train_data_ratio: float) \
    -> Tuple[List[str], List[str]]:
    """学習に使用するデータと検証に使用するデータファイルを分割する。

    Args:
        file_list (List[str]): 学習に使用するnpzファイルリスト。
        train_data_ratio (float): 学習に使用するデータの割合。

    Returns:
        Tuple[List[str], List[str]]: 学習データセットと検証データセット。
    """
    train_data_set = file_list[:int(len(file_list) * train_data_ratio)]
    test_data_set = file_list[int(len(file_list) * train_data_ratio):]

    print(f"Training data set : {train_data_set}")
    print(f"Testing data set  : {test_data_set}")

    return train_data_set, test_data_set


def apply_softmax(logits: np.array) -> np.array:
    """Softmax関数を適用する。

    Args:
        logits (np.array): Softmax関数の入力値。

    Returns:
        np.array: Softmax関数適用後の値。
    """
    shift_exp = np.exp(logits - np.max(logits))

    return shift_exp / np.sum(shift_exp)


def load_network(model_file_path: str, use_gpu: bool, gpu_num: int=-1) -> DualNet:
    """ニューラルネットワークをロードして取得する。

    Args:
        model_file_path (str): ニューラルネットワークのパラメータファイルパス。
        use_gpu (bool): GPU使用フラグ。

    Returns:
        DualNet: パラメータロード済みのニューラルネットワーク。
    """
    device = get_torch_device(use_gpu=use_gpu, gpu_num=gpu_num)
    network = DualNet(device)
    network.to(device)
    try:
        network.load_state_dict(torch.load(model_file_path))
    except Exception as e: # pylint: disable=W0702
        print(f"👺Failed to load_network {model_file_path}.")
        raise("Failed to load_network.")
    network.eval()
    torch.set_grad_enabled(False)

    return network



def load_DualNet_128_12(model_file_path: str, use_gpu: bool, gpu_num: int) -> DualNet_128_12:
    """ニューラルネットワークをロードして取得する。DualNet_128_12 版。

    Args:
        model_file_path (str): ニューラルネットワークのパラメータファイルパス。
        use_gpu (bool): GPU使用フラグ。

    Returns:
        DualNet: パラメータロード済みのニューラルネットワーク。
    """
    device = get_torch_device(use_gpu=use_gpu, gpu_num=gpu_num)
    network = DualNet_128_12(device)
    network.to(device)
    try:
        network.load_state_dict(torch.load(model_file_path))
    except Exception as e: # pylint: disable=W0702
        print(f"Failed to load_DualNet_128_12 {model_file_path}.")
        raise("Failed to load_DualNet_128_12.")
    network.eval()
    torch.set_grad_enabled(False)

    return network



def load_DualNet_256_24(model_file_path: str, use_gpu: bool, gpu_num: int) -> DualNet_128_12:
    """ニューラルネットワークをロードして取得する。DualNet_128_24 版。

    Args:
        model_file_path (str): ニューラルネットワークのパラメータファイルパス。
        use_gpu (bool): GPU使用フラグ。

    Returns:
        DualNet: パラメータロード済みのニューラルネットワーク。
    """
    device = get_torch_device(use_gpu=use_gpu, gpu_num=gpu_num)
    network = DualNet_256_24(device)
    network.to(device)
    try:
        network.load_state_dict(torch.load(model_file_path))
    except Exception as e: # pylint: disable=W0702
        print(f"Failed to load_DualNet_256_24 {model_file_path}.")
        print_err(f"Failed to load_DualNet_256_24 {model_file_path}.")
        raise Exception(f"Failed to load_DualNet_256_24., e: {e}")
    network.eval()
    torch.set_grad_enabled(False)

    return network

def load_DualNet_semeai(model_file_path: str, use_gpu: bool, gpu_num: int) -> DualNet_semeai:
    """ニューラルネットワークをロードして取得する。DualNet_semeai 版。

    Args:
        model_file_path (str): ニューラルネットワークのパラメータファイルパス。
        use_gpu (bool): GPU使用フラグ。

    Returns:
        DualNet: パラメータロード済みのニューラルネットワーク。
    """
    device = get_torch_device(use_gpu=use_gpu, gpu_num=gpu_num)
    network = DualNet_semeai(device)
    network.to(device)
    try:
        network.load_state_dict(torch.load(model_file_path))
    except Exception as e: # pylint: disable=W0702
        print(f"Failed to load_DualNet_semeai {model_file_path}.")
        raise("Failed to load_DualNet_semeai.")
    network.eval()
    torch.set_grad_enabled(False)

    return network


def load_DualNet_256_24_semeai(model_file_path: str, use_gpu: bool, gpu_num: int) -> DualNet_semeai:
    """ニューラルネットワークをロードして取得する。DualNet_semeai 版。

    Args:
        model_file_path (str): ニューラルネットワークのパラメータファイルパス。
        use_gpu (bool): GPU使用フラグ。

    Returns:
        DualNet: パラメータロード済みのニューラルネットワーク。
    """
    device = get_torch_device(use_gpu=use_gpu, gpu_num=gpu_num)
    network = DualNet_256_24_semeai(device)
    network.to(device)
    # print (f"model_file_path: {model_file_path}")#######
    # import os
    # print (f"extst???: {os.path.isfile(model_file_path)}")
    try:
        network.load_state_dict(torch.load(model_file_path))
    except Exception as e: # pylint: disable=W0702
        print(f"Failed to load_DualNet_semeai {model_file_path}.")
        raise("Failed to load_DualNet_semeai.")
    network.eval()
    torch.set_grad_enabled(False)

    return network


def choose_network(network_name: str, model_file_path: str, use_gpu: bool, gpu_num: int=-1):
    if network_name == "DualNet":
        network = load_network(model_file_path=model_file_path, use_gpu=use_gpu, gpu_num=gpu_num)
    elif network_name == "DualNet_128_12":
        network = load_DualNet_128_12(model_file_path=model_file_path, use_gpu=use_gpu, gpu_num=gpu_num)
    elif network_name == "DualNet_256_24":
        network = load_DualNet_256_24(model_file_path=model_file_path, use_gpu=use_gpu, gpu_num=gpu_num)
    elif network_name == "DualNet_semeai":
        network = load_DualNet_semeai(model_file_path=model_file_path, use_gpu=use_gpu, gpu_num=gpu_num)
    elif network_name == "DualNet_256_24_semeai":
        network = load_DualNet_256_24_semeai(model_file_path=model_file_path, use_gpu=use_gpu, gpu_num=gpu_num)
    else:
        print(f"👺network_name: {network_name} is not defined.")
        raise(f"network_name is not defined.")
    return network

TEST = "test"