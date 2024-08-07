"""碁盤のデータ定義と操作処理。
"""
from typing import List, Tuple, NoReturn
from collections import deque
import numpy as np

from board.constant import PASS, OB_SIZE, GTP_X_COORDINATE
from board.coordinate import Coordinate
from board.pattern import Pattern, copy_pattern
from board.record import Record, copy_record
from board.stone import Stone
from board.string import StringData, copy_strings
from board.zobrist_hash import affect_stone_hash, affect_string_hash
from common.print_console import print_err


class GoBoard: # pylint: disable=R0902
    """碁盤クラス
    """
    def __init__(self, board_size: int, komi: float=7.0, check_superko: bool=False):
        """碁盤クラスの初期化

        Args:
            board_size (int): 碁盤の大きさ。
            komi (float): コミの値。デフォルト値は7.0。
            check_superko (bool): 超劫の判定有効化。デフォルト値はFalse。
        """
        self.board_size = board_size
        self.board_size_with_ob = board_size + OB_SIZE * 2
        self.komi = komi

        def pos(x_coord: int, y_coord: int) -> int:
            """(x, y)座標を1次元配列のインデックスに変換する。

            Args:
                x_coord (int): X座標。
                y_coord (int): Y座標。

            Returns:
                int: _description_
            """
            return x_coord + y_coord * self.board_size_with_ob

        def get_neighbor4(pos: int) -> List[int]:
            """指定した座標の上下左右の座標を取得する。

            Args:
                pos (int): 基準となる座標。

            Returns:
                List[int]: 上下左右の座標列。
            """
            return [pos - self.board_size_with_ob, pos - 1, pos + 1, pos + self.board_size_with_ob]

        def get_cross4(pos: int) -> List[int]:
            """指定した座標の斜め方向の座標を取得する。
            """
            return [pos - self.board_size_with_ob - 1, pos - self.board_size_with_ob + 1, \
                pos + self.board_size_with_ob - 1, pos + self.board_size_with_ob + 1]

        self.board = [Stone.EMPTY] * (self.board_size_with_ob ** 2)
        self.pattern = Pattern(board_size, pos)
        self.strings = StringData(board_size, pos, get_neighbor4)
        self.record = Record()
        self.onboard_pos = [0] * (self.board_size ** 2)
        self.coordinate = Coordinate(board_size=board_size)
        self.ko_move = 0
        self.ko_pos = PASS
        self.prisoner = [0] * 2
        self.positional_hash = np.zeros(1, dtype=np.uint64)
        self.check_superko = check_superko
        self.board_start = OB_SIZE
        self.board_end = board_size + OB_SIZE - 1
        self.sym_map = [[0 for i in range(self.board_size_with_ob ** 2)] for j in range(8)]

        self.POS = pos # pylint: disable=C0103
        self.get_neighbor4 = get_neighbor4
        self.get_cross4 = get_cross4

        idx = 0
        for y_coord in range(self.board_start, self.board_end + 1):
            for x_coord in range(self.board_start, self.board_end + 1):
                coord = pos(x_coord, y_coord)
                self.onboard_pos[idx] = coord

                # そのまま
                self.sym_map[0][coord] = coord
                # 左右対称
                self.sym_map[1][coord] = pos(self.board_size_with_ob - (x_coord + 1), y_coord)
                # 上下対称
                self.sym_map[2][coord] = pos(x_coord, self.board_size_with_ob - (y_coord + 1))
                # 上下左右対称
                self.sym_map[3][coord] = pos(self.board_size_with_ob - (x_coord + 1),\
                    self.board_size_with_ob - (y_coord + 1))
                # 左上から右下方向の軸に対称
                self.sym_map[4][coord] = pos(x_coord=y_coord, y_coord=x_coord)
                # 90度反時計回りに回転
                self.sym_map[5][coord] = pos(y_coord, self.board_size_with_ob - (x_coord + 1))
                # 90度時計回りに回転
                self.sym_map[6][coord] = pos(self.board_size_with_ob - (y_coord + 1), x_coord)
                # 左下から右上方向の軸に対称
                self.sym_map[7][coord] = pos(self.board_size_with_ob - (y_coord + 1),\
                    self.board_size_with_ob - (x_coord + 1))
                idx += 1

        self.clear()


    def clear(self) -> NoReturn:
        """盤面の初期化
        """
        self.moves = 1
        self.position_hash = 0
        self.ko_move = 0
        self.ko_pos = 0
        self.prisoner = [0] * 2
        self.positional_hash.fill(0)

        for i, _ in enumerate(self.board):
            self.board[i] = Stone.OUT_OF_BOARD

        for y_coord in range(self.board_start, self.board_end + 1):
            for x_coord in range(self.board_start, self.board_end + 1):
                pos = self.POS(x_coord, y_coord)
                self.board[pos] = Stone.EMPTY

        self.pattern.clear()
        self.strings.clear()
        self.record.clear()

    # def put_stone(self, pos: int, color: Stone) -> NoReturn:#########
    def put_stone(self, pos: int, color: Stone) -> None:
        """指定された座標に指定された色の石を石を置く。

        Args:
            pos (int): 石を置く座標。
            color (Stone): 置く石の色。
        """
        if pos == PASS:
            self.record.save(self.moves, color, pos, self.positional_hash)
            self.moves += 1
            return

        opponent_color = Stone.get_opponent_color(color)

        self.board[pos] = color
        self.pattern.put_stone(pos, color)
        self.positional_hash = affect_stone_hash(self.positional_hash, pos, color)

        neighbor4 = self.get_neighbor4(pos)

        connection = []
        prisoner = 0

        for neighbor in neighbor4:
            if self.board[neighbor] == color:
                self.strings.remove_liberty(neighbor, pos)
                connection.append(self.strings.get_id(neighbor))
            elif self.board[neighbor] == opponent_color:
                self.strings.remove_liberty(neighbor, pos)
                if self.strings.get_num_liberties(neighbor) == 0:
                    removed_stones = self.strings.remove_string(self.board, neighbor)
                    prisoner += len(removed_stones)
                    for removed_pos in removed_stones:
                        self.pattern.remove_stone(removed_pos)
                    self.positional_hash = affect_string_hash(self.positional_hash, \
                        removed_stones, opponent_color)

        if color == Stone.BLACK:
            self.prisoner[0] += prisoner
        elif color == Stone.WHITE:
            self.prisoner[1] += prisoner

        if len(connection) == 0:
            self.strings.make_string(self.board, pos, color)
            if prisoner == 1 and self.strings.get_num_liberties(pos) == 1:
                self.ko_move = self.moves
                self.ko_pos = self.strings.string[self.strings.get_id(pos)].lib[0]
        elif len(connection) == 1:
            self.strings.add_stone(self.board, pos, color, connection[0])
        else:
            self.strings.connect_string(self.board, pos, color, connection)

        # 着手した時に記録
        self.record.save(self.moves, color, pos, self.positional_hash)
        self.moves += 1

    def put_handicap_stone(self, pos: int, color: Stone) -> NoReturn:
        """指定された座標に指定された色の置き石を置く。

        Args:
            pos (int): 石を置く座標。
            color (Stone): 置く石の色。
        """
        opponent_color = Stone.get_opponent_color(color)

        self.board[pos] = color
        self.pattern.put_stone(pos, color)
        self.positional_hash = affect_stone_hash(self.positional_hash, pos, color)

        neighbor4 = self.get_neighbor4(pos)

        connection = []
        prisoner = 0

        for neighbor in neighbor4:
            if self.board[neighbor] == color:
                self.strings.remove_liberty(neighbor, pos)
                connection.append(self.strings.get_id(neighbor))
            elif self.board[neighbor] == opponent_color:
                self.strings.remove_liberty(neighbor, pos)
                if self.strings.get_num_liberties(neighbor) == 0:
                    removed_stones = self.strings.remove_string(self.board, neighbor)
                    prisoner += len(removed_stones)
                    for removed_pos in removed_stones:
                        self.pattern.remove_stone(removed_pos)
                    self.positional_hash = affect_string_hash(self.positional_hash, \
                        removed_stones, opponent_color)

        if color == Stone.BLACK:
            self.prisoner[0] += prisoner
        elif color == Stone.WHITE:
            self.prisoner[1] += prisoner

        if len(connection) == 0:
            self.strings.make_string(self.board, pos, color)
            if prisoner == 1 and self.strings.get_num_liberties(pos) == 1:
                self.ko_move = self.moves
                self.ko_pos = self.strings.string[self.strings.get_id(pos)].lib[0]
        elif len(connection) == 1:
            self.strings.add_stone(self.board, pos, color, connection[0])
        else:
            self.strings.connect_string(self.board, pos, color, connection)

        # 着手した時に記録
        self.record.save_handicap(pos)

    def _is_suicide(self, pos: int, color: Stone) -> bool:
        """自殺手か否かを判定する。
        自殺手ならTrue、そうでなければFalseを返す。

        Args:
            pos (int): 確認する座標。
            color (Stone): 着手する石の色。

        Returns:
            bool: 自殺手の判定結果。自殺手ならTrue、そうでなければFalse。
        """
        other = Stone.get_opponent_color(color)

        neighbor4 = self.get_neighbor4(pos)

        for neighbor in neighbor4:
            if self.board[neighbor] is other and self.strings.get_num_liberties(neighbor) == 1:
                return False
            if self.board[neighbor] is color and self.strings.get_num_liberties(neighbor) > 1:
                return False

        return True

    def is_legal(self, pos: int, color: Stone) -> bool:
        """合法手か否かを判定する。
        合法手ならTrue、そうでなければFalseを返す。

        Args:
            pos (int): 確認する座標。
            color (Stone): 着手する石の色。

        Returns:
            bool: 合法手の判定結果。合法手ならTrue、そうでなければFalse。
        """
        # 既に石がある
        if self.board[pos] != Stone.EMPTY:
            return False

        # 自殺手
        if self.pattern.get_n_neighbors_empty(pos) == 0 and \
           self._is_suicide(pos, color):
            return False

        # 劫
        if (self.ko_pos == pos) and (self.ko_move == (self.moves - 1)):
            return False

        # 超劫の確認
        if self.check_superko and pos != PASS:
            opponent = Stone.get_opponent_color(color)
            neighbor4 = self.get_neighbor4(pos)
            neighbor_ids = [self.strings.get_id(neighbor) for neighbor in neighbor4]
            unique_ids = list(set(neighbor_ids))
            current_hash = self.positional_hash.copy()

            # 打ち上げる石があれば打ち上げたと仮定
            for string_id in unique_ids:
                if self.strings.get_num_liberties(self.strings.string[string_id].get_origin()) == 1:
                    stones = self.strings.get_stone_coordinates(string_id)
                    current_hash = affect_string_hash(current_hash, stones, opponent)
            # 石を置く
            current_hash = affect_stone_hash(current_hash, pos=pos, color=color)

            if self.record.has_same_hash(current_hash):
                return False


        return True

    def is_legal_not_eye(self, pos: int, color: Stone) -> bool:
        """合法手かつ眼でないか否かを確認する。
        合法手かつ眼でなければTrue、そうでなければFalseを返す。

        Args:
            pos (int): 確認する座標。
            color (Stone): 手番の色。

        Returns:
            bool: 判定結果。合法手かつ眼でなければTrue、そうでなければFalse。
        """
        neighbor4 = self.get_neighbor4(pos)
        if self.pattern.get_eye_color(pos) is not color or \
           self.strings.get_num_liberties(neighbor4[0]) == 1 or \
           self.strings.get_num_liberties(neighbor4[1]) == 1 or \
           self.strings.get_num_liberties(neighbor4[2]) == 1 or \
           self.strings.get_num_liberties(neighbor4[3]) == 1:
            return self.is_legal(pos, color)

        return False

    def check_self_atari_stone(self, pos: int, color: Stone) -> int:
        """アタリに突っ込んで取られる石の数を返す。取られない場合は0を返す。

        Args:
            pos (int): 評価する座標。
            color (Stone): 着手する手番の色。

        Returns:
            int: 取られる石の数
        """
        neighbor4 = self.get_neighbor4(pos)

        lib_candidate = []
        for neighbor in neighbor4:
            if self.board[neighbor] is Stone.EMPTY:
                lib_candidate.append(neighbor)

        if len(lib_candidate) > 1:
            return 0
        checked = []
        size = 0
        other = Stone.get_opponent_color(color)
        for neighbor in neighbor4:
            if self.board[neighbor] is color:
                string_id = self.strings.get_id(neighbor)
                if string_id in checked:
                    continue
                lib_candidate.extend(self.strings.string[string_id].get_liberties())
                lib_candidate = list(set(lib_candidate))
                if len(lib_candidate) >= 3:
                    return 0
                size += self.strings.string[string_id].get_size()
                checked.append(string_id)
            elif self.board[neighbor] is other:
                if self.strings.get_num_liberties(neighbor) == 1:
                    return 0

        # 石を打つ分として1を足す。
        return size + 1

    def is_complete_eye(self, pos: int, color: Stone) -> bool:
        """完全な眼形か否かを判定する。

        Args:
            pos (int): 確認する座標。
            color (Stone): 手番の色。

        Returns:
            bool: 完全な眼の判定結果
        """
        if self.pattern.get_eye_color(pos) is color:
            connection_count = 0
            edge = False

            for cross in self.get_cross4(pos):
                if self.board[cross] in [color, Stone.OUT_OF_BOARD]:
                    connection_count += 1
                elif self.board[cross] is Stone.EMPTY and \
                    self.pattern.get_eye_color(cross) is color:
                    connection_count += 1

                if self.board[cross] is Stone.OUT_OF_BOARD:
                    edge = True

            # 完全な眼の条件は下記2つのいずれかを満たすこと。
            #   1. 盤端かつ4方向の斜めの箇所がちゃんと結合していること
            #   2. 盤端ではなく、かつ4方向の斜めの箇所の内、3箇所を自分の石が専有していること
            if (edge and connection_count == 4) or (not edge and connection_count >= 3):
                return True

        return False


    def get_all_legal_pos(self, color: Stone) -> List[int]:
        """全ての合法手の座標を取得する。ただし眼は除く。

        Args:
            color (Stone): 手番の色

        Returns:
            list[int]: 合法手の座標列。
        """
        return [pos for pos in self.onboard_pos if self.is_legal(pos, color)]

    def display(self, sym: int=0) -> NoReturn:
        """盤面を表示する。
        """
        board_string = f"Move : {self.moves}\n"
        board_string += f"Prisoner(Black) : {self.prisoner[0]}\n"
        board_string += f"Prisoner(White) : {self.prisoner[1]}\n"

        board_string += "   "
        for i in range(self.board_size):
            board_string += " " + GTP_X_COORDINATE[i + 1]
        board_string += "\n"

        board_string += "  +" + "-" * (self.board_size * 2 + 1) + "+\n"

        for y_coord in range(self.board_start, self.board_end + 1):
            output = f"{self.board_size - y_coord + 1:>2d}|"
            for x_coord in range(self.board_start, self.board_end + 1):
                pos = self.get_symmetrical_coordinate(self.POS(x_coord, y_coord), sym)
                output += " " + Stone.get_char(self.board[pos])
            output += " |\n"
            board_string += output

        board_string += "  +" + "-" * (self.board_size * 2 + 1) + "+\n"

        print_err(board_string)


    def display_self_atari(self, color: Stone) -> NoReturn:
        """アタリに突っ込んだ時に取られる石の数を表示する。取られない場合は0。デバッグ用。

        Args:
            color (Stone): 手番の色。
        """
        self_atari_string = ""
        for i, pos in enumerate(self.onboard_pos):
            if self.board[pos] is Stone.EMPTY and self.is_legal(pos, color):
                print_err(self.coordinate.convert_to_gtp_format(pos))
                self_atari_string += f"{self.check_self_atari_stone(pos, color):3}"
            else:
                self_atari_string += "  0"
            if (i + 1) % self.board_size == 0:
                self_atari_string += '\n'
        print_err(self_atari_string)

    # def get_board_size(self) -> NoReturn:#########
    def get_board_size(self) -> int:
        """碁盤の大きさを取得する。

        Returns:
            int: 碁盤の大きさ
        """
        return self.board_size

    def get_board_data(self, sym: int) -> List[int]:
        """ニューラルネットワークの入力用の碁盤情報を取得する。

        Args:
            sym (int): 対称形の番号。

        Returns:
            list[int]: 空点は0, 黒石は1, 白石は2のリスト。
        """
        return [self.board[self.get_symmetrical_coordinate(pos, sym)].value \
            for pos in self.onboard_pos]

    def get_liberty_data(self, sym: int) -> List[int]:
        """ニューラルネットワークの入力用の呼吸点数の情報を取得する。

        Args:
            sym (int): 対称形の番号。

        Returns:
            List[int]: _description_
        """
        base_data = [0] * (self.board_size_with_ob ** 2)
        for index, string in enumerate(self.strings.string):
            if string.exist():
                num_liberties = string.get_num_liberties()
                coordinates = self.strings.get_stone_coordinates(index)
                for coordinate in coordinates:
                    base_data[coordinate] = num_liberties
        return [base_data[self.get_symmetrical_coordinate(pos, sym)] \
            for pos in self.onboard_pos]

    def get_symmetrical_coordinate(self, pos: int, sym: int) -> int:
        """8対称のいずれかの座標を取得する。

        Args:
            pos (int): 元の座標。
            sym (int): 対称形の指定。（0〜7）

        Returns:
            int: 指定した対称の座標。
        """
        return self.sym_map[sym][pos]

    def set_komi(self, komi: float) -> NoReturn:
        """コミを設定する。

        Args:
            komi (float): 設定するコミの値。
        """
        self.komi = komi

    def get_komi(self) -> float:
        """現在のコミの値を設定する。

        Returns:
            float: 現在のコミの値。
        """
        return self.komi

    def get_to_move(self) -> Stone:
        """手番の色を取得する。

        Returns:
            Stone: 手番の色。
        """
        if self.moves == 1:
            return Stone.BLACK
        last_move_color, _, _ = self.record.get(self.moves - 1)
        return Stone.get_opponent_color(last_move_color)

    def get_move_history(self) -> List[Tuple[Stone, int, np.array]]:
        """着手の履歴を取得する。

        Returns:
            [(Stone, int, np.array), ...]: (着手の色、座標、ハッシュ値) のリスト。
        """
        return [self.record.get(m) for m in range(1, self.moves)]

    def get_handicap_history(self) -> List[int]:
        """置き石の座標を取得する。

        Returns:
            List[int]: 置き石の座標のリスト。
        """
        return self.record.handicap_pos[:]

    def count_score(self) -> int: # pylint: disable=R0912
        """領地を簡易的にカウントする。

        Returns:
            int: 黒から見た領地の数（コミは考慮しない)。
        """
        board = self.board[:]

        # 明らかに死んでいる石は打ち上げたとみなす
        for pos in self.onboard_pos:
            if self.board[pos] in [Stone.BLACK, Stone.WHITE]:
                if self.strings.get_num_liberties(pos) == 1:
                    board[pos] = Stone.EMPTY

        already_check = [False] * len(self.board)

        # 同じ色の石に囲まれている空点をその色の領地とする。
        # 違う色が混じった場合は領地として認識しないようにする。
        for pos in self.onboard_pos: # pylint: disable=R1702
            if board[pos] is Stone.EMPTY:
                pos_list = []
                pos_queue = deque()
                pos_queue.append(pos)
                color = Stone.EMPTY
                while pos_queue:
                    coord = pos_queue.popleft()
                    if board[coord] is Stone.OUT_OF_BOARD or already_check[coord]:
                        continue
                    neighbor4 = self.get_neighbor4(coord)
                    for neighbor in neighbor4:
                        if board[neighbor] is Stone.EMPTY:
                            pos_queue.append(coord)
                        elif board[neighbor] in [Stone.BLACK, Stone.WHITE]:
                            if color is Stone.EMPTY:
                                color = board[neighbor]
                            elif color != board[neighbor]:
                                color = Stone.OUT_OF_BOARD
                    already_check[coord] = True
                    pos_list.append(coord)
                for coord in pos_list:
                    board[pos] = color
            else:
                already_check[pos] = True

        black = board.count(Stone.BLACK)
        white = board.count(Stone.WHITE)

        return black - white


def copy_board(dst: GoBoard, src: GoBoard):
    """盤面の情報をコピーする。

    Args:
        dst (GoBoard): コピー先の盤面情報のデータ。
        src (GoBoard): コピー元の盤面情報のデータ。
    """
    dst.board = src.board[:]
    copy_pattern(dst.pattern, src.pattern)
    copy_strings(dst.strings, src.strings)
    copy_record(dst.record, src.record)
    dst.ko_move = src.ko_move
    dst.ko_pos = src.ko_pos
    dst.prisoner = src.prisoner[:]
    dst.positional_hash = src.positional_hash.copy()
    dst.moves = src.moves
