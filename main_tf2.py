#coding:utf-8
from asyncio import Future
import asyncio
from asyncio.queues import Queue
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

import tensorflow as tf
import numpy as np
import os
import sys
import random
import time
import argparse
from collections import deque, defaultdict, namedtuple
import copy
from policy_value_network_tf2 import *
from policy_value_network_gpus_tf2 import *
import scipy.stats
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

def flipped_uci_labels(param):
    def repl(x):
        return "".join([(str(9 - int(a)) if a.isdigit() else a) for a in x])

    return [repl(x) for x in param]

# 创建所有合法走子UCI，size 2086
def create_uci_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    Advisor_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                      'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    Bishop_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                     'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                     'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                     'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']
    # King_labels = ['d0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9',
    #                'd7d0', 'd7d1', 'd7d2', 'd8d0', 'd8d1', 'd8d2', 'd9d0', 'd9d1', 'd9d2',
    #                'd0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9',
    #                'd0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9',
    #                'd0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9',
    #                'd0d7', 'd0d8', 'd0d9', 'd1d7', 'd1d8', 'd1d9', 'd2d7', 'd2d8', 'd2d9']

    for l1 in range(9):
        for n1 in range(10):
            destinations = [(t, n1) for t in range(9)] + \
                           [(l1, t) for t in range(10)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(9) and n2 in range(10):
                    move = letters[l1] + numbers[n1] + letters[l2] + numbers[n2]
                    labels_array.append(move)

    for p in Advisor_labels:
        labels_array.append(p)

    for p in Bishop_labels:
        labels_array.append(p)

    return labels_array

def create_position_labels():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[8 - l1] + numbers[n1]
            labels_array.append(move)
#     labels_array.reverse()
    return labels_array

def create_position_labels_reverse():
    labels_array = []
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    letters.reverse()
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for l1 in range(9):
        for n1 in range(10):
            move = letters[l1] + numbers[n1]
            labels_array.append(move)
    labels_array.reverse()
    return labels_array

class leaf_node(object):
    def __init__(self, in_parent, in_prior_p, in_state):
        self.P = in_prior_p
        self.Q = 0
        self.N = 0
        self.v = 0
        self.U = 0
        self.W = 0
        self.parent = in_parent
        self.child = {}
        self.state = in_state

    def is_leaf(self):
        return self.child == {}

    def get_Q_plus_U_new(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        # self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        U = c_puct * self.P * np.sqrt(self.parent.N) / ( 1 + self.N)
        return self.Q + U

    def get_Q_plus_U(self, c_puct):
        """Calculate and return the value for this node: a combination of leaf evaluations, Q, and
        this node's prior adjusted for its visit count, u
        c_puct -- a number in (0, inf) controlling the relative impact of values, Q, and
            prior probability, P, on this node's score.
        """
        # self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        self.U = c_puct * self.P * np.sqrt(self.parent.N) / ( 1 + self.N)
        return self.Q + self.U

    # def select_move_by_action_score(self, noise=True):
    #
    #     # P = params[self.lookup['P']]
    #     # N = params[self.lookup['N']]
    #     # Q = params[self.lookup['W']] / (N + 1e-8)
    #     # U = c_PUCT * P * np.sqrt(np.sum(N)) / (1 + N)
    #
    #     ret_a = None
    #     ret_n = None
    #     action_idx = {}
    #     action_score = []
    #     i = 0
    #     for a, n in self.child.items():
    #         U = c_PUCT * n.P * np.sqrt(n.parent.N) / ( 1 + n.N)
    #         action_idx[i] = (a, n)
    #
    #         if noise:
    #             action_score.append(n.Q + U * (0.75 * n.P + 0.25 * dirichlet([.03] * (go.N ** 2 + 1))) / (n.P + 1e-8))
    #         else:
    #             action_score.append(n.Q + U)
    #         i += 1
    #         # if(n.Q + n.U > max_Q_plus_U):
    #         #     max_Q_plus_U = n.Q + n.U
    #         #     ret_a = a
    #         #     ret_n = n
    #
    #     action_t = int(np.argmax(action_score[:-1]))
    #
    #     return ret_a, ret_n
    #     # return action_t
    def select_new(self, c_puct):
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U_new(c_puct))

    def select(self, c_puct):
        # max_Q_plus_U = 1e-10
        # ret_a = None
        # ret_n = None
        # for a, n in self.child.items():
        #     n.U = c_puct * n.P * np.sqrt(n.parent.N) / ( 1 + n.N)
        #     if(n.Q + n.U > max_Q_plus_U):
        #         max_Q_plus_U = n.Q + n.U
        #         ret_a = a
        #         ret_n = n
        # return ret_a, ret_n
        return max(self.child.items(), key=lambda node: node[1].get_Q_plus_U(c_puct))

    #@profile
    def expand(self, moves, action_probs):
        tot_p = 1e-8
        # print("action_probs : ", action_probs)
        action_probs = tf.squeeze(action_probs)  #.flatten()   #.squeeze()
        # print("expand action_probs shape : ", action_probs.shape)
        for action in moves:
            in_state = GameBoard.sim_do_action(action, self.state)
            mov_p = action_probs[label2i[action]]
            new_node = leaf_node(self, mov_p, in_state)
            self.child[action] = new_node
            tot_p += mov_p

        for a, n in self.child.items():
            n.P /= tot_p

    def back_up_value(self, value):
        self.N += 1
        self.W += value
        self.v = value
        self.Q = self.W / self.N  # node.Q += 1.0*(value - node.Q) / node.N
        self.U = c_PUCT * self.P * np.sqrt(self.parent.N) / ( 1 + self.N)
        # node = node.parent
        # value = -value

    def backup(self, value):
        node = self
        while node != None:
            node.N += 1
            node.W += value
            node.v = value
            node.Q = node.W / node.N    # node.Q += 1.0*(value - node.Q) / node.N
            node = node.parent
            value = -value

pieces_order = 'KARBNPCkarbnpc' # 9 x 10 x 14
ind = {pieces_order[i]: i for i in range(14)}

labels_array = create_uci_labels()
labels_len = len(labels_array)
flipped_labels = flipped_uci_labels(labels_array)
unflipped_index = [labels_array.index(x) for x in flipped_labels]

i2label = {i: val for i, val in enumerate(labels_array)}
label2i = {val: i for i, val in enumerate(labels_array)}

def get_pieces_count(state):
    count = 0
    for s in state:
        if s.isalpha():
            count += 1
    return count

def is_kill_move(state_prev, state_next):
    return get_pieces_count(state_prev) - get_pieces_count(state_next)

QueueItem = namedtuple("QueueItem", "feature future")
c_PUCT = 5
virtual_loss = 3
cut_off_depth = 30

class MCTS_tree(object):
    def __init__(self, in_state, in_forward, search_threads):
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.3    #0.03
        self.p_ = (1 - self.noise_eps) * 1 + self.noise_eps * np.random.dirichlet([self.dirichlet_alpha])
        self.root = leaf_node(None, self.p_, in_state)
        self.c_puct = 5    #1.5
        # self.policy_network = in_policy_network
        self.forward = in_forward
        self.node_lock = defaultdict(Lock)

        self.virtual_loss = 3
        self.now_expanding = set()
        self.expanded = set()
        self.cut_off_depth = 30
        # self.QueueItem = namedtuple("QueueItem", "feature future")
        self.sem = asyncio.Semaphore(search_threads)
        self.queue = Queue(search_threads)
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0

    def reload(self):
        self.root = leaf_node(None, self.p_,
                         "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")  # "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
        self.expanded = set()


    def Q(self, move) -> float:
        ret = 0.0
        find = False
        for a, n in self.root.child.items():
            if move == a:
                ret = n.Q
                find = True
        if(find == False):
            print("{} not exist in the child".format(move))
        return ret

    def update_tree(self, act):
        # if(act in self.root.child):
        self.expanded.discard(self.root)
        self.root = self.root.child[act]
        self.root.parent = None
        # else:
        #     self.root = leaf_node(None, self.p_, in_state)


    # def do_simulation(self, state, current_player, restrict_round):
    #     node = self.root
    #     last_state = state
    #     while(node.is_leaf() == False):
    #         # print("do_simulation while current_player : ", current_player)
    #         with self.node_lock[node]:
    #             action, node = node.select(self.c_puct)
    #         current_player = "w" if current_player == "b" else "b"
    #         if is_kill_move(last_state, node.state) == 0:
    #             restrict_round += 1
    #         else:
    #             restrict_round = 0
    #         last_state = node.state
    #
    #     positions = self.generate_inputs(node.state, current_player)
    #     positions = np.expand_dims(positions, 0)
    #     action_probs, value = self.forward(positions)
    #     if self.is_black_turn(current_player):
    #         action_probs = cchess_main.flip_policy(action_probs)
    #
    #     # print("action_probs shape : ", action_probs.shape)    #(1, 2086)
    #     with self.node_lock[node]:
    #         if(node.state.find('K') == -1 or node.state.find('k') == -1):
    #             if (node.state.find('K') == -1):
    #                 value = 1.0 if current_player == "b" else -1.0
    #             if (node.state.find('k') == -1):
    #                 value = -1.0 if current_player == "b" else 1.0
    #         elif restrict_round >= 60:
    #             value = 0.0
    #         else:
    #             moves = GameBoard.get_legal_moves(node.state, current_player)
    #             # print("current_player : ", current_player)
    #             # print(moves)
    #             node.expand(moves, action_probs)
    #
    #         # if(node.parent != None):
    #         #     node.parent.N += self.virtual_loss
    #         node.N += self.virtual_loss
    #         node.W += -self.virtual_loss
    #         node.Q = node.W / node.N
    #
    #     # time.sleep(0.1)
    #
    #     with self.node_lock[node]:
    #         # if(node.parent != None):
    #         #     node.parent.N += -self.virtual_loss# + 1
    #         node.N += -self.virtual_loss# + 1
    #         node.W += self.virtual_loss# + leaf_v
    #         # node.Q = node.W / node.N
    #
    #         node.backup(-value)

    def is_expanded(self, key) -> bool:
        """Check expanded status"""
        return key in self.expanded

    async def tree_search(self, node, current_player, restrict_round) -> float:
        """Independent MCTS, stands for one simulation"""
        self.running_simulation_num += 1

        # reduce parallel search number
        with await self.sem:
            value = await self.start_tree_search(node, current_player, restrict_round)
            # logger.debug(f"value: {value}")
            # logger.debug(f'Current running threads : {RUNNING_SIMULATION_NUM}')
            self.running_simulation_num -= 1

            return value

    async def start_tree_search(self, node, current_player, restrict_round)->float:
        """Monte Carlo Tree search Select,Expand,Evauate,Backup"""
        now_expanding = self.now_expanding

        while node in now_expanding:
            await asyncio.sleep(1e-4)

        if not self.is_expanded(node):    # and node.is_leaf()
            """is leaf node try evaluate and expand"""
            # add leaf node to expanding list
            self.now_expanding.add(node)

            positions = self.generate_inputs(node.state, current_player)
            # positions = np.expand_dims(positions, 0)

            # push extracted dihedral features of leaf node to the evaluation queue
            future = await self.push_queue(positions)  # type: Future
            await future
            action_probs, value = future.result()

            # action_probs, value = self.forward(positions)
            if self.is_black_turn(current_player):
                action_probs = cchess_main.flip_policy(action_probs)

            moves = GameBoard.get_legal_moves(node.state, current_player)
            # print("current_player : ", current_player)
            # print(moves)
            node.expand(moves, action_probs)
            self.expanded.add(node)  # node.state

            # remove leaf node from expanding list
            self.now_expanding.remove(node)

            # must invert, because alternative layer has opposite objective
            return value[0] * -1

        else:
            """node has already expanded. Enter select phase."""
            # select child node with maximum action scroe
            last_state = node.state

            action, node = node.select_new(c_PUCT)
            current_player = "w" if current_player == "b" else "b"
            if is_kill_move(last_state, node.state) == 0:
                restrict_round += 1
            else:
                restrict_round = 0
            last_state = node.state

            # action_t = self.select_move_by_action_score(key, noise=True)

            # add virtual loss
            # self.virtual_loss_do(key, action_t)
            node.N += virtual_loss
            node.W += -virtual_loss

            # evolve game board status
            # child_position = self.env_action(position, action_t)

            if (node.state.find('K') == -1 or node.state.find('k') == -1):
                if (node.state.find('K') == -1):
                    value = 1.0 if current_player == "b" else -1.0
                if (node.state.find('k') == -1):
                    value = -1.0 if current_player == "b" else 1.0
                value = value * -1
            elif restrict_round >= 60:
                value = 0.0
            else:
                value = await self.start_tree_search(node, current_player, restrict_round)  # next move
            # if node is not None:
            #     value = await self.start_tree_search(node)  # next move
            # else:
            #     # None position means illegal move
            #     value = -1

            # self.virtual_loss_undo(key, action_t)
            node.N += -virtual_loss
            node.W += virtual_loss

            # on returning search path
            # update: N, W, Q, U
            # self.back_up_value(key, action_t, value)
            node.back_up_value(value)    # -value

            # must invert
            return value * -1
            # if child_position is not None:
            #     return value * -1
            # else:
            #     # illegal move doesn't mean much for the opponent
            #     return 0

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        speed up about 45sec -> 15sec for example.
        """
        q = self.queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(1e-3)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            #logger.debug(f"predicting {len(item_list)} items")
            features = np.asarray([item.feature for item in item_list])    # asarray
            # print("prediction_worker [features.shape] before : ", features.shape)
            # shape = features.shape
            # features = features.reshape((shape[0] * shape[1], shape[2], shape[3], shape[4]))
            # print("prediction_worker [features.shape] after : ", features.shape)
            # policy_ary, value_ary = self.run_many(features)
            action_probs, value = self.forward(features)
            for p, v, item in zip(action_probs, value, item_list):
                item.future.set_result((p, v))

    async def push_queue(self, features):
        future = self.loop.create_future()
        item = QueueItem(features, future)
        await self.queue.put(item)
        return future

    #@profile
    def main(self, state, current_player, restrict_round, playouts):
        node = self.root
        if not self.is_expanded(node):    # and node.is_leaf()    # node.state
            # print('Expadning Root Node...')
            positions = self.generate_inputs(node.state, current_player)
            positions = np.expand_dims(positions, 0)
            action_probs, value = self.forward(positions)
            if self.is_black_turn(current_player):
                action_probs = cchess_main.flip_policy(action_probs)

            moves = GameBoard.get_legal_moves(node.state, current_player)
            # print("current_player : ", current_player)
            # print(moves)
            node.expand(moves, action_probs)
            self.expanded.add(node)    # node.state

        coroutine_list = []
        for _ in range(playouts):
            coroutine_list.append(self.tree_search(node, current_player, restrict_round))
        coroutine_list.append(self.prediction_worker())
        self.loop.run_until_complete(asyncio.gather(*coroutine_list))

    def do_simulation(self, state, current_player, restrict_round):
        node = self.root
        last_state = state
        while(node.is_leaf() == False):
            # print("do_simulation while current_player : ", current_player)
            action, node = node.select(self.c_puct)
            current_player = "w" if current_player == "b" else "b"
            if is_kill_move(last_state, node.state) == 0:
                restrict_round += 1
            else:
                restrict_round = 0
            last_state = node.state

        positions = self.generate_inputs(node.state, current_player)
        positions = np.expand_dims(positions, 0)
        action_probs, value = self.forward(positions)
        if self.is_black_turn(current_player):
            action_probs = cchess_main.flip_policy(action_probs)

        # print("action_probs shape : ", action_probs.shape)    #(1, 2086)

        if(node.state.find('K') == -1 or node.state.find('k') == -1):
            if (node.state.find('K') == -1):
                value = 1.0 if current_player == "b" else -1.0
            if (node.state.find('k') == -1):
                value = -1.0 if current_player == "b" else 1.0
        elif restrict_round >= 60:
            value = 0.0
        else:
            moves = GameBoard.get_legal_moves(node.state, current_player)
            # print("current_player : ", current_player)
            # print(moves)
            node.expand(moves, action_probs)

        node.backup(-value)

    def generate_inputs(self, in_state, current_player):
        state, palyer = self.try_flip(in_state, current_player, self.is_black_turn(current_player))
        return self.state_to_positions(state)

    def replace_board_tags(self, board):
        board = board.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.replace("/", "")

    # 感觉位置有点反了，当前角色的棋子在右侧，plane的后面
    def state_to_positions(self, state):
        # TODO C plain x 2
        board_state = self.replace_board_tags(state)
        pieces_plane = np.zeros(shape=(9, 10, 14), dtype=np.float32)
        for rank in range(9):    #横线
            for file in range(10):    #直线
                v = board_state[rank * 9 + file]
                if v.isalpha():
                    pieces_plane[rank][file][ind[v]] = 1
        assert pieces_plane.shape == (9, 10, 14)
        return pieces_plane


    def try_flip(self, state, current_player, flip=False):
        if not flip:
            return state, current_player

        rows = state.split('/')

        def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a

        def swapall(aa):
            return "".join([swapcase(a) for a in aa])

        return "/".join([swapall(row) for row in reversed(rows)]),  ('w' if current_player == 'b' else 'b')

    def is_black_turn(self, current_player):
        return current_player == 'b'

class GameBoard(object):
    board_pos_name = np.array(create_position_labels()).reshape(9,10).transpose()
    Ny = 10
    Nx = 9

    def __init__(self):
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"    #
        self.round = 1
        # self.players = ["w", "b"]
        self.current_player = "w"
        self.restrict_round = 0

# 小写表示黑方，大写表示红方
# [
#     "rheakaehr",
#     "         ",
#     " c     c ",
#     "p p p p p",
#     "         ",
#     "         ",
#     "P P P P P",
#     " C     C ",
#     "         ",
#     "RHEAKAEHR"
# ]
    def reload(self):
        self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"    #
        self.round = 1
        self.current_player = "w"
        self.restrict_round = 0

    @staticmethod
    def print_borad(board, action = None):
        def string_reverse(string):
            # return ''.join(string[len(string) - i] for i in range(1, len(string)+1))
            return ''.join(string[i] for i in range(len(string) - 1, -1, -1))

        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        if(action != None):
            src = action[0:2]

            src_x = int(x_trans[src[0]])
            src_y = int(src[1])

        # board = string_reverse(board)
        board = board.replace("1", " ")
        board = board.replace("2", "  ")
        board = board.replace("3", "   ")
        board = board.replace("4", "    ")
        board = board.replace("5", "     ")
        board = board.replace("6", "      ")
        board = board.replace("7", "       ")
        board = board.replace("8", "        ")
        board = board.replace("9", "         ")
        board = board.split('/')
        # board = board.replace("/", "\n")
        print("  abcdefghi")
        for i,line in enumerate(board):
            if (action != None):
                if(i == src_y):
                    s = list(line)
                    s[src_x] = 'x'
                    line = ''.join(s)
            print(i,line)
        # print(board)

    @staticmethod
    def sim_do_action(in_action, in_state):
        x_trans = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7, 'i':8}

        src = in_action[0:2]
        dst = in_action[2:4]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])

        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        # GameBoard.print_borad(in_state)
        # print("sim_do_action : ", in_action)
        # print(dst_y, dst_x, src_y, src_x)
        board_positions = GameBoard.board_to_pos_name(in_state)
        line_lst = []
        for line in board_positions:
            line_lst.append(list(line))
        lines = np.array(line_lst)
        # print(lines.shape)
        # print(board_positions[src_y])
        # print("before board_positions[dst_y] = ",board_positions[dst_y])

        lines[dst_y][dst_x] = lines[src_y][src_x]
        lines[src_y][src_x] = '1'

        board_positions[dst_y] = ''.join(lines[dst_y])
        board_positions[src_y] = ''.join(lines[src_y])

        # src_str = list(board_positions[src_y])
        # dst_str = list(board_positions[dst_y])
        # print("src_str[src_x] = ", src_str[src_x])
        # print("dst_str[dst_x] = ", dst_str[dst_x])
        # c = copy.deepcopy(src_str[src_x])
        # dst_str[dst_x] = c
        # src_str[src_x] = '1'
        # board_positions[dst_y] = ''.join(dst_str)
        # board_positions[src_y] = ''.join(src_str)
        # print("after board_positions[dst_y] = ", board_positions[dst_y])

        # board_positions[dst_y][dst_x] = board_positions[src_y][src_x]
        # board_positions[src_y][src_x] = '1'

        board = "/".join(board_positions)
        board = board.replace("111111111", "9")
        board = board.replace("11111111", "8")
        board = board.replace("1111111", "7")
        board = board.replace("111111", "6")
        board = board.replace("11111", "5")
        board = board.replace("1111", "4")
        board = board.replace("111", "3")
        board = board.replace("11", "2")

        # GameBoard.print_borad(board)
        return board

    @staticmethod
    def board_to_pos_name(board):
        board = board.replace("2", "11")
        board = board.replace("3", "111")
        board = board.replace("4", "1111")
        board = board.replace("5", "11111")
        board = board.replace("6", "111111")
        board = board.replace("7", "1111111")
        board = board.replace("8", "11111111")
        board = board.replace("9", "111111111")
        return board.split("/")

    @staticmethod
    def check_bounds(toY, toX):
        if toY < 0 or toX < 0:
            return False

        if toY >= GameBoard.Ny or toX >= GameBoard.Nx:
            return False

        return True

    @staticmethod
    def validate_move(c, upper=True):
        if (c.isalpha()):
            if (upper == True):
                if (c.islower()):
                    return True
                else:
                    return False
            else:
                if (c.isupper()):
                    return True
                else:
                    return False
        else:
            return True

    @staticmethod
    def get_legal_moves(state, current_player):
        moves = []
        k_x = None
        k_y = None

        K_x = None
        K_y = None

        face_to_face = False

        board_positions = np.array(GameBoard.board_to_pos_name(state))
        for y in range(board_positions.shape[0]):
            for x in range(len(board_positions[y])):
                if(board_positions[y][x].isalpha()):
                    if(board_positions[y][x] == 'r' and current_player == 'b'):
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].isupper()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif(board_positions[y][x] == 'R' and current_player == 'w'):
                        toY = y
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        toX = x
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (board_positions[toY][toX].isalpha()):
                                if (board_positions[toY][toX].islower()):
                                    moves.append(m)
                                break

                            moves.append(m)

                    elif ((board_positions[y][x] == 'n' or board_positions[y][x] == 'h') and current_player == 'b'):
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'N' or board_positions[y][x] == 'H') and current_player == 'w'):
                        for i in range(-1, 3, 2):
                            for j in range(-1, 3, 2):
                                toY = y + 2 * i
                                toX = x + 1 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True) and board_positions[toY - i][x].isalpha() == False:
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                                toY = y + 1 * i
                                toX = x + 2 * j
                                if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True) and board_positions[y][toX - j].isalpha() == False:
                                    moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'b' or board_positions[y][x] == 'e') and current_player == 'b'):
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 5 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif ((board_positions[y][x] == 'B' or board_positions[y][x] == 'E') and current_player == 'w'):
                        for i in range(-2, 3, 4):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x + i // 2].isalpha() == False:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 4 and \
                                            board_positions[y + i // 2][x - i // 2].isalpha() == False:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'a' and current_player == 'b'):
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'A' and current_player == 'w'):
                        for i in range(-1, 3, 2):
                            toY = y + i
                            toX = x + i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                            toY = y + i
                            toX = x - i

                            if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                        upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'k'):
                        k_x = x
                        k_y = y

                        if(current_player == 'b'):
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                                upper=False) and toY >= 7 and toX >= 3 and toX <= 5:
                                        moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'K'):
                        K_x = x
                        K_y = y

                        if(current_player == 'w'):
                            for i in range(2):
                                for sign in range(-1, 2, 2):
                                    j = 1 - i
                                    toY = y + i * sign
                                    toX = x + j * sign

                                    if GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX],
                                                                                upper=True) and toY <= 2 and toX >= 3 and toX <= 5:
                                        moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])
                    elif (board_positions[y][x] == 'c' and current_player == 'b'):
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].isupper()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'C' and current_player == 'w'):
                        toY = y
                        hits = False
                        for toX in range(x - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toX in range(x + 1, GameBoard.Nx):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        toX = x
                        hits = False
                        for toY in range(y - 1, -1, -1):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break

                        hits = False
                        for toY in range(y + 1, GameBoard.Ny):
                            m = GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX]
                            if (hits == False):
                                if (board_positions[toY][toX].isalpha()):
                                    hits = True
                                else:
                                    moves.append(m)
                            else:
                                if (board_positions[toY][toX].isalpha()):
                                    if (board_positions[toY][toX].islower()):
                                        moves.append(m)
                                    break
                    elif (board_positions[y][x] == 'p' and current_player == 'b'):
                        toY = y - 1
                        toX = x

                        if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False)):
                            moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                        if y < 5:
                            toY = y
                            toX = x + 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                            toX = x - 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=False)):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                    elif (board_positions[y][x] == 'P' and current_player == 'w'):
                        toY = y + 1
                        toX = x

                        if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True)):
                            moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                        if y > 4:
                            toY = y
                            toX = x + 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

                            toX = x - 1
                            if (GameBoard.check_bounds(toY, toX) and GameBoard.validate_move(board_positions[toY][toX], upper=True)):
                                moves.append(GameBoard.board_pos_name[y][x] + GameBoard.board_pos_name[toY][toX])

        if(K_x != None and k_x != None and K_x == k_x):
            face_to_face = True
            for i in range(K_y + 1, k_y, 1):
                if(board_positions[i][K_x].isalpha()):
                    face_to_face = False

        if(face_to_face == True):
            if(current_player == 'b'):
                moves.append(GameBoard.board_pos_name[k_y][k_x] + GameBoard.board_pos_name[K_y][K_x])
            else:
                moves.append(GameBoard.board_pos_name[K_y][K_x] + GameBoard.board_pos_name[k_y][k_x])

        return moves

def softmax(x):
    # print(x)
    probs = np.exp(x - np.max(x))
    # print(np.sum(probs))
    probs /= np.sum(probs)
    return probs

class cchess_main(object):

    def __init__(self, playout=400, in_batch_size=128, exploration = True, in_search_threads = 16, processor = "cpu", num_gpus = 1, res_block_nums = 7, human_color = 'b'):
        self.epochs = 5
        self.playout_counts = playout    #400    #800    #1600    200
        self.temperature = 1    #1e-8    1e-3
        # self.c = 1e-4
        self.batch_size = in_batch_size    #128    #512
        # self.momentum = 0.9
        self.game_batch = 400    #  Evaluation each 400 times
        # self.game_loop = 25000
        self.top_steps = 30
        self.top_temperature = 1    #2
        # self.Dirichlet = 0.3    # P(s,a) = (1 - ϵ)p_a  + ϵη_a    #self-play chapter in the paper
        self.eta = 0.03
        # self.epsilon = 0.25
        # self.v_resign = 0.05
        # self.c_puct = 5
        self.learning_rate = 0.001    #5e-3    #    0.001
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.buffer_size = 10000
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.game_borad = GameBoard()
        self.processor = processor
        # self.current_player = 'w'    #“w”表示红方，“b”表示黑方。
        self.policy_value_netowrk = policy_value_network(self.lr_callback, res_block_nums) if processor == 'cpu' else policy_value_network_gpus(num_gpus, res_block_nums)
        self.search_threads = in_search_threads
        self.mcts = MCTS_tree(self.game_borad.state, self.policy_value_netowrk.forward, self.search_threads)
        self.exploration = exploration
        self.resign_threshold = -0.8    #0.05
        self.global_step = 0
        self.kl_targ = 0.025
        self.log_file = open(os.path.join(os.getcwd(), 'log_file.txt'), 'w')
        self.human_color = human_color

    @staticmethod
    def flip_policy(prob):
        prob = tf.squeeze(prob) # .flatten()
        return np.asarray([prob[ind] for ind in unflipped_index])

    def lr_callback(self):
        return self.learning_rate * self.lr_multiplier

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        #print("training data_buffer len : ", len(self.data_buffer))
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        winner_batch = np.expand_dims(winner_batch, 1)

        start_time = time.time()
        old_probs, old_v = self.mcts.forward(state_batch)
        for i in range(self.epochs):
            # print("tf.executing_eagerly() : ", tf.executing_eagerly())
            state_batch = np.array(state_batch)
            if len(state_batch.shape) == 3:
                sp = state_batch.shape
                state_batch = np.reshape(state_batch, [1, sp[0], sp[1], sp[2]])
            if self.processor == 'cpu':
                accuracy, loss, self.global_step = self.policy_value_netowrk.train_step(state_batch, mcts_probs_batch, winner_batch,
                                                                 self.learning_rate * self.lr_multiplier)    #
            else:
                # import pickle
                # pickle.dump((state_batch, mcts_probs_batch, winner_batch, self.learning_rate * self.lr_multiplier), open('preprocess.p', 'wb'))
                with self.policy_value_netowrk.strategy.scope():
                    train_dataset = tf.data.Dataset.from_tensor_slices((state_batch, mcts_probs_batch, winner_batch)).batch(len(winner_batch))  # , self.learning_rate * self.lr_multiplier
                        # .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
                    train_iterator = self.policy_value_netowrk.strategy.make_dataset_iterator(train_dataset)
                    train_iterator.initialize()
                    accuracy, loss, self.global_step = self.policy_value_netowrk.distributed_train(train_iterator)

            new_probs, new_v = self.mcts.forward(state_batch)
            kl_tmp = old_probs * (np.log((old_probs + 1e-10) / (new_probs + 1e-10)))

            kl_lst = []
            for line in kl_tmp:
                # print("line.shape", line.shape)
                all_value = [x for x in line if str(x) != 'nan' and str(x)!= 'inf']#除去inf值
                kl_lst.append(np.sum(all_value))
            kl = np.mean(kl_lst)
            # kl = scipy.stats.entropy(old_probs, new_probs)
            # kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))

            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        self.policy_value_netowrk.save(self.global_step)
        print("train using time {} s".format(time.time() - start_time))

        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(np.array(winner_batch) - tf.squeeze(old_v)) / np.var(np.array(winner_batch)) # .flatten()
        explained_var_new = 1 - np.var(np.array(winner_batch) - tf.squeeze(new_v)) / np.var(np.array(winner_batch)) # .flatten()
        print(
            "kl:{:.5f},lr_multiplier:{:.3f},loss:{},accuracy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.lr_multiplier, loss, accuracy, explained_var_old, explained_var_new))
        self.log_file.write("kl:{:.5f},lr_multiplier:{:.3f},loss:{},accuracy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
                kl, self.lr_multiplier, loss, accuracy, explained_var_old, explained_var_new) + '\n')
        self.log_file.flush()
        # return loss, accuracy

    # def policy_evaluate(self, n_games=10):
    #     """
    #     Evaluate the trained policy by playing games against the pure MCTS player
    #     Note: this is only for monitoring the progress of training
    #     """
    #     # current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct,
    #     #                                  n_playout=self.n_playout)
    #     # pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
    #     win_cnt = defaultdict(int)
    #     for i in range(n_games):
    #         winner = self.game.start_play(start_player=i % 2)    #current_mcts_player, pure_mcts_player,
    #         win_cnt[winner] += 1
    #     win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
    #     print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(self.pure_mcts_playout_num, win_cnt[1], win_cnt[2],
    #                                                               win_cnt[-1]))
    #     return win_ratio

    def run(self):
        #self.game_loop
        batch_iter = 0
        try:
            while(True):
                batch_iter += 1
                play_data, episode_len = self.selfplay()
                print("batch i:{}, episode_len:{}".format(batch_iter, episode_len))
                extend_data = []
                # states_data = []
                for state, mcts_prob, winner in play_data:
                    states_data = self.mcts.state_to_positions(state)
                    # prob = np.zeros(labels_len)
                    # for idx in range(len(mcts_prob[0][0])):
                    #     prob[label2i[mcts_prob[0][0][idx]]] = mcts_prob[0][1][idx]
                    extend_data.append((states_data, mcts_prob, winner))
                self.data_buffer.extend(extend_data)
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                # if (batch_iter) % self.game_batch == 0:
                #     print("current self-play batch: {}".format(batch_iter))
                #     win_ratio = self.policy_evaluate()
        except KeyboardInterrupt:
            self.log_file.close()
            self.policy_value_netowrk.save(self.global_step)

    # def get_action(self, state, temperature = 1e-3):
    #     # for i in range(self.playout_counts):
    #     #     state_sim = copy.deepcopy(state)
    #     #     self.mcts.do_simulation(state_sim, self.game_borad.current_player, self.game_borad.restrict_round)
    #
    #     futures = []
    #     with ThreadPoolExecutor(max_workers=self.search_threads) as executor:
    #         for _ in range(self.playout_counts):
    #             state_sim = copy.deepcopy(state)
    #             futures.append(executor.submit(self.mcts.do_simulation, state_sim, self.game_borad.current_player, self.game_borad.restrict_round))
    #
    #     vals = [f.result() for f in futures]
    #
    #     actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
    #     actions, visits = zip(*actions_visits)
    #     probs = softmax(1.0 / temperature * np.log(visits))    #+ 1e-10
    #     move_probs = []
    #     move_probs.append([actions, probs])
    #
    #     if(self.exploration):
    #         act = np.random.choice(actions, p=0.75 * probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
    #     else:
    #         act = np.random.choice(actions, p=probs)
    #
    #     self.mcts.update_tree(act)
    #
    #     return act, move_probs

    def get_hint(self, mcts_or_net, reverse, disp_mcts_msg_handler):

        if mcts_or_net == "mcts":
            if self.mcts.root.child == {}:
                disp_mcts_msg_handler()
                self.mcts.main(self.game_borad.state, self.game_borad.current_player, self.game_borad.restrict_round,
                               self.playout_counts)

            actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
            actions, visits = zip(*actions_visits)
            # print("visits : ", visits)
            # print("np.log(visits) : ", np.log(visits))
            probs = softmax(1.0 / self.temperature * np.log(visits))  # + 1e-10

            act_prob_dict = defaultdict(float)
            for i in range(len(actions)):
                if self.human_color == 'w':
                    action = "".join(flipped_uci_labels(actions[i]))
                else:
                    action = actions[i]
                act_prob_dict[action] = probs[i]

        elif mcts_or_net == "net":
            positions = self.mcts.generate_inputs(self.game_borad.state, self.game_borad.current_player)
            positions = np.expand_dims(positions, 0)
            action_probs, value = self.mcts.forward(positions)

            if self.mcts.is_black_turn(self.game_borad.current_player):
                action_probs = cchess_main.flip_policy(action_probs)
            moves = GameBoard.get_legal_moves(self.game_borad.state, self.game_borad.current_player)

            tot_p = 1e-8
            action_probs = tf.squeeze(action_probs)  # .flatten()  # .squeeze()
            act_prob_dict = defaultdict(float)
            # print("expand action_probs shape : ", action_probs.shape)
            for action in moves:
                # in_state = GameBoard.sim_do_action(action, self.state)
                mov_p = action_probs[label2i[action]]
                if self.human_color == 'w':
                    action = "".join(flipped_uci_labels(action))
                act_prob_dict[action] = mov_p
                # new_node = leaf_node(self, mov_p, in_state)
                # self.child[action] = new_node
                tot_p += mov_p

            for a, _ in act_prob_dict.items():
                act_prob_dict[a] /= tot_p

        sorted_move_probs = sorted(act_prob_dict.items(), key=lambda item: item[1], reverse=reverse)
        # print(sorted_move_probs)

        return sorted_move_probs

    #@profile
    def get_action(self, state, temperature = 1e-3):
        # for i in range(self.playout_counts):
        #     state_sim = copy.deepcopy(state)
        #     self.mcts.do_simulation(state_sim, self.game_borad.current_player, self.game_borad.restrict_round)

        self.mcts.main(state, self.game_borad.current_player, self.game_borad.restrict_round, self.playout_counts)

        actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
        actions, visits = zip(*actions_visits)
        probs = softmax(1.0 / temperature * np.log(visits))    #+ 1e-10
        move_probs = []
        move_probs.append([actions, probs])

        if(self.exploration):
            act = np.random.choice(actions, p=0.75 * probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
        else:
            act = np.random.choice(actions, p=probs)

        win_rate = self.mcts.Q(act) # / 2.0 + 0.5
        self.mcts.update_tree(act)

        # if position.n < 30:    # self.top_steps
        #     move = select_weighted_random(position, on_board_move_prob)
        # else:
        #     move = select_most_likely(position, on_board_move_prob)

        return act, move_probs, win_rate

    def get_action_old(self, state, temperature = 1e-3):
        for i in range(self.playout_counts):
            state_sim = copy.deepcopy(state)
            self.mcts.do_simulation(state_sim, self.game_borad.current_player, self.game_borad.restrict_round)

        actions_visits = [(act, nod.N) for act, nod in self.mcts.root.child.items()]
        actions, visits = zip(*actions_visits)
        probs = softmax(1.0 / temperature * np.log(visits))    #+ 1e-10
        move_probs = []
        move_probs.append([actions, probs])

        if(self.exploration):
            act = np.random.choice(actions, p=0.75 * probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
        else:
            act = np.random.choice(actions, p=probs)

        self.mcts.update_tree(act)

        return act, move_probs

    def check_end(self):
        if (self.game_borad.state.find('K') == -1 or self.game_borad.state.find('k') == -1):
            if (self.game_borad.state.find('K') == -1):
                print("Green is Winner")
                return True, "b"
            if (self.game_borad.state.find('k') == -1):
                print("Red is Winner")
                return True, "w"
        elif self.game_borad.restrict_round >= 60:
            print("TIE! No Winners!")
            return True, "t"
        else:
            return False, ""

    def human_move(self, coord, mcts_or_net):
        win_rate = 0
        x_trans = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i'}

        src = coord[0:2]
        dst = coord[2:4]

        src_x = (x_trans[src[0]])
        src_y = str(src[1])

        dst_x = (x_trans[dst[0]])
        dst_y = str(dst[1])

        action = src_x + src_y + dst_x + dst_y

        if self.human_color == 'w':
            action = "".join(flipped_uci_labels(action))

        if mcts_or_net == "mcts":
            if self.mcts.root.child == {}:
                # self.get_action(self.game_borad.state, self.temperature)
                self.mcts.main(self.game_borad.state, self.game_borad.current_player, self.game_borad.restrict_round,
                               self.playout_counts)
            win_rate = self.mcts.Q(action) # / 2.0 + 0.5
            self.mcts.update_tree(action)

        last_state = self.game_borad.state
        # print(self.game_borad.current_player, " now take a action : ", action, "[Step {}]".format(self.game_borad.round))
        self.game_borad.state = GameBoard.sim_do_action(action, self.game_borad.state)
        self.game_borad.round += 1
        self.game_borad.current_player = "w" if self.game_borad.current_player == "b" else "b"
        if is_kill_move(last_state, self.game_borad.state) == 0:
            self.game_borad.restrict_round += 1
        else:
            self.game_borad.restrict_round = 0

        return win_rate


    def select_move(self, mcts_or_net):
        if mcts_or_net == "mcts":
            action, probs, win_rate = self.get_action(self.game_borad.state, self.temperature)
            # win_rate = self.mcts.Q(action) / 2.0 + 0.5
        elif mcts_or_net == "net":
            positions = self.mcts.generate_inputs(self.game_borad.state, self.game_borad.current_player)
            positions = np.expand_dims(positions, 0)
            action_probs, value = self.mcts.forward(positions)
            win_rate = value[0, 0] # / 2 + 0.5
            if self.mcts.is_black_turn(self.game_borad.current_player):
                action_probs = cchess_main.flip_policy(action_probs)
            moves = GameBoard.get_legal_moves(self.game_borad.state, self.game_borad.current_player)

            tot_p = 1e-8
            action_probs = tf.squeeze(action_probs)  # .flatten()  # .squeeze()
            act_prob_dict = defaultdict(float)
            # print("expand action_probs shape : ", action_probs.shape)
            for action in moves:
                # in_state = GameBoard.sim_do_action(action, self.state)
                mov_p = action_probs[label2i[action]]
                act_prob_dict[action] = mov_p
                # new_node = leaf_node(self, mov_p, in_state)
                # self.child[action] = new_node
                tot_p += mov_p

            for a, _ in act_prob_dict.items():
                act_prob_dict[a] /= tot_p

            action = max(act_prob_dict.items(), key=lambda node: node[1])[0]
            # self.mcts.update_tree(action)

        print('Win rate for player {} is {:.4f}'.format(self.game_borad.current_player, win_rate))
        last_state = self.game_borad.state
        print(self.game_borad.current_player, " now take a action : ", action, "[Step {}]".format(self.game_borad.round))    # if self.human_color == 'w' else "".join(flipped_uci_labels(action))
        self.game_borad.state = GameBoard.sim_do_action(action, self.game_borad.state)
        self.game_borad.round += 1
        self.game_borad.current_player = "w" if self.game_borad.current_player == "b" else "b"
        if is_kill_move(last_state, self.game_borad.state) == 0:
            self.game_borad.restrict_round += 1
        else:
            self.game_borad.restrict_round = 0

        self.game_borad.print_borad(self.game_borad.state)

        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        if self.human_color == 'w':
            action = "".join(flipped_uci_labels(action))

        src = action[0:2]
        dst = action[2:4]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])

        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        return (src_x, src_y, dst_x - src_x, dst_y - src_y), win_rate

    def selfplay(self):
        self.game_borad.reload()
        # p1, p2 = self.game_borad.players
        states, mcts_probs, current_players = [], [], []
        z = None
        game_over = False
        winnner = ""
        start_time = time.time()
        # self.game_borad.print_borad(self.game_borad.state)
        while(not game_over):
            action, probs, win_rate = self.get_action(self.game_borad.state, self.temperature)
            state, palyer = self.mcts.try_flip(self.game_borad.state, self.game_borad.current_player, self.mcts.is_black_turn(self.game_borad.current_player))
            states.append(state)
            prob = np.zeros(labels_len)
            if self.mcts.is_black_turn(self.game_borad.current_player):
                for idx in range(len(probs[0][0])):
                    # probs[0][0][idx] = "".join((str(9 - int(a)) if a.isdigit() else a) for a in probs[0][0][idx])
                    act = "".join((str(9 - int(a)) if a.isdigit() else a) for a in probs[0][0][idx])
                    # for idx in range(len(mcts_prob[0][0])):
                    prob[label2i[act]] = probs[0][1][idx]
            else:
                for idx in range(len(probs[0][0])):
                    prob[label2i[probs[0][0][idx]]] = probs[0][1][idx]
            mcts_probs.append(prob)
            # mcts_probs.append(probs)
            current_players.append(self.game_borad.current_player)

            last_state = self.game_borad.state
            # print(self.game_borad.current_player, " now take a action : ", action, "[Step {}]".format(self.game_borad.round))
            self.game_borad.state = GameBoard.sim_do_action(action, self.game_borad.state)
            self.game_borad.round += 1
            self.game_borad.current_player = "w" if self.game_borad.current_player == "b" else "b"
            if is_kill_move(last_state, self.game_borad.state) == 0:
                self.game_borad.restrict_round += 1
            else:
                self.game_borad.restrict_round = 0

            # self.game_borad.print_borad(self.game_borad.state, action)

            if (self.game_borad.state.find('K') == -1 or self.game_borad.state.find('k') == -1):
                z = np.zeros(len(current_players))
                if (self.game_borad.state.find('K') == -1):
                    winnner = "b"
                if (self.game_borad.state.find('k') == -1):
                    winnner = "w"
                z[np.array(current_players) == winnner] = 1.0
                z[np.array(current_players) != winnner] = -1.0
                game_over = True
                print("Game end. Winner is player : ", winnner, " In {} steps".format(self.game_borad.round - 1))
            elif self.game_borad.restrict_round >= 60:
                z = np.zeros(len(current_players))
                game_over = True
                print("Game end. Tie in {} steps".format(self.game_borad.round - 1))
            # elif(self.mcts.root.v < self.resign_threshold):
            #     pass
            # elif(self.mcts.root.Q < self.resign_threshold):
            #    pass
            if(game_over):
                # self.mcts.root = leaf_node(None, self.mcts.p_, "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")#"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
                self.mcts.reload()
        print("Using time {} s".format(time.time() - start_time))
        return zip(states, mcts_probs, z), len(z)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'play'], type=str, help='train or play')
    parser.add_argument('--ai_count', default=1, choices=[1, 2], type=int, help='choose ai player count')
    parser.add_argument('--ai_function', default='mcts', choices=['mcts', 'net'], type=str, help='mcts or net')
    parser.add_argument('--train_playout', default=400, type=int, help='mcts train playout')
    parser.add_argument('--batch_size', default=512, type=int, help='train batch_size')
    parser.add_argument('--play_playout', default=400, type=int, help='mcts play playout')
    parser.add_argument('--delay', dest='delay', action='store',
                        nargs='?', default=3, type=float, required=False,
                        help='Set how many seconds you want to delay after each move')
    parser.add_argument('--end_delay', dest='end_delay', action='store',
                        nargs='?', default=3, type=float, required=False,
                        help='Set how many seconds you want to delay after the end of game')
    parser.add_argument('--search_threads', default=16, type=int, help='search_threads')
    parser.add_argument('--processor', default='cpu', choices=['cpu', 'gpu'], type=str, help='cpu or gpu')
    parser.add_argument('--num_gpus', default=1, type=int, help='gpu counts')
    parser.add_argument('--res_block_nums', default=7, type=int, help='res_block_nums')
    parser.add_argument('--human_color', default='b', choices=['w', 'b'], type=str, help='w or b')
    args = parser.parse_args()

    if args.mode == 'train':
        train_main = cchess_main(args.train_playout, args.batch_size, True, args.search_threads, args.processor, args.num_gpus, args.res_block_nums, args.human_color)    # * args.num_gpus
        train_main.run()
    elif args.mode == 'play':
        from ChessGame_tf2 import *
        game = ChessGame(args.ai_count, args.ai_function, args.play_playout, args.delay, args.end_delay, args.batch_size,
                         args.search_threads, args.processor, args.num_gpus, args.res_block_nums, args.human_color)    # * args.num_gpus
        game.start()
