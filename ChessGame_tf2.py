from ChessBoard import *
from ChessView import ChessView
from main_tf2 import *
import tkinter

def real_coord(x):
    if x <= 50:
        return 0
    else:
        return (x-50)//40 + 1


def board_coord(x):
    return 30 + 40*x


class ChessGame:

    board = None    #ChessBoard()
    cur_round = 1
    game_mode = 1  # 0:HUMAN VS HUMAN 1:HUMAN VS AI 2:AI VS AI
    time_red = []
    time_green = []

    def __init__(self, in_ai_count, in_ai_function, in_play_playout, in_delay, in_end_delay, batch_size, search_threads,
                 processor, num_gpus, res_block_nums, human_color = "b"):
        self.human_color = human_color
        self.current_player = "w"
        self.players = {}
        self.players[self.human_color] = "human"
        ai_color = "w" if self.human_color == "b" else "b"
        self.players[ai_color] = "AI"

        ChessGame.board = ChessBoard(self.human_color == 'b')
        self.view = ChessView(self, board=ChessGame.board)
        self.view.showMsg("Loading Models...")    #"Red"    player_color
        self.view.draw_board(self.board)
        ChessGame.game_mode = in_ai_count
        self.ai_function = in_ai_function
        self.play_playout = in_play_playout
        self.delay = in_delay
        self.end_delay = in_end_delay

        self.win_rate = {}
        self.win_rate['w'] = 0.0
        self.win_rate['b'] = 0.0

        self.view.root.update()
        self.cchess_engine = cchess_main(playout=self.play_playout, in_batch_size=batch_size, exploration=False, in_search_threads=search_threads,
                                         processor=processor, num_gpus=num_gpus, res_block_nums=res_block_nums, human_color=human_color)

    def player_is_red(self):
        return self.current_player == "w"

    def start(self):
        # below added by Fei Li
        self.view.showMsg("Red")
        if self.game_mode == 1:
            print ('-----Round %d-----' % self.cur_round)
            if self.players["w"] == "AI":
                self.win_rate['w'] = self.perform_AI()
                self.view.draw_board(self.board)
                self.change_player()
        elif self.game_mode == 2:
            print('-----Round %d-----' % self.cur_round)
            self.win_rate['w'] = self.perform_AI()
            self.view.draw_board(self.board)

        self.view.start()

    def disp_mcts_msg(self):
        self.view.showMsg("MCTS Searching...")

    def callback(self, event):
        if self.game_mode == 1 and self.players[self.current_player] == "AI":
            return
        if self.game_mode == 2:
            return
        rx, ry = real_coord(event.x), real_coord(event.y)
        # print(rx, ry)
        change, coord = self.board.select(rx, ry, self.player_is_red())
        if self.view.print_text_flag == True:
            self.view.print_text_flag = False
            self.view.can.create_image(0, 0, image=self.view.img, anchor=tkinter.NW)
        self.view.draw_board(self.board)
        if self.check_end():
            self.view.root.update()
            self.quit()
            return
        if change:
            if self.cur_round == 1 and self.human_color == 'w':
                self.view.showMsg("MCTS Searching...")

            self.win_rate[self.current_player] = self.cchess_engine.human_move(coord, self.ai_function)
            if self.check_end():
                self.view.root.update()
                self.quit()
                return
            performed = self.change_player()
            if performed:
                self.view.draw_board(self.board)
                if self.check_end():
                    self.view.root.update()
                    self.quit()
                    return
                self.change_player()


    # below added by Fei Li

    def quit(self):
        time.sleep(self.end_delay)
        self.view.quit()

    def check_end(self):
        ret, winner = self.cchess_engine.check_end()
        if ret == True:
            if winner == "b":
                self.view.showMsg('*****Green Wins at Round %d*****' % self.cur_round)
                self.view.root.update()
            elif winner == "w":
                self.view.showMsg('*****Red Wins at Round %d*****' % self.cur_round)
                self.view.root.update()
            elif winner == "t":
                self.view.showMsg('*****Draw at Round %d*****' % self.cur_round)
                self.view.root.update()
        return ret

    def _check_end(self, board):
        red_king = False
        green_king = False
        pieces = board.pieces
        for (x, y) in pieces.keys():
            if pieces[x, y].is_king:
                if pieces[x, y].is_red:
                    red_king = True
                else:
                    green_king = True
        if not red_king:
            self.view.showMsg('*****Green Wins at Round %d*****' % self.cur_round)
            self.view.root.update()
            return True
        elif not green_king:
            self.view.showMsg('*****Red Wins at Round %d*****' % self.cur_round)
            self.view.root.update()
            return True
        elif self.cur_round >= 200:
            self.view.showMsg('*****Draw at Round %d*****' % self.cur_round)
            self.view.root.update()
            return True
        return False

    def change_player(self):
        self.current_player = "w" if self.current_player == "b" else "b"
        if self.current_player == "w":
            self.cur_round += 1
            print ('-----Round %d-----' % self.cur_round)
        red_msg = " ({:.4f})".format(self.win_rate['w'])
        green_msg = " ({:.4f})".format(self.win_rate['b'])
        sorted_move_probs = self.cchess_engine.get_hint(self.ai_function, True, self.disp_mcts_msg)
        # print(sorted_move_probs)
        self.view.print_all_hint(sorted_move_probs)
        # self.move_images.append(tkinter.PhotoImage(file="images/OOS.gif"))
        # self.can.create_image(board_coord(x), board_coord(y), image=self.move_images[-1])

        self.view.showMsg("Red" + red_msg + " Green" + green_msg if self.current_player == "w" else "Green" + green_msg + " Red" + red_msg)
        self.view.root.update()
        # if self.game_mode == 0:
        #     return False
        if self.game_mode == 1:
            if self.players[self.current_player] == "AI":
                self.win_rate[self.current_player] = self.perform_AI()
                return True
            return False
        elif self.game_mode == 2:
            # if self.current_player == "w":
            #     self.human_win_rate = self.perform_AI()
            # else:
            self.win_rate[self.current_player] = self.perform_AI()
            return True
        return False

    def perform_AI(self):
        print ('...AI is calculating...')
        START_TIME = time.clock()
        move, win_rate = self.cchess_engine.select_move(self.ai_function)
        time_used = time.clock() - START_TIME
        print ('...Use %fs...' % time_used)
        if self.current_player == "w":
            self.time_red.append(time_used)
        else:
            self.time_green.append(time_used)
        if move is not None:
            self.board.move(move[0], move[1], move[2], move[3])
        return win_rate

    # AI VS AI mode
    def game_mode_2(self):
        self.change_player()
        self.view.draw_board(self.board)
        self.view.root.update()
        if self.check_end():
            return True
        return False

# game = ChessGame()
# game.start()
