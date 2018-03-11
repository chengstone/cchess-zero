#coding:utf-8
from chessman.Bing import *
from chessman.Shuai import *
from chessman.Pao import *
from chessman.Shi import *
from chessman.Xiang import *
from chessman.Ma import *
from chessman.Che import *


class ChessBoard:
    pieces = dict()

    selected_piece = None

    def __init__(self, north_is_red = True):
        # self.north_is_red = north_is_red
        # north area
        ChessBoard.pieces[4, 0] = Shuai(4, 0, north_is_red, "north")

        ChessBoard.pieces[0, 3] = Bing(0, 3, north_is_red, "north")
        ChessBoard.pieces[2, 3] = Bing(2, 3, north_is_red, "north")
        ChessBoard.pieces[4, 3] = Bing(4, 3, north_is_red, "north")
        ChessBoard.pieces[6, 3] = Bing(6, 3, north_is_red, "north")
        ChessBoard.pieces[8, 3] = Bing(8, 3, north_is_red, "north")

        ChessBoard.pieces[1, 2] = Pao(1, 2, north_is_red, "north")
        ChessBoard.pieces[7, 2] = Pao(7, 2, north_is_red, "north")

        ChessBoard.pieces[3, 0] = Shi(3, 0, north_is_red, "north")
        ChessBoard.pieces[5, 0] = Shi(5, 0, north_is_red, "north")

        ChessBoard.pieces[2, 0] = Xiang(2, 0, north_is_red, "north")
        ChessBoard.pieces[6, 0] = Xiang(6, 0, north_is_red, "north")

        ChessBoard.pieces[1, 0] = Ma(1, 0, north_is_red, "north")
        ChessBoard.pieces[7, 0] = Ma(7, 0, north_is_red, "north")

        ChessBoard.pieces[0, 0] = Che(0, 0, north_is_red, "north")
        ChessBoard.pieces[8, 0] = Che(8, 0, north_is_red, "north")

        # south area
        ChessBoard.pieces[4, 9] = Shuai(4, 9, not north_is_red, "south")

        ChessBoard.pieces[0, 6] = Bing(0, 6, not north_is_red, "south")
        ChessBoard.pieces[2, 6] = Bing(2, 6, not north_is_red, "south")
        ChessBoard.pieces[4, 6] = Bing(4, 6, not north_is_red, "south")
        ChessBoard.pieces[6, 6] = Bing(6, 6, not north_is_red, "south")
        ChessBoard.pieces[8, 6] = Bing(8, 6, not north_is_red, "south")

        ChessBoard.pieces[1, 7] = Pao(1, 7, not north_is_red, "south")
        ChessBoard.pieces[7, 7] = Pao(7, 7, not north_is_red, "south")

        ChessBoard.pieces[3, 9] = Shi(3, 9, not north_is_red, "south")
        ChessBoard.pieces[5, 9] = Shi(5, 9, not north_is_red, "south")

        ChessBoard.pieces[2, 9] = Xiang(2, 9, not north_is_red, "south")
        ChessBoard.pieces[6, 9] = Xiang(6, 9, not north_is_red, "south")

        ChessBoard.pieces[1, 9] = Ma(1, 9, not north_is_red, "south")
        ChessBoard.pieces[7, 9] = Ma(7, 9, not north_is_red, "south")

        ChessBoard.pieces[0, 9] = Che(0, 9, not north_is_red, "south")
        ChessBoard.pieces[8, 9] = Che(8, 9, not north_is_red, "south")

    def can_move(self, x, y, dx, dy):
        return self.pieces[x, y].can_move(self, dx, dy)

    def move(self, x, y, dx, dy):
        return self.pieces[x, y].move(self, dx, dy)

    def remove(self, x, y):
        del self.pieces[x, y]

    def select(self, x, y, player_is_red):
        # 选中棋子
        if not self.selected_piece:
            if (x, y) in self.pieces and self.pieces[x, y].is_red == player_is_red:
                self.pieces[x, y].selected = True
                self.selected_piece = self.pieces[x, y]
            return False, None

        # 移动棋子
        if not (x, y) in self.pieces:
            if self.selected_piece:
                ox, oy = self.selected_piece.x, self.selected_piece.y
                if self.can_move(ox, oy, x-ox, y-oy):
                    self.move(ox, oy, x-ox, y-oy)
                    self.pieces[x,y].selected = False
                    self.selected_piece = None
                    return True, (ox, oy, x, y)
            return False, None

        # 同一个棋子
        if self.pieces[x, y].selected:
            return False, None

        # 吃子
        if self.pieces[x, y].is_red != player_is_red:
            ox, oy = self.selected_piece.x, self.selected_piece.y
            if self.can_move(ox, oy, x-ox, y-oy):
                self.move(ox, oy, x-ox, y-oy)
                self.pieces[x,y].selected = False
                self.selected_piece = None
                return True, (ox, oy, x, y)
            return False, None

        # 取消选中
        for key in self.pieces.keys():
            self.pieces[key].selected = False
        # 选择棋子
        self.pieces[x, y].selected = True
        self.selected_piece = self.pieces[x,y]
        return False, None