__author__ = 'Zhaoliang'
from ChessPiece import ChessPiece


class Shi(ChessPiece):

    def get_image_file_name(self):
        if self.selected:
            if self.is_red:
                return "images/RAS.gif"
            else:
                return "images/BAS.gif"
        else:
            if self.is_red:
                return "images/RA.gif"
            else:
                return "images/BA.gif"

    def get_selected_image(self):
        if self.is_red:
            return "images/RAS.gif"
        else:
            return "images/BAS.gif"

    def can_move(self, board, dx, dy):
        nx, ny = self.x + dx, self.y + dy
        if nx < 0 or nx > 8 or ny < 0 or ny > 9:
            return False
        if (nx, ny) in board.pieces:
            if board.pieces[nx, ny].is_red == self.is_red:
                #print 'blocked by yourself'
                return False
        x, y = self.x, self.y
        if not (self.is_north() and 3 <= nx <=5 and 0<= ny <=2) and\
                not (self.is_south() and 3 <= nx <= 5 and 7 <= ny <= 9):
            #print 'out of castle'
            return False
        if self.is_north() and (nx, ny) == (4, 1) or (x,y) == (4,1):
            if abs(dx)>1 or abs(dy)>1:
                #print 'too far'
                return False
        if self.is_south() and (nx, ny) == (4, 8) or (x,y) == (4,8):
            if abs(dx)>1 or abs(dy)>1:
                #print 'too far'
                return False
        #below modified by Fei Li
        if abs(dx) != 1 or abs(dy) != 1:
            #print 'no diag'
            return False
        return True

    def __init__(self, x, y, is_red, direction):
        ChessPiece.__init__(self, x, y, is_red, direction)

