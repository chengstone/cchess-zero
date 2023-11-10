import tkinter
import time

def board_coord(x):
    return 30 + 40*x

class ChessView:
    root = tkinter.Tk()
    root.title("Chinese Chess")
    root.resizable(0, 0)
    can = tkinter.Canvas(root, width=373, height=410)
    can.pack(expand=tkinter.YES, fill=tkinter.BOTH)
    img = tkinter.PhotoImage(file="images/WHITE.gif")
    can.create_image(0, 0, image=img, anchor=tkinter.NW)
    piece_images = dict()
    move_images = []
    def draw_board(self, board):
        self.piece_images.clear()
        self.move_images = []
        pieces = board.pieces
        for (x, y) in pieces.keys():
            self.piece_images[x, y] = tkinter.PhotoImage(file=pieces[x, y].get_image_file_name())
            self.can.create_image(board_coord(x), board_coord(y), image=self.piece_images[x, y])
        if board.selected_piece:
            for (x, y) in board.selected_piece.get_move_locs(board):
                self.move_images.append(tkinter.PhotoImage(file="images/OOS.gif"))
                self.can.create_image(board_coord(x), board_coord(y), image=self.move_images[-1])
                # self.can.create_text(board_coord(x), board_coord(y),text="Hello")

        # label = tkinter.Label(self.root, text='Hello　world!')
        # label.place(x=30,y=30)
        # label.pack(fill='x', expand=1)

    def disp_hint_on_board(self, action, percentage):
        board = self.board
        for key in board.pieces.keys():
            board.pieces[key].selected = False
        board.selected_piece = None

        self.can.create_image(0, 0, image=self.img, anchor=tkinter.NW)
        self.draw_board(board)
        # self.can.create_text(board_coord(self.last_text_x), board_coord(self.last_text_y), text="")
        x_trans = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8}

        src = action[0:2]
        dst = action[2:4]

        src_x = int(x_trans[src[0]])
        src_y = int(src[1])

        dst_x = int(x_trans[dst[0]])
        dst_y = int(dst[1])

        pieces = board.pieces
        if (src_x, src_y) in pieces.keys():
            self.piece_images[src_x, src_y] = tkinter.PhotoImage(file=pieces[src_x, src_y].get_selected_image())
            self.can.create_image(board_coord(src_x), board_coord(src_y), image=self.piece_images[src_x, src_y])

        if (dst_x, dst_y) in pieces.keys():
            self.piece_images[dst_x, dst_y] = tkinter.PhotoImage(file=pieces[dst_x, dst_y].get_selected_image())
            self.can.create_image(board_coord(dst_x), board_coord(dst_y), image=self.piece_images[dst_x, dst_y])
            self.can.create_text(board_coord(dst_x), board_coord(dst_y), text="{:.3f}".format(float(percentage)))
            self.last_text_x = dst_x
            self.last_text_y = dst_y
        else:
            self.move_images.append(tkinter.PhotoImage(file="images/OOS.gif"))
            self.can.create_image(board_coord(dst_x), board_coord(dst_y), image=self.move_images[-1])
            self.can.create_text(board_coord(dst_x), board_coord(dst_y),text="{:.3f}".format(float(percentage)))
            self.last_text_x = dst_x
            self.last_text_y = dst_y
            self.print_text_flag = True
        # return (src_x, src_y, dst_x - src_x, dst_y - src_y), win_rate

    def print_all_hint(self, sorted_move_probs):

        # for i in range(len(sorted_move_probs)):
        #     self.lb.insert(END, str(i * 100))

        self.lb.delete(0, "end")
        for item in sorted_move_probs:
            # print(item[0], item[1])
            self.lb.insert("end", item)
        self.lb.pack()

    def showMsg(self, msg):
        print(msg)
        self.root.title(msg)

    def printList(self, event):
        # print(self.lb.curselection())
        # print(self.lb.get(self.lb.curselection()))
        # for i in range(self.lb.size()):
        #     print(i, self.lb.selection_includes(i))
        w = event.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        print(value)
        self.disp_hint_on_board(value[0], value[1])


    def __init__(self, control, board):
        self.control = control
        if self.control.game_mode != 2:
            self.can.bind('<Button-1>', self.control.callback)

        self.lb = tkinter.Listbox(ChessView.root,selectmode="browse")
        self.scr1 = tkinter.Scrollbar(ChessView.root)
        self.lb.configure(yscrollcommand=self.scr1.set)
        self.scr1['command'] = self.lb.yview
        self.scr1.pack(side='right',fill="y")
        self.lb.pack(fill="x")

        self.lb.bind('<<ListboxSelect>>', self.printList)  # Double-    <Button-1>
        self.board = board
        self.last_text_x = 0
        self.last_text_y = 0
        self.print_text_flag = False

    # def start(self):
    #     tkinter.mainloop()
    def start(self):
        if self.control.game_mode == 2:
            self.root.update()
            time.sleep(self.control.delay)
            while True:
                game_end = self.control.game_mode_2()
                self.root.update()
                time.sleep(self.control.delay)
                if game_end:
                    time.sleep(self.control.end_delay)
                    self.quit()
                    return
        else:
            tkinter.mainloop()
            # self.root.mainloop()

    # below added by Fei Li

    def quit(self):
        self.root.quit()
