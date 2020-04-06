"""
The game of dots and boxes
"""
import numpy as np


class DotsAndBoxesBoard:

    def __init__(self, size: int):
        self.size = size
        # there should be (size+1)*size horizontal lines, and size*(size+1) vertical lines
        # initialize the board lines with a (size+1)*(size+1)*2 matrix
        # (x,y,h), h == 0 <=> horizontal lines
        self.lines = np.zeros((size + 1, size + 1, 2), dtype=np.int)
        # 0: not be conquered; 1: player1 conquered; 2: player2 conquered
        self.boxes = np.zeros((size, size), dtype=np.int)
        self.scores = [0, 0, 0]
        self.step = 0
        self.end_step = (size + 1) * size * 2
        self.available_lines = set([(x, y, 0) for x in range(size + 1) for y in range(size)]
                                   + [(x, y, 1) for x in range(size) for y in range(size + 1)])

    def _box_is_conquered(self, x, y):
        """
                    (x,y,0)
            (x,y,1) (x,y) (x,y+1,1)
                    (x+1,y,0)
        """
        return self.lines[x][y][0] == 1 and self.lines[x][y][1] == 1 \
               and self.lines[x + 1][y][0] == 1 and self.lines[x][y + 1][1] == 1

    def is_conquered(self, x, y):
        return self.boxes[x][y] > 0

    def set_line(self, line, player_id):
        """
        a conquerable line (x, y, h)
        player_id can merely be 1 or 2
        if return is True, the player should not be switched
        """
        x, y, h = line
        self.available_lines.remove((x, y, h))
        self.step += 1
        self.lines[x][y][h] = 1
        count = 0
        if x < self.size and y < self.size and self._box_is_conquered(x, y):
            self.boxes[x][y] = player_id
            count += 1
        if h == 0 and x > 0:
            # for a horizontal line, [x-1][y] may also be conquered
            if self._box_is_conquered(x - 1, y):
                self.boxes[x - 1][y] = player_id
                count += 1
        elif y > 0:
            # for a vertical line, [x][y-1] may also be conquered
            if self._box_is_conquered(x, y - 1):
                self.boxes[x][y - 1] = player_id
                count += 1
        if count > 0:
            self.scores[player_id] += count
        return count > 0

    def cur_brief_state(self):
        return self.step == self.end_step, self.scores[1], self.scores[2]

    def is_end(self):
        return self.step == self.end_step

    def state_str(self):
        def horizontal_str(x):
            h_res = '+'
            for y in range(self.size):
                if self.lines[x][y][0] > 0:
                    h_res += '---+'
                else:
                    h_res += '   +'
            return h_res

        def vertical_str(x):
            v_res = ''
            if self.lines[x][0][1] > 0:
                v_res += '|'
            else:
                v_res += ' '
            for y in range(self.size):
                if self.boxes[x][y] > 0:
                    v_res += ' %d ' % self.boxes[x][y]
                else:
                    v_res += '   '
                if self.lines[x][y + 1][1] > 0:
                    v_res += '|'
                else:
                    v_res += ' '
            return v_res

        res = ''
        for x in range(self.size):
            res += horizontal_str(x) + '\n'
            res += vertical_str(x) + '\n'
        res += horizontal_str(self.size)
        return res


class DotsAndBoxesPlayerBase:

    def __init__(self):
        self._board: DotsAndBoxesBoard = None

    def set_game_board(self, board: DotsAndBoxesBoard):
        self._board = board

    def get_line(self):
        raise NotImplementedError()


class DotsAndBoxes:

    def __init__(self, size: int, player1: DotsAndBoxesPlayerBase, player2: DotsAndBoxesPlayerBase):
        """
        Initialize the game and game board, the game board has size x size boxes
        """
        self.size = size
        self.board = DotsAndBoxesBoard(self.size)
        self.cur_player_id = 1
        player1.set_game_board(self.board)
        player2.set_game_board(self.board)
        self.players = [None, player1, player2]

    def switch_player(self):
        # only 1 or 2
        self.cur_player_id = 3 - self.cur_player_id

    def get_winner(self):
        if self.board.is_end():
            if self.board.scores[1] > self.board.scores[2]:
                return 1
            elif self.board.scores[1] < self.board.scores[2]:
                return 2
            else:
                return 0
        else:
            return -1

    def play(self, verbose=0):
        while self.get_winner() < 0:
            not_switch_player = self.board.set_line(self.players[self.cur_player_id].get_line(), self.cur_player_id)
            if verbose > 0:
                print('step: %d' % self.board.step)
                print(self.board.state_str())
            if not_switch_player:
                continue
            self.switch_player()
        return self.get_winner()
