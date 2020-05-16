import copy
import math
import random

import numpy as np

from Game import GameBase
from dots_and_boxes import DotsAndBoxesPlayerBase, DotsAndBoxes


class RandomPlayer(DotsAndBoxesPlayerBase):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, state: GameBase, **kwargs):
        return random.choice(list(state.get_available_actions()))

    def copy(self):
        return RandomPlayer()

    def __str__(self):
        return "Random {}".format(self.player)


class GreedyPlayer(DotsAndBoxesPlayerBase):
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, state: DotsAndBoxes, **kwargs):
        return_prob = kwargs.get('return_prob', 0)
        size = state.size
        board = state.board
        available_lines = copy.deepcopy(state.get_available_actions())
        # if there is a box which has only one line not having been drawn, draw it
        best_choices = []
        for x in range(size):
            for y in range(size):
                if board.boxes[x][y] == 0:
                    # check 4 lines of this box
                    four_lines = ((0, x, y), (1, x, y), (0, x + 1, y), (1, x, y + 1))
                    num_empty = 4 - (board.lines[0][x][y] + board.lines[1][x][y] + board.lines[0][x + 1][y] +
                                     board.lines[1][x][y + 1])
                    if num_empty == 1:
                        # only 1 line empty, occupy it
                        for line in four_lines:
                            if board.lines[line] == 0:
                                line_id = state.board.pos_to_line_id(line)
                                best_choices.append(line_id)
                    elif num_empty == 2:
                        # it will give advantage to opponent
                        for line in four_lines:
                            available_lines.discard(state.board.pos_to_line_id(line))

        if len(best_choices) > 0:
            choices_for_greedy = best_choices
        elif len(available_lines) > 0:
            choices_for_greedy = list(available_lines)
        else:
            choices_for_greedy = list(state.get_available_actions())

        if return_prob == 0:
            return random.choice(choices_for_greedy)
        else:
            probs = np.zeros(state.get_action_space())
            probs[choices_for_greedy] = 1.0 / len(choices_for_greedy)
            return random.choice(choices_for_greedy), probs

    def copy(self):
        return GreedyPlayer()

    def __str__(self):
        return "Greedy {}".format(self.player)


class HumanPlayer(object):
    """
    human player
    """
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, state: DotsAndBoxes, **kwargs):
        try:
            print("move format h, x, y, h=0 is the horizontal line, h=1 is the vertical line")
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            # move = board.location_to_move(location)
            move = state.board.pos_to_line_id(location)
        except Exception as e:
            move = -1
        if move == -1 or move not in state.board.available_lines:
            print("invalid move")
            move = self.get_action(state)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def edges_of_box(state: DotsAndBoxes, x, y):
    board = state.board
    num_empty = 4 - (board.lines[0][x][y] + board.lines[1][x][y] + board.lines[0][x + 1][y] +
                     board.lines[1][x][y + 1])
    return 4 - num_empty


class MoreGreedyPlayer(DotsAndBoxesPlayerBase):
    def __init__(self):
        super(MoreGreedyPlayer, self).__init__()
        self.player = None

        self.size = -1

        self.count = 0  # count of boxes in a chain
        self.loop = False

        self.move_cache = []

    def set_player_ind(self, p):
        self.player = p

    def takeedge(self, state: DotsAndBoxes, zz, x, y):
        line_id = state.board.pos_to_line_id((zz, x, y))
        self.move_cache.insert(0, line_id)
        state.act(line_id)
        return line_id

    def takebox(self, state: DotsAndBoxes, x, y):
        if state.board.lines[0][x][y] < 1:
            # return state.board.pos_to_line_id((0, x, y))
            return self.takeedge(state, 0, x, y)
        elif state.board.lines[1][x][y] < 1:
            # return state.board.pos_to_line_id((1, x, y))
            return self.takeedge(state, 1, x, y)
        elif state.board.lines[0][x+1][y] < 1:
            # return state.board.pos_to_line_id((0, x+1, y))
            return self.takeedge(state, 0, x+1, y)
        elif state.board.lines[1][x][y+1] < 1:
            # return state.board.pos_to_line_id((1, x, y+1))
            return self.takeedge(state, 1, x, y+1)

    def makemove(self, state: DotsAndBoxes):
        self.size = state.size
        self.takesafe3s(state)
        exist3s, u, v = self.sides3(state)  # box(u,v)==3, box with 3 edges
        exist1s, h, x, y = self.sides01(state)  # one safe edge
        exist_single, h_sing, x_sing, y_sing = self.singleton(state)
        exist_double, h_doub, x_doub, y_doub = self.doubleton(state)
        if exist3s:
            if exist1s:
                self.takeall3s(state)
                self.takeedge(state, h, x, y)
            else:
                self.sac(state, u, v)  # take a chain, and sacrifice some boxes (2 boxes?)

            if state.board.scores[1] + state.board.scores[2] == state.size * state.size:
                # print("Game Over.")
                pass
        elif exist1s:
            self.takeedge(state, h, x, y)
        elif exist_single:  # FIXME: stupid assumption: giving away singlton/doubleton so we can have advatage in chains
            self.takeedge(state, h_sing, x_sing, y_sing)
        elif exist_double:  # FIXME: stupid assumption
            self.takeedge(state, h_doub, x_doub, y_doub)
        else:
            self.makeanymove(state)

    def takesafe3s(self, state: DotsAndBoxes):
        """Take all singleton and doubleton 3's"""
        pass
        size = state.size
        board = state.board
        for x in range(size):
            for y in range(size):
                if edges_of_box(state, x, y) == 3:
                    # top edge is empty
                    if board.lines[0][x][y] < 1:
                        if x == 0 or edges_of_box(state, x - 1, y) != 2:  # no continuous box on top
                            # return board.pos_to_line_id((0, x, y))
                            return self.takeedge(state, 0, x, y)
                    # left edge is empty
                    elif board.lines[1][x][y] < 1:
                        if y == 0 or edges_of_box(state, x, y - 1) != 2:  # no continuous box on left
                            # return board.pos_to_line_id((1, x, y))
                            return self.takeedge(state, 1, x, y)
                    # bottom edge is empty
                    elif board.lines[0][x+1][y] < 1:
                        if x == size - 1 or edges_of_box(state, x + 1, y) != 2:  # no continuous box on bottom
                            # return board.pos_to_line_id((0, x + 1, y))
                            return self.takeedge(state, 0, x+1, y)
                    # right edge is empty
                    elif board.lines[1][x][y+1] < 1:
                        if y == size - 1 or edges_of_box(state, x, y + 1) != 2:  # no continuous box on right
                            # return board.pos_to_line_id((1, x, y + 1))
                            return self.takeedge(state, 1, x, y+1)
        return -1

    def sides3(self, state: DotsAndBoxes):
        """Return true and u,v if there is a box(u, v)=3"""
        for x in range(state.size):
            for y in range(state.size):
                if edges_of_box(state, x, y) == 3:
                    return True, x, y
        return False, -1, -1

    def takeall3s(self, state: DotsAndBoxes):
        while True:
            result, x, y = self.sides3(state)
            if result:
                line_id = self.takebox(state, x, y)  # takebox => takeedge => move_cache
                return line_id
            else:
                return -1

    def sides01(self, state: DotsAndBoxes):
        """Return true and zz, x, y if there is a safe edge(x, y)"""
        if random.random() < 0.5:
            zz = 1  # vertical
        else:
            zz = 0  # horizontal
        i = math.floor(self.size * random.random())
        j = math.floor(self.size * random.random())
        if zz == 1:
            result, coordinate = self.rand_V_edge(state, i, j)
            if result:
                x, y = coordinate
                return result, zz, x, y
            else:
                zz = 0
                result, coordinate = self.rand_H_edge(state, i, j)
                if result:
                    x, y = coordinate
                    return result, zz, x, y
        else:
            result, coordinate = self.rand_H_edge(state, i, j)
            if result:
                x, y = coordinate
                return result, zz, x, y
            else:
                zz = 1
                result, coordinate = self.rand_V_edge(state, i, j)
                if result:
                    x, y = coordinate
                    return result, zz, x, y
        return False, zz, -1, -1

    def safe_V_edge(self, state: DotsAndBoxes, x, y):  # vertical edges
        """Returns true if (x, y) is a safe edge"""
        if state.board.lines[1][x][y] < 1:  # empty edge
            if y == 0:  # left most
                if edges_of_box(state, x, y) < 2:
                    return True
            elif y == state.size:  # right most
                if edges_of_box(state, x, y-1) < 2:
                    return True
            elif edges_of_box(state, x, y) < 2 and edges_of_box(state, x, y-1) < 2:
                return True
        return False

    def safe_H_edge(self, state: DotsAndBoxes, x, y):  # horizontal edges
        """Returns true if (x, y) is a safe edge"""
        if state.board.lines[0][x][y] < 1:
            if x == 0:  # top most
                if edges_of_box(state, x, y) < 2:
                    return True
            elif x == state.size:  # bottom most
                if edges_of_box(state, x-1, y) < 2:
                    return True
            elif edges_of_box(state, x, y) < 2 and edges_of_box(state, x-1, y) < 2:
                return True
        return False

    def rand_H_edge(self, state: DotsAndBoxes, i, j):
        """randomly choose a safe horizontal edge"""
        x = i
        y = j
        while True:
            if self.safe_H_edge(state, x, y):
                return True, (x, y)
            else:
                y += 1
                if y == state.size:
                    y = 0
                    x += 1
                    if x > state.size:
                        x = 0
            if x == i and y == j:
                break
        return False, (-1, -1)

    def rand_V_edge(self, state: DotsAndBoxes, i, j):
        """randomly choose a safe vertical edge"""
        x = i
        y = j
        while True:
            if self.safe_V_edge(state, x, y):
                return True, (x, y)
            else:
                y += 1
                if y > state.size:
                    y = 0
                    x += 1
                    if x == state.size:
                        x = 0
            if x == i and y == j:
                break
        return False, (-1, -1)

    def singleton(self, state: DotsAndBoxes):  # sacrifice one box
        """Returns true and zz,x,y if edge(x,y) gives exactly 1 square away"""
        numb = 0
        for i in range(state.size):
            for j in range(state.size):
                if edges_of_box(state, i, j) == 2:
                    numb = 0
                    if state.board.lines[1][i][j] < 1:  # up edge is empty
                        if i < 1 or edges_of_box(state, i - 1, j) < 2:
                            numb += 1
                    zz = 1  # vertical
                    if state.board.lines[1][i][j] < 1:  # left
                        if j < 1 or edges_of_box(state, i, j-1) < 2:
                            numb += 1
                        if numb > 1:
                            x = i
                            y = j
                            return True, zz, x, y
                    if state.board.lines[1][i][j+1] < 1:  # right
                        if j+1 == state.size or edges_of_box(state, i, j+1) < 2:
                            numb += 1
                        if numb > 1:
                            x = i
                            y = j+1
                            return True, zz, x, y
                    zz = 0  # horizontal
                    if state.board.lines[0][i+1][j] < 1:  # down
                        if i+1 == state.size or edges_of_box(state, i+1, j) < 2:
                            numb += 1
                        if numb > 1:
                            x = i+1
                            y = j
                            return True, zz, x, y
        return False, -1, -1, -1

    def doubleton(self, state: DotsAndBoxes):  # sacrifice two boxes
        """Returns true and zz,x,y if edge(x,y) gives away exactly 2 squares"""
        zz = 1
        for i in range(state.size):
            for j in range(state.size - 1):
                #  --- ---
                # |       |    both two boxes have exactly two edges
                #  --- ---
                if edges_of_box(state, i, j) == 2 and edges_of_box(state, i, j+1) == 2 and state.board.lines[1][i][j+1] < 1:
                    if self.ldub(state, i, j) and self.rdub(state, i, j+1):
                        x = i
                        y = j+1
                        return True, zz, x, y
        zz = 0
        for i in range(state.size - 1):
            for j in range(state.size):
                #  ---
                # |   |
                #           both two boxes have exactly two edges
                # |   |
                #  ---
                if edges_of_box(state, i, j) == 2 and edges_of_box(state, i+1, j) == 2 and state.board.lines[0][i+1][j] < 1:
                    if self.udub(state, i, j) and self.ddub(state, i+1, j):
                        x = i+1
                        y = j
                        return True, zz, x, y
        return False, -1, -1, -1

    def ldub(self, state: DotsAndBoxes, i, j):  # left
        #  ---
        # |        return True if the empty edges of the 3 edges leads to a box<2
        #  ---
        """Given box(i,j)=2 and vertical edge(i,j+1)=0, returns true if the other free edge leads to a box<2"""
        if state.board.lines[1][i][j] < 1:  # left
            if j < 1 or edges_of_box(state, i, j-1) < 2:
                return True
        elif state.board.lines[0][i][j] < 1:  # up
            if i < 1 or edges_of_box(state, i-1, j) < 2:
                return True
        elif i==state.size-1 or edges_of_box(state, i+1, j) < 2:  # down
            return True
        return False

    def rdub(self, state:DotsAndBoxes, i, j):  # right
        #  ---
        #     |
        #  ---
        """Given box(i,j)=2 and vertical edge(i,j)=0, returns true if the other free edge leads to a box<2"""
        if state.board.lines[1][i][j+1] < 1:  # right
            if j+1==state.size or edges_of_box(state, i, j+1) < 2:
                return True
        elif state.board.lines[0][i][j] < 1:  # up
            if i < 1 or edges_of_box(state, i-1, j) < 2:
                return True
        elif i==state.size-1 or edges_of_box(state, i+1, j) < 2:  # down
            return True
        return False

    def udub(self, state: DotsAndBoxes, i, j):
        #  ---
        # |   |
        #
        if state.board.lines[0][i][j] < 1:  # up
            if i < 1 or edges_of_box(state, i-1, j) < 2:
                return True
        elif state.board.lines[1][i][j] < 1:  # left
            if j < 1 or edges_of_box(state, i, j-1) < 2:
                return True
        elif j==state.size-1 or edges_of_box(state, i, j+1) < 2:  # right
            return True
        return False

    def ddub(self, state: DotsAndBoxes, i, j):
        #
        # |   |
        #  ---
        if state.board.lines[0][i+1][j] < 1:  # down
            if i == state.size - 1 or edges_of_box(state, i+1, j) < 2:
                return True
        elif state.board.lines[1][i][j] < 1:  # left
            if j < 1 or edges_of_box(state, i, j-1) < 2:
                return True
        elif j==state.size-1 or edges_of_box(state, i, j+1) < 2:  # right
            return True
        return False

    def sac(self, state: DotsAndBoxes, i, j):  # box(i,j) has 3 edges
        """ The smart part
            Sacrifices two squares if there are still 3's """
        self.count = 0
        self.loop = False
        self.incount(state, 0, i, j)  # to get number of boxes in this chain starting at i,j
        if not self.loop:  # loop is a kind of special chain
            self.takeallbut(state, i, j)  # take all other 3s before handling this chain
        if self.count + state.board.scores[1] + state.board.scores[2] == state.size*state.size:  # the end of game
            self.takeall3s(state)
        else:  # not the end
            if self.loop:
                self.count -= 2  # if it's a loop, we need to sacrifice 2 more boxes
            self.outcount(state, 0, i, j)
            i = state.size
            j = state.size

    def incount(self, state: DotsAndBoxes, k, i, j):
        """
        enter with box[i][j]=3 and k=0
        returns count = number in chain starting at i,j
        k=1,2,3,4 means skip left,up,right,down.
        """
        self.count += 1
        if k != 1 and state.board.lines[1][i][j] < 1:  # not skip left, and left edge is empty
            if j > 0:  # not the left most edge
                if edges_of_box(state, i, j - 1) > 2:  # left box  FIXME: is > 2 OK ?
                    self.count += 1
                    self.loop = True
                elif edges_of_box(state, i, j - 1) > 1:
                    self.incount(state, 3, i, j-1)  # skip right side

        elif k != 2 and state.board.lines[0][i][j] < 1:  # not skip up, and up edge is empty
            if i > 0:  # not the up most edge
                if edges_of_box(state, i - 1, j) > 2:  # up box, end of chain
                    self.count += 1
                    self.loop = True
                elif edges_of_box(state, i - 1, j) > 1:  # not the end of chain
                    self.incount(state, 4, i - 1, j)  # skip down side

        elif k != 3 and state.board.lines[1][i][j + 1] < 1:  # not skip right, and right edge is empty
            if j < state.size - 1:  # not the right most edge
                if edges_of_box(state, i, j + 1) > 2:  # right box
                    self.count += 1
                    self.loop = True
                elif edges_of_box(state, i, j + 1) > 1:
                    self.incount(state, 1, i, j + 1)  # skip left side

        elif k != 4 and state.board.lines[0][i + 1][j] < 1:  # not skip down, and down edge is empty
            if i < state.size-1:  # not the down most edge
                if edges_of_box(state, i + 1, j) > 2:  # down box
                    self.count += 1
                    self.loop = True
                elif edges_of_box(state, i + 1, j) > 1:
                    self.incount(state, 2, i + 1, j)  # skip up side

    def takeallbut(self, state: DotsAndBoxes, x, y):
        """All boxes with 3 edges can be chose, expect box(x,y)"""
        result, coordinate = self.side3not(state, x, y)
        while result:
            x, y = coordinate
            self.takebox(state, x, y)
            result, coordinate = self.side3not(state, x, y)

    def side3not(self, state: DotsAndBoxes, x, y):
        """Returns any box with 3 edges expect box(x, y)"""
        size = state.size
        for i in range(size):
            for j in range(size):
                if edges_of_box(state, i, j) == 3:
                    if i != x or j != y:
                        return True, (i, j)
        return False, (-1, -1)

    def outcount(self, state: DotsAndBoxes, k, i, j):
        """Takes all but count-2 boxes and exits"""
        # k=1,2,3,4 means skip left,up,right,down.
        if self.count > 0:
            if k != 1 and state.board.lines[1][i][j] < 1:  # not skip left, and left edge is empty
                if self.count != 2:
                    # return True, state.board.pos_to_line_id((1, i, j))
                    self.takeedge(state, 1, i, j)
                self.count -= 1
                self.outcount(state, 3, i, j - 1)
            elif k != 2 and state.board.lines[0][i][j] < 1:  # not skip up, and up edge is empty
                if self.count != 2:
                    # return True, state.board.pos_to_line_id((0, i, j))
                    self.takeedge(state, 0, i, j)
                self.count -= 1
                self.outcount(state, 4, i - 1, j)
            elif k != 3 and state.board.lines[1][i][j+1] < 1:  # not skip right, and right edge is empty
                if self.count != 2:
                    # return True, state.board.pos_to_line_id((1, i, j+1))
                    self.takeedge(state, 1, i, j + 1)
                self.count -= 1
                self.outcount(state, 1, i, j + 1)
            elif k != 4 and state.board.lines[0][i+1][j] < 1:  # not skip down, and down edge is empty
                if self.count != 2:
                    # return True, state.board.pos_to_line_id((0, i+1, j))
                    self.takeedge(state, 0, i+1, j)
                self.count -= 1
                self.outcount(state, 2, i + 1, j)

    def makeanymove(self, state: DotsAndBoxes):
        x = -1
        for i in range(state.size+1):
            for j in range(state.size):
                if state.board.lines[0][i][j] < 1:  # horizontal
                    x = i
                    y = j
                    i = state.size + 1
                    j = state.size
                    # return True, (1, x, y)
                    return self.takeedge(state, 0, x, y)

        if x < 0:
            for i in range(state.size):
                for j in range(state.size+1):
                    if state.board.lines[1][i][j] < 1:  # vertical
                        x = i
                        y = j
                        i = state.size
                        j = state.size + 1
                        # return True, (1, x, y)
                        return self.takeedge(state, 1, x, y)
        return -1

    def get_action(self, state: DotsAndBoxes, **kwargs):
        return_prob = kwargs.get('return_prob', 0)
        if len(self.move_cache) < 1:
            self.makemove(state.copy())

        move = self.move_cache.pop()
        return move

    def copy(self):
        return MoreGreedyPlayer()

    def __str__(self):
        return "MoreGreedy {}".format(self.player)
