"""
The game of dots and boxes
"""
import numpy as np
from Game import GameBase, Player
import mcts


class DotsAndBoxesBoard:

    def __init__(self, size: int, copy_dict=None):
        self.size = size
        self.end_step = (size + 1) * size * 2
        self._half_split = (size + 1) * size
        if copy_dict is None:
            # there should be (size+1)*size horizontal lines, and size*(size+1) vertical lines
            # initialize the board lines with a (size+1)*(size+1)*2 matrix
            # (h,x,y), h == 0 <=> horizontal lines
            self.lines = np.zeros((2, size + 1, size + 1), dtype=np.float)
            # 0: not be conquered; 1: player1 conquered; 2: player2 conquered
            self.boxes = np.zeros((size + 1, size + 1), dtype=np.int)
            self.scores = [0, 0, 0]
            self.step = 0
            self.available_lines = set([i for i in range(self.end_step)])
            # except current state of each line, we use the line being drawn at last step as current state
            self.last_act_line = None
            self.last_act_player_id = None
        else:
            self.lines = copy_dict['lines'].copy()
            self.boxes = copy_dict['boxes'].copy()
            self.scores = copy_dict['scores'].copy()
            self.step = copy_dict['step']
            self.available_lines = copy_dict['available_lines'].copy()
            self.last_act_line = copy_dict['last_act_line']
            self.last_act_player_id = copy_dict['last_act_player_id']

    def line_id_to_pos(self, line_id):
        if line_id >= self._half_split:
            # vertical lines
            tmp = line_id - self._half_split
            x = tmp // (self.size + 1)
            y = tmp % (self.size + 1)
            h = 1
        else:
            x = line_id // self.size
            y = line_id % self.size
            h = 0
        return h, x, y

    def pos_to_line_id(self, pos):
        h, x, y = pos
        if h == 0:
            return x * self.size + y
        else:
            return self._half_split + x * (self.size + 1) + y

    def _box_is_conquered(self, x, y):
        """
                    (x,y,0)
            (x,y,1) (x,y) (x,y+1,1)
                    (x+1,y,0)
        """
        return self.lines[0][x][y] == 1 and self.lines[1][x][y] == 1 \
               and self.lines[0][x + 1][y] == 1 and self.lines[1][x][y + 1] == 1

    def is_conquered(self, x, y):
        return self.boxes[x][y] > 0

    def get_current_state(self):
        # channel first input
        # 0: horizontal lines
        # 1: vertical lines
        # 2: last line being drawn (horizontal)
        # 3: last line being drawn (vertical)
        # 4: boxes occupied map, p1
        # 5: boxes occupied map, p2
        # 6: indicate the player (do it at DotAndBoxes)
        last_act_line_map = np.zeros(self.lines.shape)
        if self.last_act_line:
            last_act_line_map[self.last_act_line] = 1.0
        boxes_map = np.zeros((2,) + self.boxes.shape)
        boxes_map[0][self.boxes == 1] = 1.0
        boxes_map[1][self.boxes == 2] = 1.0
        state = np.concatenate((self.lines, last_act_line_map, boxes_map), axis=0)
        return state

    def set_line(self, line, player_id):
        """
        a conquerable line (x, y, h)
        player_id can merely be 1 or 2
        if return is True, the player should not be switched
        """
        self.available_lines.remove(line)
        h, x, y = self.line_id_to_pos(line)
        self.last_act_line = (h, x, y)
        self.last_act_player_id = player_id
        self.step += 1
        self.lines[h][x][y] = 1
        count = 0
        if x < self.size and y < self.size and self._box_is_conquered(x, y):
            assert self.boxes[x][y] == 0
            self.boxes[x][y] = player_id
            count += 1
        if h == 0 and x > 0:
            # for a horizontal line, [x-1][y] may also be conquered
            if self._box_is_conquered(x - 1, y):
                assert self.boxes[x - 1][y] == 0
                self.boxes[x - 1][y] = player_id
                count += 1
        elif h == 1 and y > 0:
            # for a vertical line, [x][y-1] may also be conquered
            if self._box_is_conquered(x, y - 1):
                assert self.boxes[x][y - 1] == 0
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
                if self.lines[0][x][y] > 0:
                    h_res += '---+'
                else:
                    h_res += '   +'
            return h_res

        def vertical_str(x):
            v_res = ''
            if self.lines[1][x][0] > 0:
                v_res += '|'
            else:
                v_res += ' '
            for y in range(self.size):
                if self.boxes[x][y] > 0:
                    v_res += ' %d ' % self.boxes[x][y]
                else:
                    v_res += '   '
                if self.lines[1][x][y + 1] > 0:
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

    def copy(self):
        new_board = DotsAndBoxesBoard(self.size, copy_dict={
            'lines': self.lines,
            'boxes': self.boxes,
            'scores': self.scores,
            'step': self.step,
            'available_lines': self.available_lines,
            'last_act_line': self.last_act_line,
            'last_act_player_id': self.last_act_player_id
        })
        return new_board

    def state_id(self):
        sid = 0
        for index in range(self.end_step):
            h, x, y = self.line_id_to_pos(index)
            if not self.lines[h][x][y] == 0:
                sid += 1 << index
        return sid

    #def symmetrical_id(self):
    #    """
    #    One symmetrical_id corresponds to multiple state ids
    #    """


class DotsAndBoxes(GameBase):

    def __init__(self, size: int, board: DotsAndBoxesBoard = None):
        """
        Initialize the game and game board, the game board has size x size boxes
        """
        super().__init__()
        self.size = size
        if board is None:
            self.board = DotsAndBoxesBoard(self.size)
        else:
            self.board = board
        self.cur_player_id = 1
        self.players = [1, 2]  # player1 and player2

    def switch_player(self):
        """ only 1 or 2 """
        self.cur_player_id = 3 - self.cur_player_id

    @property
    def current_player_id(self):
        return self.cur_player_id

    def set_current_player(self, gamer_id):
        self.cur_player_id = gamer_id

    def get_action_space(self):
        return self.board.end_step

    def is_draw(self):
        return self.board.is_end() and self.board.scores[1] == self.board.scores[2]

    def is_end(self):
        return self.board.is_end()

    def is_playing(self):
        return not self.board.is_end()

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

    def get_available_actions(self) -> set:
        return self.board.available_lines

    def act(self, action, verbose=0):
        not_switch_player = self.board.set_line(action, self.cur_player_id)
        if verbose > 0:
            print('step: %d' % self.board.step)
            print(self.board.state_str())
        if not not_switch_player:
            self.switch_player()

    def copy(self):
        new_board = self.board.copy()
        new_game = DotsAndBoxes(self.size, new_board)
        new_game.cur_player_id = self.cur_player_id
        return new_game

    def get_current_state(self):
        state = self.board.get_current_state()
        if self.current_player_id % 2 == 1:
            player_map = np.ones((1,) + state[0].shape)
        else:
            player_map = np.zeros((1,) + state[0].shape)
        return np.concatenate((state, player_map), axis=0)

    def reset(self):
        self.board = DotsAndBoxesBoard(self.size)
        self.cur_player_id = 1

    def start_play(self, player1, player2, start_player=1, is_shown=1):
        """start a game between two players"""
        if start_player not in (1, 2):
            raise Exception('start_player should be either 1 (player1 first) '
                            'or 2 (player2 first)')

        p1, p2 = self.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}  # convert to a set
        if is_shown:
            # self.graphic(self.board, player1.player, player2.player)
            print(self.board.state_str())
        while True:
            current_player = self.current_player_id
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self)
            # self.board.do_move(move)
            self.act(move, verbose=is_shown)

            # end, winner = self.board.game_end()
            end = self.is_end()
            if end:
                winner = self.get_winner()
                if is_shown:
                    if winner != 0:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                if winner != 0:
                    print("Game end. Winner is", players[winner])
                else:
                    print("Game end. Tie")
                return winner

    def stage1(self):
        available_lines = self.get_available_actions().copy()
        for x in range(self.size):
            for y in range(self.size):
                if self.board.boxes[x][y] == 0:
                    # check 4 lines of this box
                    four_lines = ((0, x, y), (1, x, y), (0, x + 1, y), (1, x, y + 1))
                    num_empty = 4 - (
                                self.board.lines[0][x][y] + self.board.lines[1][x][y] + self.board.lines[0][x + 1][y] +
                                self.board.lines[1][x][y + 1])
                    if num_empty <= 2:
                        # it will give advantage to opponent
                        for line in four_lines:
                            available_lines.discard(self.board.pos_to_line_id(line))
        return len(available_lines) > 0

    def stage2(self):
        return not self.stage1()

    # def getPossibleActions(self):
    #     return list(self.get_available_actions())
    #
    # def takeAction(self, action):
    #     tmp = self.copy()
    #     tmp.act(action)
    #     return tmp
    #
    # def isTerminal(self):
    #     return self.is_end()
    #
    # def getReward(self):
    #     return self.board.scores[1] - self.board.scores[2]


class DotsAndBoxesPlayerBase(Player):

    def __init__(self):
        super(Player, self).__init__()
        pass

    def get_action(self, state: GameBase, **kwargs):
        raise NotImplementedError()

    def copy(self):
        """
        I do not use deepcopy here because we do not need to deepcopy a game board.
        But please copy everything needed except _board
        """
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()
