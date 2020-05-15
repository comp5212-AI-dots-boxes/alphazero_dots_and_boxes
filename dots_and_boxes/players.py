from dots_and_boxes import DotsAndBoxesPlayerBase, DotsAndBoxes, DotsAndBoxesBoard
from Game import GameBase
import random
import copy
import numpy as np

from MCTS_improve import MCTSPlayer as MCTSPlayer_improve
from MCTS import MCTSPlayer as MCTSPlayer


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


def test():
    #######################
    #  Play with pure MCTS Player
    #######################
    size = 3
    try:
        game = DotsAndBoxes(size)

        mcts_player1 = MCTSPlayer(c_puct=5, n_playout=500)  # just a simple try

        mcts_player2 = MCTSPlayer_improve(c_puct=5, n_playout=500)
        human = HumanPlayer()
        rand = RandomPlayer()

        game.start_play(mcts_player1, mcts_player2, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    for i in range(10):
        test()
