from dots_and_boxes import DotsAndBoxesPlayerBase, DotsAndBoxes, DotsAndBoxesBoard
from Game import GameBase
import random
import copy
import numpy as np


class RandomPlayer(DotsAndBoxesPlayerBase):

    def get_action(self, state: GameBase, **kwargs):
        return random.choice(list(state.get_available_actions()))

    def copy(self):
        return RandomPlayer()


class GreedyPlayer(DotsAndBoxesPlayerBase):

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

        choices_for_greedy = []
        if len(best_choices) > 0:
            choices_for_greedy = best_choices
        if len(available_lines) > 0:
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
