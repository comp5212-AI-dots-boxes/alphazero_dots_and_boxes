from dots_and_boxes import DotsAndBoxesPlayerBase
import random


class RandomPlayer(DotsAndBoxesPlayerBase):

    def get_line(self):
        return random.choice(list(self._board.available_lines))
