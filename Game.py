import numpy as np


class GameBase:

    def __init__(self):
        pass

    @property
    def current_player_id(self):
        raise NotImplementedError()

    def set_current_player(self, gamer_id):
        raise NotImplementedError()

    def get_action_space(self):
        raise NotImplementedError()

    def is_playing(self):
        raise NotImplementedError()

    def is_draw(self):
        raise NotImplementedError()

    def is_end(self):
        raise NotImplementedError()

    def get_winner(self):
        """
        :return: 0 = tie; k>1 = player k wins; -1 = is playing
        """
        raise NotImplementedError()

    def get_available_actions(self):
        raise NotImplementedError()

    def act(self, action, verbose=0):
        raise NotImplementedError()

    def copy(self):
        raise NotImplementedError()

    def get_current_state(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class Player:

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, state: GameBase, **kwargs):
        raise NotImplementedError()

    def play(self, state: GameBase, **kwargs):
        verbose = kwargs.get('verbose', 0)
        state.act(self.get_action(state), verbose=verbose)

    def __str__(self):
        raise NotImplementedError()


def self_play_with_statistics(game: GameBase, player: Player, temp=1e-3, verbose=0):
    states, mcts_probs, current_players = [], [], []
    while game.is_playing():
        action, action_probs = player.get_action(game, temp=temp, return_prob=1)
        # store the data
        states.append(game.get_current_state())
        mcts_probs.append(action_probs)
        current_players.append(game.current_player_id)
        # perform a move
        game.act(action, verbose)
    winner = game.get_winner()
    # winner from the perspective of the current player of each state
    winners_z = np.zeros(len(current_players))
    if winner != 0:
        winners_z[np.array(current_players) == winner] = 1.0
        winners_z[np.array(current_players) != winner] = -1.0

    if verbose > 0:
        if winner != 0:
            print("Game ended. Winner is player:", winner)
        else:
            print("Game ended. Tie")
    return winner, zip(states, mcts_probs, winners_z)


class GameManager:

    def __init__(self, game: GameBase, gamers: list, first_gamer_id):
        """
        :param gamers: [None, p1, p2, ...], player at index 0 will never take part in the game
        """
        self.gamers = gamers
        self.game = game
        self.game.set_current_player(first_gamer_id)

    def play(self, **kwargs):
        while self.game.is_playing():
            self.gamers[self.game.current_player_id].play(self.game, **kwargs)
        return self.game.get_winner()


if __name__ == '__main__':
    from dots_and_boxes import DotsAndBoxes
    from dots_and_boxes.players import GreedyPlayer
    from MCTS import MCTSPlayer
    # from gomoku import Gomoku
    # game = Gomoku()
    # game.init_board()

    game = DotsAndBoxes(4)
    player = MCTSPlayer(n_playout=1000)
    # random_player = RandomPlayer()
    greedy_player = GreedyPlayer()

    # self_play_with_statistics(game, player, verbose=1)
    gm = GameManager(game, [None, player, greedy_player], 1)
    print(gm.play())
