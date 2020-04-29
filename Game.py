import numpy as np
import random


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
        pass

    def get_action(self, state: GameBase, **kwargs):
        raise NotImplementedError()

    def play(self, state: GameBase, **kwargs):
        verbose = kwargs.get('verbose', 0)
        state.act(self.get_action(state), verbose=verbose)


class RandomPlayer(Player):

    def get_action(self, state: GameBase, **kwargs):
        return random.choice(list(state.get_available_actions()))


def self_play_with_statistics(game: GameBase, player: Player, temp=1e-3, verbose=0):
    states, mcts_probs, current_players = [], [], []
    while True:
        action, action_probs = player.get_action(game, temp=temp, return_prob=1)
        # action = player.get_action(game, temp=temp)
        # store the data
        states.append(game.get_current_state())
        mcts_probs.append(action_probs)
        current_players.append(game.current_player_id)
        # perform a move
        game.act(action, verbose)
        if game.is_end():
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


def auto_play_with_statistics(game: GameBase, players: list, temp=1e-3, verbose=0):
    states, mcts_probs, current_players = [], [], []
    mcts_player_id = 1
    while True:
        if mcts_player_id == game.current_player_id:
            action, action_probs = players[game.current_player_id].get_action(game, temp=temp, return_prob=1)
        else:
            action, action_probs = players[game.current_player_id].get_action(game, temp=temp, return_prob=1)
            players[mcts_player_id].mcts.update_with_move(action)
        # store the data
        states.append(game.get_current_state())
        mcts_probs.append(action_probs)
        current_players.append(game.current_player_id)
        # perform a move
        game.act(action, verbose)
        if game.is_end():
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

# if __name__ == '__main__':
#     from MCTS import MCTSPlayer
#     from dots_and_boxes.players import GreedyPlayer
#     # from gomoku import Gomoku
#     # game = Gomoku()
#     # game.init_board()
#     from dots_and_boxes import DotsAndBoxes
#     game = DotsAndBoxes(4)
#     player = MCTSPlayer(n_playout=10000)
#     random_player = RandomPlayer()
#     greedy_palyer = GreedyPlayer()
#
#     # self_play_with_statistics(game, player, verbose=1)
#     gm = GameManager(game, [None, player, greedy_palyer], 1)
#     print(gm.play())
