import random
from collections import defaultdict, deque
from multiprocessing import Queue, Process

import numpy as np

from Game import self_play_with_statistics
from dots_and_boxes import DotsAndBoxes, PolicyValueNet
# from MCTS_alphazero import MCTSPlayer
from dots_and_boxes.MCTS_alphazero_stage1 import MCTSStage1Player as MCTSPlayer
from dots_and_boxes.players import GreedyPlayer


def run_until_stage2(game: DotsAndBoxes, players, **kwargs):
    while game.stage1():
        players[game.current_player_id].play(game, **kwargs)


def self_play_until_stage2_with_statistics(game: DotsAndBoxes, player: MCTSPlayer, temp=1e-3, verbose=0):
    states, mcts_probs, current_players = [], [], []
    while game.stage1():
        action, action_probs = player.get_action(game, temp=temp, return_prob=1)
        # store the data
        states.append(game.get_current_state())
        mcts_probs.append(action_probs)
        current_players.append(game.current_player_id)
        # perform a move
        game.act(action, verbose)
    # default current player loose (not determined, but in most cases)
    winner = game.stage1_winner()
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


class TrainPipeline:
    def __init__(self, init_model=None):
        # params of the board and the game
        self.size = 3
        self.game = DotsAndBoxes(self.size)
        # self.board = self.game.board
        # training params
        self.learn_rate = 5e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 50000
        self.batch_size = 512  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.size, 1, model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.size, 1)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, 1,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)
        self.mcts_player2 = MCTSPlayer(self.policy_value_net.policy_value_fn, 2,
                                       c_puct=self.c_puct,
                                       n_playout=self.n_playout,
                                       is_selfplay=1)
        self.mcts_players = [None, self.mcts_player, self.mcts_player2]

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        return self.policy_value_net.get_equi_data(play_data)

    def collect_selfplay_data_multi_process(self, process_num=4):
        def a_process(q: Queue):
            winner, play_data = self_play_with_statistics(self.game, self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            for d in play_data:
                q.put(d)

        Q = Queue()
        processes = [Process(target=a_process, args=(Q,)) for _ in range(process_num)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        while not Q.empty():
            self.data_buffer.append(Q.get())

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        for i in range(n_games):
            self.game.reset()
            self.mcts_player.reset_player(self.game.current_player_id)
            winner, play_data = self_play_until_stage2_with_statistics(self.game, self.mcts_player, temp=self.temp)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                                axis=1)
                         )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 4 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 4 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return loss, entropy

    def policy_evaluate(self, n_games=10, test_mode=False):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, 2,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        opp_player = GreedyPlayer()
        # opp_player = RandomPlayer()
        win_cnt = defaultdict(int)
        game = DotsAndBoxes(self.size)
        for i in range(n_games):
            game.reset()
            if test_mode:
                run_until_stage2(game, [None, opp_player, current_mcts_player], verbose=1)
                winner = game.stage1_winner()
                print('winner : ' + str(winner))
            else:
                run_until_stage2(game, [None, opp_player, current_mcts_player], verbose=0)
                winner = game.stage1_winner()
            win_cnt[winner] += 1
        # The mcts player cannot win currently.
        # The best of mcts player can do in stage1 currently is behaving in the same way as GreedyPlayer
        # We may use the reword of stage2 model as the result of stage1 while training stage1 model. But it is a bit
        # too time consuming
        win_ratio = 1.0 * (win_cnt[2] + 0.5 * win_cnt[0]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[2], win_cnt[1], win_cnt[0]))
        return win_ratio

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                # self.collect_selfplay_data_multi_process(4)
                print("batch i:{}, episode_len:{}".format(
                    i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.model')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print('\n\rquit')

    def test(self):
        win_ratio = self.policy_evaluate(10, test_mode=True)


def test_model(model_file):
    training_pipeline = TrainPipeline(init_model=model_file)
    training_pipeline.test()


if __name__ == '__main__':
    # training_pipeline = TrainPipeline()
    # training_pipeline.run()
    # test_model(None)
    # test_model('3x3-player2-greedy.model')
    test_model('3x3-stage1.model')
