import pickle

import keras
import keras.backend.tensorflow_backend as tfback
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.regularizers import l2

from Game import GameBase


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    # global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus


def self_entropy(probs):
    return -np.mean(np.sum(probs * np.log(probs + 1e-10), axis=1))


class PolicyValueNet:

    def __init__(self, size, stage, model_file=None):
        self.size = size
        self._half_split = (size + 1) * size
        self.channel_size = 7
        self.input_shape = (self.channel_size, size + 1, size + 1)
        self.input_batch_shape = (-1, self.channel_size, size + 1, size + 1)
        self.action_spec = (size + 1) * size * 2
        self.l2_const = 1e-4
        if stage == 1:
            self.create_model_stage1()
        elif stage == 2:
            self.create_model_stage2()
        self._loss_train_op()

        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))
            self.model.set_weights(net_params)

    def create_model_stage1(self):
        # conv layers
        in_x = network = Input(self.input_shape)
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=256, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # action policy layers
        policy_net = Conv2D(filters=8, kernel_size=(1, 1), data_format="channels_first", activation="relu",
                            kernel_regularizer=l2(self.l2_const))(network)
        policy_net = Flatten()(policy_net)
        policy_net = Dense(128, activation="relu", kernel_regularizer=l2(self.l2_const))(policy_net)
        self.policy_net = Dense(self.action_spec, activation="softmax", kernel_regularizer=l2(self.l2_const))(
            policy_net)
        # state value layers
        value_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu",
                           kernel_regularizer=l2(self.l2_const))(network)
        value_net = Flatten()(value_net)
        value_net = Dense(128, kernel_regularizer=l2(self.l2_const))(value_net)
        value_net = Dense(32, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        self.model = keras.models.Model(in_x, [self.policy_net, self.value_net])

    def create_model_stage2(self):
        # conv layers
        # if we use the same model as stage1, the behavior becomes very weird. It will do double-cross occasionally,
        # even in the last half-open chain, and that causes the lose. Therefore, a simple model makes it behaviors as a
        # normal greedy player
        in_x = network = Input(self.input_shape)
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # action policy layers
        policy_net = Conv2D(filters=8, kernel_size=(1, 1), data_format="channels_first", activation="relu",
                            kernel_regularizer=l2(self.l2_const))(network)
        policy_net = Flatten()(policy_net)
        # policy_net = Dense(128, activation="relu", kernel_regularizer=l2(self.l2_const))(policy_net)
        self.policy_net = Dense(self.action_spec, activation="softmax", kernel_regularizer=l2(self.l2_const))(
            policy_net)
        # state value layers
        value_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu",
                           kernel_regularizer=l2(self.l2_const))(network)
        value_net = Flatten()(value_net)
        value_net = Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        # value_net = Dense(128, kernel_regularizer=l2(self.l2_const))(value_net)
        # value_net = Dense(32, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        self.model = keras.models.Model(in_x, [self.policy_net, self.value_net])

    def policy_value(self, state_input):
        results = self.model.predict_on_batch(np.array(state_input))
        return results

    def policy_value_fn(self, game: GameBase):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = list(game.get_available_actions())
        # legal_positions = list(game.get_stage1_actions())  # for stage1 training
        current_state = game.get_current_state()
        act_probs, value = self.policy_value(current_state.reshape(self.input_batch_shape))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Three loss terms：
        loss = (z - v)^2 + pi^T * log(p) + c||theta||^2
        """

        # get the train op
        losses = ['categorical_crossentropy', 'mean_squared_error']
        self.model.compile(optimizer=keras.optimizers.Adam(), loss=losses)

    def train_step(self, state_input, mcts_probs, winner, learning_rate):
        state_input_union = np.array(state_input)
        mcts_probs_union = np.array(mcts_probs)
        winner_union = np.array(winner)
        loss = self.model.evaluate(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input),
                                   verbose=0)
        action_probs, _ = self.model.predict_on_batch(state_input_union)
        entropy = self_entropy(action_probs)
        keras.backend.set_value(self.model.optimizer.lr, learning_rate)
        self.model.fit(state_input_union, [mcts_probs_union, winner_union], batch_size=len(state_input), verbose=0)
        return loss[0], entropy

    def get_policy_param(self):
        net_params = self.model.get_weights()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """

        def rot90_horizontal(lines):
            # (size+1)x(size) ==> (size)x(size+1)
            lines = np.rot90(lines[:, 0:-1], 1)
            return np.concatenate((lines, np.zeros((1, self.size + 1))), axis=0)

        def rot90_vertical(lines):
            # (size)x(size+1) ==> (size+1)x(size)
            lines = np.rot90(lines[0:-1, :], 1)
            return np.concatenate((lines, np.zeros((self.size + 1, 1))), axis=1)

        def rot90_lines(lines):
            h_lines = rot90_vertical(lines[1])
            v_lines = rot90_horizontal(lines[0])
            return np.array([h_lines, v_lines])

        def rot90_boxes(boxes):
            boxes[0][:-1, :-1] = np.rot90(boxes[0][:-1, :-1], 1)
            boxes[1][:-1, :-1] = np.rot90(boxes[1][:-1, :-1], 1)
            return boxes

        def fliplr_lines(lines):
            h_lines = np.fliplr(lines[0][:, 0:-1])
            h_lines = np.concatenate((h_lines, np.zeros((self.size + 1, 1))), axis=1)
            v_lines = np.fliplr(lines[1])
            return np.array([h_lines, v_lines])

        def fliplr_boxes(boxes):
            boxes[0][:-1, :-1] = np.fliplr(boxes[0][:-1, :-1])
            boxes[1][:-1, :-1] = np.fliplr(boxes[1][:-1, :-1])
            return boxes

        def rot90(state, mcts_prob, winner):
            new_state = np.concatenate(
                (rot90_lines(state[0:2]), rot90_lines(state[2:4]), rot90_boxes(state[4:6]), state[6:7]), axis=0)
            h_prob = mcts_prob[0:self._half_split].reshape((self.size + 1, self.size))
            h_prob = np.rot90(h_prob, 1).flatten()  # horizontal to vertical now
            v_prob = mcts_prob[self._half_split:].reshape((self.size, self.size + 1))
            v_prob = np.rot90(v_prob, 1).flatten()  # horizontal now
            new_prob = np.concatenate((v_prob, h_prob), axis=0)
            return new_state, new_prob, winner

        def fliplr(state, mcts_prob, winner):
            new_state = np.concatenate(
                (fliplr_lines(state[0:2]), fliplr_lines(state[2:4]), fliplr_boxes(state[4:6]), state[6:7]), axis=0)
            h_prob = mcts_prob[0:self._half_split].reshape((self.size + 1, self.size))
            h_prob = np.fliplr(h_prob).flatten()
            v_prob = mcts_prob[self._half_split:].reshape((self.size, self.size + 1))
            v_prob = np.fliplr(v_prob).flatten()
            new_prob = np.concatenate((h_prob, v_prob), axis=0)
            return new_state, new_prob, winner

        extend_data = []
        for _state, _mcts_prob, _winner in play_data:
            extend_data.append((_state, _mcts_prob, _winner))
            extend_data.append(fliplr(_state, _mcts_prob, _winner))
            cur_state = _state
            cur_prob = _mcts_prob
            cur_winner = _winner
            for _ in range(3):
                # rotate counterclockwise
                # rotate horizontal lines (size+1)x(size), vertical lines (size)x(size+1)
                # state is 'channel x size+1 x size+1'
                cur_state, cur_prob, cur_winner = rot90(cur_state, cur_prob, cur_winner)
                extend_data.append((cur_state, cur_prob, cur_winner))
                extend_data.append(fliplr(cur_state, cur_prob, cur_winner))
        return extend_data
