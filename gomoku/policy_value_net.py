import keras
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.regularizers import l2
import numpy as np
import pickle
from Game import GameBase

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback


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

    def __init__(self, size, model_file=None):
        self.size = size
        self.board_width = size
        self.board_height = size
        self.input_shape = (4, self.board_width, self.board_height)
        self.l2_const = 1e-4
        self.create_model()
        self._loss_train_op()

        if model_file:
            net_params = pickle.load(open(model_file, 'rb'))
            self.model.set_weights(net_params)

    def create_model(self):
        """create the policy value network """
        in_x = network = Input(self.input_shape)

        # conv layers
        network = Conv2D(filters=32, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=64, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        network = Conv2D(filters=128, kernel_size=(3, 3), padding="same", data_format="channels_first",
                         activation="relu", kernel_regularizer=l2(self.l2_const))(network)
        # action policy layers
        policy_net = Conv2D(filters=4, kernel_size=(1, 1), data_format="channels_first", activation="relu",
                            kernel_regularizer=l2(self.l2_const))(network)
        policy_net = Flatten()(policy_net)
        self.policy_net = Dense(self.board_width * self.board_height, activation="softmax",
                                kernel_regularizer=l2(self.l2_const))(policy_net)
        # state value layers
        value_net = Conv2D(filters=2, kernel_size=(1, 1), data_format="channels_first", activation="relu",
                           kernel_regularizer=l2(self.l2_const))(network)
        value_net = Flatten()(value_net)
        value_net = Dense(64, kernel_regularizer=l2(self.l2_const))(value_net)
        self.value_net = Dense(1, activation="tanh", kernel_regularizer=l2(self.l2_const))(value_net)

        self.model = keras.Model(in_x, [self.policy_net, self.value_net])

    def policy_value(self, state_input):
        results = self.model.predict_on_batch(np.array(state_input))
        return results

    def policy_value_fn(self, game: GameBase):
        """
        input: board
        output: a list of (action, probability) tuples for each available action and the score of the board state
        """
        legal_positions = list(game.get_available_actions())
        current_state = game.get_current_state()
        act_probs, value = self.policy_value(current_state.reshape(-1, 4, self.board_width, self.board_height))
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
        extend_data = []
        for state, mcts_porb, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data
