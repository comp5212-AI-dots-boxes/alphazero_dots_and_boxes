"""
The MCTS process
improved by turning tree to acyclic directed graph
"""
import numpy as np
from Game import GameBase, Player
from dots_and_boxes import DotsAndBoxes


def rollout_policy_fn(game: GameBase):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly
    available_actions = list(game.get_available_actions())
    action_probs = np.random.rand(len(available_actions))
    return zip(available_actions, action_probs)


def policy_value_fn(game: GameBase):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    available_actions = list(game.get_available_actions())
    action_probs = np.ones(len(available_actions)) / len(available_actions)
    return zip(available_actions, action_probs), 0


class StateSet(object):
    def __init__(self):
        self.state_id2node = {}  # each item is a map from state_id to TreeNode

    @staticmethod
    def state_id_act(state_id, move):
        new_id = state_id
        move_id = 1 << move
        if (move_id & state_id) != 0:
            print("WARNING: repeated move_id")
        return new_id | move_id


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, prev_move, parent, prior_p, state_id, state_set: dict):
        if parent is not None:
            self._parent = {prev_move: parent}  # map from previous move to parent node
        else:
            self._parent = {}
        self._children = {}  # a map from action to TreeNode
        self._children_n_visits = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

        self._state_id = state_id
        self._state_set = state_set
        self._prev_move = prev_move  # one of parents --prev_move--> self
        self._last_move = -1  # self --last_move--> one of children

    def add_parent(self, prev_move, parent):
        self._parent[prev_move] = parent
        # self._prev_move = prev_move

    def add_visit(self, move):
        self._n_visits += 1
        if move in self._children_n_visits.keys():
            self._children_n_visits[move] += 1

    def get_visits(self):
        return self._n_visits

    def get_path_visits(self, move):
        if move in self._children.keys():
            return self._children_n_visits[move]
        else:
            return 0

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                # get state_id of the child
                new_id = StateSet.state_id_act(self._state_id, action)
                # check if this child exists
                if new_id in self._state_set.keys():
                    node = self._state_set[new_id]
                    node.add_parent(action, self)
                    self._children[action] = node
                    self._children_n_visits[action] = 0
                else:  # else if this child dose not exist
                    self._children[action] = TreeNode(action, self, prob, new_id, self._state_set)
                    self._children_n_visits[action] = 0
                    self._state_set[new_id] = self._children[action]

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        move, child = max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))
        self._last_move = move
        child._prev_move = move
        return move, child

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self.add_visit(self._last_move)
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._prev_move in self._parent.keys():
            self._parent[self._prev_move].update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        path_visits = 0
        parent_visits = 0
        if self._prev_move in self._parent.keys():
            path_visits += self._parent[self._prev_move]._children_n_visits[self._prev_move]
            parent_visits += self._parent[self._prev_move]._n_visits

        self._u = (c_puct * self._P * np.sqrt(parent_visits) / (1 + path_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        # return self._parent is None
        return self._parent == {}


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._state_set = StateSet()
        # self._root = TreeNode(None, 1.0)
        # (prev_move, parent, prior_p, state_id, state_set: dict)
        self._root = TreeNode(-1, None, 1.0, 0, self._state_set.state_id2node)

        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state: GameBase):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)  # maximum Q + u(P)
            state.act(action)

        action_probs, _ = self._policy(state)  # equal probabilities in pure MCTS
        # Check for end of game
        if state.is_playing():
            node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state: GameBase, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = state.current_player_id
        for i in range(limit):
            if state.is_end():
                break
            action_probs = rollout_policy_fn(state)  # random in pure MCTS
            max_action = max(action_probs, key=lambda a: a[1])[0]
            state.act(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        winner = state.get_winner()
        if winner == 0:  # tie
            return 0
        else:
            return 1 if winner == player else -1

    def get_move(self, state: DotsAndBoxes):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action
        """
        # init the root node
        self._root._state_id = state.board.state_id()
        self._state_set.state_id2node[self._root._state_id] = self._root
        for n in range(self._n_playout):
            self._playout(state.copy())
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = {}
        else:
            # self._root = TreeNode(None, 1.0)
            # (prev_move, parent, prior_p, state_id, state_set: dict)
            # self._state_set.state_id2node.clear()  # FIXME: should the search graph be kept?
            self._root = TreeNode(-1, None, 1.0, 0, self._state_set.state_id2node)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(Player):
    """AI player based on MCTS"""

    def __init__(self, c_puct=5, n_playout=2000):
        super(MCTSPlayer, self).__init__()
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, game: GameBase, **kwargs):
        available_actions = game.get_available_actions()
        if len(available_actions) > 0:
            move = self.mcts.get_move(game)
            self.mcts.update_with_move(-1)
            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)
