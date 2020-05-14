"""
The MCTS process
"""
import numpy as np
from Game import GameBase, Player
from mcts import mcts


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
    next_player = []
    for action in available_actions:
        tmp = game.copy()
        tmp.act(action)
        next_player.append(tmp.current_player_id)
    return zip(available_actions, action_probs, next_player), 0


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p, player):
        self.player = player
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob, next_player in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob, next_player)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value, player):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value, player)
        if player == self.player:
            self.update(leaf_value)
        else:
            self.update(-leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, player, c_puct=5, n_playout=10000, rollout_policy=rollout_policy_fn):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self.player = player
        self._root = TreeNode(None, 1.0, self.player)
        self._policy = policy_value_fn
        self._rollout_policy = rollout_policy
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
            action, node = node.select(self._c_puct)
            state.act(action)

        player = state.current_player_id
        # Check for end of game
        if state.is_playing():
            # the action_probs should contain the next player info
            action_probs, _ = self._policy(state)
            node.expand(action_probs)
        # Evaluate the leaf node by random rollout
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value, player)

    def _evaluate_rollout(self, state: GameBase, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        player = state.current_player_id
        state.board.scores = [0, 0, 0]
        for i in range(limit):
            if state.is_end():
                break
            action_probs = self._rollout_policy(state)
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

    def get_move(self, state: GameBase):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state
        Return: the selected action
        """
        for n in range(self._n_playout):
            self._playout(state.copy())
        return max(self._root._children.items(), key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0, self.player)

    def __str__(self):
        return "MCTS"


def greedy_dots_and_boxes_rollout_policy(state: GameBase):
    size = state.size
    board = state.board
    available_lines = state.get_available_actions().copy()
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

    available_actions = list(state.get_available_actions())
    probs = np.zeros(state.get_action_space())
    probs[choices_for_greedy] = 0.5 / len(choices_for_greedy)
    probs[available_actions] += 0.5 / len(available_actions)
    return zip(available_actions, probs[available_actions])


def greedy_dots_and_boxes_rollout_policy1(state: GameBase):
    size = state.size
    board = state.board
    available_lines = state.get_available_actions().copy()
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

    available_actions = list(state.get_available_actions())
    probs = np.zeros(state.get_action_space())
    probs[choices_for_greedy] = 0.5 / len(choices_for_greedy)
    probs[available_actions] += 0.5 / len(available_actions)
    next_players = []
    for act in available_actions:
        tmp = state.copy()
        tmp.act(act)
        next_players.append(tmp.current_player_id)
    return zip(available_actions, probs[available_actions], next_players), 0


class MCTSPlayer(Player):
    """AI player based on MCTS"""

    def __init__(self, player, c_puct=5, n_playout=2000):
        super(MCTSPlayer, self).__init__()
        self.mcts = MCTS(greedy_dots_and_boxes_rollout_policy1, player, c_puct, n_playout,
                         rollout_policy=greedy_dots_and_boxes_rollout_policy)

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


class MCTSPlayer2(Player):

    def __init__(self):
        super(MCTSPlayer2, self).__init__()
        self.mcts = mcts(timeLimit=10000)

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, game: GameBase, **kwargs):
        return self.mcts.search(initialState=game)
