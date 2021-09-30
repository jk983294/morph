import math
from typing import Dict, List
import numpy as np
import torch.nn.functional as F
from rl.rcts.c4 import r4_game as game
from rl.rcts.c4 import r4_model as model


class NodeStat(object):
    def __init__(self):
        """record edge (s, a) stats"""
        self.visit_count: List[float] = []  # state_int -> [N(s, a)]
        self.value: List[float] = []  # state_int -> [W(s, a)] total action-value
        self.value_avg: List[float] = []  # state_int -> [Q(s, a)] mean action-value
        self.probs: List[float] = []  # state_int -> [P(s,a)], prior probability of actions

    def update_stat(self, action, cur_value):
        self.visit_count[action] += 1
        self.value[action] += cur_value
        self.value_avg[action] = self.value[action] / self.visit_count[action]

    def calc_visit_count_sqrt(self):
        return math.sqrt(sum(self.visit_count))

    def init_root_prob(self):
        """
        root node of the search process has an extra noise added to the probabilities to
        improve the exploration of the search process
        """
        noises = np.random.dirichlet([0.03] * game.GAME_COLS)
        self.probs = [0.75 * prob + 0.25 * noise for prob, noise in zip(self.probs, noises)]

    def choose_action(self, c_puct, cur_state):
        """action utility, which is a sum between Q(s, a) and the prior probabilities scaled to the visit count"""
        visit_count_sqrt = self.calc_visit_count_sqrt()
        action_score = [value + c_puct * prob * visit_count_sqrt / (1 + count)
                        for value, prob, count in zip(self.value_avg, self.probs, self.visit_count)]
        invalid_actions = set(range(game.GAME_COLS)) - set(game.possible_moves(cur_state))
        for invalid in invalid_actions:
            action_score[invalid] = -np.inf
        return int(np.argmax(action_score))

    def init(self, prob):
        self.visit_count = [0] * game.GAME_COLS
        self.value = [0.0] * game.GAME_COLS
        self.value_avg = [0.0] * game.GAME_COLS
        self.probs = prob

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.value_avg.clear()
        self.probs.clear()


class MCTS:
    """
    Monte-Carlo Tree Search, keeps statistics for every state encountered during the search
    """
    def __init__(self, c_puct=1.0):
        self.c_puct = c_puct  # for node selection process
        self.state2stat: Dict[int, NodeStat] = {}  # state_int -> [N(s, a)]

    def clear(self):
        self.state2stat.clear()

    def __len__(self):
        return len(self.state2stat)

    def get_stat(self, state_int) -> NodeStat:
        if state_int not in self.state2stat:
            st = NodeStat()
            self.state2stat[state_int] = st
            return st
        else:
            return self.state2stat[state_int]

    def find_leaf(self, root_state_, player: int):
        """
        Traverse the tree until the end of game or leaf node.
        keeping walking down until we reach the final game state or a yet unexplored leaf has been found.

        :param root_state_: root node state
        :param player: player to move
        :return: tuple of (value, leaf_state, player, states, actions)
            @value: the game outcome for the player at leaf or None if the final state hasn't been reached
            @leaf_state: state_int of the last state
            @player: player at the leaf node
            @states: list of states traversed
            @actions: list of actions taken
        """
        states = []  # keep track of the visited states and the executed actions, update the nodes' statistics later
        actions = []
        cur_state = root_state_
        cur_player: int = player
        value = None

        while not self.is_leaf(cur_state):
            states.append(cur_state)
            st = self.get_stat(cur_state)

            # choose action to take
            if cur_state == root_state_:
                st.init_root_prob()

            action = st.choose_action(self.c_puct, cur_state)
            actions.append(action)
            state_new, won = game.move(cur_state, action, cur_player)
            # print('find_leaf player {}, {} -> {} take action {}, get {}'.format(cur_player, cur_state, state_new, action, won))
            cur_state = state_new
            if won:
                # if somebody won the game, the value of the final state is -1 (as it is on opponent's turn)
                value = -1.0
            cur_player = int(1 - cur_player)
            # check for the draw
            if value is None and len(game.possible_moves(cur_state)) == 0:
                value = 0.0
        return value, cur_state, cur_player, states, actions

    def is_leaf(self, state_int):
        """
        The final game states (win, lose, or draw) are never added to the MCTS statistics,
        so they will always be leaf nodes
        """
        return state_int not in self.state2stat

    def search_batch(self, count, batch_size, state_int, player, net: model.R4Net, device="cpu"):
        for _ in range(count):
            self.search_minibatch(batch_size, state_int, player, net, device)

    def search_minibatch(self, batch_size, state_int, player, net: model.R4Net, device="cpu"):
        backup_queue = []
        expand_states = []
        expand_players = []
        expand_queue = []
        planned_states = set()
        for _ in range(batch_size):
            value, leaf_state, leaf_player, states, actions = self.find_leaf(state_int, player)
            if value is not None:  # final game state
                backup_queue.append((value, states, actions))
                # print('backup_queue append final value=%d, player=%d' % (value, player))
            else:  # store the leaf for later expansion
                if leaf_state not in planned_states:
                    planned_states.add(leaf_state)
                    leaf_state_lists = game.decode_binary(leaf_state)
                    expand_states.append(leaf_state_lists)
                    expand_players.append(leaf_player)
                    expand_queue.append((leaf_state, states, actions))

        # do expansion of nodes
        if expand_queue:
            """
            requires the NN to be used to get the prior probabilities of the actions and the estimated game value.
            """
            batch_v = model.state_lists_to_batch(expand_states, expand_players, device)
            logits_v, values_v = net.forward(batch_v)
            probs_v = F.softmax(logits_v, dim=1)
            values = values_v.data.cpu().numpy()[:, 0]
            probs = probs_v.data.cpu().numpy()

            # create the nodes
            for (leaf_state, states, actions), value, prob in zip(expand_queue, values, probs):
                leaf_st = self.get_stat(leaf_state)
                leaf_st.init(prob)
                backup_queue.append((value, states, actions))
                # print('expand backup_queue append leaf_state=%d, player=%d' % (leaf_state, player))

        # perform backup of the searches
        for value, states, actions in backup_queue:
            # leaf state is not stored in states and actions, so the value of the leaf will be the value of the opponent
            cur_value = -value
            for state_int, action in zip(states[::-1], actions[::-1]):
                # print('update_stat state=%d, action=%d, val=%f' % (state_int, action, cur_value))
                st = self.get_stat(state_int)
                st.update_stat(action, cur_value)
                cur_value = -cur_value

    def get_policy_value(self, state_int, tau=1):
        """
        Extract policy and action-values by the state
        :param state_int: state of the board
        :param tau: state of the board
        :return: (probs, values)
        """
        st = self.get_stat(state_int)
        if tau == 0:  # the selection becomes deterministic, as we select the most frequently visited action
            probs = [0.0] * game.GAME_COLS
            probs[np.argmax(st.visit_count)] = 1.0
        else:  # distribution improves exploration
            st.visit_count = [count ** (1.0 / tau) for count in st.visit_count]
            total = sum(st.visit_count)
            probs = [count / total for count in st.visit_count]
        values = st.value_avg
        return probs, values
