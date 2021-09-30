#!/usr/bin/env python3
import numpy as np
from rl.rcts.c4 import r4_game as game
from rl.rcts.c4 import r4_mcts as mcts
from rl.rcts.c4 import r4_model as model
import torch

from rl.rcts.c4.r4_game import decode_binary

rounds = 2
MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 8


def net_against_net(nets):
    wins, losses, draws = 0, 0, 0
    for _ in range(rounds):
        r, _ = model.play_game(mcts_stores=None, replay_buffer=None, net1=nets[0], net2=nets[1], steps_before_tau_0=0,
                               mcts_searches=MCTS_SEARCHES, mcts_batch_size=MCTS_BATCH_SIZE, device=device)
        if r > 0.5:
            wins += 1
        elif r < -0.5:
            losses += 1
        else:
            draws += 1
    print("w=%d, l=%d, d=%d" % (wins, losses, draws))


def net_against_human(net):
    human_player_idx = 1
    cur_player = 0
    result = None
    state = game.INITIAL_STATE
    mcts_store = mcts.MCTS()
    while result is None:
        if human_player_idx == cur_player:
            action = int(input("enter move action: "))
        else:
            mcts_store.search_batch(MCTS_SEARCHES, MCTS_BATCH_SIZE, state, cur_player, net, device=device)
            probs, _ = mcts_store.get_policy_value(state, tau=0)
            action = np.random.choice(game.GAME_COLS, p=probs)
        if action not in game.possible_moves(state):
            print("Impossible action selected")
        state_new, won = game.move(state, action, cur_player)
        print('player {}, {} -> {} take action {}, get {}'.format(cur_player, state, state_new, action, won))
        print(decode_binary(state_new))
        state = state_new
        if won:
            print('player {} win'.format(cur_player))
            break
        cur_player = 1 - cur_player

        # check the draw case
        if len(game.possible_moves(state)) == 0:
            print('player {} draw'.format(cur_player))
            break


if __name__ == "__main__":
    human_play = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = ["saves/r4/best_001_00400.dat", "saves/r4/best_093_50100.dat"]

    nets = []
    for fname in models:
        net = model.R4Net(model.OBS_SHAPE, game.GAME_COLS)
        net.load_state_dict(torch.load(fname, map_location=lambda storage, loc: storage))
        net = net.to(device)
        nets.append(net)

    if human_play:
        net_against_human(nets[1])
    else:
        net_against_net(nets)

