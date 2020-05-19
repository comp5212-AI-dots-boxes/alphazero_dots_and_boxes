from Game import GameManager
from dots_and_boxes import DotsAndBoxes
from dots_and_boxes.players import GreedyPlayer, MoreGreedyPlayer, MCTSPlayer as SmartMCTSPlayer, StagedMCTSPlayer

if __name__ == '__main__':
    p1 = MoreGreedyPlayer()
    # p2 = StagedMCTSPlayer(3, 2, '3x3-beat-greedy.model', '3x3-stage2.model', n_playout=400)
    p2 = SmartMCTSPlayer(3, player=2, model='3x3-best.model', n_playout=400)
    game = DotsAndBoxes(3)
    gm = GameManager(game, [None, p1, p2], 1)
    win_record = [0, 0, 0]
    for round in range(1, 11):
        print('round %d' % round)
        gm.play(verbose=1)
        winner = gm.game.get_winner()
        # 3x3 game will never get a tie
        print('round %d winner is P%d' % (round, winner))
        win_record[winner] += 1
        print('current MCTS player winning rate is %3f' % (win_record[2] / round))

        gm.game.reset()
    print('Greedy : staged-MCTS = %d : %d' % (win_record[1], win_record[2]))
