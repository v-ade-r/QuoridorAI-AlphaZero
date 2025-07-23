from Coach import Coach
from quoridor.QuoridorGame import QuoridorGame as Game
from quoridor.pytorch.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 200,    # 1000
    'numEps': 35,   # 100
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 800,     # 250
    'arenaCompare': 12,
    'cpuct': 1,
    'training': True,
    'dirichletAlpha': 0.3,
    'epsilon': 0.25,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('./temp', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'start_iter': 1,
})

if __name__ == "__main__":
    BOARD_SIZE = 9
    g = Game(BOARD_SIZE)
    nnet = nn(g, args)
    print("Game size official:", (BOARD_SIZE, BOARD_SIZE))
    print("Game size with walls:", g.getBoardSize())
    print("Game action size:", g.getActionSize())
    print("NN action size:", nnet.action_size)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
