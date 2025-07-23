from collections import deque
from Arena import Arena
from MCTS import MCTS
from utils import dotdict
import numpy as np
from pytorch_utils.utils.progress.progress.bar import Bar
from pytorch_utils.utils import AverageMeter
import time
import os
import sys
from pickle import Pickler, Unpickler
from random import shuffle

STARTING_NUM_SIMS = 5
MIDDLE_NUM_SIMS = 250
EARLY_NUM_EPISODES = 100


def display(board):
    n = board.shape[1]
    for i in range(n):
        line = ""
        for j in range(n):
            if board[0][i][j] == 1:
                line += "1"
            elif board[1][i][j] == 1:
                line += "2"
            elif board[2][i][j] == 1:
                line += "-"     # wall sign for player 1, - not to be mistaken for a horizontal wall.
            elif board[3][i][j] == 1:
                line += "|"     # wall sign for player 2 - not to be mistaken for a vertical wall.
            elif i % 2 == 0 and j % 2 == 0:
                line += "."
            else:
                line += " "
        print(line)
    print()


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.pnet = self.nnet.__class__(self.game, self.args)  # the competitor network
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overridden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Add Dirichlet noise to each root, as a new MCTS is created at every step.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        position_history = {}

        while True and episodeStep < 200:
            print(f"episode step: {episodeStep}")
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)

            # Check for position repetition (anti-loop mechanism)
            board_hash = hash(str(canonicalBoard))
            if board_hash in position_history:
                position_history[board_hash] += 1
                if position_history[board_hash] > 3:
                    print("Draw by repetition - 4-fold repetition detected")
                    # Return examples with penalty for repetition
                    return [(x[0], x[2], 0) for x in trainExamples]
            else:
                position_history[board_hash] = 1

            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)

            if np.sum(pi) == 0:
                print("End of the game, PI equals 0")
                break

            action = np.random.choice(len(pi), p=pi)
            trainExamples.append([canonicalBoard, self.curPlayer, pi, None])
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            r = self.game.getGameEnded(board, self.curPlayer)
            if r != 0:
                print(f"Game ended player {r} won! Board state:")
                self.game.print_board(board)
                return [(x[0],x[2],r*x[1]) for x in trainExamples]
        print("Game not resolveds! Board state:")
        self.game.print_board(board)
        return [(x[0], x[2], 0) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """
        original_sims = self.args.numMCTSSims
        original_eps = self.args.numEps
        for i in range(self.args.start_iter, self.args.numIters+1):
            print('------ITER ' + str(i) + '------')
            if i == 1:
                self.args.numMCTSSims = STARTING_NUM_SIMS
                self.args.numEps = EARLY_NUM_EPISODES
                print(f"ITERATION {i}: Temporarily setting numMCTSSims to "
                      f"{self.args.numMCTSSims} for a quick kickstart.")
                print(f"ITERATION {i}: Temporarily setting numEPS to {self.args.numEps}.")
            elif 1 < i < 4:
                self.args.numMCTSSims = MIDDLE_NUM_SIMS
                self.args.numEps = EARLY_NUM_EPISODES
                print(f"ITERATION {i}: Setting numMCTSSims to {self.args.numMCTSSims}.")
                print(f"ITERATION {i}: Temporarily setting numEPS to {self.args.numEps}.")
            else:
                self.args.numMCTSSims = original_sims
                self.args.numEps = original_eps
                print(f"ITERATION {i}: Setting numMCTSSims to {self.args.numMCTSSims}.")
                print(f"ITERATION {i}: Setting numEPS to {self.args.numEps}.")
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.numEps)
                end = time.time()

                for eps in range(self.args.numEps):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    epsiode_example = self.executeEpisode()
                    iterationTrainExamples += epsiode_example
                    vc1 = self.mcts.visit_count_p1
                    vc2 = self.mcts.visit_count_p2

                    def print_pawn_matrix(mat):
                        pawn_mat = mat[::2, ::2]
                        for row in pawn_mat:
                            print(row)

                    # Printing number of visits for each board cell.
                    print("=== Visit counts for player 1 ===")
                    print_pawn_matrix(vc1)
                    print("\n=== Visit counts for player -1 ===")
                    print_pawn_matrix(vc2)

                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix  = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        eps=eps+1, maxeps=self.args.numEps, et=eps_time.avg,
                        total=bar.elapsed_td, eta=bar.eta_td)
                    bar.next()
                bar.finish()

                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainExamplesHistory.pop(0)
            self.saveTrainExamples(i-1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            print(f"Number of train examples: \n{len(trainExamples)}")

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            arena_args = dotdict({**self.args})
            arena_args.epsilon = 0.0

            self.nnet.train(trainExamples)
            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(self.pnet, self.nnet, self.game, arena_args, display=display)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare, verbose=True)

            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) >= self.args.updateThreshold:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            else:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration)+".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)

    def loadTrainExamples(self):
        folder, model_filename = self.args.load_folder_file
        examplesFile = ""
        try:
            base_name = model_filename.split('.pth.tar')[0]
            iteration_num = int(base_name.split('_')[1])
            examples_iteration = iteration_num - 1

            if examples_iteration < 0:
                raise FileNotFoundError

            examples_filename = self.getCheckpointFile(examples_iteration) + ".examples"
            examplesFile = os.path.join(folder, examples_filename)

        except (ValueError, IndexError, FileNotFoundError):
            print(f"Could not determine a valid example file for {model_filename}.")

        if os.path.isfile(examplesFile):
            print(f"File with trainExamples found: {examplesFile}. Reading it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            self.skipFirstSelfPlay = True
        else:
            print(f"File with trainExamples not found at path: {examplesFile}")
            r = input("Continue without loading examples? [y|n]")
            if r != "y":
                sys.exit()
