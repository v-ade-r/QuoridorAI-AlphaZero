import numpy as np
from pytorch_utils.utils.progress.progress.bar import Bar
from pytorch_utils.utils import AverageMeter
import time
from MCTS import MCTS

np.set_printoptions(threshold=np.inf)


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """
    def __init__(self, player1, player2, game, args, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it. Is necessary for verbose
                     mode.

        """
        self.pnet, self.nnet = player1, player2
        self.game = game
        self.display = display
        self.args = args
        self.player1 = None
        self.player2 = None

    def _init_trees(self):
        self.pmcts = MCTS(self.game, self.pnet, self.args)
        self.nmcts = MCTS(self.game, self.nnet, self.args)

        self.player1 = lambda x: np.argmax(self.pmcts.getActionProb(x, temp=0))
        self.player2 = lambda x: np.argmax(self.nmcts.getActionProb(x, temp=0))

    def playGame(self, verbose=True):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        position_history = {}
        while self.game.getGameEnded(board, curPlayer) == 0 and it < 200:
            it += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            board_hash = hash(str(canonicalBoard))
            if board_hash in position_history:
                position_history[board_hash] += 1
                if position_history[board_hash] > 3:
                    print("Arena: Draw by 4-fold repetition!")
                    return 0
            else:
                position_history[board_hash] = 1
            if verbose:
                assert self.display
                print("Turn", str(it), "Player", str(curPlayer))
                self.display(board)
                if players[curPlayer+1].__name__ != '<lambda>':
                    self.display(board)

            action = players[curPlayer+1](self.game.getCanonicalForm(board, curPlayer))
            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)
            if valids[action] == 0:
                print(f"FATAL: Agent {curPlayer} chose an illegal move {action}")
                return -curPlayer

            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            if verbose and players[curPlayer+1].__name__ == '<lambda>':
                self.display(board)

        if verbose:
            assert self.display
            print("Game over: Turn", str(it), "Result", str(self.game.getGameEnded(board, 1)))
            if verbose and players[curPlayer+1].__name__ == '<lambda>':
                self.display(board)
            else:
                self.display(board)
            self.display(board)
        print(f"\nFInal Board state:\n {board}")
        return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num_half = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        self._init_trees()

        for _ in range(num_half):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg,
                total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        # Init new trees for MCTS, and swap players to change the starting player
        self._init_trees()
        self.player1, self.player2 = self.player2, self.player1

        for _ in range(num_half):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                eps=eps, maxeps=maxeps, et=eps_time.avg,
                total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()

        bar.finish()

        return oneWon, twoWon, draws
