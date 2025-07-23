import math
import numpy as np

EPS = 1e-8


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.sH = {}

        h, w = self.game.getBoardSize()
        self.visit_count_p1 = np.zeros((h, w), dtype=int)
        self.visit_count_p2 = np.zeros((h, w), dtype=int)

        self.dirichlet_alpha = getattr(self.args, "dirichletAlpha")
        self.dirichlet_epsilon = getattr(self.args, "epsilon")

    def getActionProb(self, canonicalBoard, temp=1):
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa.get((s, a), 0)
                  for a in range(self.game.getActionSize())]

        if sum(counts) == 0:
            print("Sum == 0")
            valids = self.game.getValidMoves(canonicalBoard, 1)
            print(f"Valids:\n {valids}")
            total_valid = np.sum(valids)
            return [v/total_valid for v in valids]

        if temp == 0:
            bestA = int(np.argmax(counts))
            probs = [0]*len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1./temp) for x in counts]
        s_counts = float(sum(counts))
        return [x/s_counts for x in counts]

    def search(self, canonicalBoard, counter=0):
        self.visit_count_p1 += canonicalBoard[0].astype(int)
        self.visit_count_p2 += canonicalBoard[1].astype(int)
        if counter == 0:
            self.sH = {}

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)

            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                print("All valid moves were masked, do workaround.")
                self.Ps[s] = valids / np.sum(valids)

            # adding Dirichlet noise in a root
            if counter == 0:
                valid_indices = np.where(valids == 1)[0]
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_indices))
                dir_noise = np.zeros_like(self.Ps[s])
                dir_noise[valid_indices] = noise
                self.Ps[s] = (1 - self.dirichlet_epsilon) * self.Ps[s] + self.dirichlet_epsilon * dir_noise
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # if not a root
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        for a in range(self.game.getActionSize()):
            if not valids[a]:
                continue
            if (s, a) in self.Qsa:
                u = (self.Qsa[(s, a)]
                     + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)]))
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
            if u > cur_best:
                cur_best = u
                best_act = a

        if s in self.sH or counter > 256:
            return 0
        self.sH[s] = 1

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)
        v = self.search(next_s, counter + 1)
        if v == 0:
            if self.Ns[s] > 0:
                self.Ns[s] -= 1
            return 0

        # update
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)]*self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v
