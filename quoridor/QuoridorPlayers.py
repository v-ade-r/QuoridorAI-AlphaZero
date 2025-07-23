import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanQuoridorPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        # print(valid)

        while True:
            try:
                a_dic = {'u': 0, 'd': 1, 'l': 2, 'r': 3, 'ul': 4, 'dr': 5, 'dl': 6, 'ur': 7}
                a = input()
                if a in a_dic:
                    a = a_dic[a]
                    if valid[a]:
                        break
                    else:
                        print('Invalid move, try again.')
                        continue
                elif a in ['h', 'v']:
                    b = input()
                    try:
                        x, y = [int(x) for x in b.split(' ')]
                    except ValueError:
                        print("Please enter two numbers separated by a space.")
                        continue
                    if a == 'h':
                        action = self.game.board.index_of_action(8, x * 2 - 1, y * 2)
                    else:  # a == 'v'
                        action = self.game.board.index_of_action(9, x * 2, y * 2 + 1)
                    if action < 0 or action >= len(valid) or not valid[action]:
                        print('Invalid wall placement, try again.')
                        continue
                    a = action
                    break
                else:
                    print("Invalid direction. Try again.")
            except ValueError:
                print(
                    "Invalid input.\n"
                    "Valid moves are: (u, d, r, l, ul, dr, dl, ur).\n"
                    "Valid walls are: First type h or v (for horizontal or vertical), press enter, "
                    "then type numbers separated by a space for the wall coordinates.\n"
                )
                continue
        return a


class GreedyQuoridorPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a
