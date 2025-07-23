
"""
2D GUI for the Quoridor game with an AI opponent, using Pygame.

- Left-click on a square (large square) -> move the pawn if the move is legal.
- Left-click on a horizontal bar between squares -> place a horizontal wall.
- Left-click on a vertical bar between squares -> place a vertical wall.
- After the player's move, the AI (red pawn) will automatically make its move.

"""

from typing import Optional, Tuple
import pygame
import numpy as np

from quoridor.QuoridorGame import QuoridorGame
from quoridor.pytorch.NNet import NNetWrapper as NNet
from MCTS import MCTS
from utils import dotdict

# === Graphics Configuration ===
CELL_SIZE = 60
WALL_THICKNESS = 16
BOARD_MARGIN = 40
SIDE_PANEL_WIDTH = 230
FPS = 60

C_BG = (245, 245, 245)
C_SQUARE = (222, 184, 135)
C_GRID = (139, 69, 19)
C_WALL = (50, 50, 50)
C_P1 = (20, 130, 255)  # Blue (Player)
C_P2 = (240, 70, 70)  # Red (AI)
C_BUTTON = (100, 200, 100)
C_BUTTON_TEXT = (255, 255, 255)
DISCLAIMER_AREA_HEIGHT = 80


def pixel_to_board(px: int, py: int, n: int) -> Tuple[int, int]:
    """Convert pixel coordinates to indices in the 2*n-1 × 2*n-1 grid."""
    unit = CELL_SIZE + WALL_THICKNESS
    if not (BOARD_MARGIN < px < BOARD_MARGIN + n * unit and BOARD_MARGIN < py < BOARD_MARGIN + n * unit):
        return -1, -1

    bx_block = (px - BOARD_MARGIN) // unit
    bx_rem = (px - BOARD_MARGIN) % unit
    by_block = (py - BOARD_MARGIN) // unit
    by_rem = (py - BOARD_MARGIN) % unit

    j = bx_block * 2 + (1 if bx_rem >= CELL_SIZE else 0)
    i = by_block * 2 + (1 if by_rem >= CELL_SIZE else 0)
    return int(i), int(j)


class QuoridorGUI:
    def __init__(self, n: int = 9):
        pygame.init()
        self.n = n
        self.game = QuoridorGame(n)
        self.board: Optional[np.ndarray] = None
        self.cur_player: Optional[int] = None
        self.game_over: bool = True
        self.winner_msg: str = ""
        self.p1_walls: int = 0
        self.p2_walls: int = 0

        # --- AI Initialization ---
        print("Loading AI model...")
        nn_args = dotdict({
            'numIters': 200,  # 1000
            'numEps': 35,  # 100
            'tempThreshold': 15,
            'updateThreshold': 0.55,
            'maxlenOfQueue': 200000,
            'numMCTSSims': 800,
            'arenaCompare': 12,
            'cpuct': 0.7,
            'training': False,
            'dirichletAlpha': 0.3,
            'epsilon': 0.0,

            'checkpoint': './temp/',
            'load_model': True,
            'load_folder_file': ('./temp', 'best.pth.tar'),
            'numItersForTrainExamplesHistory': 20,
            'start_iter': 1,
        })
        self.nnet = NNet(self.game, nn_args)
        self.nnet.load_checkpoint('./temp/', 'best.pth.tar')
        args = dotdict({
            'numMCTSSims': 100,
            'cpuct': 1,
            'training': False,
            'dirichletAlpha': 0.3,
            'epsilon': 0
        })
        self.mcts = MCTS(self.game, self.nnet, args)
        self.ai_player = lambda x: np.argmax(self.mcts.getActionProb(x, temp=0))
        print("AI is ready to play.")

        self.unit = CELL_SIZE + WALL_THICKNESS
        self.board_pixel_size = self.unit * self.n - WALL_THICKNESS + 2 * BOARD_MARGIN

        screen_width = self.board_pixel_size + SIDE_PANEL_WIDTH
        screen_height = self.board_pixel_size + DISCLAIMER_AREA_HEIGHT
        self.screen = pygame.display.set_mode((screen_width, screen_height))

        pygame.display.set_caption("Quoridor GUI – Player vs AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.panel_font = pygame.font.SysFont('Arial', 20, bold=True)
        self.disclaimer_font = pygame.font.SysFont('Arial', 18)

        button_x = self.board_pixel_size + (SIDE_PANEL_WIDTH - 150) // 2
        self.new_game_button_rect = pygame.Rect(button_x, 40, 150, 50)

        self.reset_game()

    def reset_game(self):
        """Resets the game to its initial state."""
        self.board = self.game.getInitBoard()
        self.game.board.pieces = self.board
        self.cur_player = 1
        self.game_over = False
        self.winner_msg = ""
        self.p1_walls = 10
        self.p2_walls = 10
        print("New game started.")

    def draw_board(self):
        self.screen.fill(C_BG)

        for r in range(self.n):
            for c in range(self.n):
                x_px = BOARD_MARGIN + c * self.unit
                y_px = BOARD_MARGIN + r * self.unit
                rect = pygame.Rect(x_px, y_px, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, C_SQUARE, rect)

        wall_mask = self.board[2] + self.board[3]
        visited = set()
        size = self.n * 2 - 1
        for i in range(size):
            for j in range(size):
                if not wall_mask[i][j] or (i, j) in visited:
                    continue
                if i % 2 == 1 and j % 2 == 0:
                    x_px = BOARD_MARGIN + (j // 2) * self.unit
                    y_px = BOARD_MARGIN + (i // 2) * self.unit + CELL_SIZE
                    w = CELL_SIZE * 2 + WALL_THICKNESS
                    rect = pygame.Rect(x_px, y_px - WALL_THICKNESS / 2, w, WALL_THICKNESS)
                    pygame.draw.rect(self.screen, C_WALL, rect)
                    visited.update({(i, j), (i, j + 2)})
                elif i % 2 == 0 and j % 2 == 1:
                    x_px = BOARD_MARGIN + (j // 2) * self.unit + CELL_SIZE
                    y_px = BOARD_MARGIN + (i // 2) * self.unit
                    h = CELL_SIZE * 2 + WALL_THICKNESS
                    rect = pygame.Rect(x_px - WALL_THICKNESS / 2, y_px, WALL_THICKNESS, h)
                    pygame.draw.rect(self.screen, C_WALL, rect)
                    visited.update({(i, j), (i + 2, j)})

        for idx, color in enumerate([C_P1, C_P2]):
            if np.sum(self.board[idx]) == 0:
                continue
            pos = np.argmax(self.board[idx])
            i = pos // (self.n * 2 - 1)
            j = pos % (self.n * 2 - 1)
            center_x = BOARD_MARGIN + (j // 2) * self.unit + CELL_SIZE // 2
            center_y = BOARD_MARGIN + (i // 2) * self.unit + CELL_SIZE // 2
            pygame.draw.circle(self.screen, color, (center_x, center_y), CELL_SIZE // 3)

        outer = pygame.Rect(BOARD_MARGIN - 3, BOARD_MARGIN - 3,
                            self.unit * self.n - WALL_THICKNESS + 6,
                            self.unit * self.n - WALL_THICKNESS + 6)
        pygame.draw.rect(self.screen, C_GRID, outer, 3)

    def draw_side_panel(self):
        # Draw New Game Button
        pygame.draw.rect(self.screen, C_BUTTON, self.new_game_button_rect, border_radius=10)
        btn_text = self.panel_font.render("New Game", True, C_BUTTON_TEXT)
        text_rect = btn_text.get_rect(center=self.new_game_button_rect.center)
        self.screen.blit(btn_text, text_rect)

        # Draw Wall Counters
        p1_text = self.panel_font.render(f"Human Player Walls: {self.p1_walls}", True, C_P1)
        self.screen.blit(p1_text, (self.new_game_button_rect.left, 120))

        p2_text = self.panel_font.render(f"AI Player Walls: {self.p2_walls}", True, C_P2)
        self.screen.blit(p2_text, (self.new_game_button_rect.left, 160))

    def draw_text_multiline(self, surface, text, pos, font, color, max_width):
        words = [word.split(' ') for word in text.splitlines()]
        space = font.size(' ')[0]
        x, y = pos

        for line in words:
            for word in line:
                word_surface = font.render(word, True, color)
                word_width, word_height = word_surface.get_size()
                if x + word_width >= pos[0] + max_width:
                    x = pos[0]
                    y += word_height
                surface.blit(word_surface, (x, y))
                x += word_width + space
            x = pos[0]
            y += word_height

    def draw_disclaimer(self):
        disclaimer_text = ("Disclaimer: The AI "
                           "model has completed only 18% of the planned training iterations and it's still relatively"
                           " retarded. It places too many random walls, but after that it usually manages to complete "
                           "a simple route.")
        text_color = (250, 0, 0)
        max_text_width = self.board_pixel_size - 40
        text_x = BOARD_MARGIN + 20
        board_bottom_y = self.unit * self.n - WALL_THICKNESS + BOARD_MARGIN
        text_y = board_bottom_y + 25

        self.draw_text_multiline(self.screen, disclaimer_text, (text_x, text_y), self.disclaimer_font, text_color,
                            max_text_width)

    def display_message(self, text, color=(0, 0, 0)):
        text_surf = self.font.render(text, True, color)
        text_rect = text_surf.get_rect(center=(self.board_pixel_size / 2, 20))
        self.screen.blit(text_surf, text_rect)

    def _find_action_for_click(self, bi: int, bj: int, valids: np.ndarray) -> Optional[int]:
        if bi == -1:
            return None  # Click was outside the board
        for action, is_valid in enumerate(valids):
            if not is_valid:
                continue

            if 0 <= action < 8:
                move = self.game.board.move_action_destination(action, self.cur_player)
                if move is None:
                    continue
                _, (tx, ty) = move
                if tx == bi and ty == bj:
                    return action

            elif action >= 8:
                num_wall_placements = (self.n - 1) * (self.n - 1)
                h_wall_start_index = 8
                v_wall_start_index = h_wall_start_index + num_wall_placements
                wall_i, wall_j = -1, -1
                if h_wall_start_index <= action < v_wall_start_index:
                    wall_idx = action - h_wall_start_index
                    row, col = divmod(wall_idx, (self.n - 1))
                    wall_i, wall_j = row * 2 + 1, col * 2
                elif v_wall_start_index <= action < self.game.getActionSize():
                    wall_idx = action - v_wall_start_index
                    row, col = divmod(wall_idx, (self.n - 1))
                    wall_i, wall_j = row * 2, col * 2 + 1
                if wall_i == bi and wall_j == bj:
                    return action
        return None

    def game_loop(self):
        running = True

        while running:
            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.new_game_button_rect.collidepoint(event.pos):
                        self.reset_game()
                        continue  # Skip the rest of the event processing for this click

                    # --- HUMAN'S TURN (Player 1) ---
                    if self.cur_player == 1 and not self.game_over:
                        valids = self.game.getValidMoves(self.game.getCanonicalForm(self.board, self.cur_player),
                                                         1)
                        mx, my = event.pos
                        bi, bj = pixel_to_board(mx, my, self.n)
                        action = self._find_action_for_click(bi, bj, valids)

                        if action is not None:
                            if action >= 8:
                                self.p1_walls -= 1
                            self.board, self.cur_player = self.game.getNextState(self.board, self.cur_player, action)
                            self.game.board.pieces = self.board

                            result = self.game.getGameEnded(self.board, 1)
                            if result != 0:
                                self.game_over = True
                                if result == 1:
                                    self.winner_msg = "You Won Human! (Blue)"
                                elif result == -1:
                                    self.winner_msg = "You Lost Human! AI Won (Red)"
                                else:
                                    self.winner_msg = "It's a Draw!"
                                print(f"Game Over! {self.winner_msg}")
                        else:
                            print("This move is invalid or an invalid location was clicked.")

            # --- AI'S TURN (Player -1) ---
            if self.cur_player == -1 and not self.game_over:
                self.draw_board()
                self.draw_side_panel()
                self.draw_disclaimer()
                self.display_message("AI's Turn (Red)...")
                pygame.display.flip()
                pygame.time.wait(100)

                canonical_board = self.game.getCanonicalForm(self.board, self.cur_player)
                action = self.ai_player(canonical_board)

                if action >= 8:
                    self.p2_walls -= 1
                self.board, self.cur_player = self.game.getNextState(self.board, self.cur_player, action)
                self.game.board.pieces = self.board

                result = self.game.getGameEnded(self.board, 1)
                if result != 0:
                    self.game_over = True
                    if result == 1:
                        self.winner_msg = "You Won Human! (Blue)"
                    elif result == -1:
                        self.winner_msg = "You Lost Human! AI Won (Red)"
                    else:
                        self.winner_msg = "It's a Draw!"
                    print(f"Game Over! {self.winner_msg}")

            # --- Drawing and Screen Update ---
            self.draw_board()
            self.draw_side_panel()
            self.draw_disclaimer()

            if self.game_over:
                self.display_message(f"Game Over! {self.winner_msg}", color=(180, 0, 0))
            else:
                turn_msg = "Your Turn (Blue)" if self.cur_player == 1 else "AI's Turn (Red)..."
                self.display_message(turn_msg)

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


if __name__ == "__main__":
    gui = QuoridorGUI(n=9)
    gui.game_loop()
