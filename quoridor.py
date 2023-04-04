from collections import deque
import random

# import numpy as np
import torch
import torch.nn.functional as F


class ActionError(Exception):
    pass


class QuoridorEnv:
    def __init__(self):
        self.board_size = 9
        self.num_fences = 10
        self.reset()

    @torch.no_grad()
    def reset(self):
        self.board = torch.zeros(self.board_size, self.board_size, dtype=torch.int8)
        self.fences = torch.zeros(
            (self.board_size - 1, self.board_size - 1), dtype=torch.int8
        )
        self.player_positions = [
            ((self.board_size - 1) // 2, 0),
            ((self.board_size - 1) // 2, self.board_size - 1),
        ]
        self.fence_counts = [self.num_fences, self.num_fences]
        self.current_player = 0
        self.board[self.player_positions[0]] = 1
        self.board[self.player_positions[1]] = 2
        self.done = False
        return self.get_state()

    def get_state(self):
        # Returns the state as a tuple containing the current player,
        # the board, the fence positions, and the remaining fence counts
        return self.current_player, self.board.detach().clone(), self.fences.detach().clone(), tuple(self.fence_counts)

    @torch.no_grad()
    def move_pawn(self, player, new_position):
        if self.is_valid_move(player, new_position):
            self.board[self.player_positions[player]] = 0
            self.board[new_position] = player + 1
            self.player_positions[player] = new_position
            return

        raise ActionError(f"Invalid move from {self.player_positions[player]} to {new_position}")

    @torch.no_grad()
    def place_fence(self, player, fence_position, fence_orientation):
        if self.is_valid_fence_placement(player, fence_position, fence_orientation):
            x, y = fence_position
            if fence_orientation == "h":
                self.fences[x, y] |= 1
            else:
                self.fences[x, y] |= 2
            self.fence_counts[player] -= 1
            return

        raise ActionError(f"Invalid fence placement at {fence_position} with orientation {fence_orientation}")

    @torch.no_grad()
    def step(self, action):
        # action is a tuple: (action_type, action_data)
        # action_type is either 'move' or 'fence'
        # action_data is either a new_position for 'move' or (fence_position, fence_orientation) for 'fence'
        action_type, action_data = action

        if action_type == "move":
            self.move_pawn(self.current_player, action_data)
            if self.has_won(self.current_player):
                self.done = True
                return self.get_state(), 100, True
        elif action_type == "fence":
            self.place_fence(self.current_player, *action_data)

        self.current_player = 1 - self.current_player

        return self.get_state(), 0, False

    @torch.no_grad()
    def is_blocked(self, position, direction):
        x, y = position
        dx, dy = direction
        nx, ny = x + dx, y + dy

        assert 0 <= x < self.board_size and 0 <= y < self.board_size
        assert abs(dx) + abs(dy) == 1

        if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
            return True

        if dx != 0:
            if y < self.board_size - 1 and self.fences[min(x, x + dx), y] & 2 != 0:
                return True
            if y > 0 and self.fences[min(x, x + dx), y - 1] & 2 != 0:
                return True
        else:
            if x < self.board_size - 1 and self.fences[x, min(y, y + dy)] & 1 != 0:
                return True
            if x > 0 and self.fences[x - 1, min(y, y + dy)] & 1 != 0:
                return True

        return False

    # Check if the given move is valid
    @torch.no_grad()
    def is_valid_move(self, player, new_position, check_opponent=True):
        opponent = 2 - player
        x, y = self.player_positions[player]
        nx, ny = new_position
        dx, dy = nx - x, ny - y

        if not (0 <= nx < self.board_size and 0 <= ny < self.board_size):
            return False

        if abs(dx) > 2 or abs(dy) > 2:
            return False

        if abs(dx) + abs(dy) == 1:
            if self.is_blocked((x, y), (dx, dy)):
                return False

        if not check_opponent:
            return True

        if self.board[new_position] != 0:
            return False

        if abs(dx) == 2 or abs(dy) == 2:  # Jump over opponent's pawn
            if self.board[(x + nx) // 2, (y + ny) // 2] != opponent:
                return False
            if self.is_blocked((x, y), (dx // 2, dy // 2)):
                return False
            if self.is_blocked((nx, ny), (-dx // 2, -dy // 2)):
                return False

        elif abs(dx) == 1 and abs(dy) == 1:  # Diagonal move
            if self.board[x, ny] == opponent:
                if self.is_blocked((x, y), (0, dy)):
                    return False
                if self.is_blocked((x, ny), (dx, 0)):
                    return False
                if not self.is_blocked((x, ny), (0, dy)):
                    return False
                return True

            elif self.board[nx, y] == opponent:
                if self.is_blocked((x, y), (dx, 0)):
                    return False
                if self.is_blocked((nx, y), (0, dy)):
                    return False
                if not self.is_blocked((nx, y), (dx, 0)):
                    return False
                return True

            return False

        return True

    # Check if the given fence placement is valid
    @torch.no_grad()
    def is_valid_fence_placement(self, player, fence_position, fence_orientation):
        if self.fence_counts[player] == 0:
            return False

        x, y = fence_position

        if not (0 <= x < self.board_size - 1 and 0 <= y < self.board_size - 1):
            return False

        if self.fences[x, y] != 0:
            return False

        if fence_orientation == "h":
            if (
                self.fences[x, y] & 1 != 0
                or (x > 0 and self.fences[x - 1, y] & 1 != 0)
                or (x < self.board_size - 2 and self.fences[x + 1, y] & 1 != 0)
            ):
                return False
        else:
            if (
                self.fences[x, y] & 2 != 0
                or (y > 0 and self.fences[x, y - 1] & 2 != 0)
                or (y < self.board_size - 2 and self.fences[x, y + 1] & 2 != 0)
            ):
                return False

        self.fences[x, y] |= 1 if fence_orientation == "h" else 2
        paths_exist = self.paths_exist()
        self.fences[x, y] &= ~(1 if fence_orientation == "h" else 2)

        return paths_exist

    # Check if there is a path from the player's current position to the opposite side
    @torch.no_grad()
    def paths_exist(self):
        checked = []
        for player in (0, 1):
            visited = torch.zeros((self.board_size, self.board_size), dtype=bool)
            queue = deque([self.player_positions[player]])
            visited[self.player_positions[player]] = True

            while queue:
                x, y = queue.popleft()

                if player == 0 and y == self.board_size - 1 or player == 1 and y == 0:
                    checked.append(player)
                    break

                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if not self.is_blocked((x, y), (dx, dy)) and not visited[nx, ny]:
                        visited[nx, ny] = True
                        queue.append((nx, ny))
            else:
                return False

        return True

    @torch.no_grad()
    def sample_action(self, valid_only=True):
        if self.fence_counts[self.current_player] > 0 and random.random() < 0.5:
            x, y = self.player_positions[self.current_player]

            moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 2), (0, -2), (2, 0), (-2, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

            moves = [(x + dx, y + dy) for dx, dy in moves if self.is_valid_move(self.current_player, (x + dx, y + dy))]

            return ("move", random.choice(moves))
        else:
            fences = [
                ((x, y), o)
                for x in range(self.board_size - 1)
                for y in range(self.board_size - 1)
                for o in ("h", "v")
            ]

            if not valid_only:
                return ("fence", random.choice(fences))

            while True:
                fence = random.choice(fences)
                if self.is_valid_fence_placement(self.current_player, *fence):
                    return ("fence", fence)

    # Check if the player has reached the opposite side
    @torch.no_grad()
    def has_won(self, player):
        _, y = self.player_positions[player]
        return y == (self.board_size - 1 if player == 0 else 0)

    @torch.no_grad()
    def is_game_over(self):
        for player in range(2):
            if self.has_won(player):
                return True
        return False

    @torch.no_grad()
    def get_winner(self):
        for player in range(2):
            if self.has_won(player):
                return player
        return None

    @torch.no_grad()
    def print_board(self):
        h_fences = F.pad(self.fences, (0, 0, 1, 1)) & 1
        v_fences = F.pad(self.fences, (1, 1, 0, 0)) & 2

        h_fences = h_fences[1:, :] | h_fences[:-1, :]
        v_fences = v_fences[:, 1:] | v_fences[:, :-1]

        print("  0   1   2   3   4   5   6   7   8")

        for y in range(self.board_size):
            row = [f'{y}']
            for x in range(self.board_size):
                if self.board[x, y] == 1:
                    row.append("1")
                elif self.board[x, y] == 2:
                    row.append("2")
                else:
                    row.append(" ")
                if x < self.board_size - 1:
                    row.append("|" if v_fences[x, y] else " ")
            print(" ".join(row))
            if y < self.board_size - 1:
                fence_row = [' ']
                for x in range(self.board_size):
                    fence_row.append("-" if h_fences[x, y] else " ")
                    if x < self.board_size - 1:
                        fence_row.append("+")
                print(" ".join(fence_row))
        print()


# Usage example:
if __name__ == "__main__":
    env = QuoridorEnv()
    env.place_fence(0, (4, 4), "h")
    env.print_board()
