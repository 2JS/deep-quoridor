import numpy as np

class QuoridorEnv:
    def __init__(self):
        self.board_size = 9
        self.num_fences = 10
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.fences = np.zeros((self.board_size - 1, self.board_size - 1), dtype=int)
        self.player_positions = [((self.board_size - 1) // 2, 0), ((self.board_size - 1) // 2, self.board_size - 1)]
        self.player_fences = [self.num_fences, self.num_fences]
        self.current_player = 0
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return self.board.copy(), self.fences.copy(), tuple(self.player_positions), tuple(self.player_fences), self.current_player

    def is_valid_move(self, move):
        x1, y1 = self.player_positions[self.current_player]
        x2, y2 = move

        if not (0 <= x2 < self.board_size and 0 <= y2 < self.board_size):
            return False

        dx, dy = x2 - x1, y2 - y1
        if abs(dx) + abs(dy) > 2:
            return False

        opponent_pos = self.player_positions[1 - self.current_player]
        if abs(dx) + abs(dy) == 1:  # Normal move or move after jumping over the opponent's pawn
            if x1 == opponent_pos[0] and y1 == opponent_pos[1] - 1 and dy == 1:
                if self.fences[y1, x1] & 1:  # There's a fence behind the opponent's pawn
                    return False
            elif x1 == opponent_pos[0] and y1 == opponent_pos[1] + 1 and dy == -1:
                if self.fences[y1 - 1, x1] & 1:
                    return False
            elif y1 == opponent_pos[1] and x1 == opponent_pos[0] - 1 and dx == 1:
                if self.fences[y1, x1] & 2:
                    return False
            elif y1 == opponent_pos[1] and x1 == opponent_pos[0] + 1 and dx == -1:
                if self.fences[y1, x1 - 1] & 2:
                    return False
            else:  # Normal move
                if dx == 1:
                    if self.fences[y1, x1] & 2:
                        return False
                    if 0 <= y1 - 1 < self.board_size - 1 and self.fences[y1 - 1, x1] & 2:
                        return False
                elif dx == -1:
                    if self.fences[y1, x1 - 1] & 2:
                        return False
                    if 0 <= y1 - 1 < self.board_size - 1 and self.fences[y1 - 1, x1 - 1] & 2:
                        return False
                elif dy == 1:
                    if self.fences[y1, x1] & 1:
                        return False
                    if 0 <= x1 - 1 < self.board_size - 1 and self.fences[y1, x1 - 1] & 1:
                        return False
                elif dy == -1:
                    if self.fences[y1 - 1, x1] & 1:
                        return False
                    if 0 <= x1 - 1 < self.board_size - 1 and self.fences[y1 - 1, x1 - 1] & 1:
                        return False

        else:  # Jumping over the opponent's pawn
            if (x1 + x2) // 2 != opponent_pos[0] or (y1 + y2) // 2 != opponent_pos[1]:
                return False

            if dx == 2:
                if self.fences[y1, x1] & 2 or self.fences[y1, x1 + 1] & 2:
                    return False
            elif dx == -2:
                if self.fences[y1, x1 - 1] & 2 or self.fences[y1, x1 - 2] & 2:
                    return False
            elif dy == 2:
                if self.fences[y1, x1] & 1 or self.fences[y1 + 1, x1] & 1:
                    return False
            elif dy == -2:
                if self.fences[y1 - 1, x1] & 1 or self.fences[y1 - 2, x1] & 1:
                    return False

        return True

    def is_valid_fence_placement(self, fence_pos, orientation):
        x, y = fence_pos
        if not (0 <= x < self.board_size - 1 and 0 <= y < self.board_size - 1):
            return False

        if self.player_fences[self.current_player] == 0:
            return False

        if orientation == 0:
            if self.fences[y, x] & 1:
                return False
            if x > 0 and self.fences[y, x - 1] & 1:
                return False
            if x < self.board_size - 2 and self.fences[y, x + 1] & 1:
                return False
        else:
            if self.fences[y, x] & 2:
                return False
            if y > 0 and self.fences[y - 1, x] & 2:
                return False
            if y < self.board_size - 2 and self.fences[y + 1, x] & 2:
                return False

        return True

    def step(self, action):
        if not self.done:
            if len(action) == 2:  # Move pawn
                if self.is_valid_move(action):
                    # Update player position
                    self.player_positions[self.current_player] = action

                    # Check if player has reached the opposite side
                    if (self.current_player == 0 and action[1] == self.board_size - 1) or (
                            self.current_player == 1 and action[1] == 0):
                        self.done = True
                        self.winner = self.current_player

            elif len(action) == 3:  # Place fence
                x, y, orientation = action
                fence_pos = y, x
                if self.is_valid_fence_placement(fence_pos, orientation):
                    # Update fence
                    if orientation == 0:  # horizontal
                        self.fences[fence_pos] = 1
                    else:  # vertical
                        self.fences[fence_pos] = 2

                    # Decrease the number of fences remaining
                    self.player_fences[self.current_player] -= 1

            # Switch to the next player
            self.current_player = 1 - self.current_player

        return self.get_state(), self.done

def print_state(state):
    board, fences, player_positions, player_fences, current_player = state
    print("Board:")
    for y in range(board.shape[0]):
        row = []
        for x in range(board.shape[1]):
            if (x, y) == player_positions[0]:
                row.append("P1")
            elif (x, y) == player_positions[1]:
                row.append("P2")
            else:
                row.append("--")
        print(" ".join(row))
    print("\nFences:")
    print(fences)
    print("\nPlayer positions:", player_positions)
    print("Player fences:", player_fences)
    print("Current player:", current_player + 1)

def parse_action(action_str):
    parts = action_str.strip().split()
    if len(parts) == 2:
        x, y = int(parts[0]), int(parts[1])
        return x, y
    elif len(parts) == 3:
        x, y, o = int(parts[0]), int(parts[1]), int(parts[2])
        return x, y, o
    else:
        raise ValueError("Invalid action format")

if __name__ == "__main__":
    env = QuoridorEnv()
    state = env.reset()
    print_state(state)

    while not env.done:
        # try:
        action_str = input("Enter your action (x y) for move or (x y o) for fence placement: ")
        action = parse_action(action_str)
        if len(action) == 2 and not env.is_valid_move(action):
            print("Invalid move. Please try again.")
        elif len(action) == 3 and not env.is_valid_fence_placement(action[:-1], action[-1]):
            print("Invalid move. Please try again.")
        else:
            state, done = env.step(action)
            print_state(state)
        # except ValueError as e:
        #     print(e)
        # except Exception as e:
        #     print("Error:", e)
        #     break
    print("Game over! Winner is Player", env.winner + 1)
