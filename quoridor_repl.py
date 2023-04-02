import random
from quoridor import QuoridorEnv

def human_input_to_move(human_input):
    tokens = human_input.split()
    if len(tokens) == 2:
        x, y = map(int, tokens)
        return (x, y)
    elif len(tokens) == 3:
        x, y, orientation = tokens
        x, y = int(x), int(y)
        return x, y, orientation
    else:
        raise ValueError("Invalid input format")

def main():
    env = QuoridorEnv()
    player_turn = 0

    while not env.is_game_over():
        print("\nCurrent board:")
        env.print_board()
        print(f"Player {player_turn + 1}'s turn")

        # if player_turn == 0:  # Human player
        while True:
            try:
                human_input = input("Enter your move (x y) or fence (x y h/v): ").strip()
                move = human_input_to_move(human_input)
                if isinstance(move, tuple) and len(move) == 2:  # Pawn move
                    if env.is_valid_move(player_turn, move):
                        env.move_pawn(player_turn, move)
                        break
                elif len(move) == 3:  # Fence placement
                    x, y, orientation = move
                    if env.is_valid_fence_placement(player_turn, (x, y), orientation):
                        env.place_fence(player_turn, (x, y), orientation)
                        break
                print("Invalid move. Please try again.")
            except ValueError as e:
                print(e)

        # else:  # Bot player (random moves)
        #     valid_moves = env.get_valid_pawn_moves(player_turn)
        #     move = random.choice(valid_moves)
        #     env.move_pawn(player_turn, move)
        #     print(f"Bot moved to {move}")

        player_turn = 1 - player_turn  # Switch turns

    print("\nGame over!")
    winner = env.get_winner()
    if winner is not None:
        print(f"Player {winner + 1} wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    main()
