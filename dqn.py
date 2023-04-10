import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from quoridor import (
    QuoridorEnv,
    ActionError,
)
from model import Model as DQN
from replay_buffer import ReplayBuffer
from tqdm import trange


device = torch.device("cpu")

# Set hyperparameters
num_episodes = 500
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.9997
target_update_freq = 10
batch_size = 128

env = QuoridorEnv()
epsilon = epsilon_start


def train_dqn(dqn, target_net, experiences, optimizer, gamma=0.99):
    states, actions, rewards, next_states, dones = experiences

    actions = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    # Compute Q-values for current states and next states
    states = tuple(
        map(
            lambda x: torch.stack(x).to(dtype=torch.float32, device=device),
            zip(*states),
        )
    )

    dqn.train()

    q_values = dqn(*states).gather(1, actions.unsqueeze(1))
    with torch.no_grad():
        next_states = tuple(
            map(
                lambda x: torch.stack(x).to(dtype=torch.float32, device=device),
                zip(*next_states),
            )
        )
        next_q_values = target_net(*next_states).max(1)[0].unsqueeze(1)

    # Compute target Q-values
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    # Update DQN model weights
    loss = F.mse_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Initialize two DQN models and target networks for each player
dqn = [DQN().to(device), DQN().to(device)]  # Replace with arguments as needed
target_net = [DQN().to(device), DQN().to(device)]  # Replace with arguments as needed
optimizer = [
    optim.Adam(dqn[0].parameters(), lr=learning_rate),
    optim.Adam(dqn[1].parameters(), lr=learning_rate),
]

for player in range(2):
    target_net[player].load_state_dict(dqn[player].state_dict())
    target_net[player].eval()

# Initialize separate replay buffers for each player
replay_buffer = [ReplayBuffer(capacity=1024), ReplayBuffer(capacity=1024)]

# Training loop
for episode in trange(num_episodes):
    state = env.reset()
    done = False

    while not done:
        for player in range(2):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.sample_action()
                player_position = env.player_positions[player]
                next_state, reward, done = env.step(action)
            else:
                with torch.no_grad():
                    dqn[player].eval()

                    player, board, fence, num_fences = state
                    player = torch.tensor(player, device=device).unsqueeze(0)
                    board = board.to(dtype=torch.float32, device=device).unsqueeze(0)
                    fence = fence.to(dtype=torch.float32, device=device).unsqueeze(0)
                    num_fences = torch.tensor(
                        num_fences, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                    out = dqn[player](player, board, fence, num_fences).cpu()

                    player_position = env.player_positions[player]

                    while True:
                        action = env.index_to_action(player, out.argmax().item())
                        try:
                            next_state, reward, done = env.step(action)
                            break
                        except ActionError:
                            out[0, out.argmax().item()] = float("-inf")


            # Store experience in the replay buffer for the current player
            replay_buffer[player].add(
                state,
                env.action_to_index(player_position, action),
                reward,
                next_state,
                done,
            )

            # Train the current player's DQN model if there are enough samples in their replay buffer
            if len(replay_buffer[player]) > batch_size:
                experiences = replay_buffer[player].sample(batch_size)
                train_dqn(
                    dqn[player], target_net[player], experiences, optimizer[player]
                )
            else:
                print(len(replay_buffer[player]))

            # Update the target network for the current player periodically
            if episode % target_update_freq == 0:
                target_net[player].load_state_dict(dqn[player].state_dict())

            state = next_state

            if done:
                break

    # Decay epsilon for epsilon-greedy action selection
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

torch.save(dqn[0].state_dict(), "trained_dqn_0.pth")
torch.save(dqn[1].state_dict(), "trained_dqn_1.pth")
