import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from quoridor import QuoridorEnv, ActionError  # Replace with your actual environment import
from model import Model as DQN  # Replace with your actual DQN model import
from replay_buffer import ReplayBuffer  # Replace with your actual replay buffer import
from tqdm import trange

device = torch.device('cuda')

# Set hyperparameters
num_episodes = 1000
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 10
batch_size = 64

# # Initialize environment, DQN model, and target network
env = QuoridorEnv()
# dqn = DQN()  # Replace with arguments as needed
# target_net = DQN()  # Replace with arguments as needed
# target_net.load_state_dict(dqn.state_dict())
# target_net.eval()

# optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
# replay_buffer = ReplayBuffer()

# # Training loop
epsilon = epsilon_start
# for episode in range(num_episodes):
#     state = env.reset()
#     done = False

#     while not done:
#         # Epsilon-greedy action selection
#         if random.random() < epsilon:
#             action = env.sample_action()
#         else:
#             with torch.no_grad():
#                 action = dqn(state).argmax().item()

#         next_state, reward, done = env.step(action)
#         replay_buffer.add(state, action, reward, next_state, done)

#         # If there are enough samples in the replay buffer, train the DQN model
#         if len(replay_buffer) > batch_size:
#             experiences = replay_buffer.sample(batch_size)
#             states, actions, rewards, next_states, dones = experiences

#             # Compute Q-values for current states and next states
#             q_values = dqn(states).gather(1, actions.unsqueeze(1))
#             with torch.no_grad():
#                 next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)

#             # Compute target Q-values
#             target_q_values = rewards + (gamma * next_q_values * (1 - dones))

#             # Update DQN model weights
#             loss = F.mse_loss(q_values, target_q_values)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # Update target network weights periodically
#         if episode % target_update_freq == 0:
#             target_net.load_state_dict(dqn.state_dict())

#     # Decay epsilon for epsilon-greedy action selection
#     epsilon = max(epsilon_end, epsilon_decay * epsilon)

# # Save the trained DQN model
# torch.save(dqn.state_dict(), "trained_dqn.pth")

import torch.nn.functional as F

def train_dqn(dqn, target_net, experiences, optimizer, gamma=0.99):
    states, actions, rewards, next_states, dones = experiences

    actions = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    # Compute Q-values for current states and next states
    states = tuple(map(lambda x: torch.stack(x).to(dtype=torch.float32, device=device), zip(*states)))
    q_values = dqn(*states).gather(1, actions.unsqueeze(1))
    with torch.no_grad():
        next_states = tuple(map(lambda x: torch.stack(x).to(dtype=torch.float32, device=device), zip(*next_states)))
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
optimizer = [optim.Adam(dqn[0].parameters(), lr=learning_rate), optim.Adam(dqn[1].parameters(), lr=learning_rate)]

for player in range(2):
    target_net[player].load_state_dict(dqn[player].state_dict())
    target_net[player].eval()

# Initialize separate replay buffers for each player
replay_buffer = [ReplayBuffer(), ReplayBuffer()]

# Training loop
for episode in trange(num_episodes):
    state = env.reset()
    done = False

    while not done:
        for player in range(2):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.sample_action()
            else:
                with torch.no_grad():
                    player, board, fence, num_fences = state
                    player = torch.tensor(player, device=device).unsqueeze(0)
                    board = board.to(dtype=torch.float32, device=device).unsqueeze(0)
                    fence = fence.to(dtype=torch.float32, device=device).unsqueeze(0)
                    num_fences = torch.tensor(num_fences, dtype=torch.float32, device=device).unsqueeze(0)

                    out = dqn[player](player, board, fence, num_fences).cpu()
                    action = env.index_to_action(player, out.argmax().item())


            player_position = env.player_positions[player]
            try:
                next_state, reward, done = env.step(action)
            except ActionError:
                next_state = state
                reward = -100
                done = True

            # Store experience in the replay buffer for the current player
            replay_buffer[player].add(state, env.action_to_index(player_position, action), reward, next_state, done)

            # Train the current player's DQN model if there are enough samples in their replay buffer
            if len(replay_buffer[player]) > batch_size:
                experiences = replay_buffer[player].sample(batch_size)
                train_dqn(dqn[player], target_net[player], experiences, optimizer[player])
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
