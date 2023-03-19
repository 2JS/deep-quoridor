import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

from quoridor import QuoridorEnv

# Define your Q-Network here
class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.board_conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.fence_conv = nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1)

        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.fc = nn.Linear(9 * 9 * 64, 8 * 8 * 2 + 8)

        self.flatten = nn.Flatten()
        self.act = nn.ReLU()


    def forward(self, board, fence):
        board = self.board_conv(board)
        fence = self.fence_conv(fence)

        x = board + fence

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))

        x = self.fc(self.flatten(x))

        return x



def epsilon_greedy_policy(q_values, epsilon):
    if random.random() < epsilon:
        return random.randint(0, len(q_values) - 1)
    else:
        return torch.argmax(q_values).item()


def train_dqn(env, num_episodes, batch_size, gamma, learning_rate, initial_epsilon, min_epsilon, epsilon_decay):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")

    # Initialize Q-Network
    q_network = QNetwork().to(device)
    target_network = QNetwork().to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    replay_memory = deque(maxlen=10000)
    epsilon = initial_epsilon

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = epsilon_greedy_policy(q_values, epsilon)

            next_state, reward, done = env.step(action)
            replay_memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(replay_memory) >= batch_size:
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones = torch.tensor(dones, dtype=torch.bool).to(device)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_network(next_states).max(1)[0].detach()
                target_q_values = rewards + gamma * next_q_values * (~dones)

                loss = F.mse_loss(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode % 100 == 0:
            target_network.load_state_dict(q_network.state_dict())

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Logging
        if episode % 10 == 0:
            print(f"Episode {episode}: Loss = {loss.item()}, Epsilon = {epsilon}")

if __name__ == "__main__":
    env = QuoridorEnv()

    # Set your hyperparameters here
    num_episodes = 1000
    batch_size = 64
    gamma = 0.99
    learning_rate = 0.001
    initial_epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay = 0.995

    train_dqn(env, num_episodes, batch_size, gamma, learning_rate, initial_epsilon, min_epsilon, epsilon_decay)
