import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNetwork(nn.Module):
    def __init__(self, alpha, input_dims, nb_actions):
        super(DQNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = 256
        self.fc2_dims = 256
        self.nb_actions = nb_actions

        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.nb_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class DQNAgent:
    def __init__(
        self,
        nb_actions,
        input_dims,
        gamma=0.995,
        alpha=0.01,
        epsilon=0.1,
        epsilon_end=0.01,
        epsilon_decay=5e-4,
        batch_size=64,
        mem_size=50000,
    ):
        self.nb_actions = nb_actions
        self.action_space = [i for i in range(self.nb_actions)]

        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.mem_size = mem_size
        self.batch_size = batch_size
        self.mem_counter = 0

        self.network = DQNetwork(alpha=self.alpha, input_dims=input_dims, nb_actions=nb_actions)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, terminal):
        index = self.mem_counter % self.mem_size  # goes back to 0 if memory full
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal

        self.mem_counter += 1

    def select_action(self, observation, mode_train=True):
        if np.random.random() > self.epsilon or not mode_train:
            state = torch.tensor(np.array(observation)).to(self.network.device)
            actions = self.network.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def can_learn(self):
        if self.mem_counter < self.batch_size:
            return False
        else:
            return True

    def learn(self):
        # learn only if we have a batch
        if not self.can_learn():
            return

        self.network.zero_grad()

        # take min of either counter or mem_size
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.network.device)
        next_state_batch = torch.tensor(self.next_state_memory[batch]).to(self.network.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.network.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.network.device)
        action_batch = self.action_memory[batch]  # no need for pytorch tensor here

        # compute the q value for this action and the next q value of all actions (note we are using the same net for qv and next qv)
        qv = self.network.forward(state_batch)[batch_index, action_batch]
        next_qv = self.network.forward(next_state_batch)
        # setting next_q_value to zero if is terminal state
        next_qv[terminal_batch] = 0.0

        # taking the max over action for next_qv
        q_target = reward_batch + self.gamma * torch.max(next_qv, dim=1)[0]

        # updating network params
        loss = self.network.loss(q_target, qv).to(self.network.device)
        loss.backward()
        self.network.optimizer.step()

        # decay eps
        self.epsilon -= self.epsilon_decay if self.epsilon > self.epsilon_end else self.epsilon_end
