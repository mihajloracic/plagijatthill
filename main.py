import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from network.simple_network import SimpleNetwork
from network.replay_memory import ReplayMemory
from network.replay_memory import Transition
from environment.environment import Environment

from collections import deque

import numpy as np
import matplotlib.pyplot as plt

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = SimpleNetwork().to(device)
    target_net = SimpleNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(100000)
    env = Environment()

    eps_start = 0.8
    eps_end = 0.1
    eps_decay = 5000
    steps_done = 0
    is_network = False

    def get_random_action():
        return [random.random(), random.random(), random.random(), 0]

    def select_action(state):
        nonlocal steps_done
        nonlocal is_network
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action = policy_net(state)
                is_network = True
                return action
        else:
            is_network = False
            return torch.tensor([get_random_action()], device=device, dtype=torch.float)

    def optimize_model(batch_size=1000, gamma=0.9):
        if len(memory) < batch_size:
            return
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        next_states = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch)
        next_state_values = torch.zeros(batch_size, device=device)
        next_state_values = target_net(next_states)
        
        expected_state_action_values = (next_state_values * gamma) + reward_batch
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.detach())

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        return loss.item()

    def train_loop(num_episodes=100000):
        rewards = deque([], num_episodes)
        states = deque([], num_episodes)

        for i_episode in range(num_episodes):
            state = torch.tensor([env.get_state()], device=device, dtype=torch.float)
            action = select_action(state)
            reward = torch.tensor([env.next_state(action[0])], device=device, dtype=torch.float)
            memory.push(state, action, torch.tensor([env.get_state()], device=device, dtype=torch.float), reward)
            
            loss = optimize_model()

            # Popunjava se za iscrtavanje grafika
            rewards.append(sum(sum(reward)))
            states.append(sum(sum(state)))

            if i_episode % 100 == 0:
                print("Loss: ", loss)
                target_net.load_state_dict(policy_net.state_dict())
        
        # Trenutno stanje - dostupnost vode
        plt.plot(np.linspace(1, len(states), len(states)), states, 'b')

        #Nagrada - ukupna dostavljena kolicina vode
        plt.plot(np.linspace(1, len(rewards), len(rewards)), rewards, '--r')
        plt.show()
    train_loop(25000)

train()