import gym
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from tensorboardX import SummaryWriter
from collections import deque
import argparse

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        
        )
      
    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, environment, device):
        self.states = []
        self.actions = []
        self.rewards = []
        self.env_name = environment
        self.env = gym.make(environment)
        self.policy = PolicyNet(self.env.observation_space.shape[0], self.env.action_space.n).to(device)
        self.device = device
        self.writter = SummaryWriter()
        
    def calculate_return(self, gamma):
        res = []
        sum_r = 0.0
        for r in reversed(self.rewards):
            sum_r *= gamma 
            sum_r += r
            res.append(sum_r)
        res = list(reversed(res))
        mean_q = np.mean(res)
        return [q - mean_q for q in res]
        
    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    def save_checkpoint(self):
        print("*****SAVING MODEL*****")
        torch.save(self.policy.state_dict(), self.env_name + "_vpg.pth")

    def choose_action(self, state):
        state = torch.tensor(state, dtype = torch.float32).to(device)
        probs = F.softmax(self.policy(state), dim = 0)
        action_probs = Categorical(probs)
        action = action_probs.sample()
        return action.item()
        
    def learn(self, num_episodes, lr, gamma):

        optimizer = optim.Adam(self.policy.parameters(), lr = lr)
        writer = SummaryWriter()
        best_score = self.env.reward_range[0]
        score_history = []
        for episode in range(num_episodes):
                done = False
                score = 0
                observation = self.env.reset()
                while not done:
                    action = agent.choose_action(observation)
                    observation_, reward, done, info = self.env.step(action)
                    agent.store_transition(observation, action, reward)
                    observation = observation_
                    score += reward
                score_history.append(score)
                writer.add_scalar("Score",score, episode)

                # Learn
                states = torch.tensor(self.states).to(device)
                actions = torch.tensor(self.actions).to(device)
                rewards = torch.tensor(self.rewards).to(device)
                
                logits = F.softmax(self.policy(states), dim = 1)
                sampler = Categorical(logits)
                log_probs = sampler.log_prob(actions)
                G = torch.tensor(self.calculate_return(gamma)).to(device)
                loss = -torch.sum(log_probs * G)
                writer.add_scalar("loss", loss.item(), episode)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                self.states.clear()
                self.actions.clear()
                self.rewards.clear()


                avg_score = np.mean(score_history[-100:])
                print('episode: ', episode,'score: %.1f' % score,
                    'average score %.1f' % avg_score)

                if avg_score >= best_score:
                    best_score = avg_score
                    self.save_checkpoint()


                if (self.env_name == 'CartPole-v1'):
                    if (avg_score >= 450):
                        break
                    
                if (self.env_name == 'LunarLander-v2'):
                    if (avg_score >= 200):
                        break

        # torch.save(agent.policy.state_dict(), self.env_name + ".pth")

    def play(self, num_episodes):

        self.policy.load_state_dict(torch.load(self.env_name + '.pth'))
        for episode in range(num_episodes):
            print("***Episode", episode + 1,"***")
            rewards = 0
            done = False
            state = self.env.reset()
            while not done:
                self.env.render()
                time.sleep(0.01)
                action = self.choose_action(state)
                new_state, reward, done, info = self.env.step(action)
                rewards += reward
                state = new_state
                if done:
                    print("Score: ", rewards )
                    self.env.reset()
                    continue

        self.env.close()
            



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help = "CartPole-v1 or LunarLander-v2", type = str)
    parser.add_argument("--learn", help = "agent learns to solve the environment", action = 'store_true')
    parser.add_argument("-g", help = "gamma: discount factor", type = int, default = 0.99 )
    parser.add_argument("-lr", help = "learning rate", type = int, default = 0.001)
    parser.add_argument("-ep", help = "number of episodes to learn", type = int, default = 1000 )
    parser.add_argument("--play", help = "number of episodes to learn", action = 'store_true' )
    args = parser.parse_args()

    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert(args.env in ['CartPole-v1', 'LunarLander-v2'])

    agent = Agent(args.env, device)

    if (args.learn):
        agent.learn(args.ep, args.lr, args.g)

    if (args.play):
        agent.play(args.ep)

    # agent.learn(1000, 0.001, 0.99)

