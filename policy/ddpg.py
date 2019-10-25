"""
    Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
    Paper: https://arxiv.org/abs/1802.09477
    Ref: https://github.com/sfujim/TD3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.noise import OUNoise, GaussNoise
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, actor_input_dim, actor_output_dim, max_action, min_action, name):
        super(Actor, self).__init__()

        setattr(self, name + "_l1", nn.Linear(actor_input_dim, 32))
        setattr(self, name + "_l2", nn.Linear(32, 32))
        setattr(self, name + "_l3", nn.Linear(32, actor_output_dim))

        self.max_action = max_action
        self.min_action = min_action
        self.name = name

    def forward(self, x):
        x = F.relu(getattr(self, self.name + "_l1")(x))
        x = F.relu(getattr(self, self.name + "_l2")(x))
        x = torch.sigmoid(getattr(self, self.name + "_l3")(x))
        return x


class Critic(nn.Module):
    def __init__(self, critic_input_dim, name):
        super(Critic, self).__init__()

        setattr(self, name + "_l1", nn.Linear(critic_input_dim, 48))
        setattr(self, name + "_l2", nn.Linear(48, 48))
        setattr(self, name + "_l3", nn.Linear(48, 1))

        setattr(self, name + "_l4", nn.Linear(critic_input_dim, 48))
        setattr(self, name + "_l5", nn.Linear(48, 48))
        setattr(self, name + "_l6", nn.Linear(48, 1))
        self.name = name

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x = F.relu(getattr(self, self.name + "_l1")(xu))
        x = F.relu(getattr(self, self.name + "_l2")(x))
        x = getattr(self, self.name + "_l3")(x)

        x1 = F.relu(getattr(self, self.name + "_l4")(xu))
        x1 = F.relu(getattr(self, self.name + "_l5")(x1))
        x1 = getattr(self, self.name + "_l6")(x1)
        return x, x1

class DDPG(object):
    def __init__(self, actor_input_dim, actor_output_dim, critic_input_dim, max_action, min_action, name, args, i_agent, env):
        # NOTE the name for target and behavior policy must be the same
        self.actor = Actor(actor_input_dim, actor_output_dim, max_action, min_action, name=name + "_actor").to(device)
        self.actor_target = Actor(actor_input_dim, actor_output_dim, max_action, min_action, name=name + "_actor").to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.critic = Critic(critic_input_dim, name=name + "_critic").to(device)
        self.critic_target = Critic(critic_input_dim, name=name + "_critic").to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.max_action = max_action
        self.min_action = min_action
        self.name = name
        self.args = args
        self.i_agent = i_agent
        self.noise = GaussNoise(action_dimension=self.args.batch_size, mu=0, std=self.args.gauss_std)
        self.nghd = env.agents[i_agent].nghd

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # return self.actor(state).cpu().data.numpy().flatten()
        return self.actor(state)

    def centralized_train(self, total_ep_count, agent_n, index, replay_buffer, iterations, batch_size, discount, tau, policy_freq):
        n_agent = len(agent_n)
        
        debug = {}
        debug["critic_loss"] = 0.
        debug["actor_loss"] = 0.

        for it in range(iterations):
            # Sample replay buffer 
            x_n, y_n, u_n, r_n, d_n = replay_buffer.centralized_sample(
                batch_size=batch_size, n_agent=n_agent) 

            state_n = [
                torch.FloatTensor(x_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]
            action_n = [
                torch.FloatTensor(u_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]
            next_state_n = [
                torch.FloatTensor(y_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]
            done_n = [
                torch.FloatTensor(1 - d_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]
            reward_n = [
                torch.FloatTensor(r_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]

            if total_ep_count > 0:
                # Select action according to policy and add clipped noise 
                next_action_n = []
                for i_agent_ in range(n_agent):
                    with torch.no_grad():
                        next_action = agent_n[i_agent_].policy.actor_target(next_state_n[i_agent_])
                        noise = self.noise.noise()
                        noise = noise.clip(-0.05,0.05)
                        noise = noise.reshape((noise.shape[0],1))
                        noise = torch.from_numpy(noise).float()
                        noise = torch.FloatTensor(noise).to(device)
                        next_action = next_action + noise
                        next_action = (next_action).clamp(0.0,1.0)
                        next_action_n.append(next_action)

                mean_action1 = [i[:,0:1] for i,j in zip(next_action_n,self.nghd) if j == 1]
                mean_action1 = torch.cat(mean_action1, dim=1)
                mean_action1 = torch.mean(mean_action1, dim=1, keepdim=True)

                mean_action2 = [i[:,1:2] for i,j in zip(next_action_n,self.nghd) if j == 1]
                mean_action2 = torch.cat(mean_action2, dim=1)
                mean_action2 = torch.mean(mean_action2, dim=1, keepdim=True)

                mean_action = torch.cat([mean_action1,mean_action2], dim=1)
                next_action_n = torch.cat(next_action_n, dim=1)
                self_action = next_action_n[:,2*index:2*index+2]
                next_action_n = torch.cat([self_action,mean_action], dim=1)

                # next_state_n = [j[0] for i,j in enumerate(zip(next_state_n,self.nghd)) if j[1] != 0 or i == self.i_agent]
                
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(torch.cat(next_state_n, dim=1), next_action_n)
                target_Q = torch.min(target_Q1,target_Q2)
                reward = reward_n[index]
                target_Q = reward + (done_n[index] * discount * target_Q).detach()

                # state_nc = [j[0] for i,j in enumerate(zip(state_n,self.nghd)) if j[1] != 0 or i == self.i_agent]

                mean_action1c = [i[:,0:1] for i,j in zip(action_n,self.nghd) if j == 1]
                mean_action1c = torch.cat(mean_action1c, dim=1)
                mean_action1c = torch.mean(mean_action1c, dim=1, keepdim=True)

                mean_action2c = [i[:,1:2] for i,j in zip(action_n,self.nghd) if j == 1]
                mean_action2c = torch.cat(mean_action2c, dim=1)
                mean_action2c = torch.mean(mean_action2c, dim=1, keepdim=True)

                mean_actionc = torch.cat([mean_action1c,mean_action2c], dim=1)
                action_nc = torch.cat(action_n, dim=1)
                self_actionc = action_nc[:,2*index:2*index+2]
                action_nc = torch.cat([self_actionc,mean_actionc], dim=1)

                # Get current Q estimates
                current_Q1, current_Q2 = self.critic(torch.cat(state_n, dim=1), action_nc)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                debug["critic_loss"] += critic_loss.cpu().data.numpy().flatten()

                # Delayed policy updates
                if it % policy_freq == 0:
                    # Compute actor loss
                    action_n = []
                    for i_agent_ in range(n_agent):
                        if i_agent_ == index:
                            with torch.enable_grad():
                                action = agent_n[i_agent_].policy.actor(state_n[i_agent_])
                                action_n.append(action)
                        else:
                            with torch.no_grad():
                                action = agent_n[i_agent_].policy.actor(state_n[i_agent_])
                                action_n.append(action)

                    mean_action1c = [i[:,0:1] for i,j in zip(action_n,self.nghd) if j == 1]
                    mean_action1c = torch.cat(mean_action1c, dim=1)
                    mean_action1c = torch.mean(mean_action1c, dim=1, keepdim=True)

                    mean_action2c = [i[:,1:2] for i,j in zip(action_n,self.nghd) if j == 1]
                    mean_action2c = torch.cat(mean_action2c, dim=1)
                    mean_action2c = torch.mean(mean_action2c, dim=1, keepdim=True)

                    mean_actionc = torch.cat([mean_action1c,mean_action2c], dim=1)
                    action_nc = torch.cat(action_n, dim=1)
                    self_actionc = action_nc[:,2*index:2*index+2]
                    action_nc = torch.cat([self_actionc,mean_actionc], dim=1)
                    
                    a1, a2 = self.critic(torch.cat(state_n, dim=1), action_nc)
                    actor_loss = -a1.mean()
                    
                    # Optimize the actor 
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()
                    debug["actor_loss"] += actor_loss.cpu().data.numpy().flatten()

                    # Update the frozen target models
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return debug

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        from collections import OrderedDict

        actor_weight = torch.load('%s/%s_actor.pth' % (directory, filename))
        critic_weight = torch.load('%s/%s_critic.pth' % (directory, filename))

        # Fix name :-)
        actor_weight_fixed = OrderedDict()
        for k, v in actor_weight.items():
            name_fixed = self.name
            for i_name, name in enumerate(k.split("_")):
                if i_name > 0:
                    name_fixed += "_" + name
            actor_weight_fixed[name_fixed] = v

        critic_weight_fixed = OrderedDict()
        for k, v in critic_weight.items():
            name_fixed = self.name
            for i_name, name in enumerate(k.split("_")):
                if i_name > 0:
                    name_fixed += "_" + name
            critic_weight_fixed[name_fixed] = v

        self.actor.load_state_dict(actor_weight_fixed)
        self.critic.load_state_dict(critic_weight_fixed)
