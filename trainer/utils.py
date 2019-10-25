import numpy as np
import torch
from misc.utils import normalize
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################
# MISC
def reset_noise(student_n):
    for student in student_n:
        student.exploration.reset()


def get_q_value_n(agent_n, agent_obs_n, agent_action_n, train_type):
    if train_type == "centralized":
        agent_obs_n = np.concatenate([agent_obs_n[i] for i in range(len(agent_n))]).reshape(1, -1)
        agent_obs_n = torch.FloatTensor(agent_obs_n).to(device)

        agent_action_n = np.concatenate([agent_action_n[i] for i in range(len(agent_n))]).reshape(1, -1)
        agent_action_n = torch.FloatTensor(agent_action_n).to(device)
        q_value_n = []
        for agent in agent_n:
            q_value1, q_value2 = agent.policy.critic(agent_obs_n, agent_action_n)
            q_value = torch.min(q_value1,q_value2)
            q_value = q_value.cpu().data.numpy().flatten()[0]
            q_value_n.append(q_value)

        return q_value_n
    elif train_type == "independent":
        # NOTE This function is for independent (i.e., not centralized) critics
        q_value_n = []
        for agent, agent_obs, agent_action in zip(agent_n, agent_obs_n, agent_action_n):
            agent_obs = torch.FloatTensor(agent_obs.reshape(1, -1)).to(device)
            agent_action = torch.FloatTensor(agent_action.reshape(1, -1)).to(device)

            q_value = agent.policy.critic(agent_obs, agent_action)
            q_value = q_value.cpu().data.numpy().flatten()[0]
            q_value_n.append(q_value)

        return q_value_n
    else:
        raise ValueError()


########################################################################################
# STUDENT
def get_student_obs(env_obs_n, ep_timesteps, ep_max_timesteps, args):
    student_obs_n = []

    if args.student_done:
        remaining_timesteps = normalize(
            value=(ep_max_timesteps - ep_timesteps), 
            min_value=0.,
            max_value=float(ep_max_timesteps))
        remaining_timesteps = np.array([remaining_timesteps])
        
        for env_obs in env_obs_n:
            student_obs = np.concatenate([env_obs, remaining_timesteps])
            student_obs_n.append(student_obs)

        return student_obs_n
    else:
        return env_obs_n

