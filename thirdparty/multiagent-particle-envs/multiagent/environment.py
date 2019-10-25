import random
import numpy as np
import math


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv():
    def __init__(self, world, reset_callback=None, reward_callback=None,
                 done_callback=None, observation_callback=None, info_callback=None,
                  shared_viewer=True):
        self.world = world
        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n = len(self.agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback

        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        if self.n == 2:
            assert reward_n[0] == reward_n[1]

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        ideal_reward = self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n, ideal_reward

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, time=None):
        agent.censor = action[0]
        agent.cw = math.ceil(16*action[1])
