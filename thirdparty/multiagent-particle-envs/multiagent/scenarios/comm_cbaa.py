import numpy as np
from multiagent.core import World, Agent
from multiagent.pose import *
from multiagent.scenario import BaseScenario
import random
from queue import *

class Scenario(BaseScenario):
    def make_world(self,args):
        self.args = args
        world = World()
        
        # set any world properties first
        self.num_agents = self.args.n_student
        self.num_tasks = self.args.n_task
        self.maxTask = self.args.maxTask

        # add agents
        world.agents = [Agent(i,self.num_tasks,self.maxTask,self.num_agents) for i in range(self.num_agents)]
        world.int_matrix = self.generate_graph(world)
        
        for i,agent in enumerate(world.agents):  
            _, agent.state[0], agent.nghd = self.secondhop(world.int_matrix,i)  
            agent.c = np.random.uniform(0,100,self.num_tasks)
        world.initializeCommDictionaries()
        world.updateCommNeighbors()

        return world

    def generate_graph(self,world):
        n = self.num_agents
        int_matrix = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                int_matrix[i][j] = np.random.randint(2)
        for i in range(n):
            for j in range(i+1,n):
                int_matrix[j][i] = int_matrix[i][j]
        for i in range(n):
            if sum(int_matrix[i]) == 0:
                while True:
                    randindex = np.random.randint(n)
                    if randindex != i:
                        break
                int_matrix[i][randindex] = 1
                int_matrix[randindex][i] = 1
        return int_matrix

    def BK_maxIS(self,int_matrix):
        M = []
        def g_neighbors(M,n):
            nbrs = []
            for i,j in enumerate(M[n]):
                if j == 1:
                    nbrs.append(i)
            return nbrs

        def check_end(S,T):
            for k in S:
                E = g_neighbors(int_matrix,k)
                if len(set(E).intersection(set(T))) == 0:
                    return False 
            return True

        def remove_nbrs(S,x,removeX):
            nbrs = g_neighbors(int_matrix,x)
            if removeX:
                nbrs.append(x)
            S = set([i for i in S if i not in nbrs])
            return S

        def findIS(P,S,T):
            if check_end(S,T):
                for i in T:
                    P.add(i)
                    S1 = remove_nbrs(S,i,False)
                    T1 = remove_nbrs(T,i,True)
                    if len(S1) == 0 and len(T1) == 0:
                        M.append(P.copy())
                    elif len(S1) != 0 and len(T1) == 0:
                        True 
                    else:
                        findIS(P,S1,T1)
                    P.discard(i)
                    S.add(i)
        P = set()
        S = set()
        T = [i for i in range(int_matrix.shape[0])]
        findIS(P,S,T)
        return M

    def secondhop(self,int_matrix,x):
        A = []
        nbrs = [j for j,i in enumerate(int_matrix[x]) if i == 1]
        for i in nbrs:
            temp = [j for j,k in enumerate(int_matrix[i]) if j not in nbrs and j != x and k == 1]
            A += temp
        second_hop = set(A)
        subgraph = np.zeros((len(second_hop),len(second_hop)))
        for i,j in enumerate(second_hop):
            for k,l in enumerate(second_hop):
                subgraph[i][k] = int_matrix[j][l]
        neighborhood = int_matrix[x]
        neighborhood = [1 if j == 1 or i in second_hop else 0 for i,j in enumerate(neighborhood)]
        max_IS = self.BK_maxIS(subgraph)
        bk_feature = [len(i) for i in max_IS]
        if len(bk_feature) != 0:
            bk_feature = sum(bk_feature)/len(bk_feature)
        else:
            bk_feature = 0
        baseline_feature = len(second_hop)
        return baseline_feature, bk_feature, neighborhood

    def ideal_reward(self, world):
        N = min(self.num_tasks, self.num_agents*self.maxTask)
        C = np.zeros((self.num_agents,self.num_tasks))
        bundle = dict()
        winning_bid = dict()
        for i in world.agents:
            C[i.agentID] = i.c
            bundle[i.agentID] = []
            winning_bid[i.agentID] = []
        for n in range(N):
            y_ = np.amax(C)
            m = np.where(C==y_)
            i = (int(m[0]),int(m[1]))
            bundle[i[0]].append(i[1])
            winning_bid[i[0]].append(y_)
            C[:,i[1]] = -np.ones(self.num_agents)
            if len(bundle[i[0]]) == self.maxTask:
                C[i[0],:] = -np.ones(num_tasks)
        reward_i = 0
        for i in range(self.num_agents):
            reward_i += sum(winning_bid[i])

        return reward_i

    def reset_world(self, world):
        world = World()
        # add agents
        world.agents = [Agent(i,self.num_tasks,self.maxTask,self.num_agents) for i in range(self.num_agents)]
        world.int_matrix = self.generate_graph(world)
        
        for i,agent in enumerate(world.agents):  
            _, agent.state[0], agent.nghd = self.secondhop(world.int_matrix,i)  
            agent.c = np.random.uniform(0,100,self.num_tasks)
        world.initializeCommDictionaries()
        world.updateCommNeighbors()

        reward_i = self.ideal_reward(world)

        return reward_i

    def reward(self, agent, world):
        rew = 0
        for i in world.agents:
            if i.agentID == agent.agentID:
                continue
            else:
                rew -= np.dot(agent.x,i.x)/(i.x.shape[0])
        temp = rew
        if world.stepct == 1:
            rew = 0
        else:
            rew -= agent.reward
        if world.stepct == self.args.ep_max_timesteps:
            rew -= agent.state[1]*0.1
        agent.reward = temp
        return rew

    def observation(self, agent, world):
        return agent.state

    def done(self, agent, world):
        if world.stepct == self.args.ep_max_timesteps:
            return True
        else:
            return False

