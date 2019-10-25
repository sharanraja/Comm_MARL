import numpy as np
from queue import *
import math
import matplotlib.pyplot as plt
from multiagent.cbba import SelectTask, UpdateTask, VoI

# properties of agent entities
class Agent(object):
    def __init__(self,agentID,nTask,maxTask,nAgent):
        # Agent information
        self.agentID = agentID

        # CBBA Variables 
        self.nTask = nTask # total number ot tasks
        self.maxTask = maxTask # upper limit on tasks assigned to an agent 
        self.bundle = [] # list of tasks assigned to agent arranged based on the CBBA iteration
        self.path = [] # allotted taks in the order to be serviced
        self.c = np.zeros(nTask) 
        self.z = -1*np.ones(nTask).astype(int) # assignment binary variable 
        self.y = np.zeros(nTask) # self estimate of the current bid 
        self.s = np.zeros(nAgent) # time stamp of last (direct or indirect) information exchange with each agent in the world
        self.x = np.zeros(nTask) 

        # Network variables 
        self.incomingBuffer = [] 
        self.incomingMsgQueue = Queue(maxsize=0)
        self.outgoingMsgQueue = Queue(maxsize=0)
        self.Time = 0
        self.newHistorySentCodes = []
        self.newHistoryRcvCodes = []
        self.currentSendCode = -10
        self.currentRcvCode = -10
        self.broadcastingEndTime = 0
        self.broadcastingBeginTime = 0
        self.isBroadcasting = False
        self.difsCounter = 0
        self.backOffCounter = 0
        self.droppedMsgs = 0
        self.transmitted = 0
        self.collisions = 0 

        self.beliefstate = dict()
        self.beliefVoI = dict() 
        self.cw = 16 # contention window size used in CSMA - varied based on VoI
        self.censor = 0
        self.nghd = None # list of agents in the neighborhood of the agent - used in mean field approximation

        self.state = [0,0,0] 
        self.reward = 0 

    # Local task allocation algorithm defined in cbba.py

# multi-agent world
class World(object):
    def __init__(self,commType="Adhoc"):
        self.agents = None
        
        self.time = 0
        self.commType = commType
        self.csma = True

        ##########  THIS CANNOT BE CHANGED!!!
        self.deltaT = 0.00001  #length of a mini-slot in seconds 
        ##########                                          
        #Comm stuff
        self.commRate = 5*10**5 #Bytes/sec
        self.commRadius = 1.5 #M, disk model

        # Color codes for different status of senders and receivers       
        self.csma_send_codes = {0:('Already Broadcasting','b'),
                                1:('Nothing to transmit','0.5'),
                                2:('Medium Busy','r'),3:('DIFS','#ffcc66'),
                                4:('BackOff','k'),
                                9:('Successful Transmission','g')}
        self.csma_rcv_codes = {0:('Already Broadcasting','m'),
                                1:('Collission','r'),
                                2:('No transmission ','0.5'),
                                3:('Receiving msg','y'),
                                9:('Successful Transmission','g')}

        #Tracking TransmissionReceptions
        self.p_comm = dict()
        self.attemptedBroadcast = dict()
        self.successfullyTransmitted = dict()
        self.successfullyReceived = dict()

        #Reward is shared by all the agents
        self.collaborative = True


        self.stepct = 0
        self.int_matrix = None

    # update state of the world
    def step(self):
        for i in range(50):
            agentI = [r.agentID for r in self.agents]
            self.time += self.deltaT
            self.time = round(self.time,5)
            np.random.shuffle(agentI)

            for ri in agentI:
                agent = self.agents[ri]
                agent.Time += self.deltaT 
                agent.Time = round(agent.Time,5)
                agent.collisions = 0

            for ri in agentI:
                agent = self.agents[ri]
                if agent.Time > agent.broadcastingEndTime:
                    agent.isBroadcasting = False
            
            for ri in agentI:
                agent = self.agents[ri]
                self.updateAgentWifi(agent)

        for ri in agentI:
            agent = self.agents[ri]
            SelectTask(agent)
            UpdateTask(agent)
            agent.x = np.asarray([1 if i == agent.agentID else 0 for i in agent.z])        
            
        for agent in self.agents:
            agent.state[1] = float((agent.censor + self.stepct * agent.state[1]) / (self.stepct + 1)) 
            for key, value in agent.beliefstate.items():
                if value == None:
                    agent.beliefVoI.pop(key, None)
                    continue
                agent.beliefVoI[key] = VoI(agent,value)
            if bool(agent.beliefVoI):
                agent.state[2] = sum(agent.beliefVoI.values())/len(agent.beliefVoI) 
            else:
                agent.state[2] = 0
            for key, value in agent.beliefstate.items():
                agent.beliefstate[key] = None
        self.stepct += 1
        

    def updateCommNeighbors(self):
        for agent in self.agents:
            for otherAgent in self.agents:
                self.p_comm[agent.agentID][otherAgent] = self.int_matrix[agent.agentID][otherAgent.agentID]
        
    def initializeCommDictionaries(self):
        self.p_comm = dict.fromkeys([r.agentID for r in self.agents])
        for k in self.p_comm.keys():
            self.p_comm[k] = dict.fromkeys([r for r in self.agents],0)
        self.attemptedBroadcast = dict.fromkeys([r.agentID for r in self.agents])
        for k in self.attemptedBroadcast.keys():
            self.attemptedBroadcast[k] =  dict.fromkeys([r.agentID for r in self.agents],0)
        self.successfullyTransmitted = dict.fromkeys([r.agentID for r in self.agents])
        for k in self.successfullyTransmitted.keys():
            self.successfullyTransmitted[k] =  dict.fromkeys([r.agentID for r in self.agents],0)
        self.successfullyReceived = dict.fromkeys([r.agentID for r in self.agents])
        for k in self.successfullyReceived.keys():
            self.successfullyReceived[k] =  dict.fromkeys([r.agentID for r in self.agents],0)

    def updateAgentWifi(self,agent):
        for agent in self.agents:
            send_code = self.csma_send(agent)   
            if send_code != agent.currentSendCode:
                agent.newHistorySentCodes.append((self.time,send_code))
                agent.currentSendCode = send_code
        for agent in self.agents:
            rcv_code = self.csma_rcv(agent)
            if rcv_code != agent.currentRcvCode:
                if rcv_code == 9: #Get rid of "receiveing" status if successful transmission
                    t = agent.newHistoryRcvCodes.pop()[0]
                    agent.newHistoryRcvCodes.append((t,9))
                else:
                    agent.newHistoryRcvCodes.append((self.time,rcv_code))
                agent.currentRcvCode = rcv_code

    def csma_send(self,agent):    
        if self.time > agent.broadcastingEndTime:  #Maintenance on broadcasting variable
            agent.isBroadcasting = False
        if agent.isBroadcasting == True and self.commType != "Ideal": #Already Broadcasting
            return 9
        if agent.outgoingMsgQueue.empty(): 
            return 1 # Nothing to Transmit
        else:
            mediumFree = self.carrierSense(agent)         #1.  Carrier Sense
            if not mediumFree:                            #2a.  Medium busy: Defer transmission
                agent.difsCounter = 0
                if agent.backOffCounter == 0: 
                    agent.backOffCounter = np.random.randint(0, agent.cw) * 10e-6 / self.deltaT # cw is chosen based on voi
                return 2
            else: #2b.  Medium is free, countdown DIFS and Backoff
                if agent.difsCounter < 3:
                    agent.difsCounter +=1
                    return 3
                else: #Entering Backoff
                    if agent.backOffCounter > 0:
                        agent.backOffCounter -=1
                        return 4
                    else: #Entering Transmission
                        msg = agent.outgoingMsgQueue.get()
                        # length = 32*msg.shape[0] #bytes
                        endTime = self.time + float(100)/self.commRate
                        endTime = round(endTime,5)
                        #Add msg to every agent in neighborhood
                        for otherAgent in self.p_comm[agent.agentID].keys():
                            if self.p_comm[agent.agentID][otherAgent]>0.9:
                                self.successfullyTransmitted[agent.agentID][otherAgent.agentID] += 1
                                otherAgent.incomingBuffer.append((endTime,msg,agent.agentID))
                        #Update sender agent status
                        agent.broadcastingEndTime = endTime
                        agent.broadcastingBeginTime = round(self.time,5)
                        agent.isBroadcasting = True
                        agent.difsCounter = 0
                        return 9

    def carrierSense(self,agent): #Check neighbors before broadcasting
        if self.commType == "Ideal":
            return True
        for otherAgent in self.p_comm[agent.agentID].keys():
            if self.p_comm[agent.agentID][otherAgent]>.2 and otherAgent.isBroadcasting and otherAgent.broadcastingBeginTime != self.time:
                return False
            elif np.random.choice(2,p=[agent.censor,1-agent.censor]) == 0:
                return False 
        return True

    #Public only to simulator not agent!
    def csma_rcv(self,agent):
        if agent.isBroadcasting and self.commType != "Ideal" and len(agent.incomingBuffer) == 0:
            return 0

        if agent.isBroadcasting and self.commType != "Ideal" and len(agent.incomingBuffer) > 0:
            agent.droppedMsgs += len(agent.incomingBuffer)
            agent.incomingBuffer = []
            agent.isBroadcasting = False
            agent.broadcastingEndTime = self.time
            agent.rebroadcast = True
            return 1
        
        numAgentsBroadcasting = 0  #Count Number of MSGS being received
        for otherAgent in self.p_comm[agent.agentID].keys():
            if (self.time - otherAgent.broadcastingEndTime<0.000001) and self.p_comm[agent.agentID][otherAgent]>0.4:
                numAgentsBroadcasting += 1
        if numAgentsBroadcasting >= 2 and self.commType != "Ideal": #Message Collission on Receiving
            agent.droppedMsgs += len(agent.incomingBuffer)
            for transmit in agent.incomingBuffer:
                self.agents[transmit[2]].collisions += 1
            agent.incomingBuffer = []
            return 1
        if len(agent.incomingBuffer) == 0: #No incoming Msgs
            return 2
        
        msgTransmitted = False
        for transmit in agent.incomingBuffer:
            if self.time > transmit[0]: #Transmission finished
                data = transmit[1]
                agent.incomingBuffer.remove(transmit)
                self.successfullyReceived[agent.agentID][transmit[2]] += 1
                agent.incomingMsgQueue.put(data)
                msgTransmitted = True
                return 9 #Successful transmission
        if msgTransmitted:
            return 9
        else:
            return 3  #Still receiving

    def OccupancyPlot(self,key,total_ep_count,reward):
        prev = 0
        figure, ax = plt.subplots(figsize=(15,10))
        for k,j in enumerate(self.agents):
            if key == 'Sent':
                codes = j.newHistorySentCodes
                color = self.csma_send_codes
            elif key == 'Rcv':
                codes = j.newHistoryRcvCodes
                color = self.csma_rcv_codes
            for i in codes:
                if prev == 0:
                    prev = i
                    continue
                ax.broken_barh(xranges=[(prev[0],i[0]-prev[0])],yrange=(k+1, 0.75),facecolors=color[prev[1]][1])
                prev = i
        figure.savefig("%s/%s_%d_%f.png"%(key,key,total_ep_count,reward))
        plt.close(figure)
        
