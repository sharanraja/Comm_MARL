import numpy as np
from queue import *

# Phase-1 of CBBA - Greedy Task Selection
def SelectTask(agent):
    h = np.asarray([1 if i >= j else 0 for i,j in zip(agent.c,agent.y)])
    while len(agent.bundle) < agent.maxTask:
        h = np.asarray([j if i not in agent.bundle else 0 for i,j in enumerate(h)])
        J = np.argmax(np.multiply(agent.c,h))
        agent.bundle.append(J)
        agent.path.append(J)
        agent.y[J] = agent.c[J]
        agent.z[J] = agent.agentID


def update(agent,a,b,j): 
    agent.y[j] = a
    agent.z[j] = b
    check = False
    for i in agent.bundle:
        if agent.z[i] != agent.agentID:
            index = agent.bundle.index(i)
            ntilde = i
            check = True
            break
    if check:
        reset = agent.bundle[index+1:].copy()
        for i in reset:
            agent.y[i] = 0
            agent.z[i] = -1
            agent.bundle.remove(i)
            agent.path.remove(i)
        agent.bundle.remove(ntilde)
        agent.path.remove(ntilde)

# Phase-2 of CBBA - Conflict resolution                 
def UpdateTask(agent):
    msgs = []    
    while agent.incomingMsgQueue.empty() == False:
        msgs.append(agent.incomingMsgQueue.get())
    for i in msgs:
        ik = i[0]
        zk = i[1]
        yk = i[2]
        sk = i[3]
        agent.beliefstate[ik] = i # latest information received from each agent in neighborhood - used in voi calculation
        for j in range(agent.nTask):
            p = agent.z[j]
            if zk[j] == ik:
                if agent.z[j] == agent.agentID:
                    if yk[j] > agent.y[j]:
                        update(agent, yk[j], zk[j], j)
                    continue
                elif agent.z[j] == ik or agent.z[j] == -1:
                    update(agent, yk[j], zk[j], j)
                    continue
                else:
                    if sk[p] > agent.s[p] or yk[j] > agent.y[j]:
                        update(agent, yk[j], zk[j], j)
                    continue

            elif zk[j] == agent.agentID:
                if agent.z[j] == agent.agentID:
                    continue
                elif agent.z[j] == ik:
                    agent.y[j] = 0
                    agent.z[j] = -1
                    continue
                elif agent.z[j] == -1:
                    continue
                else:
                    if sk[p] > agent.s[p]:
                        agent.y[j] = 0
                        agent.z[j] = -1
                    continue

            elif zk[j] == -1:
                if agent.z[j] == agent.agentID:
                    continue
                elif agent.z[j] == ik:
                    update(agent, yk[j], zk[j], j)
                    continue
                elif agent.z[j] == -1:
                    continue
                else:
                    if sk[p] > agent.s[p]:
                        update(agent, yk[j], zk[j], j)
                    continue

            else:
                m = zk[j]
                if agent.z[j] == agent.agentID:
                    if sk[m] > agent.s[m] and yk[j] > agent.y[j]:
                        update(agent, yk[j], zk[j], j)
                    continue
                elif agent.z[j] == ik:
                    if sk[m] > agent.s[m]:
                        update(agent, yk[j], zk[j], j)
                    else:
                        agent.y[j] = 0
                        agent.z[j] = -1
                    continue
                elif agent.z[j] == m:
                    if sk[m] > agent.s[m]:
                        update(agent, yk[j], zk[j], j)
                    continue
                elif agent.z[j] == -1:
                    if sk[m] > agent.s[m]:
                        update(agent, yk[j], zk[j], j)
                    continue
                else:
                    if sk[m] > agent.s[m] and sk[p] > agent.s[p]:
                        update(agent, yk[j], zk[j], j)
                        continue
                    elif sk[m] > agent.s[m] and yk[j] > agent.y[j]:
                        update(agent, yk[j], zk[j], j)
                        continue
                    elif sk[p] > agent.s[p] and agent.s[m] > sk[m]:
                        agent.y[j] = 0
                        agent.z[j] = -1
                        continue
        agent.s = np.asarray([max(i,j) for i,j in zip(sk,agent.s)])
        agent.s[ik] = agent.Time
    agent.outgoingMsgQueue.put((agent.agentID, agent.z, agent.y, agent.s))

def VoI(agent,i1): # tracks the amount of change the message from 'agent' can bring to another agent  
    a = agent.agentID
    az = agent.z
    ay = agent.y
    aS = agent.s
    b = i1[0]
    bz = i1[1]
    by = i1[2]
    bS = i1[3]
    voi = 0
    for i in range(agent.nTask):
        p = bz[i]
        if az[i] == a:
            if bz[i] == b:
                if ay[i] > by[i]:
                    voi += 1
                continue
            elif bz[i] == -1:
                voi += 1
                continue
            elif bz[i] != b and bz[i] != a:
                m = bz[i]
                if aS[m] > bS[m] or ay[i] > by[i]:
                    voi += 1
                continue
            continue
        elif az[i] == b:
            if bz[i] == b:
                continue
            elif bz[i] == a:
                voi += 1
                continue
            elif bz[i] == -1:
                continue
            else:
                if aS[p] > bS[p]:
                    voi += 1
                continue

        elif az[i] == -1:
            if bz[i] == b:
                continue
            elif bz[i] == a:
                voi += 1
                continue
            elif bz[i] == -1:
                continue
            else:
                if aS[p] > bS[p]:
                    voi += 1
                continue
                
        else:
            m = az[i]
            if bz[i] == b:
                if aS[m] > bS[m] and ay[i] > by[i]:
                    voi += 1
                continue
            elif bz[i] == a:
                if aS[m] > bS[m]:
                    voi += 1
                else:
                    voi += 1
                continue
            elif bz[i] == m:
                if aS[m] > bS[m]:
                    voi += 1
                continue
            elif bz[i] == -1:
                if aS[m] > bS[m]:
                    voi += 1
                continue
            else:
                if aS[m] > bS[m] and aS[p] > bS[p]:
                    voi += 1
                    continue
                elif aS[m] > bS[m] and ay[i] > by[i]:
                    voi += 1
                    continue
                elif aS[p] > bS[p] and bS[m] > aS[m]:
                    voi += 1
                    continue
    voi /= agent.nTask
    return voi