import copy
import numpy as np
from trainer.utils import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle 


total_step_count = 0
total_ep_count = 0
session_step_count = 0
session_ep_count = 0
reward_ratio = dict()
VoI_dict = dict()
for i in range(50):
    reward_ratio[i] = []


def eval_progress(env, student_n, n_eval, log, tb_writer, args, final):
    eval_reward = 0.
    cbaa_reward = 0.
    reward_buffer = False
    if final:
        ep_max_timesteps = 50
    else:
        ep_max_timesteps = args.ep_max_timesteps

    for i_eval in range(n_eval):
        env_obs_n, ideal_reward = env.reset()

        ep_timesteps = 0
        terminal = False

        average_action = []

        for i_student in range(args.n_student):
            average_action.append(0)

        while True:
            student_obs_n = get_student_obs(env_obs_n, ep_timesteps, ep_max_timesteps, args)
            student_action_n = []
            for student, student_obs in zip(student_n, student_obs_n):
                student_action = student.select_deterministic_action(np.array(student_obs))
                student_action_n.append(student_action)
      
            for obs,action in zip(student_obs_n,student_action_n):
                key = (obs[0],obs[1],obs[2],obs[3])
                VoI_dict[key] = action

            new_env_obs_n, reward_n, done_n, _ = env.step(copy.deepcopy(student_action_n))

            if ep_timesteps + 1 == args.ep_max_timesteps and not final:
                terminal = True

            if ep_timesteps + 1 == 50 and final:
                terminal = True

            new_student_obs_n = get_student_obs(new_env_obs_n, ep_timesteps + 1, ep_max_timesteps, args)
            student_reward_n = reward_n
            student_done = terminal if args.student_done else False
            
            terminal_reward = 0   
            temp = np.zeros(env.world.agents[0].x.shape)
            for i in env.world.agents:
                temp += i.x
            temp = np.asarray([1 if i == 1 else 0 for i in temp])
            for i in env.world.agents:
                terminal_reward += np.sum(np.multiply(np.multiply(temp,i.x),i.c))

            if final:
                reward_ratio[ep_timesteps].append(terminal_reward/ideal_reward)

            if terminal:
                cbaa_reward += terminal_reward/ideal_reward

            # For next timestep
            env_obs_n = new_env_obs_n
            eval_reward += reward_n[0]
            ep_timesteps += 1
            for i,action in enumerate(student_action_n):
                average_action[i] += action

            if terminal:
                break

    eval_reward /= n_eval
    cbaa_reward /= n_eval
    if not final:
        log[args.log_name].info("Evaluation Reward {:.5f} at episode {}".format(eval_reward, total_ep_count))
        tb_writer.add_scalar("reward/eval_reward", eval_reward, total_ep_count)
        log[args.log_name].info("Evaluation cbaa Reward {:.5f} at episode {}".format(cbaa_reward, total_ep_count))
        tb_writer.add_scalar("reward/cbaa_reward", cbaa_reward, total_ep_count)

    return eval_reward


def collect_one_traj(student_n, env, log, args, tb_writer):
    global total_step_count, total_ep_count, session_step_count, session_ep_count 

    env_obs_n, ideal_reward = env.reset()
    ep_reward = 0
    ep_timesteps = 0
    terminal = False
    reset_noise(student_n)
    reward_buffer = False

    while True:
        # Select action randomly or according to policy
        student_obs_n = get_student_obs(env_obs_n, ep_timesteps, args.ep_max_timesteps, args)

        student_action_n = [] 
        for student, student_obs in zip(student_n, student_obs_n):
            student_action = student.select_stochastic_action(np.array(student_obs), total_ep_count)
            student_action_n.append(student_action)

        # Perform action
        new_env_obs_n, env_reward_n, env_done_n, _ = env.step(copy.deepcopy(student_action_n)) 
        if ep_timesteps + 1 == args.ep_max_timesteps:
            terminal = True

        # Update student memory
        new_student_obs_n = get_student_obs(new_env_obs_n, ep_timesteps + 1, args.ep_max_timesteps, args)
        student_reward_n = env_reward_n
        student_done = terminal if args.student_done else False

        for i_student, student in enumerate(student_n):
            # NOTE Centralized` learning
            student.add_memory(
                obs=student_obs_n,
                new_obs=new_student_obs_n,
                action=student_action_n,
                reward=student_reward_n,
                done=[float(student_done) for i in range(len(student_n))])

        # For next timestep
        env_obs_n = new_env_obs_n
        ep_timesteps += 1
        total_step_count += 1
        session_step_count += 1
        ep_reward += env_reward_n[0]
        
        if terminal:
            total_ep_count += 1
            session_ep_count += 1

            log[args.log_name].info("Train episode reward {:.5f} at episode {}".format(ep_reward, total_ep_count))
            tb_writer.add_scalar("reward/train_ep_reward", ep_reward, total_ep_count)
            for i,student in enumerate(student_n):
            	tb_writer.add_scalar("action/censoring_probability" + str(student.i_agent), 
                                        student_action_n[student.i_agent][0], total_ep_count)

            return ep_reward


def train(student_n, env, log, tb_writer, args):
    while True: 
        if total_ep_count == args.total_ep_count: # Test for 200 episodes after learning
            eval_progress(
                env=env, 
                student_n=student_n, 
                n_eval=200,
                log=log, 
                tb_writer=tb_writer, 
                args=args,
                final=True)
            pickle_out = open("dict6a.pickle","wb")
            pickle.dump(reward_ratio, pickle_out)
            pickle_out.close()
            pickle_VoI = open("dictpolicy6a.pickle","wb")
            pickle.dump(VoI_dict, pickle_VoI)
            pickle_VoI.close()
            for i,student in enumerate(student_n):
                torch.save(student.policy.actor.state_dict(), "policy" + str(i) + "final.pickle")
            break

        if total_ep_count % 50 == 0: # evaluate progress after every 50 episodes
            eval_progress(
                env=env, 
                student_n=student_n, 
                n_eval=10,
                log=log, 
                tb_writer=tb_writer, 
                args=args,
                final=False)

        reward = collect_one_traj(
            student_n=student_n, 
            env=env, 
            log=log,
            args=args, 
            tb_writer=tb_writer)

        for i,student in enumerate(student_n):
            debug = student.update_policy(total_ep_count, agent_n=student_n, index=i )

            tb_writer.add_scalar(
                "loss/actor_loss" + str(student.i_agent), debug["actor_loss"], total_ep_count)
            tb_writer.add_scalar(
                "loss/critic_loss" + str(student.i_agent), debug["critic_loss"], total_ep_count)
            




