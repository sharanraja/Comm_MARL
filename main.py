import torch
import argparse
import os
import numpy as np
from misc.utils import set_log, make_env
from tensorboardX import SummaryWriter
from trainer.train import train 
from policy.ddpg import DDPG, Actor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_policy(env, log, args, name, i_agent):
    if name == "student":
        from policy.student import Student
        policy = Student(env=env, log=log, name=name, args=args, i_agent=i_agent)
    else:
        raise ValueError("Invalid name")
    return policy


def main(args):
    # Create dir
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    log = set_log(args)

    # Create env
    env = make_env(args)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize policy
    student_n = [set_policy(env, log, args, name="student", i_agent=i_agent) for i_agent in range(args.n_student)]

    # Start train
    train(
        student_n=student_n, 
        env=env, 
        log=log,
        tb_writer=tb_writer,
        args=args)
    
    if not os.path.exists("./saved_model"):
        os.makedirs("./saved_model")

    # torch.save(common_policy.policy.actor.state_dict(), "./saved_model/7agentVoIaverage.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument(
        "--discount", default=0.99, type=float, 
        help="Discount factor")
    parser.add_argument(
        "--tau", default=0.001, type=float, 
        help="Target network update rate")
    parser.add_argument(
        "--batch-size", default=100, type=int, 
        help="Batch size for both actor and critic")
    parser.add_argument(
        "--actor-lr", default=0.0001, type=float,
        help="Learning rate for actor")
    parser.add_argument(
        "--critic-lr", default=0.001, type=float,
        help="Learning rate for critic")
    parser.add_argument(
        "--reward-lr", default=0.01, type=float,
        help="Learning rate for reward estimator")
    parser.add_argument(
        "--policy-freq", default=2, type=int,
        help="Frequency of delayed policy updates")

    # Noise
    parser.add_argument(
        "--ou-theta", default=0.15, type=float, 
        help="Std of Gaussian exploration noise")
    parser.add_argument(
        "--ou-sigma", default=0.2, type=float, 
        help="Sigma for OU Noise")
    parser.add_argument(
        "--gauss-std", default=0.1, type=float, 
        help="Std of Gaussian exploration noise")

    # Student
    parser.add_argument(
        "--n-student", default=2, type=int,
        help="Number of students")
    parser.add_argument(
        "--n-task", default=100, type=int,
        help="Number of tasks")
    parser.add_argument(
        "--maxTask", default=50, type=int,
        help="Task limit for an agent")
    parser.add_argument(
        "--student-done", default=True, action="store_true",
        help="Set student done or not")
    parser.add_argument(
        "--student-noise-type", default="ou", type=str, 
        help="Noise type")
    parser.add_argument(
        "--student-train-type", default="centralized", type=str, 
        help="Train type whether to centralized or independent")

    # Env
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-max-timesteps", type=int, required=True,
        help="Episode is terminated when max timestep is reached.")
    parser.add_argument(
        "--total-ep-count", type=int, required=True,
        help="Training is terminated when total_ep_count is reached.")



    # Misc
    parser.add_argument(
        "--seed", default=0, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")

    args = parser.parse_args()

    # Set log name
    args.log_name = \
        "env::%s_seed::%s_student_noise_type::%s_student_train_type::%s_batch::%s_prefix::%s_log" % (
            args.env_name, str(args.seed), args.student_noise_type, args.student_train_type, args.batch_size, args.prefix)

    main(args=args)
