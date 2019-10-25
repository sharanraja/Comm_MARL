import logging
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args):
    log = {}                                                                                                                                        
    set_logger(
        logger_name=args.log_name, 
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    # Log arguments
    for (name, value) in vars(args).items():
        log[args.log_name].info("{}: {}".format(name, value))

    return log


def make_env(args):
    # Modified from: https://github.com/openai/maddpg/blob/master/experiments/train.py
    scenario = scenarios.load(args.env_name + ".py").Scenario()
    world = scenario.make_world(args)
    done_callback = None

    env = MultiAgentEnv(
        world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        done_callback=scenario.done)

    return env


def normalize(value, min_value, max_value):
    return 2. * (value - min_value) / float(max_value - min_value) - 1.
