import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

# 主函数运行
def run(_run, _config, _log):
    # check args sanity
    # 大概就是核实和验证一下参数，比如核实本机是否有cuda
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    # _log.info(args.device)
    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")

    # 把实验的参数转换成字符串在控制台输出
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token

    # 判断是否要用tensorboard进行日志打印
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # 日志打印使用默认的scared
    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    # 在这个函数里面开启训练
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    # 并行训练模式下才有用， 启动所有除了主线程外的所有线程
    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")


    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


# 主要运行的函数
# 这里面会创建一系列进程来收集数据
def run_sequential(args, logger):
    # Init runner so we can get env info
    # 创建一个运行器实例
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    # 获取环境信息
    env_info = runner.get_env_info()
    # 用这个信息更新参数
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    # 状态的维度
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }

    groups = {
        "agents": args.n_agents
    }

    # 用来对动作数据进行预处理，转换成独热编码
    preprocess = {
        # 动作的独热编码
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    # 经验回放缓冲
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    # logger.console_logger.info(args)
    # Setup multiagent controller here

    # 创建多智能体控制器实例
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    # 给运行器设置好数据格式、数据的groups、预处理函数、控多智能体控制器
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    # 定义一个学习器, q_learn学习器
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # 判断是否要加载训练好的模型
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    # 注意：后面带t的，都是指时间步，带time的应该都是指真实时间
    episode = 0
    # 最后一次测试的时间步
    last_test_T = -args.test_interval - 1
    # 最后一次打印日志的时间步
    last_log_T = 0
    # 模型保存的时间
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    # 当运行时间步没达到最大时
    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        # 在这里启动游戏，获得一个episode_batch的数据

        # episode_batch里只有一个episode的数据
        episode_batch = runner.run(test_mode=False)

        # 把这个episode_batch数据放进缓冲区里
        buffer.insert_episode_batch(episode_batch)

        # 收集了32个episode后，才可以采样，因为一个batch_size是32个episode的数据
        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            # 返回这32个episodes里面最长的那个序列的长度
            max_ep_t = episode_sample.max_t_filled()

            # 对这32个episodes的数据按照最长序列的长度进行切片
            episode_sample = episode_sample[:, :max_ep_t]

            # 有显卡的话将数据发送到显卡
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    # 当参数“use_cuda”设置为True且无cuda可用时，设置成False
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        # 注意 // 表示除完后向下取整
        # 作用是保证test_nepisode是batch_size_run的整数倍
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config