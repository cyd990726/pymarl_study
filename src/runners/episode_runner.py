from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # 定义一个环境的属性
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        # 一个episode的最大长度
        self.episode_limit = self.env.episode_limit

        # 记录一个episode的长度
        self.t = 0
        # 目前已经训练的所有episode中的总共的步数
        self.t_env = 0

        # 记录训练模式下每个episode的return值
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        # partial函数是给函数EpisodeBatch预先固定好值
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    # 这个函数启动一次游戏，获得一个episode的数据
    def run(self, test_mode=False):
        #初始化一次游戏
        self.reset()

        terminated = False
        episode_return = 0

        # 设置隐藏层
        self.mac.init_hidden(batch_size=self.batch_size)

        # 这里执行一个episode
        while not terminated:
            # batch中每一个时间步都存了一个pre_transition_data和post_transition_data

            # 当前状态，可选动作，观测
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }
            # 插入过渡前的数据
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1

            # 智能体挑选动作，得到的action是一个1*5的张量，因为agents为5，每个元素表示每个智能体的动作
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # 执行动作
            # reward是一个浮点值，terminated是一个bool值，env_info是一个字典{‘battle_won’:False, 'dead_allies':0, 'dead_enemies':0}
            reward, terminated, env_info = self.env.step(actions[0])

            # 当前episode的累积奖励
            episode_return += reward

            # 将actions， reward， terminated打包成字典
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            # 插入过渡后的数据
            self.batch.update(post_transition_data, ts=self.t)

            # self.t是一个episode的步长
            self.t += 1

        # 执行完一个episode后的状态，可选动作，观测
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        # 根据最后一个存储的transition挑选动作,很好奇为什么要这样
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)


        # 看当前是测试模式还是训练模式，测试模式的话就修改test_stats，训练模式的话就修改train_stats
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        # 打印日志的前缀
        log_prefix = "test_" if test_mode else ""

        # set(cur_states) | set(env_info)的意思是将两个字典的key合并成一个set
        # 即将当前episodes的env_info加入cur_state中
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})

        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0) # cur_states.get("ep_length", 0)表示没有ep_length这个字段的话返回0

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        # 打印日志，训练模式下步数至少10000才打印一次
        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
