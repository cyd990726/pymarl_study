# 这只是一个接口，里面的方法都必须要由子类实现
# StarCraft2Env就是继承的这个接口，并且实现了其中的方法

class MultiAgentEnv(object):

    def step(self, actions):
        """ Returns reward, terminated, info """
        raise NotImplementedError

    # 以一个列表的方式返回所有智能体的观测
    def get_obs(self):
        """ Returns all agent observations in a list """
        raise NotImplementedError

    # 根据agent_id返回指定智能体的观测
    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise NotImplementedError

    # 返回观测的shape。即若观测是一个1*4的张量，则返回元组(1, 4)
    def get_obs_size(self):
        """ Returns the shape of the observation """
        raise NotImplementedError

    # 应该是返回全局的状态
    def get_state(self):
        raise NotImplementedError

    # 返回全局状态的shape
    def get_state_size(self):
        """ Returns the shape of the state"""
        raise NotImplementedError

    # 返回当前状态下可选动作
    def get_avail_actions(self):
        raise NotImplementedError

    # 根据agent_id返回指定智能体的可选动作
    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        raise NotImplementedError

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        raise NotImplementedError

    # 初始化一次游戏，返回初始的观测和state
    def reset(self):
        """ Returns initial observations and states"""
        raise NotImplementedError

    # 渲染
    def render(self):
        raise NotImplementedError

    # 关闭环境
    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    # 保存经验回放？
    def save_replay(self):
        raise NotImplementedError

    # 获取环境信息，比如状态空间的大小， 每个观测的大小， 总共动作的数目，智能体的数目， 每个episode的极限长度
    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
