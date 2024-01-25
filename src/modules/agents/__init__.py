# 将类放在字典里的目的就是为了解耦，方便以后对模块进行增加
REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent