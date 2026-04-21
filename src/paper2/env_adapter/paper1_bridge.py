class Paper1EnvBridge:
    """
    将来用于把论文1修正后的环境适配为 Paper2EnvProtocol。
    当前阶段先留占位，不实现具体逻辑。
    """

    def __init__(self, env):
        self.env = env

    def reset(self, seed=None):
        raise NotImplementedError("Wait for paper1 fixed environment.")

    def step(self, action):
        raise NotImplementedError("Wait for paper1 fixed environment.")

    def get_aircraft_state(self):
        raise NotImplementedError("Wait for paper1 fixed environment.")

    def get_target_truth(self):
        raise NotImplementedError("Wait for paper1 fixed environment.")

    def get_no_fly_zones(self):
        raise NotImplementedError("Wait for paper1 fixed environment.")