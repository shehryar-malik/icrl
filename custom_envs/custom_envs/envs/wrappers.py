import gym


class BrokenJoint(gym.Wrapper):
    """Wrapper that disables one coordinate of the action, setting it to 0."""
    def __init__(self, env, broken_joint=None):
        super(BrokenJoint, self).__init__(env)
        # Change dtype of observation to be float32.
        self.observation_space = gym.spaces.Box(
                low=self.observation_space.low,
                high=self.observation_space.high,
                dtype=np.float32,
        )
        if broken_joint is not None:
            assert 0 <= broken_joint <= len(self.action_space.low) - 1
        self.broken_joint = broken_joint

    def step(self, action):
        action = action.copy()
        if self.broken_joint is not None:
            action[self.broken_joint] = 0

        return super(BrokenJoint, self).step(action)


