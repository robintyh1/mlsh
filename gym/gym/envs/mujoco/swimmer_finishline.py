import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SwimmerFinishLineEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    swimmer gets reward for running fast in the first dimension
    running forward allows the agent to get a bonus reward after crossing
    a finishing line
    """
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'swimmer_finishline.xml', 4)
        utils.EzPickle.__init__(self)

    def _step(self, a):

        self.do_simulation(a, self.frame_skip)

        xvelocity = self.sim.data.qvel[0]
        reward = xvelocity ** 2
        xpos = self.sim.data.qpos[0]
        if xpos > 10.0:
            reward += 10.0
        ob = self._get_obs()
        return ob, reward, False, {'pos': self.sim.data.qpos[:2]}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()
