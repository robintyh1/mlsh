import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class HalfCheetahFinishLineEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah_finishline.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        xvel = self.sim.data.qvel.flat[0]
        reward = xvel ** 2
        xpos = self.sim.data.qpos.flat[0]
        if xpos > 50:
            reward += 1    
        done = False
        return ob, reward, done, {'pos': self.sim.data.qpos.flat[:2]}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
