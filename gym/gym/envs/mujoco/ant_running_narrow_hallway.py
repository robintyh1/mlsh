import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntRunNarrowHallwayEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Ant running env with complete state (qpos included)
    """
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant_running_narrow_hallway.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        velocity = self.sim.data.qvel[:2]
        reward = np.sum(velocity**2)
        done = False
        return ob, reward, done, {'pos': self.sim.data.qpos.flat[:2]}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
