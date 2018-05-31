import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntFourgoalsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant_fourgoals.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):

        goal_pos = np.array([[14.0, 0.0],[-14.0, 0.0],[0.0, 14.0],[0.0, -14.0]])

        self.do_simulation(a, self.frame_skip)
        pos = self.get_body_com("torso")[:2]
        
        dist = [np.linalg.norm(pos-goal) for goal in goal_pos]
        dist_reward = np.exp(-np.min(dist)**2 / 100.0)
        #survive_reward = 1.0
        reward = dist_reward# + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(pos=self.sim.data.qpos.flat[0])

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
