import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntFinishLineVerticalEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant_finishline.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):
        #xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        #xposafter = self.get_body_com("torso")[0]
        #forward_reward = (xposafter - xposbefore)/self.dt
        goal = np.array([0.0, 5.0])
        pos = self.get_body_com("torso")[:2]
        forward_reward = np.exp(-(np.linalg.norm(pos-goal)**2) / 10.0)
        ctrl_cost = 0.0 #.5 * np.square(a).sum()
        contact_cost = 0.0 #0.5 * 1e-3 * np.sum(
            # np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()

        return ob, reward, done, {'pos':self.sim.data.qpos.flat[:2]}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],  # full state
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
