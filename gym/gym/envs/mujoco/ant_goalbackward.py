import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntGoalBackwardEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant_smallfourgoals.xml', 5)
        utils.EzPickle.__init__(self)

    def _step(self, a):

        #goal_pos = np.array([[5.0, 0.0],[-5.0, 0.0],[0.0, 5.0],[0.0, -5.0]])
        goal_pos = np.array([[-5.0, 0.0]])

        self.do_simulation(a, self.frame_skip)
        pos = self.get_body_com("torso")[:2]
        
        dist = [np.linalg.norm(pos-goal) for goal in goal_pos]
        #dist_reward = np.exp(-np.min(dist)**2 / 10.0)
        dist_reward = np.exp(-np.min(dist)**2 / 2.0)
        forward_reward = dist_reward
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = forward_reward #forward_reward - ctrl_cost - contact_cost + survive_reward
        #state = self.state_vector()
        #notdone = np.isfinite(state).all() \
        #    and state[2] >= 0.2 and state[2] <= 1.0
        #done = not notdone
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(pos=self.sim.data.qpos.flat[:2])

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
