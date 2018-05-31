import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

#goallist = np.array([[0.1,0.1],[-0.1,0.1],[0.1,-0.1],[-0.1,-0.1]])
goallist = np.array([[0,0.12],[0,-0.12]])

class ReacherMultigoalEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    cumulative_vec = np.array([0.0, 0.0])

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_multigoal.xml', 2)

    def _step(self, a):
        #vec = self.get_body_com("fingertip")-self.get_body_com("target")
        pos = self.get_body_com("fingertip")[:2]
        veclist = [np.linalg.norm(pos-goal) for goal in goallist]
        self.cumulative_vec += veclist
        reward_dist = - np.min(veclist)
        #reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        #if np.linalg.norm(goallist[0] - pos) < 0.04:
        #     reward += 1.0
        #return ob, reward, done, {'pos':self.get_body_com("fingertip")[:2]}  #dict(pos=pos, reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return ob, reward, done, {'pos':self.cumulative_vec}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        #while True:
        #    self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
        #    if np.linalg.norm(self.goal) < 2:
        #        break
        #goallist = np.array([[0.1,0.1],[0.1,-0.1]])
        #idx = np.random.choice(np.arange(goallist.shape[0]))
        #self.goal = goallist[idx]       
        self.goal = 0
        self.cumulative_vec = np.array([0.0, 0.0])

        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            #self.sim.data.qpos.flat[2:],  # goal pos
            self.sim.data.qvel.flat[:2],
            #self.get_body_com("fingertip") - self.get_body_com("target")
            self.get_body_com("fingertip")
        ])
