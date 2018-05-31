from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from gym.envs.mujoco.ant import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.reacher import ReacherEnv
from gym.envs.mujoco.swimmer import SwimmerEnv
from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
from gym.envs.mujoco.pusher import PusherEnv
from gym.envs.mujoco.thrower import ThrowerEnv
from gym.envs.mujoco.striker import StrikerEnv

from gym.envs.mujoco.swimmer_bandits import SwimmerBanditsEnv
from gym.envs.mujoco.ant_bandits import AntBanditsEnv
from gym.envs.mujoco.obstacles import Obstacles

from gym.envs.mujoco.ant_movement import AntMovementEnv
from gym.envs.mujoco.ant_obstacles import AntObstaclesEnv
from gym.envs.mujoco.ant_obstaclesbig import AntObstaclesBigEnv
from gym.envs.mujoco.ant_obstaclesgen import AntObstaclesGenEnv
from gym.envs.mujoco.humanoid_course import HumanoidCourseEnv
from gym.envs.mujoco.humanoid_seq import HumanoidSeqEnv

from gym.envs.mujoco.ant_trial import AntTrialEnv
from gym.envs.mujoco.ant_running import AntRunEnv
from gym.envs.mujoco.ant_running_full import AntRunFullEnv
from gym.envs.mujoco.ant_running_wide_hallway import AntRunWideHallwayEnv
from gym.envs.mujoco.ant_running_narrow_hallway import AntRunNarrowHallwayEnv
from gym.envs.mujoco.ant_running_Umaze import AntRunUMazeEnv

from gym.envs.mujoco.swimmer_finishline import SwimmerFinishLineEnv
from gym.envs.mujoco.half_cheetah_finishline import HalfCheetahFinishLineEnv
from gym.envs.mujoco.ant_finishline import AntFinishLineEnv
from gym.envs.mujoco.ant_finishline_vertical import AntFinishLineVerticalEnv
from gym.envs.mujoco.ant_running_fixedinit import AntRunFixedInitEnv

from gym.envs.mujoco.ant_backward import AntBackwardEnv
from gym.envs.mujoco.ant_upward import AntUpwardEnv
from gym.envs.mujoco.ant_downward import AntDownwardEnv

from gym.envs.mujoco.ant_fourgoals import AntFourgoalsEnv
from gym.envs.mujoco.ant_smallfourgoals import AntSmallFourgoalsEnv

from gym.envs.mujoco.ant_goalforward import AntGoalForwardEnv
from gym.envs.mujoco.ant_goalupward import AntGoalUpwardEnv
from gym.envs.mujoco.ant_goaldownward import AntGoalDownwardEnv
from gym.envs.mujoco.ant_goalbackward import AntGoalBackwardEnv

from gym.envs.mujoco.ant_finishlinebonus import AntFinishLineBonusEnv
from gym.envs.mujoco.ant_finishlinebonuswall import AntFinishLineBonusWallEnv

from gym.envs.mujoco.pusher_multigoal_hard import PusherMultigoalHardEnv
from gym.envs.mujoco.pusher_multigoal import PusherMultigoalEnv
from gym.envs.mujoco.thrower_multigoal import ThrowerMultigoalEnv
from gym.envs.mujoco.reacher_multigoals import ReacherMultigoalEnv
