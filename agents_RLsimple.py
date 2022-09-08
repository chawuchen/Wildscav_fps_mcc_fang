import random
import numpy as np
from typing import NamedTuple
from inspirai_fps.gamecore import AgentState
from inspirai_fps.utils import get_position, get_picth_yaw, get_distance
import math
from collections import defaultdict
from ppo import PPO
import torch

SEARCHING_TIMES = 150
DEPTH_MAP_FAR = 200
DEPTH_MAP_WIDTH = 64
DEPTH_MAP_HEIGHT = 36
MAX_FAR = 10
FIELD_SIZE = 300
MAX_POINT_NUM = 5000
FX = 55.425
CAMERA_HEIGHT = 1.49581 - 0.75
THICKNESS = 1
CLOCKWISE_SPEED = 9
LOCAL_SIZE = 20
SUPPLY_INTERVAVL = 30
HEIGHT_THRESHOLD = 2
GAUSS_PARAMETER = 10
JUMP_INT = 30
CHECK_INT = 100
TRIES_PER_BLOCK = 1
# DO NOT MODIFY THIS CLASS

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class NavigationAction(NamedTuple):
    walk_dir: float
    walk_speed: float
    turn_lr_delta: float
    look_ud_delta: float
    jump: bool


# DO NOT MODIFY THIS CLASS
class SupplyGatherAction(NamedTuple):
    walk_dir: float
    walk_speed: float
    turn_lr_delta: float
    look_ud_delta: float
    jump: bool
    pickup: bool


# DO NOT MODIFY THIS CLASS
class SupplyBattleAction(NamedTuple):
    walk_dir: float
    walk_speed: float
    turn_lr_delta: float
    look_ud_delta: float
    jump: bool
    pickup: bool
    attack: bool
    reload: bool


def get_pitch_yaw(x, y, z):
    pitch = np.arctan2(y, (x ** 2 + z ** 2) ** 0.5) / np.pi * 180
    yaw = np.arctan2(x, z) / np.pi * 180
    return pitch, yaw


class AgentNavigation:
    """
    This is a template of an agent for the navigation task.
    TODO: Modify the code in this class to implement your agent here.
    """

    def __init__(self, episode_info) -> None:
        has_continuous_action_space = False  # continuous action space; else discrete
        action_std = 0.6  # starting std for action distribution (Multivariate Normal)
        #####################################################

        ## Note : print/log frequencies should be > than max_ep_len

        ################ PPO hyperparameters ################

        K_epochs = 80  # update policy for K epochs in one PPO update
        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.99  # discount factor
        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network

        #####################################################
        state_dim = 6
        lstm_out = 64
        action_dim = 4

        self.ppo_agent = PPO(state_dim, lstm_out, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                             has_continuous_action_space, action_std)
        self.ppo_agent.load("./submission/PPO_agent.pth")
        self.hx = torch.zeros(lstm_out).to(device)
        self.cx = torch.zeros(lstm_out).to(device)
        self.map2d = np.zeros((FIELD_SIZE * 2, FIELD_SIZE * 2))
        self.target_location = episode_info['target_location']
        self.start_location = episode_info['start_location']
        self.pos = self.start_location
        self.old_pos = self.pos
        self.last_walk_dir = 0
        self.clockwise = 0
        self.map2d_dir = np.full((FIELD_SIZE * 2, FIELD_SIZE * 2, action_dim), TRIES_PER_BLOCK)
        self.last_walk_idx = [0, 0, 0, 0]
        self.pitch = 0

    def get_local_map(self, p1):
        return np.copy(
            self.map2d[p1[0] - LOCAL_SIZE: p1[0] + LOCAL_SIZE + 1, p1[1] - LOCAL_SIZE: p1[1] + LOCAL_SIZE + 1])

    def reset_map2d(self):
        p1 = [int(self.pos[0]) + FIELD_SIZE, int(self.pos[2]) + FIELD_SIZE]
        self.map2d[p1[0] - LOCAL_SIZE: p1[0] + LOCAL_SIZE + 1, p1[1] - LOCAL_SIZE: p1[1] + LOCAL_SIZE + 1] = 0

    def get_legal_action(self):
        cur = self.pos
        p1 = [int(cur[0]) + FIELD_SIZE, int(cur[2]) + FIELD_SIZE]
        self.map2d[p1[0]][p1[1]] = 0
        direction = [v2 - v1 for v1, v2 in zip(self.pos, self.target_location)]
        direction = get_pitch_yaw(*direction)[1]
        direction = direction % 360
        direction = (- ((int(direction) + 45) // 90)) % 4
        legal_action = self.map2d_dir[p1[0]][p1[1]]
        legal_action[np.where(legal_action > 0)] = 1
        legal_action[np.where(legal_action <= 0)] = 0
        if min(self.map2d[p1[0]][p1[1] + 1], self.map2d[p1[0] + 1][p1[1]], self.map2d[p1[0]][p1[1] - 1],
               self.map2d[p1[0] - 1][p1[1]]) == 1:
            self.reset_map2d()
        if self.map2d[p1[0]][p1[1] + 1] == 1:
            legal_action[direction] = 0
        if self.map2d[p1[0] + 1][p1[1]] == 1:
            legal_action[(direction + 1) % 4] = 0
        if self.map2d[p1[0]][p1[1] - 1] == 1:
            legal_action[(direction + 2) % 4] = 0
        if self.map2d[p1[0] - 1][p1[1]] == 1:
            legal_action[(direction + 3) % 4] = 0
        if self.pitch > 30:
            legal_action[0] = 0
        blocked = self.if_blocked(self.pos, self.target_location)
        if blocked:
            legal_action[0] = 0
        if max(legal_action) <= 0:

            self.map2d_dir[p1[0]][p1[1]] += TRIES_PER_BLOCK
            legal_action = self.map2d_dir[p1[0]][p1[1]]
            if self.map2d[p1[0]][p1[1] + 1] == 1:
                legal_action[direction] = 0
            if self.map2d[p1[0] + 1][p1[1]] == 1:
                legal_action[(direction + 1) % 4] = 0
            if self.map2d[p1[0]][p1[1] - 1] == 1:
                legal_action[(direction + 2) % 4] = 0
            if self.map2d[p1[0] - 1][p1[1]] == 1:
                legal_action[(direction + 3) % 4] = 0
            legal_action = np.copy(legal_action)
            legal_action[(self.last_walk_idx.index(1) + 2) % 4] = 0

            if blocked:
                legal_action[0] = 0
            if max(legal_action) <= 0:
                legal_action += TRIES_PER_BLOCK
                self.reset_map2d()
        legal_action[np.where(legal_action > 0)] = 1
        legal_action[np.where(legal_action <= 0)] = 0
        return legal_action

    def if_blocked(self, cur, tar):
        line_len = max(abs(int(cur[0]) - int(tar[0])), abs(int(cur[2]) - int(tar[2])))
        if line_len < 2:
            line_len = 2
        line1 = np.linspace(cur[0] + FIELD_SIZE, tar[0] + FIELD_SIZE, line_len)
        line2 = np.linspace(cur[2] + FIELD_SIZE, tar[2] + FIELD_SIZE, line_len)
        for i in range(line_len):
            if self.map2d[int(line1[i])][int(line2[i])] == 1:
                return True
        return False

    def obstacles_on_eyes(self, state):
        """Define a simple navigation policy"""

        depth = state.depth_map[DEPTH_MAP_HEIGHT // 2 - 2:DEPTH_MAP_HEIGHT // 2 + 2, :]
        analyse = []
        for i in range(DEPTH_MAP_WIDTH):
            if max(depth[:, i]) - min(depth[:, i]) < 0.05 and depth[3][i] < MAX_FAR:
                analyse.append(
                    [i - (DEPTH_MAP_WIDTH - 1) / 2, DEPTH_MAP_HEIGHT // 2 - (DEPTH_MAP_HEIGHT - 1) / 2, depth[3][i]])
        if len(analyse) == 0:
            return 0
        analyse = np.array(analyse)
        map_x_pos = analyse[:, 0].flatten()
        map_y_pos = analyse[:, 1].flatten()
        flatten_dmap = analyse[:, 2].flatten()
        map_x_pos = np.multiply(map_x_pos, flatten_dmap) / FX
        map_y_pos = np.multiply(map_y_pos, flatten_dmap) / - FX
        x_d_y_cam = np.array([map_x_pos, flatten_dmap, map_y_pos, np.full(np.shape(map_y_pos), 1)])
        pos_x, pos_y, pos_z = self.pos
        camera_yaw = state.yaw / 180 * math.pi

        transformation_matrix = np.array([[math.cos(camera_yaw), math.sin(camera_yaw), 0, pos_x],
                                          [-math.sin(camera_yaw), math.cos(camera_yaw), 0, pos_z],
                                          [0, 0, 1, pos_y + CAMERA_HEIGHT]])
        point_cloud = np.dot(transformation_matrix, x_d_y_cam).transpose()

        for i in range(point_cloud.shape[0]):
            self.map2d[int(point_cloud[i][0]) + FIELD_SIZE][int(point_cloud[i][1]) + FIELD_SIZE] = 1
        return 0

    def state_to_obs(self, state):

        depth = state.depth_map[DEPTH_MAP_HEIGHT // 2 - 2:DEPTH_MAP_HEIGHT // 2 + 2, DEPTH_MAP_WIDTH // 2]
        see_obstacle = 0
        if max(depth) - min(depth) < 0.03 and min(depth) < 3:
            see_obstacle = 1
        change = [v2 - v1 for v1, v2 in zip(self.old_pos, self.pos)]
        moved = get_distance(self.old_pos, self.pos)
        self.old_pos = self.pos
        direction = [v2 - v1 for v1, v2 in zip(self.pos, self.target_location)]
        yaw = get_pitch_yaw(*direction)[1]
        one_hot = [0, 0, 0, 0, 0, 0, 0, 0]
        hot = int((int(yaw) + 22.5) % 360 // 45)
        one_hot[hot] = 1
        walk_obstacle = 0
        if moved * math.cos((self.last_walk_dir - get_pitch_yaw(*change)[1]) / 180 * math.pi) <= (
                0.3 if self.pos[1] > -4 else 0.2):
            walk_obstacle = 1
        feature_ex = np.array([see_obstacle, walk_obstacle] + self.last_walk_idx)
        return feature_ex

    def get_turn_lr(self, camera_yaw, yaw):
        if (camera_yaw - yaw) % 360 < 180:
            self.clockwise = - ((camera_yaw - yaw) % 360) / 5
        else:
            self.clockwise = (360 - (camera_yaw - yaw) % 360) / 5

    def index_to_action(self, index, direct_yaw, walk_speed=10):
        if index > 4:
            jump = True
        else:
            jump = False
        yaw_list = [direct_yaw, direct_yaw + 90, direct_yaw + 180, direct_yaw + 270]
        yaw = yaw_list[index % 4]
        return [yaw, walk_speed, self.clockwise, 0, jump]

    @staticmethod
    def get_distance(a, b):
        return np.linalg.norm([v2 - v1 for v1, v2 in zip(a, b)])

    def set_obstacle(self, direction, cur_pos):
        pos = [int(cur_pos[0]) + FIELD_SIZE, int(cur_pos[1]), int(cur_pos[2]) + FIELD_SIZE]
        direction = direction % 360
        value = 1
        if 0 <= direction < 45 or 315 <= direction < 360:
            self.map2d[pos[0]][pos[2] + 1] = value
        elif 45 <= direction < 135:
            self.map2d[pos[0] + 1][pos[2]] = value
        elif 135 <= direction < 225:
            self.map2d[pos[0]][pos[2] - 1] = value
        elif 225 <= direction < 315:
            self.map2d[pos[0] - 1][pos[2]] = value

    def my_policy(self, state, ts):
        self.state = state
        self.pos = get_position(state)
        self.obstacles_on_eyes(state)
        last_direction = [v2 - v1 for v1, v2 in zip(self.old_pos, self.pos)]
        p1 = [int(self.old_pos[0]) + FIELD_SIZE, int(self.old_pos[2]) + FIELD_SIZE]
        p2 = [int(self.pos[0]) + FIELD_SIZE, int(self.pos[2]) + FIELD_SIZE]
        if p1 != p2:
            self.map2d_dir[p1[0]][p1[1]][self.last_walk_idx.index(1)] -= 1
        x = self.get_distance(self.pos, self.old_pos) \
            * math.cos((self.last_walk_dir - get_pitch_yaw(*last_direction)[1]) / 180 * math.pi)
        direction = [v2 - v1 for v1, v2 in zip(self.pos, self.target_location)]
        self.pitch = get_pitch_yaw(*direction)[0]
        if ts > 20 and x <= (0.3 if self.pos[1] > -4 else 0.2):
            self.set_obstacle(self.last_walk_dir, self.old_pos)
        self.old_pos = self.pos
        legal_action = self.get_legal_action()
        legal_action = torch.tensor(legal_action).to(device)
        action, self.hx, self.cx = self.ppo_agent.select_action(self.state_to_obs(state), self.hx, self.cx,
                                                                legal_action, testing=True)

        yaw = get_pitch_yaw(*direction)[1]
        self.get_turn_lr(state.yaw, yaw)
        self.last_walk_idx = [0, 0, 0, 0]
        self.last_walk_idx[action] = 1
        true_action = self.index_to_action(action, direct_yaw=yaw)

        self.last_walk_dir = true_action[0]
        return true_action

    def act(self, ts: int, state: AgentState) -> NavigationAction:
        walk_dir, walk_speed, turn_lr, look_ud, jump = self.my_policy(state, ts)

        return NavigationAction(
            walk_dir=walk_dir,
            walk_speed=walk_speed,
            turn_lr_delta=turn_lr,
            look_ud_delta=look_ud,
            jump=jump,
        )
