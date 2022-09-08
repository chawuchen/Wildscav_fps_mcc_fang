import random
import numpy as np
from typing import NamedTuple
from inspirai_fps.gamecore import AgentState
from inspirai_fps.utils import get_position, get_picth_yaw
import math

# Hyperparameter
DEPTH_MAP_FAR = 200
DEPTH_MAP_WIDTH = 64
DEPTH_MAP_HEIGHT = 36
MAX_FAR = 10
FIELD_SIZE = 250
MAX_POINT_NUM = 5000
FX = 55.425
CAMERA_HEIGHT = 1.49581 - 0.75
THICKNESS = 1
LOCAL_SIZE = 20
SUPPLY_INTERVAVL = 30
HEIGHT_THRESHOLD = 1
GAUSS_PARAMETER = 10


# DO NOT MODIFY THIS CLASS
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
    pitch = np.arctan2(y, (x**2 + z**2) ** 0.5) / np.pi * 180
    yaw = np.arctan2(x, z) / np.pi * 180
    return pitch, yaw


class AgentNavigation:
    """
    This is a template of an agent for the navigation task.
    TODO: Modify the code in this class to implement your agent here.
    """
    def __init__(self, episode_info) -> None:
        self.s = 0
        self.count = 0
        self.checkcount = 0
        self.episode_info = episode_info
        self.last_walk_dir = 0
        self.clockwise = 0  # actually yaw speed
        self.target_location = episode_info['target_location']
        self.start_location = episode_info['start_location']
        self.pos = self.start_location
        self.checkpoint = [self.start_location]
        self.map2d = np.zeros((FIELD_SIZE * 2, FIELD_SIZE * 2))
        self.old_position = self.start_location
        self.checkpoint_pos = self.checkpoint[-1]

    def reset_map2d(self):
        self.map2d = np.zeros((FIELD_SIZE * 2, FIELD_SIZE * 2))

    def get_turn_lr(self, camera_yaw, yaw):  # Adjust camera_yaw to yaw in one step
        if (camera_yaw - yaw) % 360 < 180:
            self.clockwise = - ((camera_yaw - yaw) % 360) / 5
        else:
            self.clockwise = (360 - (camera_yaw - yaw) % 360) / 5

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

    def set_obstacle(self, direction, cur_pos):
        pos = [int(cur_pos[0]) + FIELD_SIZE,int(cur_pos[1]),int(cur_pos[2]) + FIELD_SIZE]
        direction = direction % 360
        if 0 <= direction < 45 or 315 <= direction < 360:
            self.map2d[pos[0]][pos[2] + 1] = self.map2d[pos[0]-1][pos[2] + 1] = self.map2d[pos[0]+1][pos[2] + 1] = 1
            self.map2d[pos[0]][pos[2] + 2] = self.map2d[pos[0]-1][pos[2] + 2] = self.map2d[pos[0]+1][pos[2] + 2] = 1
        elif 45 <= direction < 135:
            self.map2d[pos[0]+1][pos[2]] = self.map2d[pos[0]+1][pos[2] + 1] = self.map2d[pos[0]+1][pos[2] - 1] = 1
            self.map2d[pos[0]+2][pos[2]] = self.map2d[pos[0]+2][pos[2] + 1] = self.map2d[pos[0]+2][pos[2] - 1] = 1
        elif 135 <= direction < 225:
            self.map2d[pos[0]][pos[2]-1] = self.map2d[pos[0]+1][pos[2]-1] = self.map2d[pos[0]-1][pos[2]-1] = 1
            self.map2d[pos[0]][pos[2]-2] = self.map2d[pos[0]+1][pos[2]-2] = self.map2d[pos[0]-1][pos[2]-2] = 1
        elif 225 <= direction < 315:
            self.map2d[pos[0]-1][pos[2]] = self.map2d[pos[0]-1][pos[2]+1] = self.map2d[pos[0]-1][pos[2]-1] = 1
            self.map2d[pos[0]-2][pos[2]] = self.map2d[pos[0]-2][pos[2]+1] = self.map2d[pos[0]-2][pos[2]-1] = 1

    @staticmethod
    def get_target_local(p1, p2):
        x = p2[0] - p1[0]
        z = p2[1] - p1[1]
        if max(abs(x), abs(z)) > LOCAL_SIZE:
            t = max(abs(x), abs(z))
            x = x / t
            z = z / t
            return [LOCAL_SIZE + int(LOCAL_SIZE * x), LOCAL_SIZE + int(LOCAL_SIZE * z)]
        else:
            return [LOCAL_SIZE + x, LOCAL_SIZE + z]

    def get_local_map(self, p1):
        return np.copy(self.map2d[p1[0] - LOCAL_SIZE: p1[0] + LOCAL_SIZE + 1, p1[1] - LOCAL_SIZE: p1[1] + LOCAL_SIZE + 1])

    def searching_dir(self, cur, tar):  # using Astar
        p1 = [int(cur[0]) + FIELD_SIZE, int(cur[2]) + FIELD_SIZE]  # current position sampling
        p2 = [int(tar[0]) + FIELD_SIZE, int(tar[2]) + FIELD_SIZE]  # target position sampling

        if p1 == p2: # if sampled in the same block, though very rare
            direction = [v2 - v1 for v1, v2 in zip(self.pos, self.target_location)]
            return get_pitch_yaw(*direction)[1]

        localmap = self.get_local_map(p1)
        localmap[LOCAL_SIZE][LOCAL_SIZE] = 0
        local_tar = self.get_target_local(p1, p2)
        localmap[local_tar[0]][local_tar[1]] = 0
        heuristic = np.zeros(localmap.shape)
        width = LOCAL_SIZE * 2 + 1

        for i in range(width):
            for j in range(width):
                heuristic[i][j] = ((i - LOCAL_SIZE) ** 2 + (j - LOCAL_SIZE) ** 2) ** 0.5
                if localmap[i][j]:
                    heuristic[i][j] = 9999
        delta = [[-1, 0], [0, -1], [0, 1], [1, 0]]
        cost = [1, 1, 1, 1]
        close_matrix = np.zeros(localmap.shape)
        close_matrix[local_tar[0]][local_tar[1]] = 1
        action_matrix = np.full(localmap.shape, -1, dtype=np.int)
        x, y = local_tar[0], local_tar[1]
        g = 0
        f = g + heuristic[x][y]
        cell = [[f, g, x, y]]
        found = False
        resign = False
        yaw = [90, 0, 180, 270]
        while not found and not resign:
            if len(cell) == 0:
                return -1
            else:
                cell.sort()
                cell.reverse()
                next_node = cell.pop()
                f, g, x, y = next_node
                if x == LOCAL_SIZE and y == LOCAL_SIZE:
                    return yaw[action_matrix[LOCAL_SIZE][LOCAL_SIZE]]
                else:
                    for i in range(len(delta)):
                        x2 = x + delta[i][0]
                        y2 = y + delta[i][1]
                        if 0 <= x2 < len(heuristic) and 0 <= y2 < len(heuristic):
                            if close_matrix[x2][y2] == 0 and localmap[x2][y2] == 0:
                                g2 = g + cost[i]
                                f2 = g2 + heuristic[x2][y2]
                                cell.append([f2, g2, x2, y2])
                                close_matrix[x2][y2] = 1
                                action_matrix[x2][y2] = i

    @staticmethod
    def get_distance(a, b):
        return np.linalg.norm([v2 - v1 for v1, v2 in zip(a, b)])

    def my_policy(self, state, ts):
        """Define a simple navigation policy"""
        pos = np.asarray(get_position(state))
        self.pos = pos
        cur_pos = [state.position_x, state.position_y, state.position_z]

        # detect obstacles
        depth = state.depth_map[DEPTH_MAP_HEIGHT//2 - 2:DEPTH_MAP_HEIGHT//2 + 2, DEPTH_MAP_WIDTH//2]
        see_obstacle = False
        walk_obstacle = False
        if max(depth) - min(depth) < 0.05 and min(depth) < 10:
            # 0.05 to decide whether a slope. 10 is large enough to doge buildings
            see_obstacle = True
        jump = False
        last_direction = [v2 - v1 for v1, v2 in zip(self.old_position, cur_pos)]
        if ts > 20 and self.get_distance(cur_pos, self.old_position)\
                * math.cos((self.last_walk_dir - get_pitch_yaw(*last_direction)[1]) / 180 * math.pi) <= (0.3 if self.pos[1] > -4 else 0.2):
            walk_obstacle = True

        camera_yaw = state.yaw
        walk_speed = 10
        direction = [v2 - v1 for v1, v2 in zip(cur_pos, self.target_location)]
        blocked = self.if_blocked(cur_pos, self.target_location)
        yaw = get_pitch_yaw(*direction)[1]

        if self.s == 0:    # directly move mode
            if walk_obstacle or blocked or see_obstacle:
                if walk_obstacle and not see_obstacle:
                    self.count = 0
                    jump = True
                    yaw += random.gauss(0, 20)
                else:
                    self.s = 1
                    self.count = 20
                    self.checkcount = 0
                    self.reset_map2d()
                    self.set_obstacle(self.last_walk_dir, cur_pos)
                    yaw = self.searching_dir(cur_pos, self.target_location)
            else:
                self.count += 1
                if self.count == 10:
                    self.checkpoint_pos = self.checkpoint[-1]
                    if self.get_distance(self.checkpoint_pos, cur_pos) < 3:
                        self.s = 2
                        self.count = 10
                        if len(self.checkpoint) > 1:
                            self.checkpoint.pop()
                        direction = [v2 - v1 for v1, v2 in zip(cur_pos, self.checkpoint_pos)]
                        yaw = get_pitch_yaw(*direction)[1]
                    else:
                        self.checkpoint.append(cur_pos)
                        self.count = 0

        elif self.s == 1:  # Astar move mode
            self.checkcount += 1
            if self.checkcount == 40:
                self.checkpoint_pos = self.checkpoint[-1]
                if self.get_distance(self.checkpoint_pos, cur_pos) < 5:
                    self.s = 2
                    self.count = 20
                    if len(self.checkpoint) > 1:
                        self.checkpoint.pop()
                    direction = [v2 - v1 for v1, v2 in zip(cur_pos, self.checkpoint_pos)]
                    yaw = get_pitch_yaw(*direction)[1]
                else:
                    self.checkpoint.append(cur_pos)
                    self.checkcount = 0
            else:
                if walk_obstacle:
                    self.set_obstacle(self.last_walk_dir, cur_pos)
                    self.count = 20
                elif self.count == 0:
                    self.s = 0
                else:
                    self.count -= 1
                yaw = self.searching_dir(cur_pos, self.target_location)
                if yaw == -1:
                    yaw = random.randint(0, 360)
                    self.reset_map2d()
                self.last_walk_dir = yaw
            return [yaw, walk_speed, 0, 0, jump]

        elif self.s == 2:  # retrieve
            if self.count == 0:
                self.s = 0
            else:
                self.count -= 1
                direction = [v2 - v1 for v1, v2 in zip(cur_pos, self.checkpoint_pos)]
                yaw = get_pitch_yaw(*direction)[1]

        self.get_turn_lr(camera_yaw, yaw)
        return [yaw, walk_speed, self.clockwise, 0, jump]

    def act(self, ts: int, state: AgentState) -> NavigationAction:
        walk_dir, walk_speed, turn_lr, look_ud, jump = self.my_policy(state, ts)

        return NavigationAction(
            walk_dir=walk_dir,
            walk_speed=walk_speed,
            turn_lr_delta=turn_lr,
            look_ud_delta=look_ud,
            jump=jump,
        )
