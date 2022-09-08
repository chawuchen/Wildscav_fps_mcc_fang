import random
import numpy as np
from typing import NamedTuple
from inspirai_fps.gamecore import AgentState
from inspirai_fps.utils import get_position, get_picth_yaw, get_distance
import math
from collections import defaultdict

SEARCHING_TIMES = 300
DEPTH_MAP_FAR = 200
DEPTH_MAP_WIDTH = 64
DEPTH_MAP_HEIGHT = 36
MAX_FAR = 20
FIELD_SIZE = 300
MAX_POINT_NUM = 5000
FX = 55.425
CAMERA_HEIGHT = 1.49581 - 0.75
LOCAL_SIZE = 20
SUPPLY_INTERVAVL = 30
HEIGHT_THRESHOLD = 2
GAUSS_PARAMETER = 10
JUMP_INT = 30
CHECK_INT = 100
STAIR_HEIGHT = 3


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
    pitch = np.arctan2(y, (x ** 2 + z ** 2) ** 0.5) / np.pi * 180
    yaw = np.arctan2(x, z) / np.pi * 180
    return pitch, yaw


def get_stair(pos):
    return min(max(int((pos[1] - 1) / STAIR_HEIGHT), 0), 3)


class AgentNavigation:
    """
    This is a template of an agent for the navigation task.
    TODO: Modify the code in this class to implement your agent here.
    """

    def __init__(self, episode_info, supply_mode=False) -> None:
        self.s = 0
        self.count = 0
        self.checkcount = 0
        self.episode_info = episode_info
        self.last_walk_dir = 0
        self.pos = None
        self.clockwise = 0
        self.target_location = episode_info['target_location']
        self.start_location = episode_info['start_location']
        self.checkpoint = [self.start_location]
        self.map2d_stairs = np.zeros((FIELD_SIZE * 2, FIELD_SIZE * 2, 4))
        self.map2d = self.map2d_stairs[:, :, 0]
        self.direction_map = np.zeros((FIELD_SIZE * 4, FIELD_SIZE * 4, 4))
        self.old_position = self.start_location
        self.supply_mode = supply_mode
        self.jump_interval = 0
        self.stairs = get_stair(self.start_location)
        self.on_floor_once = self.stairs == 0
        self.entrance = [self.start_location for i in range(4)]
        self.stair_change = False
        self.back_mode = False
        self.checkpoint_pos = self.checkpoint[-1]

    def reset_map2d(self):
        p1 = [int(self.pos[0]) + FIELD_SIZE, int(self.pos[2]) + FIELD_SIZE]
        self.map2d[p1[0] - LOCAL_SIZE: p1[0] + LOCAL_SIZE + 1, p1[1] - LOCAL_SIZE: p1[1] + LOCAL_SIZE + 1] = 0

    def set_target_location(self, tar, back_mode=False):
        self.target_location = tar
        if back_mode:
            self.back_mode = True

    def get_turn_lr(self, camera_yaw, yaw):
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

    def obstacles_on_eyes(self, state):
        """Define a simple navigation policy"""
        depth = state.depth_map[DEPTH_MAP_HEIGHT // 2 - 2:DEPTH_MAP_HEIGHT // 2 + 2, :]
        analyse = []
        for i in range(DEPTH_MAP_WIDTH):
            if max(depth[:, i]) - min(depth[:, i]) < 0.05 and depth[3][i] < MAX_FAR:
                analyse.append([i - (DEPTH_MAP_WIDTH - 1) / 2, DEPTH_MAP_HEIGHT // 2 - (DEPTH_MAP_HEIGHT - 1) / 2,
                                depth[3][i]])
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
            self.map2d_stairs[int(point_cloud[i][0])][int(point_cloud[i][1])][0] = 1
        return 0

    def set_obstacle(self, direction, cur_pos, value=1):
        pos = [int(cur_pos[0]) + FIELD_SIZE, int(cur_pos[1]), int(cur_pos[2]) + FIELD_SIZE]
        direction = direction % 360
        if self.supply_mode:
            if 0 <= direction < 45 or 315 <= direction < 360:
                self.map2d[pos[0]][pos[2] + 1] = value
            elif 45 <= direction < 135:
                self.map2d[pos[0] + 1][pos[2]] = value
            elif 135 <= direction < 225:
                self.map2d[pos[0]][pos[2] - 1] = value
            elif 225 <= direction < 315:
                self.map2d[pos[0] - 1][pos[2]] = value
        else:
            if 0 <= direction < 45 or 315 <= direction < 360:
                self.map2d[pos[0]][pos[2] + 1] = self.map2d[pos[0] - 1][pos[2] + 1] = self.map2d[pos[0] + 1][
                    pos[2] + 1] = 1
                self.map2d[pos[0]][pos[2] + 2] = self.map2d[pos[0] - 1][pos[2] + 2] = self.map2d[pos[0] + 1][
                    pos[2] + 2] = 1
            elif 45 <= direction < 135:
                self.map2d[pos[0] + 1][pos[2]] = self.map2d[pos[0] + 1][pos[2] + 1] = self.map2d[pos[0] + 1][
                    pos[2] - 1] = 1
                self.map2d[pos[0] + 2][pos[2]] = self.map2d[pos[0] + 2][pos[2] + 1] = self.map2d[pos[0] + 2][
                    pos[2] - 1] = 1
            elif 135 <= direction < 225:
                self.map2d[pos[0]][pos[2] - 1] = self.map2d[pos[0] + 1][pos[2] - 1] = self.map2d[pos[0] - 1][
                    pos[2] - 1] = 1
                self.map2d[pos[0]][pos[2] - 2] = self.map2d[pos[0] + 1][pos[2] - 2] = self.map2d[pos[0] - 1][
                    pos[2] - 2] = 1
            elif 225 <= direction < 315:
                self.map2d[pos[0] - 1][pos[2]] = self.map2d[pos[0] - 1][pos[2] + 1] = self.map2d[pos[0] - 1][
                    pos[2] - 1] = 1
                self.map2d[pos[0] - 2][pos[2]] = self.map2d[pos[0] - 2][pos[2] + 1] = self.map2d[pos[0] - 2][
                    pos[2] - 1] = 1

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
        return np.copy(self.map2d[p1[0] - LOCAL_SIZE: p1[0] + LOCAL_SIZE + 1,
                       p1[1] - LOCAL_SIZE: p1[1] + LOCAL_SIZE + 1])

    def searching_dir(self, cur, tar):
        if self.back_mode:
            return self.get_yaw_from_direction(cur)

        p1 = [int(cur[0]) + FIELD_SIZE, int(cur[2]) + FIELD_SIZE]
        p2 = [int(tar[0]) + FIELD_SIZE, int(tar[2]) + FIELD_SIZE]
        left = min(p1[0], p2[0])
        right = max(p1[0], p2[0])
        up = min(p1[1], p2[1])
        down = max(p1[1], p2[1])
        cover = np.max(self.map2d[left:right + 1, up:down + 1])
        localmap = self.get_local_map(p1)
        localmap[LOCAL_SIZE][LOCAL_SIZE] = 0
        local_tar = self.get_target_local(p1, p2)
        if localmap[local_tar[0]][local_tar[1]] != 0 and max(abs(local_tar[0] - LOCAL_SIZE),
                                                             abs(local_tar[1] - LOCAL_SIZE)) < 20:
            return "-1"
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) <= 1 or cover < 0.5:
            direction = [v2 - v1 for v1, v2 in zip(self.pos, self.target_location)]
            return get_pitch_yaw(*direction)[1] + random.gauss(0, 15)
        localmap[local_tar[0]][local_tar[1]] = 0

        if not self.stair_change:
            self.map2d[p1[0]][p1[1]] = 0

        heuristic = np.zeros(localmap.shape)
        width = LOCAL_SIZE * 2 + 1
        for i in range(width):
            for j in range(width):
                heuristic[i][j] = abs(i - LOCAL_SIZE) + abs(j - LOCAL_SIZE)
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
                return "-1"
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

    def get_distance(self, a, b):
        return np.linalg.norm([v2 - v1 for v1, v2 in zip(a, b)])

    def draw_direction(self, old, cur):
        if [old[0], old[2]] != [cur[0], cur[2]] and self.direction_map[cur[0]][cur[2]][self.stairs] == 0:
            reversed_direction = [v2 - v1 for v1, v2 in zip(cur, old)]
            self.direction_map[cur[0]][cur[2]][self.stairs] = get_pitch_yaw(*reversed_direction)[1] + 0.1
            if self.direction_map[old[0]][old[2]][self.stairs] == 0:
                self.direction_map[old[0]][old[2]][self.stairs] = get_pitch_yaw(*reversed_direction)[1] + 0.1

        k = [[0, 1], [1, 0], [1, 1], [0, -1], [-1, 0], [-1, -1], [1, -1], [-1, 1]]
        for each in k:
            if self.direction_map[cur[0] + each[0]][cur[2] + each[1]][self.stairs] == 0:
                new_pos = [cur[0] + each[0], 0, cur[2] + each[1]]
                reversed_direction = [v2 - v1 for v1, v2 in zip(new_pos, cur)]
                self.direction_map[new_pos[0]][new_pos[2]][self.stairs] = get_pitch_yaw(*reversed_direction)[1] \
                    if get_pitch_yaw(*reversed_direction)[1] != 0 else 0.1

    def get_yaw_from_direction(self, cur):
        c = self.direction_map[int(cur[0] * 2) + 2 * FIELD_SIZE][int(cur[2] * 2) + 2 * FIELD_SIZE][self.stairs]
        if c == 0 or abs(int(cur[0] * 2) - int(self.entrance[self.stairs][0] * 2)) + \
                abs(int(cur[2] * 2) - int(self.entrance[self.stairs][2] * 2)) <= 1:
            direction = [v2 - v1 for v1, v2 in zip(cur, self.entrance[self.stairs])]
            c = get_pitch_yaw(*direction)[1] + random.gauss(0, 15)
        return c

    def set_last_walk_dir(self, walk_dir):
        self.last_walk_dir = walk_dir

    def my_policy(self, state, ts):
        """Define a simple navigation policy"""

        self.stair_change = False
        pos = np.asarray(get_position(state))
        self.pos = pos
        self.jump_interval += 1
        cur_pos = [state.position_x, state.position_y, state.position_z]

        depth = state.depth_map[DEPTH_MAP_HEIGHT // 2 - 2:DEPTH_MAP_HEIGHT // 2 + 2, DEPTH_MAP_WIDTH // 2]
        see_obstacle = False
        walk_obstacle = False
        if max(depth) - min(depth) < 0.05 and min(depth) < 3:
            see_obstacle = True
        jump = False
        last_direction = [v2 - v1 for v1, v2 in zip(self.old_position, cur_pos)]
        if ts > 20 and self.get_distance(cur_pos, self.old_position) \
                * math.cos((self.last_walk_dir - get_pitch_yaw(*last_direction)[1]) / 180 *
                           math.pi) <= (0.4 if pos[1] > -2 else 0.2):
            walk_obstacle = True

        camera_yaw = state.yaw
        walk_speed = 10
        direction = [v2 - v1 for v1, v2 in zip(cur_pos, self.target_location)]
        blocked = self.if_blocked(cur_pos, self.target_location)
        yaw = get_pitch_yaw(*direction)[1]

        # deal with "upstairs" problem
        if self.stairs == 0:
            self.on_floor_once = True
        if get_stair(self.old_position) != get_stair(pos):
            self.stairs = get_stair(pos)
            self.map2d = self.map2d_stairs[:, :, self.stairs]
            if get_stair(self.old_position) > get_stair(pos):
                self.map2d[int(self.old_position[0]) + FIELD_SIZE][int(self.old_position[2]) + FIELD_SIZE] = 1
                self.direction_map[:, :, get_stair(self.old_position)] = 0
                self.set_obstacle((self.last_walk_dir + 180) % 360, cur_pos)
            else:
                self.map2d[int(self.old_position[0]) + FIELD_SIZE][int(self.old_position[2]) + FIELD_SIZE] = 1
                self.entrance[self.stairs] = self.old_position
            self.stair_change = True
        if self.stairs != 0:
            self.map2d[int(self.entrance[self.stairs][0]) + FIELD_SIZE][
                int(self.entrance[self.stairs][2]) + FIELD_SIZE] = 0
            p1 = [int(self.old_position[0] * 2) + FIELD_SIZE, 0, int(self.old_position[2] * 2) + FIELD_SIZE]
            p2 = [int(cur_pos[0] * 2) + FIELD_SIZE, 0, int(cur_pos[2] * 2) + FIELD_SIZE]
            self.draw_direction(p1, p2)
        else:  # using visual detection if not upstairs
            if abs(state.pitch) < 0:
                self.obstacles_on_eyes(state)

        if self.s == 0:
            if walk_obstacle or blocked or see_obstacle or self.stairs != 0:
                self.s = 1
                self.count = 20
                self.checkcount = 0
                if walk_obstacle:
                    # self.reset_map2d()
                    self.set_obstacle(self.last_walk_dir, self.old_position)
                yaw = self.searching_dir(cur_pos, self.target_location)
                if yaw == "-1":
                    self.reset_map2d()
            else:
                self.count += 1
                if self.count == CHECK_INT:
                    self.checkpoint_pos = self.checkpoint[-1]
                    if self.get_distance(self.checkpoint_pos, cur_pos) < 1:
                        if not self.supply_mode:
                            self.s = 2
                            self.count = CHECK_INT
                            if len(self.checkpoint) > 1:
                                self.checkpoint.pop()
                            direction = [v2 - v1 for v1, v2 in zip(cur_pos, self.checkpoint_pos)]
                            yaw = get_pitch_yaw(*direction)[1]
                        else:
                            if self.jump_interval >= JUMP_INT:
                                jump = True
                                self.jump_interval = 0
                    else:
                        self.checkpoint.append(cur_pos)
                        self.count = 0

        elif self.s == 1:
            self.checkcount += 1
            if self.checkcount == CHECK_INT:
                self.checkpoint_pos = self.checkpoint[-1]
                if self.get_distance(self.checkpoint_pos, cur_pos) < 1:
                    if not self.supply_mode:
                        self.s = 2
                        self.count = CHECK_INT
                        if len(self.checkpoint) > 1:
                            self.checkpoint.pop()
                        direction = [v2 - v1 for v1, v2 in zip(cur_pos, self.checkpoint_pos)]
                        yaw = get_pitch_yaw(*direction)[1]
                    else:
                        if self.jump_interval >= JUMP_INT:
                            jump = True
                            self.jump_interval = 0
                else:
                    self.checkpoint.append(cur_pos)
                    self.checkcount = 0
            else:
                if walk_obstacle or see_obstacle:
                    if walk_obstacle:
                        self.set_obstacle(self.last_walk_dir, self.old_position)
                    self.count = 20
                elif self.count == 0:
                    self.s = 0
                else:
                    self.count -= 1
                yaw = self.searching_dir(cur_pos, self.target_location)
                if yaw == "-1":
                    self.reset_map2d()
            self.old_position = cur_pos
            return [yaw, walk_speed, 0, 0, jump]

        elif self.s == 2:
            if self.count == 0:
                self.s = 0
            else:
                self.count -= 1
                direction = [v2 - v1 for v1, v2 in zip(cur_pos, self.checkpoint_pos)]
                yaw = get_pitch_yaw(*direction)[1]

        if yaw != "-1":
            self.get_turn_lr(camera_yaw, yaw)
        if self.back_mode and self.jump_interval >= JUMP_INT:
            jump = True
            self.jump_interval = 0

        self.old_position = cur_pos
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


class AgentSupplyGathering:
    """
    This is a template of an agent for the supply gathering task.
    TODO: Modify the code in this class to implement your agent here.
    """

    def __init__(self, episode_info) -> None:
        self.episode_info = episode_info
        self.episode_info['target_location'] = episode_info['supply_heatmap_center']
        self.Navagent = AgentNavigation(episode_info, supply_mode=True)
        self.searching_times = defaultdict(int)
        self.current_tar = []
        self.fake_ts = 0
        self.radius = 100
        self.origin_center = episode_info['supply_heatmap_center'].copy()
        self.supplies_stairs = [[], [], [], []]
        self.supply_id_stairs = [[], [], [], []]
        self.supplies = self.supplies_stairs[0]
        self.supply_ids = self.supply_id_stairs[0]

    # trying hamilton path, but not applied at last
    '''
    def find_shortest_ham_path(self, supplies, pos): 
        N = len(supplies) + 1
        mat = np.zeros((N, N))
        for i in range(1, N-1):
            for j in range(i+1, N):
                mat[i][j] = mat[j][i] = get_distance(supplies[i-1][:-1], supplies[j-1][:-1])
        for i in range(1, N):
            mat[0][i] = mat[i][0] = get_distance(pos, supplies[i-1][:-1])

        _, path = self.solve_ham_path(mat, [k for k in range(N)])
        return supplies[path[1]-1]

    def solve_ham_path(self, mat, path):
        if len(path) == 2:
            return mat[path[0]][path[1]], path
        else:
            N = len(path)
            min_w = self.calc_ham_path(mat, path)
            best = path.copy()
            for i in range(2, N):
                path[1], path[i] = path[i], path[1]
                tmp_w, tmp_p = self.solve_ham_path(mat, path[1:])
                if tmp_w + mat[path[0]][path[1]] < min_w:
                    best = path.copy()
                    min_w = tmp_w + mat[path[0]][path[1]]
                path[1], path[i] = path[i], path[1]
            return min_w, best

    def calc_ham_path(self, mat, path):
        N = len(path)
        sum = 0
        for i in range(N-1):
            sum += mat[path[i]][path[i+1]]
        return sum
    '''

    def update_supplies_info(self, pos, stair):
        self.supplies = self.supplies_stairs[stair]
        self.supply_ids = self.supply_id_stairs[stair]
        for i in range(1, 4):
            self.supply_id_stairs[i].clear()
            self.supplies_stairs[i].clear()
        if len(self.supplies) > 0:
            N = len(self.supplies)
            for i in range(N - 1, -1, -1):
                if get_distance(pos, self.supplies[i][:-1]) < 3 or \
                        self.searching_times[self.supplies[i][-1]] >= SEARCHING_TIMES:
                    idx = self.supplies[i][-1]
                    self.supplies.pop(i)
                    self.supply_ids.remove(idx)

    def act(self, ts: int, state: AgentState) -> SupplyGatherAction:
        pos = get_position(state)
        if ts == 0:
            print("origin_center:{}, radius:{}".format(self.origin_center, self.radius))
        if ts % 50 == 0:
            print(pos, self.current_tar)
        stair = get_stair(pos)

        self.update_supplies_info(pos, stair)
        self.episode_info['supply_heatmap_center'][1] = pos[1]
        if get_distance(pos, self.episode_info['supply_heatmap_center']) < 10:
            self.episode_info['supply_heatmap_center'][0] = self.origin_center[0] + random.randint(-self.radius,
                                                                                                   self.radius)
            self.episode_info['supply_heatmap_center'][2] = self.origin_center[0] + random.randint(-self.radius,
                                                                                                   self.radius)
            while max(abs(self.episode_info['supply_heatmap_center'][0]),
                      abs(self.episode_info['supply_heatmap_center'][2])) > FIELD_SIZE \
                    or self.Navagent.map2d[int(self.episode_info['supply_heatmap_center'][0]
                                               ) + FIELD_SIZE][
                int(self.episode_info['supply_heatmap_center'][2]) + FIELD_SIZE] != 0:
                self.episode_info['supply_heatmap_center'][0] = self.origin_center[0] + random.randint(-self.radius,
                                                                                                       self.radius)
                self.episode_info['supply_heatmap_center'][2] = self.origin_center[0] + random.randint(-self.radius,
                                                                                                       self.radius)
        if state.supply_states:
            supplies = list(state.supply_states.values())
            for i in range(len(supplies)):
                supplies[i] = np.asarray(get_position(supplies[i]) + [supplies[i].id])
            for each in supplies:
                if self.searching_times[int(each[-1])] < SEARCHING_TIMES and int(each[-1]) not in self.supply_ids \
                        and get_stair(each[:3]) == stair:
                    self.supplies_stairs[get_stair(each[:3])].append(each)
                    self.supply_id_stairs[get_stair(each[:3])].append(int(each[-1]))

        if len(self.supplies) > 0:
            if len(self.supplies) < 1:
                tar = self.episode_info['supply_heatmap_center']
                self.Navagent.set_target_location(tar)
                if len(self.current_tar) > 0:
                    self.current_tar = []
                    self.fake_ts = ts
            else:
                tar = self.supplies[0]
                if len(self.supplies) > 1:
                    self.supplies.sort(
                        key=lambda supply_pos: np.linalg.norm([v2 - v1 for v1, v2 in zip(pos, supply_pos[:-1])]))
                    tar = self.supplies[0]
                if get_distance(tar[:-1], pos) < 20:
                    self.searching_times[int(tar[-1])] += 1
                if len(self.current_tar) == 0 or self.searching_times[
                    int(self.current_tar[-1])] >= SEARCHING_TIMES or int(
                    self.current_tar[-1]) not in self.supply_ids or get_stair(self.current_tar[:3]) != stair:
                    self.Navagent.set_target_location(tar[:-1])
                    self.fake_ts = ts
                    self.current_tar = tar
        else:
            if stair == 0 or get_distance(self.origin_center, pos) >= 80:
                self.Navagent.set_target_location(self.episode_info['supply_heatmap_center'])
            else:
                self.Navagent.set_target_location(self.Navagent.entrance[stair],
                                                  back_mode=get_distance(self.origin_center,
                                                                         pos) < 80 and self.Navagent.on_floor_once)
            if len(self.current_tar) > 0:
                self.current_tar = []
                self.fake_ts = ts

        walk_dir, walk_speed, turn_lr, look_ud, jump = self.Navagent.my_policy(state, ts - self.fake_ts)
        if walk_dir in [0, 90, 180, 270]:
            walk_speed = 10
        if self.Navagent.back_mode or not self.Navagent.on_floor_once:
            walk_speed = 6
        if walk_dir == "-1":
            if len(self.current_tar) > 0:
                self.searching_times[int(self.current_tar[-1])] = SEARCHING_TIMES
            walk_dir = random.randint(0, 360)
        if walk_dir in [0, 90, 180, 270] or self.Navagent.back_mode:
            walk_dir += random.randint(-15, 15)
        self.Navagent.set_last_walk_dir(walk_dir)
        self.Navagent.back_mode = False
        return SupplyGatherAction(
            walk_dir=walk_dir,
            walk_speed=walk_speed,
            turn_lr_delta=turn_lr,
            look_ud_delta=look_ud,
            jump=jump,
            pickup=True,
        )


class AgentSupplyBattle:
    """
    This is a template of an agent for the supply battle task.
    TODO: Modify the code in this class to implement your agent here.
    """

    def __init__(self, episode_info) -> None:
        self.episode_info = episode_info
        self.episode_info['target_location'] = episode_info['supply_heatmap_center']
        self.Navagent = AgentNavigation(episode_info, supply_mode=True)
        self.searching_times = defaultdict(int)
        self.current_tar = []
        self.fake_ts = 0
        self.radius = 100
        self.origin_center = episode_info['supply_heatmap_center'].copy()
        self.supplies_stairs = [[], [], [], []]
        self.supplie_id_stairs = [[], [], [], []]
        self.supplies = self.supplies_stairs[0]
        self.supplie_ids = self.supplie_id_stairs[0]

    def update_supplies_info(self, pos, stair):
        self.supplies = self.supplies_stairs[stair]
        self.supplie_ids = self.supplie_id_stairs[stair]
        for i in range(1, 4):
            self.supplie_id_stairs[i].clear()
            self.supplies_stairs[i].clear()
        if len(self.supplies) > 0:
            N = len(self.supplies)
            for i in range(N - 1, -1, -1):
                if get_distance(pos, self.supplies[i][:-1]) < 3 or \
                        self.searching_times[self.supplies[i][-1]] >= SEARCHING_TIMES:
                    idx = self.supplies[i][-1]
                    self.supplies.pop(i)
                    self.supplie_ids.remove(idx)

    def act(self, ts: int, state: AgentState) -> SupplyBattleAction:
        pos = get_position(state)
        if ts == 0:
            print("origin_center:{}, radius:{}".format(self.origin_center, self.radius))
        if ts % 50 == 0:
            print(pos, self.current_tar)
        stair = get_stair(pos)

        self.update_supplies_info(pos, stair)
        self.episode_info['supply_heatmap_center'][1] = pos[1]
        if get_distance(pos, self.episode_info['supply_heatmap_center']) < 10:
            self.episode_info['supply_heatmap_center'][0] = self.origin_center[0] + random.randint(-self.radius,
                                                                                                   self.radius)
            self.episode_info['supply_heatmap_center'][2] = self.origin_center[0] + random.randint(-self.radius,
                                                                                                   self.radius)
            while max(abs(self.episode_info['supply_heatmap_center'][0]),
                      abs(self.episode_info['supply_heatmap_center'][2])) > FIELD_SIZE \
                    or self.Navagent.map2d[int(self.episode_info['supply_heatmap_center'][0]
                                               ) + FIELD_SIZE][
                int(self.episode_info['supply_heatmap_center'][2]) + FIELD_SIZE] != 0:
                self.episode_info['supply_heatmap_center'][0] = self.origin_center[0] + random.randint(-self.radius,
                                                                                                       self.radius)
                self.episode_info['supply_heatmap_center'][2] = self.origin_center[0] + random.randint(-self.radius,
                                                                                                       self.radius)
        if state.supply_states:
            supplies = list(state.supply_states.values())
            for i in range(len(supplies)):
                supplies[i] = np.asarray(get_position(supplies[i]) + [supplies[i].id])
            for each in supplies:
                if self.searching_times[int(each[-1])] < SEARCHING_TIMES and int(each[-1]) not in self.supplie_ids \
                        and get_stair(each[:3]) == stair:
                    self.supplies_stairs[get_stair(each[:3])].append(each)
                    self.supplie_id_stairs[get_stair(each[:3])].append(int(each[-1]))

        if len(self.supplies) > 0:
            if len(self.supplies) < 1:
                tar = self.episode_info['supply_heatmap_center']
                self.Navagent.set_target_location(tar)
                if len(self.current_tar) > 0:
                    self.current_tar = []
                    self.fake_ts = ts
            else:
                tar = self.supplies[0]
                if len(self.supplies) > 1:
                    self.supplies.sort(
                        key=lambda supply_pos: np.linalg.norm([v2 - v1 for v1, v2 in zip(pos, supply_pos[:-1])]))
                    tar = self.supplies[0]
                if get_distance(tar[:-1], pos) < 20:
                    self.searching_times[int(tar[-1])] += 1
                    c = self.searching_times[int(tar[-1])]
                    if c > 200:
                        print(1)
                if len(self.current_tar) == 0 or self.searching_times[
                    int(self.current_tar[-1])] >= SEARCHING_TIMES or int(
                    self.current_tar[-1]) not in self.supplie_ids or get_stair(self.current_tar[:3]) != stair:
                    self.Navagent.set_target_location(tar[:-1])
                    self.fake_ts = ts
                    self.current_tar = tar
        else:
            if stair == 0 or get_distance(self.origin_center, pos) >= 80:
                self.Navagent.set_target_location(self.episode_info['supply_heatmap_center'])
            else:
                self.Navagent.set_target_location(self.Navagent.entrance[stair],
                                                  back_mode=get_distance(self.origin_center,
                                                                         pos) < 80 and self.Navagent.on_floor_once)
            if len(self.current_tar) > 0:
                self.current_tar = []
                self.fake_ts = ts

        walk_dir, walk_speed, turn_lr, look_ud, jump = self.Navagent.my_policy(state, ts - self.fake_ts)
        if walk_dir in [0, 90, 180, 270]:
            walk_speed = 10
        if self.Navagent.back_mode or not self.Navagent.on_floor_once:
            walk_speed = 6
        if walk_dir == "-1":
            if len(self.current_tar) > 0:
                self.searching_times[int(self.current_tar[-1])] = SEARCHING_TIMES
            walk_dir = random.randint(0, 360)
        if walk_dir in [0, 90, 180, 270] or self.Navagent.back_mode:
            walk_dir += random.randint(-15, 15)
        self.Navagent.set_last_walk_dir(walk_dir)
        self.Navagent.back_mode = False

        attack = False
        if state.enemy_states:
            enemy_info = list(state.enemy_states.values())[0]
            target = np.asarray(get_position(enemy_info))
            direction = target - pos
            direction = direction / np.linalg.norm(direction)
            aim_pitch, aim_yaw = get_picth_yaw(*direction)

            diff_pitch = aim_pitch - state.pitch
            diff_yaw = aim_yaw - state.yaw
            if abs(diff_pitch) < 5 and abs(diff_yaw) < 5:
                attack = True

            skip_frames = self.episode_info["time_step_per_action"]
            rotate_speed_decay = 0.5
            turn_lr = diff_yaw / skip_frames * rotate_speed_decay
            look_ud = diff_pitch / skip_frames * rotate_speed_decay
        else:
            aim_pitch = 0
            skip_frames = self.episode_info["time_step_per_action"]
            rotate_speed_decay = 0.5
            diff_pitch = aim_pitch - state.pitch
            look_ud = diff_pitch / skip_frames * rotate_speed_decay

        return SupplyBattleAction(
            walk_dir=walk_dir,
            walk_speed=walk_speed,
            turn_lr_delta=turn_lr,
            look_ud_delta=look_ud,
            jump=jump,
            pickup=True,
            attack=attack,
            reload=state.weapon_ammo < 5 and state.spare_ammo > 0,
        )
