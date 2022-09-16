import gym
import numpy as np
import math
from gym import spaces
from inspirai_fps.utils import get_distance, get_position
from inspirai_fps.gamecore import Game, ActionVariable
import random
from utils import get_pitch_yaw

BASE_WORKER_PORT = 50000
DEPTH_MAP_FAR = 200
DEPTH_MAP_WIDTH = 64
DEPTH_MAP_HEIGHT = 36
MAX_FAR = 10
MAX_POINT_NUM = 5000
FX = 55.425
CAMERA_HEIGHT = 1.49581 - 0.75
THICKNESS = 1
CLOCKWISE_SPEED = 9
COLLISION_THRESHOLD = 0.003
FIELD_SIZE = 250
LOCAL_SIZE = 20
ACTION_DIM = 4

TRIES_PER_BLOCK = 1

DEFAULT_PAYLOAD = {
    "id": None,
    "current_episode": 0,
    "total_episodes": 0,
    "average_time_use": 0,
    "average_time_punish": 0,
    "average_time_total": 0,
    "success_rate": 0,
    "average_supply": 0,
}


class BaseEnv(gym.Env):
    def __init__(self, config):
        super().__init__()

        self.record = config.get("record", False)
        self.replay_suffix = config.get("replay_suffix", "")
        self.print_log = config.get("detailed_log", False)

        self.seed(config["random_seed"])
        self.server_port = BASE_WORKER_PORT
        print(f">>> New instance {self} on port: {self.server_port}")

        self.game = Game(map_dir=config["map_dir"], engine_dir=config["engine_dir"])
        self.game.turn_on_depth_map()
        self.game.set_depth_map_size(DEPTH_MAP_WIDTH, DEPTH_MAP_HEIGHT, DEPTH_MAP_FAR)
        self.game.set_map_id(random.randint(1, 100))
        self.game.set_episode_timeout(config["timeout"])
        self.game.set_random_seed(config["random_seed"])
        self.game.random_start_location()
        self.game.random_target_location()
        self.start_location = self.game.get_start_location()
        self.target_location = self.game.get_target_location()
        self.old_distance = get_distance(self.start_location, self.target_location) / 100

        self.s = 0
        self.count = 0
        self.checkcount = 0
        self.last_walk_dir = 0
        self.last_walk_idx = [0, 0, 0, 0]
        self.pos = self.start_location
        self.clockwise = 0
        self.checkpoint = [self.start_location]
        self.map2d = np.zeros((FIELD_SIZE * 2, FIELD_SIZE * 2))
        self.map2d_dir = np.full((FIELD_SIZE * 2, FIELD_SIZE * 2, ACTION_DIM), TRIES_PER_BLOCK)

        self.info = DEFAULT_PAYLOAD.copy()
        self.point_cloud = None

    def reset_map2d(self):
        p1 = [int(self.pos[0]) + FIELD_SIZE, int(self.pos[2]) + FIELD_SIZE]
        self.map2d[p1[0] - LOCAL_SIZE: p1[0] + LOCAL_SIZE + 1, p1[1] - LOCAL_SIZE: p1[1] + LOCAL_SIZE + 1] = 0

    def get_target_local(self, p1, p2):
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
        return np.copy(
            self.map2d[p1[0] - LOCAL_SIZE: p1[0] + LOCAL_SIZE + 1, p1[1] - LOCAL_SIZE: p1[1] + LOCAL_SIZE + 1])

    def get_legal_action(self):
        cur = self.pos
        p1 = [int(cur[0]) + FIELD_SIZE, int(cur[2]) + FIELD_SIZE]
        local_map = self.get_local_map(p1)
        legal_action = np.copy(self.map2d_dir[p1[0]][p1[1]])
        legal_action[np.where(legal_action > 0)] = 1
        legal_action[np.where(legal_action <= 0)] = 0
        if self.pitch > 30:
            legal_action[0] = 0
        if max(legal_action) <= 0:
            legal_action = [0, 0, 1, 0]
            self.map2d_dir[p1[0]][p1[1]] += TRIES_PER_BLOCK
            self.reset_map2d()
        return legal_action

    def reset(self):
        print("Reset for a new game ...")
        self._reset_game_config()
        if self.record:
            self.game.turn_on_record()
        self.map2d = np.zeros((FIELD_SIZE * 2, FIELD_SIZE * 2))
        self.map2d_dir = np.full((FIELD_SIZE * 2, FIELD_SIZE * 2, ACTION_DIM), TRIES_PER_BLOCK)
        self.game.set_game_replay_suffix(self.replay_suffix)
        self.game.set_map_id(random.randint(1, 100))
        self.game.random_start_location()
        self.game.random_target_location()
        self.start_location = self.game.get_start_location()
        self.target_location = self.game.get_target_location()
        self.old_distance = get_distance(self.start_location, self.target_location) / 100
        self.old_pos = self.start_location
        self.pos = self.start_location
        self.s = 0
        self.count = 0
        self.checkcount = 0
        self.last_walk_dir = 0
        self.last_walk_idx = [0, 0, 0, 0]
        self.clockwise = 0
        self.pitch = 0
        self.checkpoint = [self.start_location]

        self.game.new_episode()
        self.state = self.game.get_state()
        self.running_steps = 0

        self.episode_info = {
            "start_location": self.game.get_start_location(),
            "target_location": self.game.get_target_location(),
            "time_step_per_action": self.game.time_step_per_action,
        }

        # del self.point_cloud
        return self._get_obs()

    def get_turn_lr(self, camera_yaw, yaw):
        if (camera_yaw - yaw) % 360 < 180:
            self.clockwise = - ((camera_yaw - yaw) % 360) / 5
        else:
            self.clockwise = (360 - (camera_yaw - yaw) % 360) / 5

    def close(self):
        self.game.close()
        return super().close()

    def render(self, mode="replay"):
        return None

    def _reset_game_config(self):
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()


class NavigationBaseEnv(BaseEnv):
    def __init__(self, config):
        super().__init__(config)

        self.start_range = config["start_range"]
        self.start_hight = config["start_hight"]
        self.trigger_range = 1
        self.clockwise = 0
        self.game.set_game_mode(Game.MODE_NAVIGATION)

    def _reset_game_config(self):
        self.start_location = self.game.get_start_location()

    def index_to_action(self, index, direct_yaw, camera_yaw, walk_speed=10):
        if index > 4:
            jump = True
        else:
            jump = False
        yaw_list = [direct_yaw, direct_yaw + 90, direct_yaw + 180, direct_yaw + 270]
        yaw = yaw_list[index % 4] % 360
        return [yaw, walk_speed, self.clockwise, 0, jump]

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

    def get_target_local(self, p1, p2):
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
        return self.map2d[p1[0] - LOCAL_SIZE: p1[0] + LOCAL_SIZE + 1, p1[1] - LOCAL_SIZE: p1[1] + LOCAL_SIZE + 1]

    def searching_dir(self, cur, tar):
        p1 = [int(cur[0]) + FIELD_SIZE, int(cur[2]) + FIELD_SIZE]
        p2 = [int(tar[0]) + FIELD_SIZE, int(tar[2]) + FIELD_SIZE]
        if p1 == p2:
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

    def step(self, action, ts=0):
        direction = [v2 - v1 for v1, v2 in zip(self.pos, self.target_location)]
        yaw = get_pitch_yaw(*direction)[1]
        true_action = self.index_to_action(action, direct_yaw=yaw, camera_yaw=self.state.yaw)
        action_cmd = true_action
        self.game.make_action({0: action_cmd})
        self.state = self.game.get_state()
        self.obstacles_on_eyes(self.state)
        self.old_pos = self.pos
        self.pos = [self.state.position_x, self.state.position_y, self.state.position_z]
        last_direction = [v2 - v1 for v1, v2 in zip(self.old_pos, self.pos)]
        p1 = [int(self.old_pos[0]) + FIELD_SIZE, int(self.old_pos[2]) + FIELD_SIZE]
        p2 = [int(self.pos[0]) + FIELD_SIZE, int(self.pos[2]) + FIELD_SIZE]
        self.last_walk_idx = [0, 0, 0, 0]
        self.last_walk_idx[action] = 1
        if p1 != p2:
            self.map2d_dir[p1[0]][p1[1]][self.last_walk_idx.index(1)] -= 1
        x = self.get_distance(self.pos, self.old_pos) \
            * math.cos((self.last_walk_dir - get_pitch_yaw(*last_direction)[1]) / 180 * math.pi)
        direction = [v2 - v1 for v1, v2 in zip(self.pos, self.target_location)]
        self.pitch = get_pitch_yaw(*direction)[0]

        if ts > 20 and x <= (0.2 if self.pos[1] > -4 else 0.2):
            self.set_obstacle(self.last_walk_dir, self.old_pos)
            self.map2d_dir[p1[0]][p1[1]][self.last_walk_idx.index(1)] = -9999

        done = self.game.is_episode_finished()
        reward = 0
        self.running_steps += 1
        tar_pos = self.target_location
        obstacle = False

        if np.max(self.map2d_dir[p1[0]][p1[1]]) <= 0:
            reward = 0
        if done:
            if get_distance(self.pos, tar_pos) <= self.trigger_range:
                reward += 1000

            self.info = self.game.get_game_result()

            if self.print_log:
                Start = np.round(np.asarray(self.start_location), 2).tolist()
                Target = np.round(np.asarray(self.target_location), 2).tolist()
                End = np.round(np.asarray(get_position(self.state)), 2).tolist()
                Step = self.running_steps
                Reward = reward
                print(f"{Start=}\t{Target=}\t{End=}\t{Step=}\t{Reward=}")

        else:
            dis = get_distance(self.pos, tar_pos) / 100
            reward = -0.005 + self.old_distance - dis
            if obstacle:  # Collision penalty
                reward -= 1
            self.old_distance = dis

        return self._get_obs(), reward, done, {}

    def _sample_start_location(self):
        raise NotImplementedError()

    def _action_process(self, action):
        raise NotImplementedError()

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
            self.map2d[int(point_cloud[i][0])][int(point_cloud[i][1])] = 1
        return 0

class NavigationEnvSimple(NavigationBaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.action_pools = {
            ActionVariable.WALK_DIR: [0, 45, 90, 135, 180, 225, 270, 315],
            ActionVariable.WALK_SPEED: [0, 3, 10],
            ActionVariable.TURN_LR_DELTA: [0, 2, -2],
            ActionVariable.LOOK_UD_DELTA: [0],
            ActionVariable.JUMP: [True, False]
        }
        self.action_space = spaces.MultiDiscrete([len(pool) for pool in self.action_pools.values()])
        self.observation_space = spaces.Box(low=-1, high=1, shape=((64 * 32 + 3 + 3 + 1),), dtype=np.float32)

        self.game.set_available_actions([action_name for action_name in self.action_pools.keys()])
        self.game.init()

    def _get_obs(self):

        local_map = self.get_local_map([int(self.pos[0]) + FIELD_SIZE, int(self.pos[2]) + FIELD_SIZE])
        map2d = local_map.flatten()
        depth_map = np.array(self.state.depth_map).flatten()
        depth = self.state.depth_map[DEPTH_MAP_HEIGHT // 2 - 2:DEPTH_MAP_HEIGHT // 2 + 2, :]
        see_obstacle = np.zeros(DEPTH_MAP_WIDTH)
        c = (np.max(depth, axis=0) - np.min(depth, axis=0)) < 0.03
        d = np.min(depth, axis=0) < 3
        see_obstacle[np.where(c * d)] = 1
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
        feature_ex = np.append(map2d, np.array(self.last_walk_idx))
        return feature_ex

    def _action_process(self, action):
        action_values = list(self.action_pools.values())
        return [action_values[i][action[i]] for i in range(len(action))]

    def _sample_start_location(self):
        angle = np.random.uniform(0, 360)
        distance_to_trigger = abs(np.random.normal(scale=self.start_range))
        vec_len = self.trigger_range + distance_to_trigger
        dx = np.sin(angle) * vec_len
        dz = np.cos(angle) * vec_len
        x = self.target_location[0] + dx
        z = self.target_location[2] + dz
        return [x, self.start_hight, z]


