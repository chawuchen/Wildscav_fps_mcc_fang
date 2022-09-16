import numpy as np

def get_pitch_yaw(x, y, z):
    pitch = np.arctan2(y, (x**2 + z**2) ** 0.5) / np.pi * 180
    yaw = np.arctan2(x, z) / np.pi * 180
    return pitch, yaw

def index_to_action(index, direct_yaw, camera_yaw, walk_speed = 10):
    if index > 8:
        jump = True
    else:
        jump = False
    yaw_list = [0, 45, 90, 135, 180, 225, 270, 315, direct_yaw]
    yaw = yaw_list[index % 9]

    return [yaw, walk_speed, ]
