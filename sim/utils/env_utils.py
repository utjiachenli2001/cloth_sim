import numpy as np
import warp as wp

def xyzw_to_wxyz(pose: np.ndarray):
    if len(pose.shape) == 1 and pose.shape[0] >= 7:
        new_pose = pose.copy()
        new_pose[3] = pose[6]
        new_pose[4] = pose[3]
        new_pose[5] = pose[4]
        new_pose[6] = pose[5]
    elif len(pose.shape) == 1 and pose.shape[0] == 4:
        new_pose = pose.copy()
        new_pose[0] = pose[3]
        new_pose[1] = pose[0]
        new_pose[2] = pose[1]
        new_pose[3] = pose[2]
    elif pose.shape[1] >= 7:
        new_pose = pose.copy()
        new_pose[:, 3] = pose[:, 6]
        new_pose[:, 4] = pose[:, 3]
        new_pose[:, 5] = pose[:, 4]
        new_pose[:, 6] = pose[:, 5]
    elif pose.shape[1] == 4:
        new_pose = pose.copy()
        new_pose[:, 0] = pose[:, 3]
        new_pose[:, 1] = pose[:, 0]
        new_pose[:, 2] = pose[:, 1]
        new_pose[:, 3] = pose[:, 2]
    return new_pose


def wxyz_to_xyzw(pose: np.ndarray):
    if len(pose.shape) == 1 and pose.shape[0] >= 7:
        new_pose = pose.copy()
        new_pose[6] = pose[3]
        new_pose[3] = pose[4]
        new_pose[4] = pose[5]
        new_pose[5] = pose[6]
    elif len(pose.shape) == 1 and pose.shape[0] == 4:
        new_pose = pose.copy()
        new_pose[3] = pose[0]
        new_pose[0] = pose[1]
        new_pose[1] = pose[2]
        new_pose[2] = pose[3]
    elif pose.shape[1] >= 7:
        new_pose = pose.copy()
        new_pose[:, 6] = pose[:, 3]
        new_pose[:, 3] = pose[:, 4]
        new_pose[:, 4] = pose[:, 5]
        new_pose[:, 5] = pose[:, 6]
    elif pose.shape[1] == 4:
        new_pose = pose.copy()
        new_pose[:, 3] = pose[:, 0]
        new_pose[:, 0] = pose[:, 1]
        new_pose[:, 1] = pose[:, 2]
        new_pose[:, 2] = pose[:, 3]
    return new_pose


def quat_to_vec4(q: wp.quat) -> wp.vec4:
    """Convert a quaternion to a vec4."""
    return wp.vec4(q[0], q[1], q[2], q[3])
