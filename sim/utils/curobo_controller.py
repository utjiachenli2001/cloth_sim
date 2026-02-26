from pathlib import Path
import torch
import numpy as np
import transforms3d as t3d

from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from curobo.util import logger


class CuroboController:

    def __init__(self, cfg, frame_dt, b2w):
        super().__init__()
        logger.setup_logger(level="error", logger_name="curobo")

        self.cfg = cfg
        self.dt = frame_dt
        self.b2w = b2w
        self.yml_path = str(Path(__file__).parents[2] / 'experiments/assets/robots/ARX-X5/curobo.yml')

        self.joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
        ]

        # motion generation
        world_config = {
            "cuboid": {
                "table": {
                    "dims": [0.15, 0.10, 0.01],  # x, y, z
                    "pose": [
                        0.70, 0.25, 0.05,
                        1, 0, 0, 0,
                    ],  # x, y, z, qw, qx, qy, qz
                },
            }
        }
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.yml_path,
            world_config,
            interpolation_dt=self.dt,
            num_trajopt_seeds=1,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.yml_path,
            world_config,
            interpolation_dt=self.dt,
            num_trajopt_seeds=1,
            num_graph_seeds=1,
        )
        self.motion_gen_batch = MotionGen(motion_gen_config)
        self.motion_gen_batch.warmup(batch=10)  # rotate_num

    def plan_path(
        self,
        curr_joint_pos,
        target_pose,
        constraint_pose=None,
    ):
        """
        Plan a trajectory for a single target pose.
        Input:
            - curr_joint_pos: List of current joint angles (J)
            - target_pose: target end-effector pose (xyz, quat, gripper)
        Output:
            - result['status']: "Success" or "Fail"
            - result['position']: numpy array of joint positions with shape (T x J)
                where T is number of waypoints, J is number of joints
            - result['velocity']: numpy array of joint velocities with same shape as position
        """

        t2b = self.world_to_base(self.b2w, target_pose[:-1])
        goal_pose_of_ee = Pose.from_list(t2b.tolist())

        target_gripper_pos = target_pose[-1]
        curr_gripper_pos = curr_joint_pos[-2:].mean()

        start_joint_states = JointState.from_position(
            torch.from_numpy(curr_joint_pos[:-2]).to(torch.float32).cuda().reshape(1, -1),
            joint_names=self.joint_names,
        )
        # plan
        plan_config = MotionGenPlanConfig(max_attempts=10)
        if constraint_pose is not None:
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
            )
            plan_config.pose_cost_metric = pose_cost_metric

        result = self.motion_gen.plan_single(start_joint_states, goal_pose_of_ee, plan_config)

        # output
        res_result = dict()
        if result.success.item() == False:
            res_result["status"] = "Fail"
            return res_result
        else:
            res_result["status"] = "Success"
            res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))  # (T, 6)
            res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))  # (T, 6)

            num_step = res_result["position"].shape[0]
            delta_gripper_pos = target_gripper_pos - curr_gripper_pos
            gripper_step_size = delta_gripper_pos / num_step
            gripper_result = np.linspace(curr_gripper_pos, target_gripper_pos, num_step)
            res_result["position"] = np.concatenate(
                [res_result["position"], gripper_result.reshape(-1, 1), gripper_result.reshape(-1, 1)],
                axis=1,
            )  # (T, 8)
            gripper_velocity = gripper_step_size / self.dt * np.ones((num_step, 1))
            gripper_velocity[0] = 0.0  # start from 0 velocity
            gripper_velocity[-1] = 0.0  # end with 0 velocity
            res_result["velocity"] = np.concatenate(
                [res_result["velocity"], gripper_velocity, gripper_velocity],
                axis=1,
            )  # (T, 8)
            return res_result

    def plan_batch(
        self,
        curr_joint_pos,
        target_pose_list,
        constraint_pose=None,
    ):
        """
        Plan a batch of trajectories for multiple target poses.

        Input:
            - curr_joint_pos: List of current joint angles (1 x N)
            - target_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

        Output:
            - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
            - result['position']: numpy array of joint positions with shape (N x T x J)
                where N is number of target poses, T is number of waypoints, J is number of joints
            - result['velocity']: numpy array of joint velocities with same shape as position
        """

        raise NotImplementedError("This function is under testing and not ready for use.")

        num_poses = len(target_pose_list)
        # transformation from world to arm's base
        world_base_pose = np.concatenate([
            np.array(self.robot_origin_pose.p),
            np.array(self.robot_origin_pose.q),
        ])
        poses_list = []
        for target_pose in target_pose_list:
            world_target_pose = np.concatenate([np.array(target_pose.p), np.array(target_pose.q)])
            base_target_pose_p, base_target_pose_q = self.world_to_base(world_base_pose, world_target_pose)

            base_target_pose_list = list(base_target_pose_p) + list(base_target_pose_q)
            poses_list.append(base_target_pose_list)

        poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
        goal_pose_of_ee = Pose(poses_cuda[:, :3], poses_cuda[:, 3:])
        joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
        joint_angles = [curr_joint_pos[index] for index in joint_indices]
        joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
        joint_angles_cuda = (torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1))
        joint_angles_cuda = torch.cat([joint_angles_cuda] * num_poses, dim=0)
        start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)
        # plan
        plan_config = MotionGenPlanConfig(max_attempts=10)
        if constraint_pose is not None:
            pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
            )
            plan_config.pose_cost_metric = pose_cost_metric

        try:
            result = self.motion_gen_batch.plan_batch(start_joint_states, goal_pose_of_ee, plan_config)
        except Exception as e:
            return {"status": ["Failure" for i in range(10)]}

        # output
        res_result = dict()
        # Convert boolean success values to "Success"/"Failure" strings
        success_array = result.success.cpu().numpy()
        status_array = np.array(["Success" if s else "Failure" for s in success_array], dtype=object)
        res_result["status"] = status_array

        if np.all(res_result["status"] == "Failure"):
            return res_result

        res_result["position"] = np.array(result.interpolated_plan.position.to("cpu"))
        res_result["velocity"] = np.array(result.interpolated_plan.velocity.to("cpu"))
        return res_result

    def world_to_base(self, base_pose, target_pose):
        '''
            transform target pose from world frame to base frame
            base_pose: np.array([x, y, z, qw, qx, qy, qz])
            target_pose: np.array([x, y, z, qw, qx, qy, qz])
        '''
        base_p, base_q = base_pose[0:3], base_pose[3:]
        target_p, target_q = target_pose[0:3], target_pose[3:]
        rel_p = target_p - base_p
        base_q_R = t3d.quaternions.quat2mat(base_q)
        target_q_R = t3d.quaternions.quat2mat(target_q)
        t2b_p = base_q_R.T @ rel_p
        t2b_q = t3d.quaternions.mat2quat(base_q_R.T @ target_q_R)
        t2b = np.array([t2b_p[0], t2b_p[1], t2b_p[2], t2b_q[0], t2b_q[1], t2b_q[2], t2b_q[3]])
        return t2b
