import numpy as np
import warp as wp
from pathlib import Path
import torch
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

from .base_controller import BaseController
from ..utils.env_utils import xyzw_to_wxyz


class CuroboController(BaseController):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        print("Warning: CuroboController currently only supports ARX-X5 robot with 6-DOF arm and 2-DOF gripper!")

    def initialize_resources(self, env) -> None:
        self.yml_path = str(Path(__file__).parents[2] / 'experiments/assets/robots/ARX-X5/curobo.yml')
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

        self.num_joints_per_robot = env.num_arm_joints + env.num_gripper_joints

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
            interpolation_dt=env.frame_dt,
            num_trajopt_seeds=1,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup()
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.yml_path,
            world_config,
            interpolation_dt=env.frame_dt,
            num_trajopt_seeds=1,
            num_graph_seeds=1,
        )
        self.motion_gen_batch = MotionGen(motion_gen_config)
        self.motion_gen_batch.warmup(batch=10)  # rotate_num

        self._curobo_prev_target = None
        self._curobo_plan_step = 0
        self._curobo_paths = [{}] * len(env.b2w_list)

    def compute(self, env, target: np.ndarray) -> None:
        if self._curobo_prev_target is None or not np.allclose(self._curobo_prev_target, target):
            self._plan(env, target)
            self._curobo_plan_step = 0
            self._curobo_prev_target = target.copy()

        target_pos_list = []
        target_vel_list = []

        for ri in range(env.num_robot):
            path = self._curobo_paths[ri]
            step = min(self._curobo_plan_step, len(path['position']) - 1)
            target_pos_list.append(path['position'][step])
            target_vel_list.append(path['velocity'][step])

        target_pos = np.concatenate(target_pos_list)
        target_vel = np.concatenate(target_vel_list)

        env.control.joint_target_pos.assign(wp.array(target_pos, dtype=float))
        env.control.joint_target_vel.assign(wp.array(target_vel, dtype=float))

        self._curobo_plan_step += 1

    def _plan(self, env, target: np.ndarray) -> None:
        joint_q = env.state_0.joint_q.numpy()
        target_pose = target.reshape(env.num_robot, 8)
        target_pose_wxyz = xyzw_to_wxyz(target_pose)

        for ri in range(env.num_robot):
            result = self.controller_plan_path(
                env=env,
                curr_joint_pos=joint_q[ri * self.num_joints_per_robot : (ri + 1) * self.num_joints_per_robot],
                target_pose=target_pose_wxyz[ri],
                b2w=env.b2w_list[ri]
            )
            if result['status'] != 'Success':
                raise Exception(f'CuRobo planning failed for robot {ri}!')
            self._curobo_paths[ri] = result

    def controller_plan_path(
        self,
        env,
        curr_joint_pos,
        target_pose,
        b2w,
        constraint_pose=None,
    ):
        t2b = self.world_to_base(b2w, target_pose[:-1])
        goal_pose_of_ee = Pose.from_list(t2b.tolist())

        target_gripper_pos = target_pose[-1]
        curr_gripper_pos = curr_joint_pos[env.num_arm_joints:env.num_arm_joints + env.num_gripper_joints].mean()

        start_joint_states = JointState.from_position(
            torch.from_numpy(curr_joint_pos[:env.num_arm_joints]).to(torch.float32).cuda().reshape(1, -1),
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
            gripper_velocity = gripper_step_size / env.frame_dt * np.ones((num_step, 1))
            gripper_velocity[0] = 0.0  # start from 0 velocity
            gripper_velocity[-1] = 0.0  # end with 0 velocity
            res_result["velocity"] = np.concatenate(
                [res_result["velocity"], gripper_velocity, gripper_velocity],
                axis=1,
            )  # (T, 8)
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
