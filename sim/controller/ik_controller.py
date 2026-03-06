import numpy as np
import warp as wp
import newton
import newton.ik as ik

from .base_controller import BaseController
from ..utils.env_utils import quat_to_vec4, broadcast_ik_solution_kernel


class IKController(BaseController):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def initialize_resources(self, env) -> None:
        self.setup_ik(env)

        self.graph_ik_list = []
        if self.cfg.env.use_graph:
            for ri in range(env.num_robot):
                with wp.ScopedCapture() as capture:
                    self.ik_solvers[ri].step(
                        self.joint_q_ik_list[ri], self.joint_q_ik_list[ri],
                        iterations=self.ik_iters,
                    )
                self.graph_ik_list.append(capture.graph)

    def setup_ik(self, env) -> None:
        assert len(env._single_arm_models) == env.num_robot, (
            f"Expected {env.num_robot} single-arm models, got {len(env._single_arm_models)}. "
            f"Ensure controller='ik' was set before _build_scene() ran."
        )

        self.ik_iters = 24
        self.pos_objs = []
        self.rot_objs = []
        self.obj_joint_limits_list = []
        self.joint_q_ik_list = []
        self.ik_solvers = []

        for model_single in env._single_arm_models:
            state_single = model_single.state()
            body_q_np = state_single.body_q.numpy()
            ee_tf = wp.transform(*body_q_np[env.endeffector_id])

            pos_obj = ik.IKObjectivePosition(
                link_index=env.endeffector_id,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array([wp.transform_get_translation(ee_tf)], dtype=wp.vec3),
            )
            rot_obj = ik.IKObjectiveRotation(
                link_index=env.endeffector_id,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=wp.array([quat_to_vec4(wp.transform_get_rotation(ee_tf))], dtype=wp.vec4),
            )
            obj_joint_limits = ik.IKObjectiveJointLimit(
                joint_limit_lower=model_single.joint_limit_lower,
                joint_limit_upper=model_single.joint_limit_upper,
            )
            joint_q_ik = wp.array(model_single.joint_q, shape=(1, model_single.joint_coord_count))
            ik_solver = ik.IKSolver(
                model=model_single,
                n_problems=1,
                objectives=[pos_obj, rot_obj, obj_joint_limits],
                lambda_initial=0.1,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )

            self.pos_objs.append(pos_obj)
            self.rot_objs.append(rot_obj)
            self.obj_joint_limits_list.append(obj_joint_limits)
            self.joint_q_ik_list.append(joint_q_ik)
            self.ik_solvers.append(ik_solver)

    def compute(self, env, target: np.ndarray) -> None:
        for ri in range(env.num_robot):
            target_position = target[ri * 8 : ri * 8 + 3]
            target_rotation = target[ri * 8 + 3 : ri * 8 + 7]  # xyzw
            gripper_target = target[ri * 8 + 7]

            self.pos_objs[ri].set_target_positions(wp.array([target_position], dtype=wp.vec3))
            self.rot_objs[ri].set_target_rotations(wp.array([quat_to_vec4(target_rotation)], dtype=wp.vec4))

            if self.graph_ik_list:
                wp.capture_launch(self.graph_ik_list[ri])
            else:
                self.ik_solvers[ri].step(
                    self.joint_q_ik_list[ri], self.joint_q_ik_list[ri],
                    iterations=self.ik_iters,
                )

            wp.launch(
                broadcast_ik_solution_kernel,
                dim=1,
                inputs=[
                    self.joint_q_ik_list[ri],
                    gripper_target,
                    env.num_arm_joints,
                    env.num_gripper_joints,
                    ri,
                    env.control.joint_target_pos  # output
                ],
            )
