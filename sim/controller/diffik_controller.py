import numpy as np
import warp as wp

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[2] / "newton"))
import newton

from .base_controller import BaseController
from ..utils.env_utils import compute_ee_delta, compute_body_out


class DiffIKController(BaseController):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def initialize_resources(self, env) -> None:
        self._J_out_dim = 6 * env.num_robot
        self._J_in_dim = (env.num_arm_joints + env.num_gripper_joints) * env.num_robot
        self.num_joints_per_robot = env.num_arm_joints + env.num_gripper_joints
        self.bodies_per_robot = env._single_arm_models[0].body_count
        self.use_null_space = env.num_arm_joints > 6
        self.use_deadband = True

        def onehot(i, out_dim):
            return wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)

        self._J_one_hots = [onehot(i, self._J_out_dim) for i in range(self._J_out_dim)]
        self._temp_state = env.model.state(requires_grad=True)
        self._body_out = wp.empty(self._J_out_dim, dtype=float, requires_grad=True)
        self._J_flat = wp.empty(self._J_out_dim * self._J_in_dim, dtype=float)
        self._ee_delta = wp.empty(env.num_robot, dtype=wp.spatial_vector)

    def compute(self, env, target: np.ndarray) -> None:
        target_wp = wp.array(
            [wp.transform(*target[i * 8 : i * 8 + 7]) for i in range(env.num_robot)],
            dtype=wp.transform,
        )
        wp.launch(
            compute_ee_delta,
            dim=env.num_robot,
            inputs=[
                env.state_0.body_q,
                env.endeffector_offset,
                env.endeffector_id,
                self.bodies_per_robot,
                target_wp,
            ],
            outputs=[self._ee_delta],
        )
        ee_delta = self._ee_delta.numpy()  # (num_robot, 6)

        xyz_all = ee_delta[:, :3]
        omega_all = ee_delta[:, 3:]
        dist_all = np.linalg.norm(xyz_all, axis=1)
        w_sq = np.clip(1.0 - np.sum(omega_all ** 2, axis=1), 0.0, 1.0)
        theta_all = 2 * np.arccos(np.sqrt(w_sq))

        self._compute_body_jacobian(env)
        J_full = self._J_flat.numpy().reshape(-1, self._J_in_dim)
        q_full = env.state_0.joint_q.numpy()

        qd_list = []
        q_target_list = []

        for ri in range(env.num_robot):
            xyz = xyz_all[ri]
            omega = omega_all[ri]
            theta = theta_all[ri]
            dist = dist_all[ri]

            u = np.zeros(3) if theta < 1e-6 else omega / np.sin(theta / 2)

            alpha = 0.1
            xd = np.zeros(6, dtype=np.float32)
            xd[:3] = xyz * alpha / env.frame_dt
            xd[3:] = u * theta * alpha / env.frame_dt

            J = J_full[ri * 6 : (ri + 1) * 6,
                       ri * self.num_joints_per_robot : (ri + 1) * self.num_joints_per_robot]
            q = q_full[ri * self.num_joints_per_robot : (ri + 1) * self.num_joints_per_robot]
            gripper_target = target[ri * 8 + 7]

            lambda_damping = 0.01
            J_inv = J.T @ np.linalg.pinv(
                J @ J.T + (lambda_damping ** 2) * np.eye(6, dtype=np.float32)
            )

            if self.use_null_space:
                I = np.eye(J.shape[1], dtype=np.float32)
                N = I - J_inv @ J
                q_des = q.copy()
                q_des[1:env.num_arm_joints] = [0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0]
                qd = J_inv @ xd + N @ (1.0 * (q_des - q))
            else:
                qd = J_inv @ xd

            if self.use_deadband and dist < 0.01 and theta < 0.05:
                qd[:-2] *= 0.0

            qd[-2] = min(abs(gripper_target - q[-2]), 0.01) / env.frame_dt
            qd[-1] = min(abs(gripper_target - q[-1]), 0.01) / env.frame_dt
            if gripper_target < q[-2]:
                qd[-2] *= -1.0
            if gripper_target < q[-1]:
                qd[-1] *= -1.0

            q_target_list.append(q + qd * env.frame_dt)
            qd_list.append(qd)

        qd_full = np.concatenate(qd_list)
        q_target_full = np.concatenate(q_target_list)

        qd_pad = np.zeros_like(env.control.joint_target_vel.numpy())
        q_pad = np.zeros_like(env.control.joint_target_pos.numpy())
        qd_pad[:qd_full.shape[0]] = qd_full
        q_pad[:q_target_full.shape[0]] = q_target_full

        env.control.joint_target_vel.assign(qd_pad)
        env.control.joint_target_pos.assign(q_pad)

    def _compute_body_jacobian(self, env) -> None:
        """Compute the Jacobian of EE velocity w.r.t. joint_q via autodiff."""
        joint_q = env.state_0.joint_q
        joint_qd = env.state_0.joint_qd
        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        tape = wp.Tape()
        with tape:
            newton.eval_fk(env.model, joint_q, joint_qd, self._temp_state)
            wp.launch(
                compute_body_out,
                dim=env.num_robot,
                inputs=[
                    self._temp_state.body_qd,
                    env.endeffector_id,
                    self.bodies_per_robot,
                    env.endeffector_offset,
                ],
                outputs=[self._body_out],
            )

        for i in range(self._J_out_dim):
            tape.backward(grads={self._body_out: self._J_one_hots[i]})
            wp.copy(
                self._J_flat[i * self._J_in_dim : (i + 1) * self._J_in_dim],
                joint_qd.grad[:self._J_in_dim],
            )
            tape.zero()
