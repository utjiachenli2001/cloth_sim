from pathlib import Path
from typing import Optional
import numpy as np
import warp as wp
import time
from pxr import Usd, UsdGeom
import sys
sys.path.append(str(Path(__file__).parents[2] / "newton"))
import newton
import newton.ik as ik
from newton._src.geometry.utils import load_mesh
from newton.solvers import style3d
from newton.viewer import ViewerGL
from ..utils.env_utils import xyzw_to_wxyz, wxyz_to_xyzw, quat_to_vec4
from ..utils.curobo_controller import CuroboController


class ClothEnvARXV1:
    def __init__(self, cfg, viewer):
        self.cfg = cfg
        self.viewer = viewer

        self.num_robot = 2

        # simulation parameters
        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 15
        self.iterations = 10
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.collide_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # body-cloth contact
        self.cloth_particle_radius = 0.003  # style3d
        self.cloth_body_contact_margin = 0.01
        self.soft_contact_ke = 100
        self.soft_contact_kd = 2e-3

        # cloth-cloth contact
        self.self_contact_radius = 0.002
        self.self_contact_margin = 0.003

        # friction
        self.robot_friction = 10.0
        self.table_friction = 0.25
        self.cloth_friction = 0.25

        self.scene = newton.ModelBuilder()

        newton.solvers.SolverMuJoCo.register_custom_attributes(self.scene)

        if self.add_cloth:
            newton.solvers.SolverStyle3D.register_custom_attributes(self.scene)

        self.scene.default_shape_cfg.ke = 5.0e4
        self.scene.default_shape_cfg.kd = 5.0e2
        self.scene.default_shape_cfg.kf = 1.0e3
        self.scene.default_shape_cfg.mu = self.table_friction

        self.num_arm_joints = 6
        self.num_gripper_joints = 2

        # left arm
        arx_left = newton.ModelBuilder()
        arx_left.default_shape_cfg.mu = self.robot_friction
        arx_left.rigid_contact_margin = 0.001  # 1mm, down from default 10cm

        arx_left.add_urdf(
            'experiments/assets/robots/ARX-X5/X5A.urdf',
            xform=wp.transform(
                (0.5, -0.55, 0.01),
                wp.quat(0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)),
            ),
            floating=False,
            scale=1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=True,
        )
        arx_left.joint_q = [0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.01, 0.01]
        self.left_b2w = np.array([0.5, -0.55, 0.01, np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])  # wxyz

        arx_left.joint_target_kd[:self.num_arm_joints] = [200.0] * self.num_arm_joints
        arx_left.joint_target_ke[:self.num_arm_joints] = [5000.0] * self.num_arm_joints
        arx_left.joint_target_kd[self.num_arm_joints:] = [200.0] * self.num_gripper_joints
        arx_left.joint_target_ke[self.num_arm_joints:] = [5000.0] * self.num_gripper_joints

        if self.cfg.env.controller == 'ik':
            left_arm_model = arx_left.finalize()

        self.scene.add_builder(arx_left)

        self.bodies_per_robot = arx_left.body_count
        self.dof_per_robot = arx_left.joint_dof_count
        self.endeffector_id = arx_left.body_count - 3

        # right arm
        arx_right = newton.ModelBuilder()
        arx_right.default_shape_cfg.mu = self.robot_friction
        arx_right.rigid_contact_margin = 0.001  # 1mm, down from default 10cm

        arx_right.add_urdf(
            'experiments/assets/robots/ARX-X5/X5A.urdf',
            xform=wp.transform(
                (0.5, 0.55, 0.01),
                wp.quat(0.0, 0.0, -np.sin(np.pi / 4), np.cos(np.pi / 4)),
            ),
            floating=False,
            scale=1,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=True,
        )
        arx_right.joint_q = [0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.01, 0.01]
        self.right_b2w = np.array([0.5, 0.55, 0.01, np.cos(np.pi / 4), 0.0, 0.0, -np.sin(np.pi / 4)])  # wxyz

        arx_right.joint_target_kd[:self.num_arm_joints] = [200.0] * self.num_arm_joints
        arx_right.joint_target_ke[:self.num_arm_joints] = [5000.0] * self.num_arm_joints
        arx_right.joint_target_kd[self.num_arm_joints:] = [200.0] * self.num_gripper_joints
        arx_right.joint_target_ke[self.num_arm_joints:] = [5000.0] * self.num_gripper_joints

        if self.cfg.env.controller == 'ik':
            right_arm_model = arx_right.finalize()

        self.scene.add_builder(arx_right)

        assert self.bodies_per_robot == arx_right.body_count
        assert self.dof_per_robot == arx_right.joint_dof_count
        assert self.endeffector_id == arx_right.body_count - 3

        self._single_arm_models = [left_arm_model, right_arm_model]

        # add board
        mesh_points, mesh_indices = load_mesh('experiments/assets/meshes/board.stl')
        mesh = newton.Mesh(mesh_points, mesh_indices)
        self.scene.add_shape_mesh(
            body=-1,
            xform=wp.transform(
                wp.vec3(0.70, 0.0, 0.10),
                wp.quat_identity(),
            ),
            mesh=mesh,
            scale=wp.vec3(1.0, 1.0, 1.0),
            label='board',
        )

        # add the garment
        garment_usd_name = "Female_T_Shirt"
        asset_path = newton.utils.download_asset("style3d")
        usd_stage = Usd.Stage.Open(f"{asset_path}/garments/{garment_usd_name}.usd")
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath(f"/Root/{garment_usd_name}/Root_Garment"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())
        prim = UsdGeom.PrimvarsAPI(usd_geom.GetPrim()).GetPrimvar("st")
        mesh_uv_indices = np.array(prim.GetIndices())
        mesh_uv = np.array(prim.Get()) * 1e-3

        mesh_points -= mesh_points.mean(axis=0)

        vertices = [wp.vec3(v) for v in mesh_points]

        if self.add_cloth:
            style3d.add_cloth_mesh(
                self.scene,
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2),
                pos=wp.vec3(0.0, 0.0, 0.1),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.5,
                scale=1.0,
                tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e2) * 5.0,
                edge_aniso_ke=wp.vec3(1.0e-6, 1.0e-6, 1.0e-6) * 4.0 * 5.0,
                panel_verts=mesh_uv.tolist(),
                panel_indices=mesh_uv_indices.tolist(),
                particle_radius=self.cloth_particle_radius,
            )

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.cloth_friction

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=500,
            nconmax=500,
            iterations=15,
            ls_iterations=100,
            impratio=1000.0,
        )

        # initialize cloth solver
        self.cloth_solver = None
        if self.add_cloth:
            self.cloth_solver = newton.solvers.SolverStyle3D(
                model=self.model,
                iterations=self.iterations,
            )
            self.cloth_solver.precompute(self.scene)
            self.cloth_solver.collision.radius = 3.5e-3

        # initialize contacts
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            reduce_contacts=True,
            broad_phase="explicit",
        )
        self.contacts = self.collision_pipeline.contacts()

        # initialize control
        # newton class for control
        self.control = self.model.control()
        # IK (newton) or curobo
        if self.cfg.env.controller == 'ik':
            self.graph_ik_list = []
            self.setup_ik()
            for ri in range(self.num_robot):
                with wp.ScopedCapture() as capture:
                    self.ik_solvers[ri].step(
                        self.joint_q_ik_list[ri], self.joint_q_ik_list[ri], iterations=self.ik_iters
                    )
                self.graph_ik_list.append(capture.graph)

        elif self.cfg.env.controller == 'curobo':
            self.prev_target = None
            self.left_path = {}
            self.right_path = {}
            self.plan_step = 0
            self.left_controller = CuroboController(cfg, self.frame_dt, self.left_b2w)
            self.right_controller = CuroboController(cfg, self.frame_dt, self.right_b2w)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3((1.5, 0.0, 0.75)), pitch=-30, yaw=180)  # x: left-right, y: forward-backward, z: up-down

        # create Warp arrays for gravity so we can swap Model.gravity during
        # a simulation running under CUDA graph capture
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)  # used for the robot solver
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)  # used for the cloth solver

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # graph capture
        self.graph = None
        if self.add_cloth:
            self.capture()

    def capture(self):
        if self.cfg.env.use_graph and wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def get_state(self):
        return self.state_0.numpy()

    def setup_ik(self):
        assert len(self._single_arm_models) == self.num_robot, \
            f"Expected {self.num_robot} single-arm models, got {len(self._single_arm_models)}. " \
            f"Ensure controller='ik' was set before _build_scene() ran."

        self.ik_iters = 24
        self.pos_objs = []
        self.rot_objs = []
        self.obj_joint_limits_list = []
        self.joint_q_ik_list = []
        self.ik_solvers = []

        for model_single in self._single_arm_models:
            state_single = model_single.state()
            body_q_np = state_single.body_q.numpy()
            ee_tf = wp.transform(*body_q_np[self.endeffector_id])

            pos_obj = ik.IKObjectivePosition(
                link_index=self.endeffector_id,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=wp.array([wp.transform_get_translation(ee_tf)], dtype=wp.vec3),
            )
            rot_obj = ik.IKObjectiveRotation(
                link_index=self.endeffector_id,
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

    def generate_joint_control_ik(self, state_in, target):
        for ri in range(self.num_robot):
            target_position = target[ri * 8 : ri * 8 + 3]
            target_rotation = target[ri * 8 + 3 : ri * 8 + 7]  # xyzw
            gripper_target = target[ri * 8 + 7]

            self.pos_objs[ri].set_target_positions(wp.array([target_position], dtype=wp.vec3))
            self.rot_objs[ri].set_target_rotations(
                wp.array([quat_to_vec4(target_rotation)], dtype=wp.vec4)
            )

            if self.graph_ik_list:
                wp.capture_launch(self.graph_ik_list[ri])
            else:
                self.ik_solvers[ri].step(
                    self.joint_q_ik_list[ri], self.joint_q_ik_list[ri], iterations=self.ik_iters
                )

            ik_solution = self.joint_q_ik_list[ri].flatten()  # shape: (coord_count_per_arm,)

            offset = ri * self.dof_per_robot
            wp.copy(self.control.joint_target_pos[offset : offset + self.num_arm_joints], ik_solution[:self.num_arm_joints])
            wp.copy(self.control.joint_target_pos[offset + self.num_arm_joints : offset + self.dof_per_robot], wp.array([gripper_target, gripper_target]))

    def generate_joint_control_curobo(self, state_in, target):
        if self.prev_target is None or not np.allclose(self.prev_target, target):
            self.left_path, self.right_path = self.plan_path(state_in, target)
            self.plan_step = 0
            self.prev_target = target.copy()
        
        current_step_left = min(self.plan_step, len(self.left_path['position']) - 1)
        current_step_right = min(self.plan_step, len(self.right_path['position']) - 1)
        left_target_pos = self.left_path['position'][current_step_left]
        right_target_pos = self.right_path['position'][current_step_right]

        left_target_vel = self.left_path['velocity'][current_step_left]
        right_target_vel = self.right_path['velocity'][current_step_right]

        target_pos = np.concatenate([left_target_pos, right_target_pos], axis=0)
        target_vel = np.concatenate([left_target_vel, right_target_vel], axis=0)

        wp.copy(self.control.joint_target_pos, wp.array(target_pos, dtype=float))
        wp.copy(self.control.joint_target_vel, wp.array(target_vel, dtype=float))
        
        self.plan_step += 1

    def plan_path(self, state_in, target):
        # use curobo controller
        joint_q = state_in.joint_q.numpy()
        target_pose = target.reshape(self.num_robot, 8)
        target_pose = xyzw_to_wxyz(target_pose)

        left_path_result = self.left_controller.plan_path(
            curr_joint_pos=joint_q[:8],
            target_pose=target_pose[0],
        )
        right_path_result = self.right_controller.plan_path(
            curr_joint_pos=joint_q[8:],
            target_pose=target_pose[1],
        )
        if left_path_result['status'] != 'Success' or right_path_result['status'] != 'Success':
            print('Planning failed!')
            import ipdb; ipdb.set_trace()
        
        return left_path_result, right_path_result

    def step(self, action: Optional[dict] = None):
        if action is not None and 'target' in action:
            if self.cfg.env.controller == 'ik':
                self.generate_joint_control_ik(self.state_0, action['target'])

            elif self.cfg.env.controller == 'curobo':
                self.generate_joint_control_curobo(self.state_0, action['target'])

        else:
            raise ValueError("Action must contain 'target' key.")

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def simulate(self):
        if self.add_cloth:
            self.cloth_solver.rebuild_bvh(self.state_0)

        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if _step % self.collide_substeps == 0:
                self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                # self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.model.shape_contact_pair_count = 0

                self.robot_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

                # restore original settings
                self.model.particle_count = particle_count
                # self.model.gravity.assign(self.gravity_earth)

            if particle_count > 0:
                self.state_0.particle_f.zero_()  # prevent solver fighting

            # cloth sim
            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()
