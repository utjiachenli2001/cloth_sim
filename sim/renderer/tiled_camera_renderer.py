from .base_renderer import BaseRenderer
import warp as wp
import numpy as np
import OpenGL.GL as gl
import ctypes
from newton.viewer import ViewerGL
from newton.sensors import SensorTiledCamera

from ..utils.render_utils import shape_index_to_semantic_rgb, shape_index_to_random_rgb


class TiledCameraRenderer(BaseRenderer):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.image_output = 0
        self.texture_id = 0
        self.num_worlds_per_row = 1
        self.num_worlds_per_col = 1
        self.num_worlds_total = self.num_worlds_per_row * self.num_worlds_per_col

        self.ui_padding = 10
        self.ui_side_panel_width = 300

    def initialize_resources(self, env):
        sensor_render_width = 1280
        sensor_render_height = 720

        if isinstance(env.viewer, ViewerGL) and env.viewer.ui is not None:
            self.display_size = env.viewer.ui.io.display_size
            env.viewer.register_ui_callback(self.display, position="free")
            env.viewer.register_ui_callback(self.gui, position="side")

            sensor_render_width = int(self.display_size[0] // self.num_worlds_per_row)
            sensor_render_height = int(self.display_size[1] // self.num_worlds_per_col)

        self.sensor_render_width = sensor_render_width
        self.sensor_render_height = sensor_render_height

        self.tiled_camera_sensor = SensorTiledCamera(
            model=env.model,
            options=SensorTiledCamera.Options(
                default_light=True,
                default_light_shadows=True,
                colors_per_shape=self.cfg.renderer.colors_per_shape,
                checkerboard_texture=True,
                backface_culling=False,
            ),
        )

        fov = 45.0
        if isinstance(env.viewer, ViewerGL):
            fov = env.viewer.camera.fov

        self.camera_rays = self.tiled_camera_sensor.compute_pinhole_camera_rays(
            width=sensor_render_width,
            height=sensor_render_height,
            camera_fovs=np.deg2rad(fov)
        )
        self.tiled_camera_sensor_color_image = self.tiled_camera_sensor.create_color_image_output(
            width=sensor_render_width,
            height=sensor_render_height,
            camera_count=1,
        )
        self.tiled_camera_sensor_depth_image = self.tiled_camera_sensor.create_depth_image_output(
            width=sensor_render_width,
            height=sensor_render_height,
            camera_count=1,
        )
        self.tiled_camera_sensor_normal_image = self.tiled_camera_sensor.create_normal_image_output(
            width=sensor_render_width,
            height=sensor_render_height,
            camera_count=1,
        )
        self.tiled_camera_sensor_shape_index_image = self.tiled_camera_sensor.create_shape_index_image_output(
            width=sensor_render_width,
            height=sensor_render_height,
            camera_count=1,
        )

        if isinstance(env.viewer, ViewerGL):
            self.create_texture()

    def render(self, env, return_renderings=False):  # default viewer rendering
        if env.viewer is None:
            return {}
        render_result = self.render_sensors(env, return_renderings)
        env.viewer.begin_frame(env.sim_time)
        env.viewer.log_state(env.state_0)
        env.viewer.end_frame()
        return render_result

    def render_sensors(self, env, return_renderings=False):
        self.tiled_camera_sensor.update(
            env.state_0,
            self.get_camera_transforms(env),
            self.camera_rays,
            color_image=self.tiled_camera_sensor_color_image,
            depth_image=self.tiled_camera_sensor_depth_image,
            normal_image=self.tiled_camera_sensor_normal_image,
            shape_index_image=self.tiled_camera_sensor_shape_index_image,
        )
        self.update_texture()

        if return_renderings:
            rgba_buffer = wp.array(
                shape=(
                    self.num_worlds_per_col * self.sensor_render_height,
                    self.num_worlds_per_row * self.sensor_render_width,
                    4,
                ),
                dtype=wp.uint8,
            )
            if self.image_output == 0:  # color
                self.tiled_camera_sensor.flatten_color_image_to_rgba(
                    self.tiled_camera_sensor_color_image, rgba_buffer, self.num_worlds_per_row
                )
            elif self.image_output == 3:  # semantic
                wp.launch(
                    shape_index_to_semantic_rgb,
                    self.tiled_camera_sensor_shape_index_image.shape,
                    [self.tiled_camera_sensor_shape_index_image, self.semantic_colors],
                    [self.tiled_camera_sensor_shape_index_image],
                )
                self.tiled_camera_sensor.flatten_color_image_to_rgba(
                    self.tiled_camera_sensor_shape_index_image, rgba_buffer, self.num_worlds_per_row
                )
            else:
                raise NotImplementedError("Only color and semantic image outputs are supported.")
            results = {
                'rgba': rgba_buffer.numpy()  # (H, W, 4)
            }
        else:
            results = {}
        return results

    def get_camera_transforms(self, env) -> wp.array(dtype=wp.transformf):
        if isinstance(env.viewer, ViewerGL):
            return wp.array(
                [
                    [
                        wp.transformf(
                            env.viewer.camera.pos,
                            wp.quat_from_matrix(wp.mat33f(env.viewer.camera.get_view_matrix().reshape(4, 4)[:3, :3])),
                        )
                    ]
                    * self.num_worlds_total
                ],
                dtype=wp.transformf,
            )
        return wp.array(
            [[wp.transformf(wp.vec3f(10.0, 0.0, 2.0), wp.quatf(0.5, 0.5, 0.5, 0.5))] * self.num_worlds_total],
            dtype=wp.transformf,
        )

    def create_texture(self):
        width = self.sensor_render_width * self.num_worlds_per_row
        height = self.sensor_render_height * self.num_worlds_per_col

        self.texture_id = gl.glGenTextures(1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, width, height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        self.pixel_buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glBufferData(gl.GL_PIXEL_UNPACK_BUFFER, width * height * 4, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)

        self.texture_buffer = wp.RegisteredGLBuffer(self.pixel_buffer)

    def update_texture(self):
        if not self.texture_id:
            return

        texture_buffer = self.texture_buffer.map(
            dtype=wp.uint8,
            shape=(
                self.num_worlds_per_col * self.sensor_render_height,
                self.num_worlds_per_row * self.sensor_render_width,
                4,
            ),
        )
        if self.image_output == 0:
            self.tiled_camera_sensor.flatten_color_image_to_rgba(
                self.tiled_camera_sensor_color_image, texture_buffer, self.num_worlds_per_row
            )
        elif self.image_output == 1:
            self.tiled_camera_sensor.flatten_depth_image_to_rgba(
                self.tiled_camera_sensor_depth_image, texture_buffer, self.num_worlds_per_row
            )
        elif self.image_output == 2:
            self.tiled_camera_sensor.flatten_normal_image_to_rgba(
                self.tiled_camera_sensor_normal_image, texture_buffer, self.num_worlds_per_row
            )
        elif self.image_output == 3:
            wp.launch(
                shape_index_to_semantic_rgb,
                self.tiled_camera_sensor_shape_index_image.shape,
                [self.tiled_camera_sensor_shape_index_image, self.semantic_colors],
                [self.tiled_camera_sensor_shape_index_image],
            )
            self.tiled_camera_sensor.flatten_color_image_to_rgba(
                self.tiled_camera_sensor_shape_index_image, texture_buffer, self.num_worlds_per_row
            )
        elif self.image_output == 4:
            wp.launch(
                shape_index_to_random_rgb,
                self.tiled_camera_sensor_shape_index_image.shape,
                [self.tiled_camera_sensor_shape_index_image],
                [self.tiled_camera_sensor_shape_index_image],
            )
            self.tiled_camera_sensor.flatten_color_image_to_rgba(
                self.tiled_camera_sensor_shape_index_image, texture_buffer, self.num_worlds_per_row
            )
        self.texture_buffer.unmap()

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, self.pixel_buffer)
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.sensor_render_width * self.num_worlds_per_row,
            self.sensor_render_height * self.num_worlds_per_col,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            ctypes.c_void_p(0),
        )
        gl.glBindBuffer(gl.GL_PIXEL_UNPACK_BUFFER, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def gui(self, ui):
        if ui.radio_button("Show Color Output", self.image_output == 0):
            self.image_output = 0
        if ui.radio_button("Show Depth Output", self.image_output == 1):
            self.image_output = 1
        if ui.radio_button("Show Normal Output", self.image_output == 2):
            self.image_output = 2
        if ui.radio_button("Show Semantic Output", self.image_output == 3):
            self.image_output = 3
        if ui.radio_button("Show Shape Index Output", self.image_output == 4):
            self.image_output = 4

    def display(self, imgui):
        line_color = imgui.get_color_u32(imgui.Col_.window_bg)

        width = self.display_size[0] - self.ui_side_panel_width - self.ui_padding * 4
        height = self.display_size[1] - self.ui_padding * 2

        imgui.set_next_window_pos(imgui.ImVec2(0, 0))
        imgui.set_next_window_size(self.display_size)

        flags = (
            imgui.WindowFlags_.no_title_bar.value
            | imgui.WindowFlags_.no_mouse_inputs.value
            | imgui.WindowFlags_.no_bring_to_front_on_focus.value
            | imgui.WindowFlags_.no_scrollbar.value
        )

        if imgui.begin("Sensors", flags=flags):
            pos_x = self.ui_side_panel_width + self.ui_padding * 2
            pos_y = self.ui_padding

            if self.texture_id > 0:
                imgui.set_cursor_pos(imgui.ImVec2(pos_x, pos_y))
                imgui.image(imgui.ImTextureRef(self.texture_id), imgui.ImVec2(width, height))

            draw_list = imgui.get_window_draw_list()
            for x in range(1, self.num_worlds_per_row):
                draw_list.add_line(
                    imgui.ImVec2(pos_x + x * (width / self.num_worlds_per_row), pos_y),
                    imgui.ImVec2(pos_x + x * (width / self.num_worlds_per_row), pos_y + height),
                    line_color,
                    2.0,
                )
            for y in range(1, self.num_worlds_per_col):
                draw_list.add_line(
                    imgui.ImVec2(pos_x, pos_y + y * (height / self.num_worlds_per_col)),
                    imgui.ImVec2(pos_x + width, pos_y + y * (height / self.num_worlds_per_col)),
                    line_color,
                    2.0,
                )

        imgui.end()
