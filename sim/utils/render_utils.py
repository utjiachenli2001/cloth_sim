import warp as wp
import torch


@wp.kernel
def shape_index_to_semantic_rgb(
    shape_indices: wp.array(dtype=wp.uint32, ndim=3),
    colors: wp.array(dtype=wp.uint32),
    rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    world_id, camera_id, pixel_id = wp.tid()
    shape_index = shape_indices[world_id, camera_id, pixel_id]
    if shape_index < colors.shape[0]:
        rgba[world_id, camera_id, pixel_id] = colors[shape_index]
    else:
        rgba[world_id, camera_id, pixel_id] = wp.uint32(0xFF000000)


@wp.kernel
def shape_index_to_random_rgb(
    shape_indices: wp.array(dtype=wp.uint32, ndim=3),
    rgba: wp.array(dtype=wp.uint32, ndim=3),
):
    world_id, camera_id, pixel_id = wp.tid()
    shape_index = shape_indices[world_id, camera_id, pixel_id]
    random_color = wp.randi(wp.rand_init(12345, wp.int32(shape_index)))
    rgba[world_id, camera_id, pixel_id] = wp.uint32(random_color) | wp.uint32(0xFF000000)


def alpha_over(top, bottom, eps=1e-8):
    # top, bottom: (..., H, W, 4) RGBA, straight alpha in [0,1]
    Ct, At = top[..., :3], top[..., 3:4]
    Cb, Ab = bottom[..., :3], bottom[..., 3:4]

    Ao = At + Ab * (1 - At)
    Co = (Ct * At + Cb * Ab * (1 - At)) / (Ao + eps)

    out = torch.cat([Co, Ao], dim=-1)
    # optional: if Ao==0, force Co=0
    out[..., :3] = torch.where(Ao > eps, out[..., :3], torch.zeros_like(out[..., :3]))
    return out
