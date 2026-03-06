# Cloth Manipulation Simulation Environment

This repository contains a physics-based simulation environment for **cloth manipulation tasks**.

The environment supports contact-rich interactions between deformable cloth and robot URDFs, and is intended to be used for demo collection, policy training, and evaluation. It is designed to support the [2026 WBCD competition](https://wbcdcompetition.github.io/). Please refer to the details of the deformable manipulation task in the competition [here](https://wbcdcompetition.github.io/competition-tracks.html#dm).

The simulator is based on [NVIDIA Newton](https://github.com/newton-physics/newton). Although Newton has flexible support for deformable objects like clothes, it is still in **active beta development** stage. Thus, this repo is still subject to updates and needs case-specific integration into existing robot learning frameworks. Please be aware when adopting this environment for your own use.

A video showing the WBCD deformable maipulation task:

https://github.com/user-attachments/assets/c8cdb991-998f-4cec-9853-c7a65b4a8f7d

---

## Installation

```
# clone the repo
git clone --recurse-submodules git@github.com:kywind/cloth_sim.git
cd cloth_sim

# create and activate a python venv
uv venv --python=3.11
source .venv/bin/activate

# install newton
cd newton
pip install -e ".[examples]"
cd ..

# (optionally) verify installation by running newton examples
cd newton
python -m newton.examples robot_h1
cd ..

# install main dependencies
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# additional packages
uv pip install opencv-python omegaconf hydra-core pynput transforms3d

# for curobo
mkdir third-party
cd third-party
git clone git@github.com:NVlabs/curobo.git
cd curobo
uv pip install --no-build-isolation -e .

# version fix
uv pip install imgui_bundle==1.92.5
uv pip install --upgrade warp-lang
```

---

## Usage

```
### Launch examples

# normal (ik controller, default gl renderer)
python experiments/demo.py

# headless
python experiments/demo.py env.headless=True save_state=True

# save video after simulation with save_state=True (with example dir)
python experiments/make_video.py log/experiments/output_demo/20260305-212500 --fps 60 --output log/experiments/output_demo/20260305-212500.mp4

### Example options

# tiled camera renderer (fast, parallelizable rendering, random shape colors, can perform depth rendering)
python experiments/demo.py renderer=tiled_camera_renderer

# tiled camera renderer + random shape color disabled
python experiments/demo.py renderer=tiled_camera_renderer renderer.colors_per_shape=False

# curobo controller
python experiments/demo.py controller=curobo

# diffik controller
python experiments/demo.py controller=diffik
```

All examples run a manually defined robot action trajectory. The headless + save video command should output a video as follows:

https://github.com/user-attachments/assets/aff80727-78e5-4b54-a5f4-3e10f637680f

Using the tiled camera renderer + save video should output a video as follows:

https://github.com/user-attachments/assets/82cde720-dcd2-4403-98d3-fc936d01b77d

With random shape color disabled:

https://github.com/user-attachments/assets/28c0057b-819f-40bd-8fd0-b26fe70b200b

---

## Customize

Object and robot position configs could be configured in ```cfg/env/cloth_env_ARX.yaml```.

---

## Contact

For any questions regarding the repository, please create issues or contact the author, [Kaifeng Zhang](https://kywind.github.io/).
