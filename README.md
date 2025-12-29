# M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place
### [project website](https://m2-t2.github.io) &emsp; &emsp; [arxiv paper](https://arxiv.org/abs/2311.00926) &emsp; &emsp; [model weights](https://drive.google.com/drive/folders/1qlvHVi1-Jk4ET-NyHwnqZOxALVy9kTO5)
![robot](figures/real_robot.gif)

This repository is a fork of M2T2. Please see [README_OLD.md](README_OLD.md) for the original README. We have included instructions for installing M2T2 and running it as an inference server.

## Installation

We use [pixi](https://pixi.prefix.dev/) to manage the Python environment and the dependencies. If you don't already have it installed, then you can run.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Then, follow the instructions below to setup M2T2:

```bash
# Clone the repository
git clone https://github.com/williamshen-nz/M2T2.git
cd M2T2

# Install dependencies (this may take a few minutes)
pixi install
pixi run setup

# Download pretrained weights
pixi run download-weights
```

### Running the Demo

```bash
# Run the full demo with meshcat visualization
pixi run demo
```

Go to Meshcat on http://127.0.0.1:7000/static/ in your browser, and wait to see the grasp predictions. It should look like the figure below.

<img src="figures/demo.jpg" width="450">

**Tip**: Press `Ctrl+B` then `D` to detach from the tmux session.

## Running the Inference Server

```bash
# Start the server (default: http://0.0.0.0:8123)
pixi run server

# Customize server configuration
pixi run server -- --port 8080 --checkpoint weights/m2t2.pth

# See all options
pixi run server -- --help
```


## Citation

If you find this work useful, please consider citing the original authors:

```bibtex
@inproceedings{yuan2023m2t2,
  title     = {M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place},
  author    = {Yuan, Wentao and Murali, Adithyavairavan and Mousavian, Arsalan and Fox, Dieter},
  booktitle = {7th Annual Conference on Robot Learning},
  year      = {2023}
}
```