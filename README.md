# M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place

**M2T2** is a unified transformer model for learning different low-level action primitives on complex open-world scenes. Given a raw point cloud observation, M2T2 reasons about contact points and predicts collision-free gripper poses for different action modes, including 6-DoF object-centric grasping and orientation-aware placement.

### Links
- 🌐 [Project Website](https://m2-t2.github.io)
- 📄 [arXiv Paper](https://arxiv.org/abs/2311.00926)
- 🔗 [Model Weights](https://huggingface.co/wentao-yuan/m2t2)

![robot](figures/real_robot.gif)

## Quick Start

### Prerequisites
- [Pixi](https://pixi.sh) package manager
- NVIDIA GPU with CUDA 12.x drivers
- `tmux` for running the demo

### Installation

```bash
# Clone the repository
git clone https://github.com/williamshen-nz/M2T2.git
cd M2T2

# Install dependencies (this may take a few minutes)
pixi install

# Build and install M2T2 and PointNet++
pixi run setup

# Download pretrained weights
pixi run download-weights
```

### Running the Demo

```bash
# Run the full demo with meshcat visualization
pixi run demo
```

This starts a tmux session with three panes:
- **Pane 0**: M2T2 client demo
- **Pane 1**: M2T2 server
- **Pane 2**: Meshcat visualization server

Open http://127.0.0.1:7000/static/ in your browser to see the grasp predictions.

**Tip**: Press `Ctrl+B` then `D` to detach from the tmux session.

## CUDA Compatibility

The default configuration uses PyTorch with CUDA 12.1, which works with any CUDA 12.x driver (12.0, 12.1, 12.2, etc.).

If you have CUDA 11.x, edit `pixi.toml` and change:
```toml
torch = "==2.5.1+cu121"
torchvision = "==0.20.1+cu121"
...
extra-index-urls = ["https://download.pytorch.org/whl/cu121"]
```

to:
```toml
torch = "==2.0.1+cu118"
torchvision = "==0.15.2+cu118"
...
extra-index-urls = ["https://download.pytorch.org/whl/cu118"]
```

Check your CUDA version with `nvidia-smi`.

## Supported GPUs

M2T2 supports a wide range of NVIDIA GPUs including:
- RTX 3070, 3080, 3090
- RTX 4090
- RTX 5090
- A6000 series
- And more

The PointNet++ CUDA extensions will automatically compile for your specific GPU architecture.

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{yuan2023m2t2,
  title     = {M2T2: Multi-Task Masked Transformer for Object-centric Pick and Place},
  author    = {Yuan, Wentao and Murali, Adithyavairavan and Mousavian, Arsalan and Fox, Dieter},
  booktitle = {7th Annual Conference on Robot Learning},
  year      = {2023}
}
```

## Additional Documentation

For detailed training instructions, data format, and advanced usage, see [README_OLD.md](README_OLD.md).

## License

MIT Software License
