# Stable Motion Primitives via Imitation and Contrastive Learning
[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)

Code accompanying the paper: "Stable Motion Primitives via Imitation and Contrastive Learning" (T-RO).
For details, please refer to https://arxiv.org/pdf/2302.10017.pdf. 

The current version of the paper can be cited using the following reference:
```bibtex
@article{PerezDattari2023TRO,
  author  = {P\'{e}rez-Dattari, Rodrigo AND Kober, Jens},
  journal = {IEEE Transactions on Robotics},
  title   = {Stable Motion Primitives via Imitation and Contrastive Learning},
  year    = {2023},
  pages   = {1--20},
  doi     = {10.1109/TRO.2023.3289597},
  code    = {https://github.com/rperezdattari/Stable-Motion-Primitives-via-Imitation-and-Contrastive-Learning},
  file    = {https://arxiv.org/pdf/2302.10017.pdf},
  project = {FlexCRAFT},
  video   = {https://youtu.be/OM-2edHBRfc},
  oa      = {green},
}
```
## Teaser: executing learned motion for multiple initial conditions
<p align="center">
    <img src="./media/s_shape_animation.gif" width="50%" height="50%"/>
</p>

## Options
This repository allows learning dynamical systems of multiple dimensions and orders.

### First-order 2-dimensional dynamical systems
<p float="left">
  <img src="./media/1st_order_2D.png" width="40%" height="40%"/>
  <img src="./media/1st_order_2D_diffeo.png" width="40%" height="40%"/> 
</p>

### Second-order 2-dimensional dynamical systems
<img src="./media/2nd_order_2D.png" width="40%" height="40%"/>

### First-order 3-dimensional dynamical systems
<img src="./media/1st_order_3D.png" width="60%" height="60%"/>

### First-order N-dimensional dynamical systems
<img src="./media/1st_order_ND.png" width="35%" height="35%"/>

## Robot Experiments
This repository contains simulated experiments; however, this framework has also been tested using a KUKA LBR iiwa robot manipulator. These results are shown in https://youtu.be/OM-2edHBRfc.
<p align="center">
    <img src="./media/robot_demo.gif"/>
</p>

## Installation with poetry

You can install the package using poetry.
```bash
poetry install
```

Enter the virtual environment using:
```bash
poetry shell
```

Requirements can be found at `pyproject.toml`.
`
## Usage
In the folder `src` run:

### Training
```bash
  python train.py --params <params_file_name>
```
The parameter files required for the argument `params_file_name` can be found in the folder `params`.

### Simulate learned 2D motion
```bash
  python simulate_ds.py
```

### Hyperparameter Optimization
```bash
  python run_optuna.py --params <params_file_name>
```

## Troubleshooting

If you run into problems of any kind, don't hesitate to [open an issue](https://github.com/rperezdattari/Stable-Motion-Primitives-via-Imitation-and-Contrastive-Learning/issues) on this repository.
