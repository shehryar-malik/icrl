**Status:** Archive (code is provided as-is, no updates expected)

This repository contains the code for the paper Inverse Constrained Reinforcement Learning (ICML 2021) [[link]](https://arxiv.org/abs/2011.09999). This codebase is built on top of the [stablebaselines3](https://github.com/DLR-RM/stable-baselines3) repository.

## Code Dependency

It is recommended to use Python 3.8 to run this code within a virtual environment. Within the virtual environment, run the following commands to download the essential python packages for this codebase. You will also need to setup [mujoco](https://github.com/openai/mujoco-py).

```bash
pip install 'mujoco-py<2.1,>=2.0'
pip install wandb==0.10.12 torch==1.5.0 gym==0.15.7 matplotlib==3.3.2 numpy==1.17.5 cloudpickle==1.2.2 tqdm pandas pillow psutil mpl-scatter-density
pip install -e ./custom_envs # To access custom environments through gym interface
```

In addition, you will need to setup a (free) [wandb](www.wandb.ai) account. Note that this codebase is based on a fork of stable-baseline3 which is also provided with the code.

To run any experiment present in the code, go to the main directory and run a command from the following list. Experiments typically take 2-3 hours to complete.

## Running Constraint Learning Experiments (Section 4.1)

### Lap Grid World

```bash
# ICRL
python run_me.py icrl -p ICRL-FE2 --group LapGrid-ICRL -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -tk 0.01 -cl 20 -clr 0.003 -ft 0.5e5 -ni 10 -bi 20 -dno -dnr -dnc

# Binary Classifier
python run_me.py icrl -p ICRL-FE2 --group LapGrid-Glag --train_gail_lambda -nis -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -tk 0.01 -cl 20 -crc 0.5 -clr 0.01 -ft 10000 --n_steps 2000 -ni 12 -bi 10 -dno -dnr -dnc

# GAIL-Constraint
python run_me.py gail -p ICRL-FE2 --group LapGrid-GLC -er 20 -ep icrl/expert_data/LGW -tei LGW-v0 -eei CLGW-v0 -tk 0.01 -dl 20 -dlr 0.01 -t 120000 --n_steps 2000 -dno -dnr -lc
```

### Half Cheetah

```bash
# ICRL
python run_me.py icrl -p ICRL-FE2 --group HC-ICRL -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 20 -bi 10 -ft 2e5 -ni 30 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5

# Binary Classifier
python run_me.py icrl -p ICRL-FE2 --group HC-Glag --train_gail_lambda -nis -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -cl 30 -bi 10 -ft 2e5 -ni 30 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -clr 0.05 -aclr 0.9 -crc 0.5 -psis -ctkno 2.5

# GAIL-Constraint
python run_me.py gail -p ICRL-FE2 --group HC-GLC -er 10 -ep icrl/expert_data/HCWithPos-New -tk 0.01 -t 4e6 -tei HCWithPos-v0 -eei HCWithPosTest-v0 -dl 30 -dlr 0.003 -lc
```
### Ant Wall

```bash
# ICRL
python run_me.py icrl -p ICRL-FE2 --group AntWall-ICRL -ep icrl/expert_data/AntWall -er 45 -cl 40 40 -clr 0.005 -aclr 0.9 -crc 0.6 -bi 5 -ft 2e5 -ni 20 -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -piv 0.1 -plr 0.05 -psis -tk 0.02 -ctkno 2.5

# Binary Classifier
python run_me.py icrl -p ICRL-FE2 --group AntWall-GLag --train_gail_lambda -nis -ep icrl/expert_data/AntWall -er 45 -cl 40 40 -clr 0.005 -aclr 0.9 -crc 0.6 -bi 5 -ft 2e5 -ni 20 -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --cost_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -piv 0.1 -plr 0.05 -psis -tk 0.02 -ctkno 2.5

# GAIL-Constraint
python run_me.py gail -p ICRL-FE2 --group AntWall-GLC -ep icrl/expert_data/AntWall -er 45 -dl 40 40 -dlr 0.005 -t 4e6 -tei AntWall-v0 -eei AntWallTest-v0 --batch_size 128 --reward_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -lc
```

## Running Constraint Transfer Experiments (Section 4.2)

### Ant Wall To Point Circle

```bash
# ICRL
python run_me.py cpg -p ICRL-FE2 --group Point-CT-ICRL --cn_path ./icrl/expert_data/ConstraintTransfer/ICRL/Point/files/best_cn_model.pt -cosd 0 1 -casd -1 -tei PointCircle-v0 -eei PointCircleTestBack-v0 -tk 0.01 -t 1.5e6 -plr 1.0

# Binary Classifier
python run_me.py cpg -p ICRL-FE2 --group Point-CT-Glag -cosd 0 1 -casd -1 --load_gail --cn_path ./icrl/expert_data/ConstraintTransfer/GAIL-PPO/Point/files/best_cn_model.pt -tk 0.01 -t 1.5e6 -tei PointCircle-v0 -eei PointCircleTestBack-v0

# GAIL-Constraint
python run_me.py gail -p ICRL-FE2 --group Point-CT-GLC -dosd 0 1 -dasd -1 --freeze_gail_weights --gail_path ./icrl/expert_data/ConstraintTransfer/GAIL/Point/files/gail_discriminator.pt -ep icrl/expert_data/AntWall -er 1 -tk 0.01 -t 1.5e5 -tei PointCircle-v0 -eei PointCircleTestBack-v0
```

### Ant Wall To Ant Broken

```bash
# ICRL
python run_me.py cpg -p ICRL-FE2 --group AntBroken-CT-ICRL --cn_path ./icrl/expert_data/ConstraintTransfer/ICRL/AntBroken/files/best_cn_model.pt -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0 -tk 0.01 --batch_size 128 --reward_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -t 2e6 -plr 1.0

# Binary Classifier
python run_me.py cpg -p ICRL-FE2 --group AntBroken-CT-GLag --load_gail --cn_path ./icrl/expert_data/ConstraintTransfer/GAIL/AntBroken/files/gail_discriminator.pt -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0 -tk 0.01 --batch_size 128 --reward_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -t 3e6 -plr 1.0

# GAIL-Constraint
python run_me.py gail -p ICRL-FE2 --group AntBroken-CT-GLC --freeze_gail_weights --gail_path ./icrl/expert_data/ConstraintTransfer/GAIL/AntBroken/files/gail_discriminator.pt -ep icrl/expert_data/AntWall -er 2 -tei AntWallBroken-v0 -eei AntWallBrokenTest-v0 -tk 0.01 --batch_size 128 --reward_gae_lambda 0.9 --n_epochs 20 --learning_rate 3e-5 --clip_range 0.4 -t 2e6
```


If you find this paper useful, please cite it as:
```

@InProceedings{pmlr-v139-malik21a,
  title = 	 {Inverse Constrained Reinforcement Learning},
  author =       {Shehryar Malik and Usman Anwar and Alireza Aghasi and Ali Ahmed},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {7390--7399},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/malik21a/malik21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/malik21a.html},
}
```
