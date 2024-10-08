



## Clone and compile

### Clone the repo.
We need `--recursive` to get the correct submodule
```shell
git clone --recursive 
```

### Install dependencies
First Install MuJoCo

Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz)

Extract the downloaded mujoco210 directory into `~/.mujoco/mujoco210`.

### Create conda env

First create a conda env with name `sts`.
```shell
conda create --name sts python=3.9
```

Then, source `set_env.sh` to activate `sts` conda env. It also setup several important paths such as `MUJOCO_PY_MUJOCO_PATH` and add current project folder to `PYTHONPATH`.
Note that if the conda env has a different name, you will need to manually modify the `set_env.sh`.
You also need to modify the `set_env.sh` if the mujoco is not installed at the default location.

```shell
# NOTE: run this once per shell before running any script from this repo
source set_env.sh
```

Then install python dependencies
```shell
# first install pytorch with correct cuda version, in our case we use torch 2.1 with cu121
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# then install extra dependencies from requirement.txt
pip install -r requirements.txt
```
If the command above does not work for your versions.
Please check out `tools/core_packages.txt` for a list of commands to manually install relavent packages.


### Compile C++ code
We have a C++ module in the common utils that requires compilation
```shell
cd common_utils
make
```

### Trouble Shooting
Later when running the training commands, if we encounter the following error
```shell
ImportError: .../libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```
Then we can force the conda to use the system c++ lib.
Use these command to symlink the system c++ lib into conda env. To find `PATH_TO_CONDA_ENV`, run `echo ${CONDA_PREFIX:-"$(dirname $(which conda))/../"}`.

```shell
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/bin/../lib/libstdc++.so
ln -sf /lib/x86_64-linux-gnu/libstdc++.so.6 PATH_TO_CONDA_ENV/bin/../lib/libstdc++.so.6
```

## Reproduce Results

Remember to run `source set_env.sh`  once per shell before running any script from this repo.







### Metaworld

#### sketch-to-skill

Train RL policy using the BC policy provided in `release` folder
```shell
# assembly
python mw_main/train_rl_mw_pofd.py --config_path release/cfgs/metaworld/ibrl_basic.yaml --bc_policy ButtonPress



If you want to train BC policy from scratch
```shell
python mw_main/train_bc_mw.py --dataset.path Assembly --save_dir SAVE_DIR
```

#### IBRL

Train RL policy using the BC policy provided in `release` folder
```shell
# assembly
python mw_main/train_rl_mw.py --config_path release/cfgs/metaworld/ibrl_basic.yaml --bc_policy ButtonPress



If you want to train BC policy from scratch
```shell
python mw_main/train_bc_mw.py --dataset.path Assembly --save_dir SAVE_DIR
```

