# master-project
A fork from Daniel S' master project. Most features stripped away, focusing on
the ability to tinker with the code and trying new things.

# Requirements
Install anaconda, miniconda, or miniforge. Then initialize the environment using

```
conda env create -f conda.yaml
```

This environment contains all dependencies necessary to run the Qube2 gym_brt
environment and the simulated ipm_furuta environment.

If running on Ubuntu 18, it is also necessary to install the
[HIL driver](https://github.com/quanser/hil_sdk_linux_x86_64). _Tip: Unplug the
power from the pendulum, then connect USB, and then reconnect power again._

# Usage

Train and plot performance:

```
python src/deeprl.py -i 500000 -tp
```
