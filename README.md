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

Test that the physical pendulum works by running

```
python submodules/quanser-openai-driver/tests/test.py --env QubeSwingupEnv --controller flip
```

# Usage

Train and plot performance:

```
python src/deeprl.py -i 500000 -tp
```


# Examples
Some parameter settings that produce decent results for certain environments.
Note the training is often highly seed sensitive, so the right seed for the
randomness is crucial.

## FurutaODE (IPM)
Default MLP (256_128_relu) and seed 1000.

```
python src/deeprl.py -E ipm -M mlp -s 1000 -i 500000 -tp
```

## Qube2
To be verified...

```
python src/deeprl.py -E qube2.sim -M mlp_64_64 -i 1000000 -s 1000 --learning-rate 4e-4 -tp
```

