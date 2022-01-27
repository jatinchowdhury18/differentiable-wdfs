# Differentiable Wave Digital Filters

This repository contains an exploration of
implementing differentiable wave digital filters
(WDFs), in an effort to integrate WDFs with
neural networks, and allow for the automatic
optimisation of WDF parameters.

## Organization

The repository is organized as follows:
```
diode_dataset/  # Dataset used for training models of diode circuits
modules/        # Third-party libraries
plugin/         # Audio plugin (JUCE/C++) containing real-time WDF models
wdf_py/         # Differentiable WDF library, and scripts for training WDFs
```

## Settings up the Python environment

The code in this repo has been tested using Python
version 3.9.5. While it is possible to just run
this code on your machine as-is, we reccomend
using `virtualenv` to help manage dependencies
and versions.

```bash
# create virtualenv (only need to do this once)
virtualenv --python=python3.9 env

# enter virtualenv
source env/bin/activate

# install requirements
pip install -r requirements.txt

# do your stuff...

# leave virtualenv
deactivate
```
