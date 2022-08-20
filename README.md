# Differentiable Wave Digital Filters

[![Paper](https://zenodo.org/badge/DOI/10.5281/zenodo.6797472.svg)](https://zenodo.org/record/6797472)

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

## Building the Audio Plugin

The audio plugin can be built using the CMake build system.

```bash
# Clone the repository
$ git clone https://github.com/jatinchowdhury18/differentiable-wdfs.git
$ cd differentiable-wdfs

# initialize and set up submodules
$ git submodule update --init --recursive

# build with CMake
$ cmake -Bbuild
$ cmake --build build --parallel 4
```

If you'd like to make an optimized "release" build, it is suggested to use some slightly different build commands:
```bash
$ cmake -Bbuild -DCMAKE_BUILD_TYPE=Release
$ cmake --build build --config Release --parallel 4
```

The resulting builds can be found in the `build/plugin/DifferentiableWDFs_artefacts` directory.

## Citation

If you are using this code as part of an academic work, please cite the repository as follows:
```
@InProceedings{chowdhury:clarke:diffwdfs:2022,
    author = {Jatin Chowdhury and Christopher Johann Clarke},
    title = {Emulating Diode Circuits with Differentiable Wave Digital Filters},
    booktitle = {19th Sound and Music Computing Conference},
    year = {2022},
    pages = {2-9},
    url = {https://zenodo.org/record/6566846},
}
```

# License
The code in this repository is licensed under BSD 3-clause license.
