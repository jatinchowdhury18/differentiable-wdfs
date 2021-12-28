# Differentiable Wave Digital Filters

This repository contains an exploration of
implementing differentiable wave digital filters
(WDFs), in an effort to integrate WDFs with
neural networks, and allow for the automatic
optimisation of WDF parameters.

TODO:
- Diode clipper 1N4148 data with potentiometer, ~10k - ~100k (3-4 steps)
- Diode clipper data with pot and multiple diodes in each direction
- Try training diode models of different sizes (2x8, 2x16, 4x8, maybe 2x4?)
- Tube screamer data with the same 1N4148 diodes
- Plugin model for diode clipper with different models
- find Kurt's tube screamer paper
- look at Stefano D'Angelo's op-amp paper
- try to reduce model size (grid-search, NEAT)

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
