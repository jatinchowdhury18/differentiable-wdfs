# Differentiable Wave Digital Filters

This repository contains an exploration of
implementing differentiable wave digital filters
(WDFs), in an effort to integrate WDFs with
neural networks, and allow for the automatic
optimisation of WDF parameters.

- 1U-1D diode clipper
  - [x] pretraining
  - [ ] fully trained models
  - [x] real-time implementation
- NU-ND diode clipper
  - [x] pretraining
  - [ ] fully trained models
- 1U-1D tube screamer (w/ same diodes)
  - [ ] pretraining
  - [ ] fully trained models

- Chris:
  - pictures of measurement setup
  - note measurement devices
  - build tube screamer
  - tube screamer data w/ different resistor values
  - work on training for full circuit, 1U-1D (w/ different model sizes)

- Jatin:
  - generate R-type scattering matrix for tube screamer
  - work on training for full circuit. NU-ND (w/ 2x8 models)
  - keep working on plugin
  - work on benchmarks w/ plugin

TODO:
- Tube screamer data with the same 1N4148 diodes
- Plugin model for diode clipper with different models
- set up benckmarks for different model sizes + WDF implementation

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
