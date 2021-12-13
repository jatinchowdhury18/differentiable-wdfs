# Differentiable Wave Digital Filters

This repository contains an exploration of
implementing differentiable wave digital filters
(WDFs), in an effort to integrate WDFs with
neural networks, and allow for the automatic
optimisation of WDF parameters.

## Circuits to train
- diode clipper with 1N4148 and pot. from ~10k - ~100k (3-4 steps)
- diode clipper with [X DIODE] and pot. from ~10k - ~100k (3-4 steps)
- diode clipper with 2x2 1N4148 and pot. from ~10k - ~100k (3-4 steps)
- tube screamer
  - buy one
  - Keith's data
  - build one?

Wish list:
- diode clipper with germanium diodes

TODO:
- record circuit data
- train full circuit models
- build a plugin for trained models
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
