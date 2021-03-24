# MAFTRL
Code for [*Multi-agent Fault-tolerant Reinforcement Learning with Noisy Environments*](https://ieeexplore.ieee.org/document/9359223) (ICPADS 2020)

## Requirements
* Python 3.6.10
* [OpenAI baselines](https://github.com/openai/baselines), version: 0.1.6
* [Multi-agent Particle Environments](https://github.com/shariqiqbal2810/multiagent-particle-envs)
* [PyTorch](http://pytorch.org/), version: 1.5.1
* [OpenAI Gym](https://github.com/openai/gym), version: 0.10.5
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 1.14.0 and [TensorboardX](https://github.com/lanpa/tensorboardX), version: 2.0

The versions are just what I used and not necessarily strict requirements.

## How to Run

All training code is contained within `main.py / main-oc.py`. To view options simply run:

```
python main.py --help
or
python main-oc.py --help
```

The "Unreliable Environment" from our paper is referred to as `unreliable_spread.py` in this repo. You can get the result by run:

```
python main.py unreliable_spread maftrl
or
python main-oc.py unreliable_spread maftrl-oc
```

## Citing our work

If you use this repo in your work, please consider citing the corresponding paper:

```
@INPROCEEDINGS{9359223,
  author={C. {Luo} and X. {Liu} and X. {Chen} and J. {Luo}},
  booktitle={2020 IEEE 26th International Conference on Parallel and Distributed Systems (ICPADS)}, 
  title={Multi-agent Fault-tolerant Reinforcement Learning with Noisy Environments}, 
  year={2020},
  volume={},
  number={},
  pages={164-171},
  doi={10.1109/ICPADS51040.2020.00031}}
```
