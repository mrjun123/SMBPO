# Enhancing Stability of Probabilistic Model-based Reinforcement Learning by Adaptive Noise Filtering

Code to reproduce the experiments in Enhancing Stability of Probabilistic Model-based Reinforcement Learning by Adaptive Noise Filtering. This paper is currently submitted to IEEE Transactions on Neural Networks and Learning Systems (TNNLS) for peer review.

![method.png](https://raw.githubusercontent.com/mrjun123/SMBPO/main/images/method.png)

Please feel free to contact us regarding to the details of implementing SMBPO. (Wenjun Huang: wj.huang1@siat.ac.cn Yunduan Cui: cuiyunduan@gmail.com)
## Running Experiments

Experiment for a specific configuration can be run using:

```python
python main.py --config hand_reach
```

The specific configuration file is located in the `configs` directory and the default configuration file can be located in the root directory `default_config.json` was found, which allows you to modify the experimental parameters.

## Logging

We use Tensorboard to record experimental data, you can view runs with:

```python
tensorboard --logdir ./runs/ --port=6006 --host=0.0.0.0
```

