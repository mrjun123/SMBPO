from torch.utils.tensorboard import SummaryWriter
# from visualdl import LogWriter
import faulthandler
faulthandler.enable()
from dotmap import DotMap
import mymbrl.experiments as experiments

from mymbrl.utils import new_run_directory, merge_dict
import os
import pprint
import numpy as np
import torch
import random
import argparse
import json
import optuna
import setproctitle
# import paddle

def main(config, args):

    run_dir_path = 'runs/mymbrl'
    if config_name != 'custom':
        if config['experiment']['noise'] > 0:
            run_dir_path = 'runs/noise'+str(int(config['experiment']['noise']*100))+'/'+config_name+'/mymbrl'
        else:
            run_dir_path = 'runs/'+config_name+'/mymbrl'
        if config['test']:
            run_dir_path += '-test'
        model_train_freq = config['agent']['model_train_freq']
        if model_train_freq != 1000:
            run_dir_path += f'-{model_train_freq}-'
    run_dir = new_run_directory(run_dir_path)
    setproctitle.setproctitle(str(run_dir))
    writer = SummaryWriter(run_dir)
    # writer = LogWriter(logdir=os.path.join(run_dir, "train"))

    writer.add_text("config", json.dumps(config))
    writer.add_text("commit", args.commit)
    # writer.add_hparams(config, config)
    
    # 写入配置
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(pprint.pformat(config))
    config = DotMap(config)
    config.run_dir = run_dir

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    # paddle.seed(config.random_seed)
    random.seed(config.random_seed)
    print(config)

    experiment_name = config.experiment.name
    experiment = experiments.get_item(experiment_name)
    run_experiment = experiment(config, writer)
    exp_reward = run_experiment.run()
    writer.close()
    return exp_reward

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='This is mymbrl.')
    parser.add_argument('--config', type=str, default='custom')
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--commit', type=str, default='no')

    args = parser.parse_args()
    config_name = args.config
    noise = args.noise
    seed = args.seed
    
    with open("configs/"+config_name+".json", 'r', encoding='UTF-8') as f:
        config = json.loads(f.read())
    with open("default_config.json", 'r', encoding='UTF-8') as f:
        default_config = json.loads(f.read())

    config = merge_dict(default_config, config)

    if noise > 0:
        config['experiment']['noise'] = noise
    if seed != -1:
        config['random_seed'] = seed

    if not len(config['optuna']['suggests']):
        main(config, args)
    else:
        storage_name = "sqlite:///optuna.db"
        study = optuna.create_study(
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=3), direction="maximize",
            study_name="fashion_mnist_torch", storage=storage_name, load_if_exists=True
        )
        study.optimize(main, n_trials=20, timeout=1200)
    