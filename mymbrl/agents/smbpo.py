from .agent import Agent
import mymbrl.controllers as controllers
import mymbrl.models as models
import mymbrl.envs as envs
import mymbrl.dataloaders as dataloaders
import torch, numpy as np
import torch.nn as nn
from scipy.stats import norm
import time
from joblib import Parallel, delayed
import dill
import os
from mymbrl.utils import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import parallel_backend
import random
import datetime
import mymbrl.optimizers as optimizers
from mymbrl.utils import shuffle_rows
import itertools
import inspect

class AgentItem(Agent):

    def __init__(self, config, env, writer):
        """
        Controller: MPC
        """
        config.run_env = env
        self.config = config
        self.env = env
        self.writer = writer
        self.exp_epoch = 0
        self._max_epochs_since_update = config.agent.max_epochs_since_update
        
        Model = models.get_item(config.agent.model)
        in_a_features = env.action_space.shape[0]
        in_s_features = env.MODEL_IN - in_a_features
        
        self.model = Model(
            ensemble_size=config.agent.ensemble_size,
            in_a_features=in_a_features,
            in_s_features=in_s_features,
            out_features=env.MODEL_OUT*2,
            hidden_size=config.agent.dynamics_hidden_size, 
            device=config.device
        )
        
        self.model = self.model.to(config.device)

        self.dynamics_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.agent.dynamics_lr, 
            weight_decay=self.config.agent.dynamics_weight_decay
        )

        Dataloader = dataloaders.get_item('smbpo')
        self.dataloader = Dataloader(device=self.config.device)

        # HoldDataloader = dataloaders.get_item('free_capacity')
        # self.hold_dataloader = HoldDataloader()

        self._epochs_since_update = 0
        self.improvement_rate = config.agent.improvement_rate
        
        Controller = controllers.get_item(config.agent.controller)
        self.controller = Controller(
            self,
            writer=writer
        )

        self.elite_model_idxes = np.arange(self.config.agent.elite_size).tolist()
        self.model.set_elite_index(self.elite_model_idxes)
        
    def save(self, num):
        super(AgentItem, self).save(num)
        path = os.path.join(self.config.run_dir, f'epoch({num})_dynamic_model_weights.pth')
        torch.save(self.model.state_dict(), path)

    def train(self, return_log=False, epoch_reward=0):
        """
        训练一个agent
        """

        start_train_mini_epoch = self.config.agent.start_train_mini_epoch

        holdout_ratio = self.config.agent.holdout_ratio
        dynamics_model = self.model
        num_nets = dynamics_model.num_nets
        num_particles = self.config.agent.num_particles
        net_particles = num_particles // num_nets

        self._snapshots = {i: (None, 1e10) for i in range(num_nets)}
        
        for param in dynamics_model.parameters():
            param.requires_grad = True
        
        if self.config.agent.reset_model:
            raise ValueError("reset_model is fail in mbpo")
        dynamics_model.train()

        i = 0

        all_data_len = self.dataloader.len()
        num_holdout = int(all_data_len * holdout_ratio)
        data_len = all_data_len - num_holdout

        indices = np.random.permutation(all_data_len)
        wo_hold_indices = indices[:data_len]
        hold_idxs = indices[data_len:]
        self.dataloader.set_hold_idxs(hold_idxs)
        
        if self.config.agent.dynamics_type == "bootstrap":
            all_idxs = np.random.randint(data_len, size=[num_nets, data_len])
        elif self.config.agent.dynamics_type == "naive":
            all_idxs = np.arange(data_len)
            all_idxs = all_idxs.reshape(1,-1)
            all_idxs = np.tile(all_idxs, [num_nets, 1])
        
        idxs = wo_hold_indices[all_idxs]

        batch_size = self.config.agent.train_batch_size
        num_batch = idxs.shape[-1] // batch_size

        obs_all, acs_all, next_obs_all = self.dataloader.get_x_y_all()
        obs_all, acs_all, next_obs_all = obs_all.to(self.config.device), acs_all.to(self.config.device), next_obs_all.to(self.config.device)

        if hasattr(self.env, 'obs_model_preproc'):
            obs_all = self.env.obs_model_preproc(obs_all)
            next_obs_all = self.env.obs_model_preproc(next_obs_all)

        if self.config.agent.fit_input:
            if not self.config.agent.only_fit_first or (self.config.agent.only_fit_first and not dynamics_model.fit_input.item() > 0.5):
                fit_inputs = torch.cat([self.env.obs_preproc(obs_all), acs_all], dim=-1)
                dynamics_model.fit_input_stats(fit_inputs)
        
        holdout_s, holdout_a = obs_all[hold_idxs, :], acs_all[hold_idxs, :]
        holdout_labels = next_obs_all[hold_idxs, :]
        
        holdout_s = holdout_s.unsqueeze(0).expand(num_nets, -1, -1)
        holdout_a = holdout_a.unsqueeze(0).expand(num_nets, -1, -1)
        holdout_labels = holdout_labels.unsqueeze(0).expand(num_nets, -1, -1)
        hold_num_batch = hold_idxs.shape[-1] // batch_size
        
        mes_loss_log = []
        train_loss_log = []
        hold_mes_loss_log = []
        for epoch in itertools.count():
            idxs = shuffle_rows(idxs)
            for batch_num in range(num_batch):
                
                batch_idxs = idxs[:, batch_num * batch_size : (batch_num + 1) * batch_size]
                obs, acs, next_obs = obs_all[batch_idxs, :], acs_all[batch_idxs, :], next_obs_all[batch_idxs, :]

                loss = self.config.agent.dynamics_weight_decay_rate * dynamics_model.compute_decays()
                # loss += 0.01 * (dynamics_model.max_logvar.sum() - dynamics_model.min_logvar.sum())
                loss += 0.01 * torch.abs(dynamics_model.max_logvar - dynamics_model.min_logvar).sum()/num_nets
                # loss += 0.01 * (dynamics_model.max_logvar.sum() - dynamics_model.min_logvar.sum())/num_nets
                
                s, a = obs, acs

                dynamics_model.select_mask(batch_size)
                
                es = self.env.obs_preproc(s)
                es = dynamics_model.encoder(es)
                ea = dynamics_model.encoder_a(a)
                emean, logvar = dynamics_model.dynamic(es, ea, ret_logvar=True)
                emean = self.env.obs_postproc(es.detach(), emean)
                # emean = self.env.obs_postproc(es, emean)
                # mean = dynamics_model.dedecoder(emean)

                with torch.no_grad():
                    y1e = dynamics_model.encoder(next_obs)
                # 新增
                # logvar = dynamics_model.encoder_var(logvar, is_log=True)

                inv_var = torch.exp(-logvar)
                mes_loss = ((emean - y1e) ** 2)
                
                mes_loss_sum = mes_loss.mean(-1).mean(-1).sum()

                train_losses = mes_loss * inv_var + logvar

                mes_loss_print = [mes_loss_sum.item()]
                mes_loss_print.append(dynamics_model.en_std)
                # mes_loss_print.append(dynamics_model.en_b)

                train_losses = train_losses.mean(-1).mean(-1).sum()
                loss += train_losses
                loss.backward()

                if return_log:
                    mes_loss_log.append(mes_loss_sum.item())
                    train_loss_log.append(train_losses.item())

                nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=20, norm_type=2)
                self.dynamics_optimizer.step()
                self.dynamics_optimizer.zero_grad()
                i += 1
                if i % 100 == 0:
                    print('epoch', self.exp_epoch, 'step', i, 'train_loss', loss.item(), 'MESLoss', mes_loss_print)

            with torch.no_grad():

                holdout_mse_losses = np.zeros((num_nets))
                holdout_mse_losses2 = np.zeros((holdout_labels.shape[-1]))
                for batch_num in range(hold_num_batch):
                    # num_nets, -1, -1
                    holdout_s_batch = holdout_s[:, batch_num * batch_size : (batch_num + 1) * batch_size, :]
                    holdout_a_batch = holdout_a[:, batch_num * batch_size : (batch_num + 1) * batch_size, :]

                    # pre_holdout_s_batch = holdout_s_batch
                    holdout_labels_batch = holdout_labels[:, batch_num * batch_size : (batch_num + 1) * batch_size, :]


                    pre_holdout_s_batch = self.env.obs_preproc(holdout_s_batch)
                    pre_holdout_es_batch = dynamics_model.encoder(pre_holdout_s_batch)
                    
                    holdout_ea_batch = dynamics_model.encoder_a(holdout_a_batch)
                    
                    emean, elogvar = dynamics_model.dynamic(pre_holdout_es_batch, holdout_ea_batch, ret_logvar=True)

                    emean = self.env.obs_postproc(pre_holdout_es_batch, emean)

                    mean = dynamics_model.dedecoder(emean)

                    mes_loss = ((mean - holdout_labels_batch) ** 2)

                    holdout_mse_losses_batch = mes_loss.mean(-1).mean(-1)
                    holdout_mse_losses_batch = holdout_mse_losses_batch.detach().cpu().numpy()
                    holdout_mse_losses += holdout_mse_losses_batch

                    holdout_mse_losses2 += mes_loss.mean(0).mean(0).detach().cpu().numpy()

                holdout_mse_losses = holdout_mse_losses / hold_num_batch
                if return_log:
                    hold_mes_loss_log.append(holdout_mse_losses.sum().item())
                # if i % 100 == 0:
                print('epoch', self.exp_epoch, 'step', i, 'test_loss', holdout_mse_losses.sum().item())
                sorted_loss_idx = np.argsort(holdout_mse_losses)
                if self.config.agent.elite_size < self.config.agent.ensemble_size:
                    self.elite_model_idxes = sorted_loss_idx[:self.config.agent.elite_size].tolist()
                break_train = self._save_best(epoch, holdout_mse_losses)
                if break_train:
                    break

        num_nets = self.config.agent.elite_size
        net_particles = num_particles // num_nets
        self.model.set_elite_index(self.elite_model_idxes)
        
        self.controller.train_epoch(epoch_reward=epoch_reward)
        # for param in dynamics_model.parameters():
        #     param.requires_grad = False
        print('max_var', dynamics_model.max_logvar.exp().mean())
        print('min_var', dynamics_model.min_logvar.exp().mean())

        # 重制epoch lr
        if self.exp_epoch in self.config.agent.lr_scheduler:
            self.dynamics_scheduler.step()
            self.improvement_rate *= self.config.agent.dynamics_lr_gamma
        if return_log:
            return train_loss_log, mes_loss_log, hold_mes_loss_log

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > self.improvement_rate:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                # improvement = (best - current) / best

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1
        if self._epochs_since_update > self._max_epochs_since_update:
            return True
        else:
            return False
        
    def sample(self, states, evaluation=False, train_step=True):
        """
        根据当前环境获取一个动作
        """
        # self.model.eval()
        action = self.controller.sample(states, self.exp_epoch, self.exp_step, evaluation,train_step)
        # print('action', action.max(), action.min())
        return action


    def add_data(self, states, actions, indexs=[], path_done=True):

        assert states.shape[0] == actions.shape[0] + 1

        obs = states[:-1]
        next_obs = states[1:]
        acs = actions

        self.dataloader.push(obs, acs, next_obs, path_done=path_done)
    

    def prediction(self, states, action, t=0, sample_epoch=0, add_var=False, is_nopt=False, use_model=None,aleatoric='config', need_input_preproc=True):
        
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.config.device).float()
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, device=self.config.device).float()
        if(states.dim() == 1):
            states = states.unsqueeze(0).tile(self.config.agent.num_particles, 1).float()
        if(action.dim() == 1):
            action = action.unsqueeze(0).tile(states.shape[0], 1).float()
        
        if is_nopt:
            expand_to_ts_format = self._expand_to_ts_format
            flatten_to_matrix = self._flatten_to_matrix
        else:
            expand_to_ts_format = self._expand_to_ts_format_dmbpo
            flatten_to_matrix = self._flatten_to_matrix_dmbpo

        states = expand_to_ts_format(states)
        action = expand_to_ts_format(action)

        s, a = states, action
        if hasattr(self.env, 'obs_model_preproc'):
            s = self.env.obs_model_preproc(s)

        # model = self.model
        if use_model is not None:
            model = use_model
        else:
            model = self.model
        

        if aleatoric == 'config':
            aleatoric = self.config.agent.aleatoric
        
        if aleatoric == 'pets1':
            np.random.shuffle(self.elite_model_idxes)
            model.set_elite_index(self.elite_model_idxes)

        if model.lin0_w_e is None:
            model.set_elite_index(self.elite_model_idxes)
        pre_s = self.env.obs_preproc(s)
        es = model.encoder(pre_s, elite=True)
        # pre_es = self.env.obs_preproc(es)
        ea = model.encoder_a(a, elite=True)
        if self.config.agent.elite_size == self.config.agent.ensemble_size:
            emean, var = model.dynamic(es, ea)
        else:
            emean, var = model.elite_dynamic(es, ea)
        
        emean = self.env.obs_postproc(es, emean)
        emean = model.dedecoder(emean, elite=True)

        # new delete
        var = model.dedecoder_var(var, is_log=False, elite=True)

        if 'pets' in aleatoric or add_var:
            predictions = emean + torch.randn_like(emean, device=self.config.device) * var.sqrt()
        else:
            predictions = emean

        predictions = flatten_to_matrix(predictions)
        states = flatten_to_matrix(states)
        if hasattr(self.env, 'obs_model_postproc'):
            predictions = self.env.obs_model_postproc(states, predictions)

        return predictions
    
    def _expand_to_ts_format_dmbpo(self, mat):
        dim = mat.shape[-1]
        num_nets = self.config.agent.elite_size
        reshaped = mat.view(num_nets, -1, dim)
        return reshaped
    
    def _flatten_to_matrix_dmbpo(self, ts_fmt_arr):

        dim = ts_fmt_arr.shape[-1]
        num_nets = self.config.agent.elite_size
        reshaped = ts_fmt_arr.view(-1, dim)
        return reshaped

    def _expand_to_ts_format(self, mat):
        dim = mat.shape[-1]
        num_nets = self.config.agent.elite_size
        reshaped = mat.view(-1, num_nets, self.config.agent.num_particles // num_nets, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(num_nets, -1, dim)
        return reshaped
    
    def _flatten_to_matrix(self, ts_fmt_arr):

        dim = ts_fmt_arr.shape[-1]
        num_nets = self.config.agent.elite_size
        reshaped = ts_fmt_arr.view(num_nets, -1, self.config.agent.num_particles // num_nets, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(-1, dim)
        return reshaped
