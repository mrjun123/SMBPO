import torch
import numpy as np
from mymbrl.utils import has_input_param
from mymbrl.envs.utils import termination_fn

class Controller:
    def __init__(self, agent, is_torch=True, writer=None):
        self.config = agent.config
        self.is_torch = is_torch
        self.agent = agent
        self.prediction = agent.prediction
        self.env = agent.env
        # self.dataloader = agent.dataloader
        self.writer = writer
        self.exp_epoch = -1

        self.mpc_obs = [0] * (self.config.agent.predict_length + 1)
        self.mpc_acs = [0] * self.config.agent.predict_length
        self.mpc_nonterm_masks = [0] * self.config.agent.predict_length

        self.net_weight = torch.ones(self.config.agent.elite_size, device=self.config.device)
        
    def set_epoch(self, exp_epoch):
        self.exp_epoch = exp_epoch
        
    def set_step(self, exp_step):
        self.exp_step = exp_step

    def end_episode(self, step, episode_reward):
        pass
        
    def add_data_step(self, cur_state, action, reward, next_state, done, is_start):
        pass
    def add_two_step_data(self, pre_state, pre_action, state, action, reward, next_state, done, is_start, is_end):
        pass

    def train_epoch(self, epoch_reward):
        pass
        
    def train_step(self):
        pass
    def save(self, num):
        pass
        
    def sample(self, states, epoch=-1, step=-1):
        self.epoch = epoch
        self.step = step

    def weight_mpc_cost_fun(self, ac_seqs, cur_obs, return_torch = True, sample_epoch = -1, solution=False):
        nopt = ac_seqs.shape[1]
        npart = self.config.agent.num_particles

        if not isinstance(cur_obs, torch.Tensor):
            cur_obs = torch.from_numpy(cur_obs).float().to(self.config.device)
        obs_dim = cur_obs.ndim
        if obs_dim == 2 and cur_obs.shape[0] == npart:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt, -1, -1).reshape(nopt * npart, -1)
        elif obs_dim == 1:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt * npart, -1)
        else:
            raise ValueError("step states shape error!")

        costs = torch.zeros(nopt, npart, device=self.config.device)
        for t in range(self.config.agent.predict_length):
            cur_acs = ac_seqs[t]
            next_obs = self.prediction(cur_obs, cur_acs, t, sample_epoch)
            net_obs = self.agent._expand_to_ts_format(next_obs)
            obs_cost = self.env.obs_cost_fn_cost(net_obs)
            obs_cost = obs_cost*self.net_weight.unsqueeze(1).expand(obs_cost.shape[0], obs_cost.shape[1])
            obs_cost = obs_cost.unsqueeze(2)
            obs_cost = self.agent._flatten_to_matrix(obs_cost).squeeze(-1)

            cost = obs_cost + self.env.ac_cost_fn_cost(cur_acs.reshape(-1, cur_acs.shape[-1]))
            cost = cost.view(-1, npart)
            costs += cost
            cur_obs = next_obs

        costs[costs != costs] = 1e6

        if return_torch:
            return costs.mean(dim=1)
        return costs.mean(dim=1).detach().cpu().numpy()

    def mpc_cost_fun(self, ac_seqs, cur_obs, return_torch = True, sample_epoch = -1, solution=False, last_q=False):
        # (400, 25, 20, 4)
        nopt = ac_seqs.shape[1]
        npart = self.config.agent.num_particles

        # if not isinstance(ac_seqs, torch.Tensor):
        #     ac_seqs = torch.from_numpy(ac_seqs).float().to(self.config.device)
        
        if not isinstance(cur_obs, torch.Tensor):
            cur_obs = torch.from_numpy(cur_obs).float().to(self.config.device)
        obs_dim = cur_obs.ndim
        if obs_dim == 2 and cur_obs.shape[0] == npart:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt, -1, -1).reshape(nopt * npart, -1)
        elif obs_dim == 1:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt * npart, -1)
        else:
            raise ValueError("step states shape error!")

        costs = torch.zeros(nopt, npart, device=self.config.device)
        
        if solution:
            self.mpc_obs[0] = cur_obs.detach().cpu().numpy()
        pre_nonterm_mask = (torch.zeros(cur_obs.shape[0], device=self.config.device) == 0)

        for t in range(self.config.agent.predict_length):
            cur_acs = ac_seqs[t]
            next_obs = self.prediction(cur_obs, cur_acs, t, sample_epoch, is_nopt=True)
            cost = self.env.obs_cost_fn_cost(next_obs) + self.env.ac_cost_fn_cost(cur_acs.reshape(-1, cur_acs.shape[-1]))
            cost[pre_nonterm_mask == False] = 0
            cost = cost.view(-1, npart)
            
            terminals = termination_fn(self.config.env, cur_obs.detach().cpu().numpy(), cur_acs.reshape(-1, cur_acs.shape[-1]).detach().cpu().numpy(), next_obs.detach().cpu().numpy())
            nonterm_mask = ~terminals.squeeze(-1)
            pre_nonterm_mask[nonterm_mask == False] = False

            costs += cost

            cur_obs = next_obs
            
            if solution:
                self.mpc_acs[t] = cur_acs.detach().cpu().numpy()
                self.mpc_obs[t+1] = cur_obs.detach().cpu().numpy()
                self.mpc_nonterm_masks[t] = pre_nonterm_mask.detach().cpu().numpy()

            # 记录预测值
            # if solution:
            #     self.mpc_obs[t] = next_obs.detach().clone()

            if solution and t == 0:
                self.pre_pred_obs = next_obs.detach().clone()

        # 长期reward
        if last_q:
            q_cost = self.citic_q(cur_obs)
            q_cost[pre_nonterm_mask == False] = 0
            q_cost = q_cost.view(-1, npart)
            costs += q_cost
        costs[costs != costs] = 1e6

        if return_torch:
            return costs.mean(dim=1)
        return costs.mean(dim=1).detach().cpu().numpy()
    
    def citic_q(self, state_batch):
        state_action, next_state_log_pi, _ = self.policy.sample(state_batch)
        qf1_next_target, qf2_next_target = self.SAC_agent.critic_target(state_batch, state_action)
        q = torch.min(qf1_next_target, qf2_next_target) - self.SAC_agent.alpha * next_state_log_pi
        return q

    def mpc_done_cost_fun(self, ac_seqs, cur_obs, return_torch = True, sample_epoch = -1, solution=False):
        # (400, 25, 20, 4)
        nopt = ac_seqs.shape[1]
        npart = self.config.agent.num_particles

        # if not isinstance(ac_seqs, torch.Tensor):
        #     ac_seqs = torch.from_numpy(ac_seqs).float().to(self.config.device)
        
        if not isinstance(cur_obs, torch.Tensor):
            cur_obs = torch.from_numpy(cur_obs).float().to(self.config.device)
        obs_dim = cur_obs.ndim
        if obs_dim == 2 and cur_obs.shape[0] == npart:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt, -1, -1).reshape(nopt * npart, -1)
        elif obs_dim == 1:
            cur_obs = cur_obs[None]
            cur_obs = cur_obs.expand(nopt * npart, -1)
        else:
            raise ValueError("step states shape error!")

        costs = torch.zeros(nopt, npart, device=self.config.device)
        pre_obs_info = None
        pre_acs_info = None
        for t in range(self.config.agent.predict_length):
            cur_acs = ac_seqs[t]
            next_obs = self.prediction(cur_obs, cur_acs, t, sample_epoch, print_info=solution)
            obs_cost, pre_obs_info = self.env.obs_cost_fn_cost_done(next_obs, t=t, pre_obs_info=pre_obs_info)
            if has_input_param(self.env.ac_cost_fn_cost, 't'):
                acs_cost, pre_acs_info = self.env.ac_cost_fn_cost(cur_acs.reshape(-1, cur_acs.shape[-1]), t=t, pre_acs_info=pre_acs_info)
            else:
                acs_cost = self.env.ac_cost_fn_cost(cur_acs.reshape(-1, cur_acs.shape[-1]))
            cost = obs_cost + acs_cost
            cost = cost.view(-1, npart)
            costs += cost
            cur_obs = next_obs
            # 记录预测值
            if solution and t == 0:
                self.pre_pred_obs = next_obs.detach().clone()
        costs[costs != costs] = 1e6

        if return_torch:
            return costs.mean(dim=1)
        return costs.mean(dim=1).detach().cpu().numpy()
    
    def mpc_cost_fun2(self, ac_seqs, states, return_torch = True, sample_epoch = -1, solution=False):
        # (400, 25, 20, 4)
        nopt = ac_seqs.shape[1]
        npart = self.config.agent.num_particles

        # if not isinstance(ac_seqs, torch.Tensor):
        #     ac_seqs = torch.from_numpy(ac_seqs).float().to(self.config.device)
        # ac_seqs = ac_seqs.transpose(0, 1).contiguous()

        if not isinstance(states, torch.Tensor):
            cur_obs = torch.from_numpy(states).float().to(self.config.device)
        cur_obs = cur_obs[None]
        cur_obs = cur_obs.expand(nopt * npart, -1)

        costs = torch.zeros(nopt, npart, device=self.config.device)
        for t in range(self.config.agent.predict_length):
            cur_acs = ac_seqs[t]
            if t == 0:
                next_obs, next_obs_reward = self.prediction(cur_obs, cur_acs, t, sample_epoch, print_info=solution,return_reward_states=True)
            else:
                next_obs = self.prediction(cur_obs, cur_acs, t, sample_epoch, print_info=solution)
                next_obs_reward = next_obs
            cost = self.env.obs_cost_fn_cost(next_obs_reward) + self.env.ac_cost_fn_cost(cur_acs.reshape(-1, cur_acs.shape[-1]))
            cost = cost.view(-1, npart)
            costs += cost
            cur_obs = next_obs
            # 记录预测值
            if solution and t == 0:
                self.pre_pred_obs = next_obs_reward.detach().clone()

        costs[costs != costs] = 1e6

        if return_torch:
            return costs.mean(dim=1)
        return costs.mean(dim=1).detach().cpu().numpy()
    
    def policy_cost_fun(self, policy, cur_obs, return_torch = True, sample_epoch = -1):
        # (400, 25, 20, 4)
        
        nopt = cur_obs.shape[0]
        npart = self.config.agent.num_particles

        if not isinstance(cur_obs, torch.Tensor):
            cur_obs = torch.from_numpy(cur_obs).to(self.config.device)

        cur_obs = cur_obs.float()
        costs = torch.zeros(nopt//npart, npart, device=self.config.device)
        # print('costs.shape', costs.shape)
        for t in range(self.config.agent.predict_length):
            cur_acs = policy(cur_obs)
            cur_acs = cur_acs.float()
            next_obs = self.prediction(cur_obs, cur_acs, t, sample_epoch)
            cost = self.env.obs_cost_fn_cost(next_obs) + self.env.ac_cost_fn_cost(cur_acs.reshape(-1, cur_acs.shape[-1]))
            cost = cost.view(-1, npart)
            # print('cost.shape', cost.shape)
            costs += cost
            cur_obs = next_obs

        costs[costs != costs] = 1e6

        if return_torch:
            return costs.mean(dim=1)
        return costs.mean(dim=1).detach().cpu().numpy()

    
    