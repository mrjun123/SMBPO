import torch
import torch.nn as nn
import numpy as np
import time
from mymbrl.mfrl.crossq.base_agent_original import Agent
from mymbrl.mfrl.crossq.utils import soft_update, hard_update
from typing import Tuple

class SAC(Agent):
    def __init__(self,
                 state_dim: int,
                 action_space,
                 args: dict,
                 agent,
                 device='cuda',
                 print_update_every=100) -> None:
        """
        SAC class
        state_dim: number of state features
        action_dim: number of action features
        config_agent: config for agent
        device: device used for training
        """
        action_dim = action_space.shape[0]
        config_agent = args.crossq_config
        self.device = device

        super().__init__(state_dim,
                         action_dim,
                         config_agent,
                         device,
                         print_update_every)
        self.td3 = False
        self.max = -1e8
        self.min = 1e8
        self.target_max = 0
        self.target_min = 0
        self.update_target_num = 0
                         

    def update_critic(self,
                      state_batch: torch.tensor,
                      action_batch: torch.tensor,
                      reward_batch: torch.tensor,
                      next_state_batch: torch.tensor,
                      mask_batch: torch.tensor,
                      time_to_print: bool,
                      full_env
                      ) -> Tuple[torch.tensor, torch.tensor, float, float, float]:
        """
        Updates critic with Soft Bellman equation
        :param state_batch: the state batch extracted from memory
        :param action_batch: the action batch extracted from memory
        :param reward_batch: the reward batch extracted from memory
        :param next_state_batch: the next state batch batch extracted from memory
        :param mask_batch: mask of dones
        :param time_to_print: bool used for metrics
        :return: float loss for Q1, float loss for Q2
        """
        with torch.no_grad():
            next_state_action_batch, next_state_log_pi_batch, mean = self.policy.sample(next_state_batch, False, False)
            if self.td3:
                target_policy_noise = 0.1
                noise_clip = 0.5
                noise = (torch.randn_like(mean) * target_policy_noise).clamp(-noise_clip, noise_clip)
                next_state_action_batch = mean + noise
                next_state_action_batch = next_state_action_batch.clamp(-1, 1)

        if self.config_agent['crossqstyle']:
            
            # (bsz x 2, nstate)
            cat_states = torch.cat([state_batch, next_state_batch], 0)
            # (bsz x 2, nact)
            cat_actions = torch.cat([action_batch, next_state_action_batch], 0)

            # (bsz x 2, 1)
            # if not full_env:
            self.critic.train() # switch to training - to update BN statistics if any
            qfull1, qfull2 = self.critic(cat_states, cat_actions)
            self.critic.eval() # switch to eval
            # Separating Q
            q1, q1next = torch.chunk(qfull1, chunks=2, dim=0)
            q2, q2next = torch.chunk(qfull2, chunks=2, dim=0)
            if self.td3:
                min_qnext = torch.min(q1next, q2next)
            else:
                min_qnext = torch.min(q1next, q2next)

                clamp_q_rate = self.config_agent['clamp_q_rate']
                clamp_q_start = self.config_agent['clamp_q_start']

                # if self.num_updates == clamp_q_start:
                #     self.target_max = self.max
                #     self.target_min = self.min
                min_qnext = min_qnext - self.alpha * next_state_log_pi_batch

                if self.config_agent['clamp_q']:
                    target_in = self.target_max-self.target_min
                    min_qnext = min_qnext.clamp(self.target_min - target_in*clamp_q_rate, self.target_max + target_in*clamp_q_rate)
                    
            next_qvalue = (reward_batch + mask_batch * self.config_agent['gamma'] * min_qnext).detach()
            self.max = max(self.max, float(next_qvalue.max()))
            self.min = min(self.min, float(next_qvalue.min()))
        else:
            with torch.no_grad():
                if self.config_agent['remove_target_network']:
                    q1next_target, q2next_target = self.critic(next_state_batch, next_state_action_batch)
                else:
                    q1next_target, q2next_target = self.critic_target(next_state_batch, next_state_action_batch)
                min_qnext = torch.min(q1next_target, q2next_target) - self.alpha * next_state_log_pi_batch
                next_qvalue = reward_batch + mask_batch * self.config_agent['gamma'] * min_qnext
            self.critic.train() # switch to training - to update BN statistics if any
            q1, q2 = self.critic(state_batch, action_batch)
            self.critic.eval() # switch to eval
        q_loss, q1_loss, q2_loss = self.calculate_q_loss(q1, q2, next_qvalue)
        # Default
        self.critic_optim.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=20, norm_type=2)
        self.critic_optim.step()
        self.critic_optim.zero_grad()

        if time_to_print:
            track_q1 = q1.mean().detach().cpu().item()
            track_q2 = q2.mean().detach().cpu().item()
            track_next_qvalue = next_qvalue.mean().detach().cpu().item()
        else:
            track_q1 = None
            track_q2 = None
            track_next_qvalue = None

        return q1_loss, q2_loss, track_q1, track_q2, track_next_qvalue

    ###################################################################################################################
    def calculate_policy_loss(self, log_pi: torch.tensor, qf1_pi: torch.tensor, qf2_pi: torch.tensor) -> torch.tensor:
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        if self.td3:
            policy_loss = torch.mean(- min_qf_pi)
        else:
            policy_loss = torch.mean((self.alpha * log_pi) - min_qf_pi)
        return policy_loss

    def update_actor(self, state_batch: torch.tensor, full_env) -> Tuple[torch.tensor, torch.tensor]:
        """
        Actor update
        + state_batch: the state batch extracted from memory
        """
        # print('state_batch.shape', state_batch.shape)
        # policy accepts two bools, first if we are in evaluation mode, second if we are in loop
        # if not full_env:
        self.policy.train() # switch to training - to update BN statistics if any
        pi, log_pi, mean = self.policy.sample(state_batch, False, False)
        self.policy.eval()

        if self.td3:
            # target_policy_noise = 0.2
            # noise_clip = 0.5
            # noise = (torch.randn_like(mean) * target_policy_noise).clamp(-noise_clip, noise_clip)
            pi = mean
        # with torch.no_grad():
        qf1_pi, qf2_pi = self.critic(state_batch, pi)

        policy_loss = self.calculate_policy_loss(log_pi, qf1_pi, qf2_pi)

        self.policy_optim.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=20, norm_type=2)
        self.policy_optim.step()

        if self.config_agent['automatic_entropy_tuning']:
            # print('log_pi.shape', log_pi.shape)
            # log_pi = log_pi.reshape(self.config_agent['batch_size'], -1).sum(dim=-1, keepdim=True)

            # 为负值，采样约接近绝对值越小
            multiplier = (log_pi + self.target_entropy).detach()
            alpha_loss = -(self.log_alpha * multiplier).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_(self.log_alpha, max_norm=20, norm_type=2)
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0

        return alpha_loss, policy_loss

    # def train(self):
    def model_update(self):
        if self.update_target_num>0:
            self.target_max = self.max
            self.target_min = self.min
            print('target_max', self.target_max, 'target_min', self.target_min)
        self.update_target_num += 1
        # pass
    def update_parameters(self, data, train_batch_size, i, full_env=False):
        # print('update', i)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = data
        # print(batch_state.shape, batch_action.shape, batch_reward.shape, batch_next_state.shape, batch_done.shape)
        batch_state = torch.tensor(batch_state, device=self.device).float()
        batch_action = torch.tensor(batch_action, device=self.device).float()
        batch_reward = torch.tensor(batch_reward, device=self.device).float().unsqueeze(-1)
        batch_next_state = torch.tensor(batch_next_state, device=self.device).float()
        batch_done = torch.tensor(batch_done, device=self.device).float().unsqueeze(-1)
        # print(batch_state.shape, batch_action.shape, batch_reward.shape, batch_next_state.shape, batch_done.shape)
        self.update(batch_state, batch_action, batch_reward, batch_next_state, batch_done, full_env)

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, full_env) -> dict:
        """
        Agent update step
        """
        self.print_counter += 1 # counter for metrics
        
        self.num_updates += 1
        # print('state_batch.shape', state_batch.shape)

        # Update critic
        q1_loss, q2_loss, track_q1, track_q2, track_next_qvalue = self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, mask_batch, self.print_counter > self.print_update_every, full_env)
        # print('self.num_updates', self.num_updates)
        # Update actor
        if self.num_updates % self.config_agent['update_critic_to_policy_ratio'] == 0:
            alpha_loss, pi_loss = self.update_actor(state_batch, full_env)
        else:
            alpha_loss, pi_loss = None, None
        
        return None
        # Update target and batchnorm
        if self.num_updates % self.config_agent['target_update_interval'] == 0 and not self.config_agent['remove_target_network'] and not self.config_agent['crossqstyle']:
            soft_update(self.critic_target, self.critic, self.config_agent['tau'])
            if len(self.batch_norm_stats) > 0:
                hard_update(self.batch_norm_stats, self.batch_norm_stats_target)
        
        
        if self.print_counter > self.print_update_every and self.num_updates % self.config_agent['update_critic_to_policy_ratio'] == 0:
            if self.config_agent['automatic_entropy_tuning']:
                alpha_log = self.alpha.clone().item()
            else:
                alpha_log = torch.tensor(self.alpha).item()

            self.print_counter = 0

            statistics = {
                "q1_loss": q1_loss.detach().cpu().numpy(),
                "q2_loss": q2_loss.detach().cpu().numpy(),
                "pi_loss": pi_loss.detach().cpu().numpy(),
                "alpha_loss": alpha_loss.detach().cpu().numpy(),
                "alpha_log": alpha_log,
                "track_q1": track_q1,
                "track_q2": track_q2,
                "track_next_qvalue": track_next_qvalue
            }

            return statistics
        else:
            return None

    def episode_reset(self):
        pass
    @torch.no_grad()
    def select_action(self, s: np.ndarray, eval=False, use_checkpoint=False):
        """
        Selects action based on the policy
        """
        evaluation = eval
        state = torch.FloatTensor(s).unsqueeze(0).to(device=self.device)
        policy = self.policy
        if evaluation and use_checkpoint: 
            policy = self.checkpoint_policy
        policy.eval()
        if self.td3:
            action = policy.sample(state, sample_for_loop=True, evaluation=True)
            if not evaluation:
                target_policy_noise = 0.1
                noise_clip = 0.5
                noise = (torch.randn_like(action) * target_policy_noise).clamp(-noise_clip, noise_clip)
                action = action + noise
                action = action.clamp(-1, 1)
        else:
            action = policy.sample(state, evaluation, sample_for_loop=True)

        return action.cpu().numpy()

    

