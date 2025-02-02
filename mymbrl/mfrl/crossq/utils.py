import torch
import numpy as np
import random
from mymbrl.mfrl.crossq.models import QNetwork, GaussianPolicy, QNetworkEncoder, GaussianPolicyEncoder
from torch.optim.adam import Adam
from collections import deque
import matplotlib.pyplot as plt
import pickle
from typing import Tuple

class ReplayMemory:
    """
    Replay buffer
    """
    def __init__(self, capacity: int) -> None:
        """
        :param capacity of the replay buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.precision = torch.float32

    def push(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray, done: bool) -> None:
        """
        Push data into replay buffer
        :param state: state array
        :param action: action array
        :param reward: reward array
        :param next_state: next state array
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (torch.tensor(state,  dtype=self.precision),
                                      torch.tensor(action,  dtype=self.precision),
                                      torch.tensor(reward,  dtype=self.precision),
                                      torch.tensor(next_state, dtype=self.precision),
                                      torch.tensor(float(not done), dtype=self.precision))
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size:int) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Sample a batch of state, action, reward and next state
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask = map(torch.stack, zip(*batch))
        return state, action, reward, next_state, mask

    def __len__(self) -> int:
        return len(self.buffer)

    def reset(self) -> None:
        self.buffer = []
        self.position = 0

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    if type(target) == list:
        for target_param, param in zip(target, source):
            target_param.data.copy_(param.data)
    else:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    for target_buffer, buffer in zip(target.buffers(), source.buffers()):
        target_buffer.data.copy_(buffer.data)

def get_parameters_by_name(model: torch.nn.Module, included_names):
    """
    Extract parameters from the state dict of ``model``
    if the name contains one of the strings in ``included_names``.

    :param model: the model where the parameters come from.
    :param included_names: substrings of names to include.
    :return: List of parameters values (Pytorch tensors)
        that matches the queried names.
    """
    return [param for name, param in model.state_dict().items() if any([key in name for key in included_names])]




def initialise(state_dim: int,
               action_dim: int,
               device: int,
               config_agent: dict
               ):
    """
    Initialises models
    Usually we have an actor which predicts action that maximises return
    And a critic which predicts return given state and action and current policy
    :param state_dim: dimension for state
    :param action_dim: dimension for action
    :param device: device used for the models
    :param agent_type: "SAC" or "TD3"
    :param crossQstyle: if we use the upgraded version of SAC -> crossQ
    :param num_layers_actor: number of layers for actor
    :param num_layers_critic: number of layers for critic
    :param hidden_dim_critic: number of layers for actor
    :param hidden_dim_actor: number of layers for critic
    :param lr: learning rate for critic and actor
    :param use_batch_norm_critic: if we use batch norm for critic
    :param use_batch_norm_policy: if we use batch norm for policy
    :param beta1: beta1 for ADAM (actor+critic)
    :param beta2: beta2 for ADAM (actor+critic)
    :param beta1_alpha: beta1 for ADAM (alpha)
    :param beta2_alpha: beta2 for ADAM (alpha)
    :param lr_alpha: learning rate alpha
    :param automatic_entropy_tuning: if True alpha is adjusted based on target entropy
    :param alpha: starting value for alpha
    :param entropy_factor: if we multiply target entropy by a factor
    :param activation_policy: activation for actor hidden layers.
    :param activation_critic: activation for critic hidden layers.
    :param bn_momentum: momentum for batch normalization. The value is 1-value of the paper due to pytorch implmentation.
    :param bn_mode: "bn" (batch normalization) or "brn" (batch renormalization) 
    :param if no target networks (without crossQ)
    """

    agent_type=config_agent['agent_type']
    crossQstyle=config_agent['crossqstyle']
    num_layers_actor=config_agent['num_layers_actor']
    num_layers_critic=config_agent['num_layers_critic']
    hidden_dim_critic=config_agent['hidden_dim_critic']
    hidden_dim_actor=config_agent['hidden_dim_actor']
    lr=config_agent['lr']
    lr_actor=config_agent['lr_actor']
    use_batch_norm_critic=config_agent['use_batch_norm_critic']
    use_batch_norm_policy=config_agent['use_batch_norm_policy']
    beta1=config_agent['beta1']
    beta2=config_agent['beta2']
    beta1_actor=config_agent['beta1_actor']
    beta2_actor=config_agent['beta2_actor']
    beta1_alpha=config_agent['beta1_alpha']
    beta2_alpha=config_agent['beta2_alpha']
    lr_alpha=config_agent['lr_alpha']
    automatic_entropy_tuning=config_agent['automatic_entropy_tuning']
    alpha=config_agent['alpha']
    entropy_factor=config_agent['entropy_factor']
    activation_policy=config_agent['activation_policy']
    activation_critic=config_agent['activation_critic']
    bn_momentum=config_agent['bn_momentum']
    bn_mode=config_agent['bn_mode']
    remove_target_network=config_agent['remove_target_network']
    target_entropy, log_alpha, alpha_optim = None, None, None
    if agent_type == "sac":
        policy = GaussianPolicyEncoder(num_inputs=state_dim,
                                action_dim=action_dim,
                                num_layers=num_layers_actor,
                                activation=activation_policy,
                                hidden_dim=hidden_dim_actor,
                                use_batch_norm=use_batch_norm_policy,
                                bn_momentum=bn_momentum,
                                bn_mode=bn_mode).to(device)
        
        fixed_policy = GaussianPolicyEncoder(num_inputs=state_dim,
                                action_dim=action_dim,
                                num_layers=num_layers_actor,
                                activation=activation_policy,
                                hidden_dim=hidden_dim_actor,
                                use_batch_norm=use_batch_norm_policy,
                                bn_momentum=bn_momentum,
                                bn_mode=bn_mode).to(device)
        hard_update(fixed_policy, policy)
        critic = QNetworkEncoder(state_dim=state_dim,
                          action_dim=action_dim,
                          num_layers=num_layers_critic,
                          activation=activation_critic,
                          hidden_dim=hidden_dim_critic,
                          use_batch_norm=use_batch_norm_critic,
                          bn_momentum=bn_momentum,
                          bn_mode=bn_mode).to(device)
        critic_target = QNetworkEncoder(state_dim=state_dim,
                          action_dim=action_dim,
                          num_layers=num_layers_critic,
                          activation=activation_critic,
                          hidden_dim=hidden_dim_critic,
                          use_batch_norm=use_batch_norm_critic,
                          bn_momentum=bn_momentum,
                          bn_mode=bn_mode).to(device)
        hard_update(critic, critic_target)

        # actor optim
        policy_optim = Adam(policy.parameters(),
                            lr=lr_actor,
                            betas=(beta1_actor, beta2_actor))

        critic_optim = Adam(critic.parameters(),
                            lr=lr,
                            betas=(beta1, beta2))
        # Alpha
        if automatic_entropy_tuning:
            target_entropy = -torch.prod(torch.FloatTensor(action_dim).to(device)).item() * entropy_factor
            log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device, dtype=torch.float32)

            alpha_optim = Adam([log_alpha], lr=lr_alpha,
                                 betas=(beta1_alpha, beta2_alpha))
        else:
            target_entropy, log_alpha, alpha_optim = None, None, None
    else:
        raise NotImplementedError

    return target_entropy, log_alpha, alpha_optim,\
           policy, fixed_policy, critic, critic_target,\
           policy_optim, critic_optim
