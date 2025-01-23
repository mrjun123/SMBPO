import numpy as np
import gym
import torch
from torch import nn as nn
from torch.nn import functional as F
from mymbrl.utils import swish, get_affine_params, get_affine_params_uniform
import random
import math

def AvgL1Norm(x, eps=1e-8):
	return x/x.abs().mean(-1,keepdim=True).clamp(min=eps)
class DynamicModel(nn.Module):

    def __init__(self, ensemble_size, in_s_features, in_a_features, out_features, hidden_size=200, device="cpu"):
        super().__init__()

        in_features = in_s_features + in_a_features

        self.batch_size = 30

        self.num_nets = ensemble_size

        self.hidden_size = hidden_size
        self.elite_index = None

        self.hidden1_mask = None
        self.hidden2_mask = None
        self.hidden3_mask = None
        self.hidden4_mask = None

        self.hidden_mask_indexs = None
        self.hidden1_mask_select = None
        self.hidden2_mask_select = None
        self.hidden3_mask_select = None
        self.hidden4_mask_select = None

        self.in_features = in_features
        self.out_features = out_features
        self.in_s_features = in_s_features

        self.lin0_w_e = None

        self.lin0_w, self.lin0_b = get_affine_params(ensemble_size, in_features, hidden_size)
        self.lin1_w, self.lin1_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin2_w, self.lin2_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin3_w, self.lin3_b = get_affine_params(ensemble_size, hidden_size, hidden_size)
        self.lin4_w, self.lin4_b = get_affine_params(ensemble_size, hidden_size, out_features)

        self.fit_input = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.inputs_mu = nn.Parameter(torch.zeros(in_features).to(device), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(in_features).to(device), requires_grad=False)

        self.max_logvar = nn.Parameter(torch.ones(1, out_features // 2, dtype=torch.float32).to(device) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, out_features // 2, dtype=torch.float32).to(device) * 10.0)

        self.en_std_o = nn.Parameter(torch.ones(ensemble_size, 1, in_s_features + in_a_features , dtype=torch.float32).to(device))
        self.en_b = nn.Parameter(torch.zeros(ensemble_size, 1, in_s_features + in_a_features, dtype=torch.float32).to(device))

    @property
    def en_std_e(self):
        en_std = F.softplus(self.en_std_o_e)
        return en_std
    
    @property
    def en_std(self):
        en_std = F.softplus(self.en_std_o)
        return en_std

    def fit_input_stats(self, data):
        
        data = data.reshape(-1, data.shape[-1])
        mu = torch.mean(data, dim=0, keepdim=False)
        sigma = torch.std(data, dim=0, keepdim=False)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu = nn.Parameter(mu, requires_grad=False)
        self.inputs_sigma = nn.Parameter(sigma, requires_grad=False)
        self.fit_input = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        print('self.inputs_mu', self.inputs_mu)
        print('self.inputs_sigma', self.inputs_sigma)

    def set_elite_index(self, elite_index):
        self.elite_index = elite_index
        self.lin0_w_e, self.lin0_b_e = self.lin0_w[elite_index,:,:], self.lin0_b[elite_index,:,:]
        self.lin1_w_e, self.lin1_b_e = self.lin1_w[elite_index,:,:], self.lin1_b[elite_index,:,:]
        self.lin2_w_e, self.lin2_b_e = self.lin2_w[elite_index,:,:], self.lin2_b[elite_index,:,:]
        self.lin3_w_e, self.lin3_b_e = self.lin3_w[elite_index,:,:], self.lin3_b[elite_index,:,:]
        self.lin4_w_e, self.lin4_b_e = self.lin4_w[elite_index,:,:], self.lin4_b[elite_index,:,:]

        self.en_std_o_e, self.en_b_e = self.en_std_o[elite_index,:,:], self.en_b[elite_index,:,:]

        # self.min_logvar_e, self.max_logvar_e = self.min_logvar[elite_index,:,:], self.max_logvar[elite_index,:,:]

    def encoder(self, s, elite=False):
        en_std = self.en_std
        en_b = self.en_b
        if elite:
            en_std = self.en_std_e
            en_b = self.en_b_e

        en_std = en_std[..., :self.in_s_features]
        en_b = en_b[..., :self.in_s_features]
        s_new = s

        if isinstance(s, np.ndarray):
            if self.fit_input.item() > 0.5:
                s_new = (s - self.inputs_mu[:self.in_s_features].detach().cpu().numpy()) / self.inputs_sigma[:self.in_s_features].detach().cpu().numpy()
            en_std = en_std.detach().cpu().numpy()
            s_new = s_new * en_std + en_b.detach().cpu().numpy()
        else:
            if self.fit_input.item() > 0.5:
                s_new = (s - self.inputs_mu[:self.in_s_features]) / self.inputs_sigma[:self.in_s_features]
            s_new = s_new*en_std + en_b

        return s_new

    def encoder_a(self, a, elite=False):

        en_std = self.en_std
        en_b = self.en_b
        if elite:
            en_std = self.en_std_e
            en_b = self.en_b_e

        en_std = en_std[..., self.in_s_features:]
        en_b = en_b[..., self.in_s_features:]

        if self.fit_input.item() > 0.5:
            a = (a - self.inputs_mu[self.in_s_features:]) / self.inputs_sigma[self.in_s_features:]

        if isinstance(a, np.ndarray):
            en_std = en_std.detach().cpu().numpy()
            a = a*en_std + en_b.detach().cpu().numpy()
        else:
            a = a*en_std + en_b

        return a
    
    def dedecoder(self, s, elite=False):

        en_std = self.en_std
        en_b = self.en_b
        if elite:
            en_std = self.en_std_e
            en_b = self.en_b_e

        en_std = en_std[..., :self.in_s_features]
        en_b = en_b[..., :self.in_s_features]

        if isinstance(s, np.ndarray):
            new_s = (s-en_b.detach().cpu().numpy())/en_std.detach().cpu().numpy()
            if self.fit_input.item() > 0.5:
                new_s = new_s * self.inputs_sigma[:self.in_s_features].detach().cpu().numpy() + self.inputs_mu[:self.in_s_features].detach().cpu().numpy()
        else:
            new_s = (s-en_b)/en_std
            if self.fit_input.item() > 0.5:
                new_s = new_s * self.inputs_sigma[:self.in_s_features] + self.inputs_mu[:self.in_s_features]

        return new_s
      
    def dedecoder_var(self, var, is_log=True, detach=False, elite=False):

        en_std = self.en_std
        if elite:
            en_std = self.en_std_e

        # return var
        if detach:
            en_std = en_std.detach()
        else:
            en_std = en_std
        
        en_std = en_std[..., :self.in_s_features]
        
        if is_log:
            logvar = var
            logvar = logvar - 2*en_std.log()
            new_var = logvar
            if self.fit_input.item() > 0.5:
                new_var = new_var + self.inputs_sigma[:self.in_s_features].log()*2
        else:
            new_var = var/(en_std**2)
            if self.fit_input.item() > 0.5:
                new_var = new_var * (self.inputs_sigma[:self.in_s_features]**2)
        return new_var
    
    def compute_decays(self):

        # var_decays = 0.000025 * ((self.en_var.sqrt() - 1) ** 2).sum() * (self.hidden_size/(self.in_features)) / 2.0
        # var_decays = 0.000025 * (self.en_var).sum() * (self.hidden_size/(self.in_features)) / 2.0
        var_decays = 0.000025 * (self.en_std ** 2).sum() / 2.0
        lin0_decays = 0.000025 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00005 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.000075 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.000075 * (self.lin3_w ** 2).sum() / 2.0
        lin4_decays = 0.0001 * (self.lin4_w ** 2).sum() / 2.0
        # lin4_decays = 0.0001 * (self.lin4_w ** 2).sum() * (self.hidden_size/(self.out_features)) / 2.0

        return var_decays + lin0_decays + lin1_decays + lin2_decays + lin3_decays + lin4_decays

    def forward(self, s, a, ret_logvar=False, open_dropout=True):
        s = self.encoder(s)
        mean, var = self.dynamic(s, a, ret_logvar, open_dropout)
        return mean, var

    def dynamic(self, s, a, ret_logvar=False, open_dropout=True):

        inputs = torch.cat([s, a], dim=-1)
        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b
        
        inputs = swish(inputs)
        inputs = inputs.matmul(self.lin4_w) + self.lin4_b 

        mean = inputs[:, :, :self.out_features // 2]
        logvar = inputs[:, :, self.out_features // 2:]

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)
    
    def elite_dynamic(self, s, a, ret_logvar=False, open_dropout=True):

        inputs = torch.cat([s, a], dim=-1)
        
        inputs = inputs.matmul(self.lin0_w_e) + self.lin0_b_e
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w_e) + self.lin1_b_e
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w_e) + self.lin2_b_e
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w_e) + self.lin3_b_e
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin4_w_e) + self.lin4_b_e

        mean = inputs[:, :, :self.out_features // 2]
        logvar = inputs[:, :, self.out_features // 2:]

        # logvar = self.max_logvar_e - F.softplus(self.max_logvar_e - logvar)
        # logvar = self.min_logvar_e + F.softplus(logvar - self.min_logvar_e)

        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if ret_logvar:
            return mean, logvar

        return mean, torch.exp(logvar)
    
   
    def reset_weights(self):
        device = self.get_param_device()
        self.lin0_w, self.lin0_b = get_affine_params(self.num_nets, self.in_features, self.hidden_size, device=device)
        self.lin1_w, self.lin1_b = get_affine_params(self.num_nets, self.hidden_size, self.hidden_size, device=device)
        self.lin2_w, self.lin2_b = get_affine_params(self.num_nets, self.hidden_size, self.hidden_size, device=device)
        self.lin3_w, self.lin3_b = get_affine_params(self.num_nets, self.hidden_size, self.hidden_size, device=device)
        self.lin4_w, self.lin4_b = get_affine_params(self.num_nets, self.hidden_size, self.out_features, device=device)
        
        self.max_logvar = nn.Parameter(torch.ones(1, self.out_features // 2, dtype=torch.float32).to(device) / 2.0)
        self.min_logvar = nn.Parameter(- torch.ones(1, self.out_features // 2, dtype=torch.float32).to(device) * 10.0)
    
    def get_param_device(self):
        return next(self.parameters()).device


class StandardScaler(object):
    def __init__(self):
        self.is_fit = False
        pass

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        self.is_fit = True
        data = data.reshape(-1, data.shape[-1])
        self.mu = torch.mean(data, dim=0, keepdim=True)
        self.std = torch.std(data, dim=0, keepdim=True)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        if not self.is_fit:
            return data
        return (data - self.mu) / self.std

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.std * data + self.mu

