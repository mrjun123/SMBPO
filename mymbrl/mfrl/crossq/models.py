import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union
# from mymbrl.mfrl.crossq.batchrenorm import BatchRenorm1d
from mymbrl.mfrl.crossq.avgl1norm import AvgL1Norm

# importlib.reload(BatchRenorm1d)
import os
if os.path.exists(os.path.join('mymbrl/mfrl/crossq/__pycache__/', 'batchrenorm.cpython-39.pyc')):
    os.remove(os.path.join('mymbrl/mfrl/crossq/__pycache__/', 'batchrenorm.cpython-39.pyc'))
from mymbrl.mfrl.crossq.batchrenorm import BatchRenorm1d

def weights_init_(m, gain=1) -> None:
    """
    Method for initialisation of weights with xavier uniform
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

class QNetworkEncoderBack(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int=256,
                 num_layers: int =2,
                 activation: str = "relu",
                 bn_momentum:float=0.9,
                 use_batch_norm:bool=False,
                 bn_mode: str="bn") -> None:
        super(QNetworkEncoder, self).__init__()
        """
        QNetwork for SAC/CrossQ
        :param state_dim: number of state features
        :param action_dim: number of action features
        :param hidden_dim: number of units in hidden layers
        :param num_layers: number of hidden layers
        :param activation: activation choice ["relu" or "leaky_relu"]
        :param use_batch_norm: if True batch norm is used between layers
        :param bn_momentum: the momentum value for batch norm
        :param bn_mode: "bn" or "brn"
        """

        # Activations
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise NotImplementedError
        
        if bn_mode == "bn":
            BN = nn.BatchNorm1d
        elif bn_mode == "brn":
            BN = BatchRenorm1d
        else:
            raise NotImplementedError

        self.BN = BN
        # Layers
        self.q1_list = nn.ModuleList()
        self.q2_list = nn.ModuleList()

        # BN layer 0 - according to the code of crossQ
        if use_batch_norm:
            self.q1_list.append(BN(state_dim + action_dim, momentum=bn_momentum))
            self.q2_list.append(BN(state_dim + action_dim, momentum=bn_momentum))
        
        sz_dim = 256

        self.q1_list.append(nn.Linear(int((state_dim + action_dim)), sz_dim))
        self.q1_list.append(self.activation)

        self.q2_list.append(nn.Linear(int((state_dim + action_dim)), sz_dim))
        self.q2_list.append(self.activation)

        if use_batch_norm:
            self.q1_list.append(BN(sz_dim, momentum=bn_momentum))
            self.q2_list.append(BN(sz_dim, momentum=bn_momentum))

        self.q1_list.append(nn.Linear(sz_dim, hidden_dim))
        self.q1_list.append(self.activation)

        self.q2_list.append(nn.Linear(sz_dim, hidden_dim))
        self.q2_list.append(self.activation)
        if use_batch_norm:
            self.q1_list.append(BN(hidden_dim, momentum=bn_momentum))
            self.q2_list.append(BN(hidden_dim, momentum=bn_momentum))
        
        for i in range(num_layers - 1):
            self.q1_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.q1_list.append(self.activation)

            self.q2_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.q2_list.append(self.activation)
            if use_batch_norm:
                self.q1_list.append(BN(hidden_dim))
                self.q2_list.append(BN(hidden_dim))

        self.q1_list.append(nn.Linear(hidden_dim, 1))
        self.q2_list.append(nn.Linear(hidden_dim, 1))

        # Weight initialization
        # self.apply(weights_init_)

    def forward(self, state, action, zsa, zs, lap_priority=None) -> Tuple[torch.tensor, torch.tensor]:
        """
        Recieve as input state and action to compute Q-values
        :param state: input state
        :param action: Input action
        """

        # embeddings = torch.cat([zsa, zs], -1)

        # Concatenate
        x = torch.cat([state, action], -1)
        # BN
        x1 = self.q1_list[0](x, lap_priority)
        # NN
        x1 = self.q1_list[1](x)
        ## Relu
        x1 = self.q1_list[2](x1)
        x1 = x1 + zsa

        # x1 = torch.cat([x1, embeddings], -1)
        
        x2 = self.q2_list[0](x, lap_priority)
        x2 = self.q2_list[1](x)
        x2 = self.q2_list[2](x2)
        x2 = x2 + zsa
        # x2 = torch.cat([x2, embeddings], -1)

        for i in range(3, len(self.q1_list)):
            if isinstance(self.q1_list[i], self.BN):
                x1 = self.q1_list[i](x1, lap_priority)
                x2 = self.q2_list[i](x2, lap_priority)
            else:
                x1 = self.q1_list[i](x1)
                x2 = self.q2_list[i](x2)

        return x1, x2

    def to(self, device):
        return super(QNetworkEncoder, self).to(device)


class QNetworkEncoder(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int=256,
                 num_layers: int =2,
                 activation: str = "relu",
                 bn_momentum:float=0.9,
                 use_batch_norm:bool=False,
                 bn_mode: str="bn") -> None:
        super(QNetworkEncoder, self).__init__()
        """
        QNetwork for SAC/CrossQ
        :param state_dim: number of state features
        :param action_dim: number of action features
        :param hidden_dim: number of units in hidden layers
        :param num_layers: number of hidden layers
        :param activation: activation choice ["relu" or "leaky_relu"]
        :param use_batch_norm: if True batch norm is used between layers
        :param bn_momentum: the momentum value for batch norm
        :param bn_mode: "bn" or "brn"
        """

        # Activations
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise NotImplementedError
        
        if bn_mode == "bn":
            BN = nn.BatchNorm1d
        elif bn_mode == "brn":
            BN = BatchRenorm1d
        else:
            raise NotImplementedError

        self.BN = BN
        # Layers
        self.q1_list = nn.ModuleList()
        self.q2_list = nn.ModuleList()

        self.state_dim = state_dim
        # BN layer 0 - according to the code of crossQ
        if use_batch_norm:
            self.q1_list.append(BN(state_dim + action_dim, momentum=bn_momentum))
            self.q2_list.append(BN(state_dim + action_dim, momentum=bn_momentum))

        self.q1_list.append(nn.Linear(int((state_dim + action_dim)), hidden_dim))
        self.q2_list.append(nn.Linear(int((state_dim + action_dim)), hidden_dim))

        self.q1_list.append(self.activation)
        self.q2_list.append(self.activation)

        # sz_dim = 256
        sz_dim = state_dim
        sz_dim_input = state_dim
        
        if use_batch_norm:
            self.q1_list.append(BN(hidden_dim, momentum=bn_momentum))
            self.q2_list.append(BN(hidden_dim, momentum=bn_momentum))

        self.q1_list.append(nn.Linear(hidden_dim + sz_dim_input, hidden_dim))
        self.q1_list.append(self.activation)

        self.q2_list.append(nn.Linear(hidden_dim + sz_dim_input, hidden_dim))
        self.q2_list.append(self.activation)

        # self.q1_list.append(BN(hidden_dim, momentum=bn_momentum))
        # self.q2_list.append(BN(hidden_dim, momentum=bn_momentum))
        
        for i in range(num_layers - 1):
            self.q1_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.q1_list.append(self.activation)

            self.q2_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.q2_list.append(self.activation)
            if use_batch_norm:
                self.q1_list.append(BN(hidden_dim, momentum=bn_momentum))
                self.q2_list.append(BN(hidden_dim, momentum=bn_momentum))

        self.q1_list.append(nn.Linear(hidden_dim, 1))
        self.q2_list.append(nn.Linear(hidden_dim, 1))

        # Weight initialization
        # self.apply(weights_init_)

    def forward(self, state, action, zsa, zs, lap_priority=None) -> Tuple[torch.tensor, torch.tensor]:
        """
        Recieve as input state and action to compute Q-values
        :param state: input state
        :param action: Input action
        """

        x = torch.cat([state, action], -1)

        # BN
        x1 = self.q1_list[0](x, lap_priority)
        # NN
        x1 = self.q1_list[1](x1)
        ## Relu
        x1 = self.q1_list[2](x1)
        # BN
        x1 = self.q1_list[3](x1)
        x1 = torch.cat([x1, zsa], dim=-1)
        # x1 = zsa
        # x1 = self.q1_list[4](x1)
        # x1 = self.q1_list[5](x1)

        # x1 = x1*embeddings_var + embeddings
        # x1 = torch.cat([x1, action], -1)

        # x2 = x
        
        x2 = self.q2_list[0](x, lap_priority)
        x2 = self.q2_list[1](x2)
        x2 = self.q2_list[2](x2)
        x2 = self.q2_list[3](x2)
        x2 = torch.cat([x2, zsa], dim=-1)
        # x2 = zsa
        # x2 = self.q2_list[4](x2)
        # x2 = self.q2_list[5](x2)

        # x2 = x2*embeddings_var + embeddings
        # x2 = torch.cat([x2, action], -1)

        for i in range(4, len(self.q1_list)):
            if isinstance(self.q1_list[i], self.BN):
                x1 = self.q1_list[i](x1, lap_priority)
                x2 = self.q2_list[i](x2, lap_priority)
            else:
                x1 = self.q1_list[i](x1)
                x2 = self.q2_list[i](x2)

        return x1, x2

    def to(self, device):
        return super(QNetworkEncoder, self).to(device)

class QNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int=256,
                 num_layers: int =2,
                 activation: str = "relu",
                 bn_momentum:float=0.9,
                 use_batch_norm:bool=False,
                 bn_mode: str="bn",
                 bn_warmup_steps: int=100e3) -> None:
        super(QNetwork, self).__init__()
        """
        QNetwork for SAC/CrossQ
        :param state_dim: number of state features
        :param action_dim: number of action features
        :param hidden_dim: number of units in hidden layers
        :param num_layers: number of hidden layers
        :param activation: activation choice ["relu" or "leaky_relu"]
        :param use_batch_norm: if True batch norm is used between layers
        :param bn_momentum: the momentum value for batch norm
        :param bn_mode: "bn" or "brn"
        """

        # Activations
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError
        
        if bn_mode == "bn":
            BN = nn.BatchNorm1d
        elif bn_mode == "brn":
            BN = BatchRenorm1d
        else:
            raise NotImplementedError

        # Layers
        self.q1_list = nn.ModuleList()
        self.q2_list = nn.ModuleList()

        # BN layer 0 - according to the code of crossQ
        if use_batch_norm:
            self.q1_list.append(BN(state_dim + action_dim, momentum=bn_momentum, warmup_steps=bn_warmup_steps))
            self.q2_list.append(BN(state_dim + action_dim, momentum=bn_momentum, warmup_steps=bn_warmup_steps))

        self.q1_list.append(nn.Linear(int((state_dim + action_dim)), hidden_dim))
        self.q1_list.append(self.activation)

        self.q2_list.append(nn.Linear(int((state_dim + action_dim)), hidden_dim))
        self.q2_list.append(self.activation)
        
        if use_batch_norm:
            self.q1_list.append(BN(hidden_dim, momentum=bn_momentum, warmup_steps=bn_warmup_steps))
            self.q2_list.append(BN(hidden_dim, momentum=bn_momentum, warmup_steps=bn_warmup_steps))

        for i in range(num_layers - 1):
            self.q1_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.q1_list.append(self.activation)
            self.q2_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.q2_list.append(self.activation)
            if use_batch_norm:
                self.q1_list.append(BN(hidden_dim, momentum=bn_momentum, warmup_steps=bn_warmup_steps))
                self.q2_list.append(BN(hidden_dim, momentum=bn_momentum, warmup_steps=bn_warmup_steps))

        self.q1_list.append(nn.Linear(hidden_dim, 1))
        self.q2_list.append(nn.Linear(hidden_dim, 1))

        # Weight initialization
        self.apply(weights_init_)

    def forward(self, state, action) -> Tuple[torch.tensor, torch.tensor]:
        """
        Recieve as input state and action to compute Q-values
        :param state: input state
        :param action: Input action
        """

        # Concatenate
        x = torch.cat([state, action], 1)
        x1 = self.q1_list[0](x)
        x2 = self.q2_list[0](x)
        for i in range(1, len(self.q1_list)):
            x1 = self.q1_list[i](x1)
            x2 = self.q2_list[i](x2)

        return x1, x2

    def to(self, device):
        return super(QNetwork, self).to(device)

    
class GaussianPolicyEncoder(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 activation: str = "relu",
                 LOG_SIG_MIN: int = -20,
                 LOG_SIG_MAX: int = 2,
                 eps: float = 1e-7,
                 use_batch_norm:bool = False,
                 bn_momentum:float=0.9,
                 bn_mode: str="bn"
                 ) -> None:
        """
        Gaussian Policy for SAC/CrossQ
        :param num_inputs: number of inputs
        :param action_dim: number of actions
        :param hidden_dim: number of units in hidden layers
        :param num_layers: number of hidden layers
        :param activation: activation choice ["relu" or "leaky_relu"]
        :param LOG_SIG_MIN: min for normal
        :param LOG_SIG_MAX: max for normal
        :param eps: small value for stability
        :param use_batch_norm: if True batch norm is used between layers
        :param bn_momentum: the momentum value for batch norm
        :param bn_mode: "bn" or "brn"
        """
        super(GaussianPolicyEncoder, self).__init__()
        # Activations
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError
        self.use_batch_norm=use_batch_norm
        if use_batch_norm:
            print("Using batchnorm in policy")

        if bn_mode == "bn":
            BN = nn.BatchNorm1d
        elif bn_mode == "brn":
            BN = BatchRenorm1d
        else:
            raise NotImplementedError
        # Layers
        self.pi_list = nn.ModuleList()
        # layer 0 batch norm according to CrossQ
        if use_batch_norm:
            self.pi_list.append(BN(num_inputs, momentum=bn_momentum))
        self.pi_list.append(nn.Linear(num_inputs, hidden_dim))
        self.pi_list.append(self.activation)

        # sz_dim = 256
        sz_dim = num_inputs
        if use_batch_norm:
            self.pi_list.append(BN(hidden_dim + sz_dim, momentum=bn_momentum))
        
        # self.pi_list.append(nn.Linear(hidden_dim, hidden_dim))
        # self.pi_list.append(nn.Linear(hidden_dim, hidden_dim))
        # self.pi_list.append(nn.Linear(sz_dim, hidden_dim))
        # self.pi_list.append(self.activation)
        for i in range(num_layers - 1):
            self.pi_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.pi_list.append(self.activation)
            if use_batch_norm:
                self.pi_list.append(BN(hidden_dim, momentum=bn_momentum))
        self.mean_out = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std_out = torch.nn.Linear(hidden_dim, action_dim)

        # Weight initialization
        # self.apply(weights_init_)

        self.LOG_SIG_MAX = LOG_SIG_MAX
        self.LOG_SIG_MIN = LOG_SIG_MIN
        self.eps = eps
    
    def forward(self, state: torch.tensor, zs: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        # Layers
        # a = AvgL1Norm(self.l0(state))
        # a = torch.cat([a, zs], 1)
        for i in range(len(self.pi_list)):
            if (not self.use_batch_norm and i == 2) or (self.use_batch_norm and i == 3):
                # state = torch.cat([state, zs], dim=1)
                pass
                # state = zs
                # state = self.pi_list[i](state)
            # else:
            state = self.pi_list[i](state)
        mean, log_std = self.mean_out(state), self.log_std_out(state)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.tensor, zs: torch.tensor, evaluation:bool=False, sample_for_loop: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Recieve as input state in 2D format.
        :param state: state input
        :param evaluation: if we are exploring or exploiting.
        :param sample_for_loop: if we are in the loop or updating.
        """

        if sample_for_loop:

            # For the loop we do less operation and we do not track the gradients.
            with torch.no_grad():
                mean, log_std = self.forward(state, zs)
                if evaluation:
                    mean = torch.tanh(mean)
                    return mean.squeeze(0)
                else:
                    std = log_std.exp()
                    normal = Normal(mean, std)
                    x_t = normal.rsample()
                    y_t = torch.tanh(x_t)
                    return y_t.squeeze(0)
        else:
            mean, log_std = self.forward(state, zs)
            std = log_std.exp()

            normal = Normal(mean, std)

            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            mean = torch.tanh(mean)
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log((1 - y_t.pow(2).clamp(min=0, max=1)) + self.eps)
            log_prob = log_prob.sum(1, keepdim=True)

            return y_t, log_prob, mean

    def to(self, device):
        return super(GaussianPolicyEncoder, self).to(device)
    
class GaussianPolicy(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 activation: str = "relu",
                 LOG_SIG_MIN: int = -20,
                 LOG_SIG_MAX: int = 2,
                 eps: float = 1e-7,
                 use_batch_norm:bool = False,
                 bn_momentum:float=0.9,
                 bn_mode: str="bn",
                 bn_warmup_steps: int=100e3
                 ) -> None:
        """
        Gaussian Policy for SAC/CrossQ
        :param num_inputs: number of inputs
        :param action_dim: number of actions
        :param hidden_dim: number of units in hidden layers
        :param num_layers: number of hidden layers
        :param activation: activation choice ["relu" or "leaky_relu"]
        :param LOG_SIG_MIN: min for normal
        :param LOG_SIG_MAX: max for normal
        :param eps: small value for stability
        :param use_batch_norm: if True batch norm is used between layers
        :param bn_momentum: the momentum value for batch norm
        :param bn_mode: "bn" or "brn"
        """
        super(GaussianPolicy, self).__init__()
        # Activations
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        if use_batch_norm:
            print("Using batchnorm in policy")

        if bn_mode == "bn":
            BN = nn.BatchNorm1d
        elif bn_mode == "brn":
            BN = BatchRenorm1d
        else:
            raise NotImplementedError
        # Layers
        self.pi_list = nn.ModuleList()
        # layer 0 batch norm according to CrossQ
        if use_batch_norm:
            self.pi_list.append(BN(num_inputs, momentum=bn_momentum, warmup_steps=bn_warmup_steps))
        self.pi_list.append(nn.Linear(num_inputs, hidden_dim))
        self.pi_list.append(self.activation)
        if use_batch_norm:
            self.pi_list.append(BN(hidden_dim, momentum=bn_momentum, warmup_steps=bn_warmup_steps))
        for i in range(num_layers - 1):
            self.pi_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.pi_list.append(self.activation)
            if use_batch_norm:
                self.pi_list.append(BN(hidden_dim, momentum=bn_momentum, warmup_steps=bn_warmup_steps))
        self.mean_out = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std_out = torch.nn.Linear(hidden_dim, action_dim)

        # Weight initialization
        self.apply(weights_init_)

        self.LOG_SIG_MAX = LOG_SIG_MAX
        self.LOG_SIG_MIN = LOG_SIG_MIN
        self.eps = eps
    
    def forward(self, state: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        # Layers
        for i in range(len(self.pi_list)):
            # print(i)
            state = self.pi_list[i](state)

        mean, log_std = self.mean_out(state), self.log_std_out(state)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.tensor, evaluation:bool=False, sample_for_loop: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Recieve as input state in 2D format.
        :param state: state input
        :param evaluation: if we are exploring or exploiting.
        :param sample_for_loop: if we are in the loop or updating.
        """

        if sample_for_loop:

            # For the loop we do less operation and we do not track the gradients.
            with torch.no_grad():
                # print('state', state)
                mean, log_std = self.forward(state)
                if evaluation:
                    mean = torch.tanh(mean)
                    return mean.squeeze(0)
                else:
                    std = log_std.exp()
                    # print('mean', mean, 'std', std)
                    normal = Normal(mean, std)
                    x_t = normal.rsample()
                    y_t = torch.tanh(x_t)
                    return y_t.squeeze(0)
        else:
            mean, log_std = self.forward(state)
            std = log_std.exp()

            normal = Normal(mean, std)

            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            mean = torch.tanh(mean)
            log_prob = normal.log_prob(x_t)
            log_prob -= torch.log((1 - y_t.pow(2).clamp(min=0, max=1)) + self.eps)
            log_prob = log_prob.sum(1, keepdim=True)

            return y_t, log_prob, mean

    # def to(self, device):
    #     return super(GaussianPolicy, self).to(device)