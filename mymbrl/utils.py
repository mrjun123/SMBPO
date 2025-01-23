from pathlib import Path
from typing import Union
import numpy as np
import torch
from torch import nn as nn
import warnings
import math
import pickle
import os
import inspect

class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
        warmup_steps: int = 10000
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_var", torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum
        self.warmup_steps = warmup_steps

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        """
        Scales standard deviation
        """
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        """
        Scales mean
        """
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 5.0
        )

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        '''
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        '''
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            # x=r(x^−μ)/σ+d # changing input with running mean, std, dynamic upper limit r, dynamic shift limit d 
            # μ, σ, r, d updated as:
            # -> μ = μ + momentum * (input.mean(0))
            # -> σ = σ + momentum * (input.std(0) + eps)
            # -> r = clip(input.std(0)/σ, !/rmax, rmax)
            # -> d = clip((input.mean(0) - μ)/σ, -dmax, dmax)
            # Also: optional masking
            # Also: counter "num_batches_tracked"
            # Note: The introduction of r and d mitigates some of the issues of BN, especially with small BZ or significant shifts in the input distribution. 
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0) 
                batch_var = z.var(0, unbiased=False)
            else:
                batch_mean = x.mean(dims)
                batch_var = x.var(dims, unbiased=False)
            
            # Adding warm up
            warmed_up_factor = (self.num_batches_tracked >= self.warmup_steps).float()

            running_std = torch.sqrt(self.running_var.view_as(batch_var) + self.eps)
            r = ((batch_var/ running_std).clamp_(1 / self.rmax, self.rmax)).detach()  
            d = (((batch_mean - self.running_mean.view_as(batch_mean))/ running_std).clamp_(-self.dmax, self.dmax)).detach() 
            if warmed_up_factor:
                x =  (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            else:
                x = r * ((x - batch_mean) / torch.sqrt(batch_var + self.eps)) + d
            # Pytorch convention (1-beta)*estimated + beta*observed
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
            self.num_batches_tracked += 1
        else: # x=r(x^−μpop​ )/σpop​ +d # running mean and std
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        if self.affine: # Step 3 affine transform: y=γx+β
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")

class LogDict():
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.file_path = os.path.join(path, f'{name}.pkl')
        self.data = {}
        pass
    
    def log(self, key_name, value):
        if key_name in self.data:
            self.data[key_name].append(value)
        else:
            self.data[key_name] = [value]
    
    def save(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.data, f)

def logsumexp(input, dim=None, keepdim=False, weights=False):
    max_val, _ = torch.max(input, dim, keepdim=True)
    max_val = max_val.detach()
    output = input - max_val  # 减去最大值
    output = torch.exp(output)  # 指数化
    if not isinstance(weights, bool):
        output = output*weights
    sum_val = torch.sum(output, dim, keepdim=keepdim)  # 求和
    log_sum_exp = torch.log(sum_val) + max_val.squeeze(dim)  # 对数化并加回最大值
    return log_sum_exp

def has_input_param(func, param_name):
    parameters = inspect.signature(func).parameters
    return param_name in parameters

def merge_dict(dict1, dict2):
    '''
    合并字典
    :return:
    '''
    if not isinstance(dict1,dict) or not isinstance(dict2,dict):
        return dict1
    for key,info in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], dict):
                dict1[key] = merge_dict(dict1[key],info)
            elif isinstance(dict1[key], list):
                # dict1[key].extend(info)
                dict1[key] = info
            else:
                dict1[key] = info
        else :
            dict1[key] = info
    return dict1

def shuffle_rows(arr):
    idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
    return arr[np.arange(arr.shape[0])[:, None], idxs]

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean = 0., std = 1., a = -2., b = 2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_affine_params_uniform(ensemble_size, in_features, out_features, device='cpu'):
    # w = torch.empty(ensemble_size, in_features, out_features, device=device)
    # torch.nn.init.trunc_normal_(w, std=1.0 / (2.0 * np.sqrt(in_features)))
    # w = nn.Parameter(w)
    # b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32, device=device))

    w = torch.empty(ensemble_size, in_features, out_features, device=device)
    nn.init.kaiming_uniform_(w, a=np.sqrt(5))

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
    bound = 1 / np.sqrt(fan_in)
    b = nn.Parameter(torch.empty(ensemble_size, 1, out_features, device=device).uniform_(-bound, bound))

    w = nn.Parameter(w)

    return w, b

def get_affine_params_uniform_2d(in_features, out_features, device='cpu'):
    w = torch.empty(in_features, out_features, device=device)
    nn.init.kaiming_uniform_(w, a=np.sqrt(5))

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
    bound = 1 / np.sqrt(fan_in)
    b = nn.Parameter(torch.empty(1, out_features, device=device).uniform_(-bound, bound))

    w = nn.Parameter(w)

    return w, b

def get_affine_params(ensemble_size, in_features, out_features, device='cpu'):
    w = torch.empty(ensemble_size, in_features, out_features, device=device)
    torch.nn.init.trunc_normal_(w, std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)
    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32, device=device))

    return w, b

def get_affine_params_2d(in_features, out_features, device='cpu'):
    w = torch.empty(in_features, out_features, device=device)
    torch.nn.init.trunc_normal_(w, std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)
    b = nn.Parameter(torch.zeros(1, out_features, dtype=torch.float32, device=device))

    return w, b

def swish(x):
    return x * torch.sigmoid(x)

def new_run_directory(path: Union[str, Path]):
    path = Path(path)
    run_name = path.name
    previous_runs = path.parent.glob(f'{run_name}*')
    # Remove run_name from run directories
    prev_run_nums = map(lambda prev_path: prev_path.name[len(run_name):], previous_runs)
    # Remove those runs that do not have a number at the end
    prev_run_nums = filter(lambda prev_run_num: prev_run_num.isdigit(), prev_run_nums)
    # Convert to int and sort
    prev_run_nums = sorted(map(lambda prev_run_num: int(prev_run_num), prev_run_nums))
    # If there are any previous runs
    if prev_run_nums:
        new_run_num = int(prev_run_nums[-1]) + 1
    else:
        new_run_num = 1
    new_run_dir = Path(str(path) + str(new_run_num))
    return new_run_dir

