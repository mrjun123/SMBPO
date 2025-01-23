import torch

# from https://github.com/ludvb/batchrenorm

__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]

class BatchRenorm(torch.jit.ScriptModule):
# class BatchRenorm(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
        warmup_steps: int = 100000
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
        self.step_update_num = 1

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        """
        Scales standard deviation
        """
        # return torch.tensor(3, device=self.num_batches_tracked.device)
        # epoch < 5 -> 1, epoch > 40 -> 3
        # min_r = 1.0
        # max_r = 3.0
        # start_step = self.warmup_steps*0.1
        # center_step = self.warmup_steps*0.8

        # progress = ((self.num_batches_tracked - start_step) / center_step).clamp(0, 1.0)
        # r = min_r + progress*(max_r-min_r)
        # return r

        return ((2 * (self.num_batches_tracked / (self.step_update_num*1000)) + 25)/ 35).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        """
        Scales mean
        """
        # return torch.tensor(5, device=self.num_batches_tracked.device)
        # epoch < 5 -> 0, epoch > 25 -> 5

        # min_d = 0.0
        # max_d = 5.0
        # start_step = self.warmup_steps*0.1
        # center_step = self.warmup_steps*0.8

        # progress = ((self.num_batches_tracked - start_step) / center_step).clamp(0, 1.0)
        # d = min_d + progress*(max_d-min_d)
        # return d

        return ((5 * (self.num_batches_tracked / (self.step_update_num*1000)) - 25)/ 20).clamp_(
            0.0, 5.0
        )

    def forward(self, x: torch.Tensor, priority=None, mask = None) -> torch.Tensor:
        '''
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        '''
        self._check_input_dim(x)
        ndim = x.ndim
        x_shape = x.shape
        if ndim > 2:
            x = x.reshape(-1, x_shape[-1])
        if self.training:
            # Adding warm up
            warmed_up_factor = (self.num_batches_tracked >= self.warmup_steps).float()
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0) 
                batch_var = z.var(0, unbiased=False)
            elif priority is not None:
                priority = priority.reshape(priority.shape[0], 1)
                weight = (1/priority)

                total_weight = torch.sum(weight, dim=0)
                weighted_mean = torch.sum(x*weight, dim=0) / total_weight

                # 计算加权方差
                differences = x - weighted_mean
                squared_differences = differences ** 2
                weighted_variance = torch.sum(squared_differences * weight, dim=0) / total_weight
                batch_mean = weighted_mean
                batch_var = weighted_variance

            else:
                batch_mean = x.mean(dims)
                batch_var = x.var(dims, unbiased=False)
            
            # warmed_up_factor = True
            if not warmed_up_factor:
                # if self.training:
                x =  (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            else:
                batch_std = torch.sqrt(batch_var + self.eps)
                running_std = torch.sqrt(self.running_var.view_as(batch_var) + self.eps)
                # r = ((batch_var/ running_std).clamp_(1 / self.rmax, self.rmax)).detach()  
                d = (((batch_mean - self.running_mean.view_as(batch_mean))/ running_std).clamp_(-self.dmax, self.dmax)).detach() 

                r = (batch_std / running_std).detach().clamp_(1 / self.rmax, self.rmax).detach()  

                x = r * (x - batch_mean) / torch.sqrt(batch_var + self.eps) + d
                # x = r * ((x - batch_mean) / torch.sqrt(batch_var + self.eps)) + d
            # Pytorch convention (1-beta)*estimated + beta*observed
            # if self.training:
            self.num_batches_tracked += 1
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.detach()
            
        else: # x=r(x^−μpop​ )/σpop​ +d # running mean and std
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        if self.affine: # Step 3 affine transform: y=γx+β
            x = self.weight * x + self.bias
        if ndim > 2:
            x = x.reshape(x_shape[0], -1, x_shape[-1])
        return x

# @torch.jit.script
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