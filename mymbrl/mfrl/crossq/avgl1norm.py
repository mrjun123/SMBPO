import torch

# from https://github.com/ludvb/batchrenorm

__all__ = ["BatchRenorm1d", "BatchRenorm2d", "BatchRenorm3d"]

class AvgL1Norm(torch.jit.ScriptModule):
    def __init__(
        self,
        eps=1e-8
    ):
        self.eps = eps
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x/x.abs().mean(-1,keepdim=True).clamp(min=self.eps)
