import torch
from botorch.test_functions.base import BaseTestProblem


class MovableSphere(BaseTestProblem):
    def __init__(self, dim: int, center_t: float, negate: bool = False):
        self.dim = dim
        self._bounds = [(0.0, 1.0)] * dim
        self.continuous_inds = list(range(dim))
        self._optimizers = None
        super().__init__(noise_std=None, negate=negate)
        x_star = torch.full((dim,), 0.5 + 0.5 * center_t, dtype=torch.float64)
        self.register_buffer("x_star", x_star)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return -(X - self.x_star).square().sum(dim=-1)
