from typing import List, Optional

import torch
from botorch.test_functions.base import BaseTestProblem
from botorch.test_functions.synthetic import SyntheticTestFunction


class EmbeddedTestFunction(BaseTestProblem):
    def __init__(
        self,
        inner_fn: SyntheticTestFunction,
        d_ambient: int,
        negate: bool = True,
        noise_std: Optional[float] = None,
        permute: bool = False,
        seed: Optional[int] = None,
    ):
       
        if inner_fn.negate:
            raise ValueError(
            )
        d_effective = inner_fn.dim
        if d_ambient < d_effective:
            raise ValueError(f"d_ambient ({d_ambient}) must be >= inner_fn.dim ({d_effective})")

        self.dim = d_ambient

        if permute:
            if seed is None:
                raise ValueError("permute=True requires an explicit seed, so active_dims stays fixed for the run")
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(d_ambient, generator=g)
            active_dims: List[int] = sorted(perm[:d_effective].tolist())
        else:
            active_dims = list(range(d_effective))
        bounds = [(0.0, 1.0)] * d_ambient
        for pos, dim_idx in enumerate(active_dims):
            bounds[dim_idx] = inner_fn._bounds[pos]
        self._bounds = bounds
        self._optimizers = None
        self.continuous_inds = list(range(d_ambient))

        super().__init__(noise_std=noise_std, negate=negate)
        self.inner_fn = inner_fn
        self.active_dims = active_dims
        self.d_effective = d_effective

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        X_active = X[..., self.active_dims]
        return self.inner_fn.evaluate_true(X_active)


class Sphere(SyntheticTestFunction):

    def __init__(self, dim: int = 2, negate: bool = False):
        self.dim = dim
        self._bounds = [(-5.0, 5.0)] * dim
        self.continuous_inds = list(range(dim))
        self._optimizers = [tuple([0.0] * dim)]
        super().__init__(negate=negate)

    def _evaluate_true(self, X: torch.Tensor) -> torch.Tensor:
        return (X**2).sum(dim=-1)
