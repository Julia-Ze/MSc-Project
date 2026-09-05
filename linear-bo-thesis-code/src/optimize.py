"""
Main module for running Bayesian optimization loops.
https://github.com/colmont/linear-bo
"""

import json
import logging
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from typing import Optional, Tuple

import botorch
import gpytorch
import hydra
import numpy as np
import torch
import tqdm as tqdm
from botorch.acquisition import LogExpectedImprovement as LogEI
from botorch.generation import gen_candidates_scipy
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ConstantKernel, LinearKernel
from gpytorch.priors import LogNormalPrior
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from src.kernels.spherical_linear import SphericalLinearKernel

TOL = 1e-2


class CenteredLinearKernel(LinearKernel):
    def forward(self, x1, x2, diag=False, **params):
        return super().forward(x1 - 0.5, x2 - 0.5, diag=diag, **params)


def build_kernel(kernel_type: str, d: int) -> gpytorch.kernels.Kernel:
    if kernel_type == "spherical":
        return SphericalLinearKernel(ard_num_dims=d)

    elif kernel_type == "linear_noproj":
        from src.kernels.spherical_linear_noproj import SphericalLinearKernel as SphericalLinearKernelNoProj
        return SphericalLinearKernelNoProj(ard_num_dims=d)

    elif kernel_type == "linear":
        return ConstantKernel() + CenteredLinearKernel()

    elif kernel_type == "ard_linear":
        return ConstantKernel() + CenteredLinearKernel(ard_num_dims=d)

    elif kernel_type == "rbf_sphere":
        from src.kernels.rbf_sphere import RBFSphereKernel
        return RBFSphereKernel(ard_num_dims=d)

    elif kernel_type == "vanilla_rbf":
        from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
        return get_covar_module_with_dim_scaled_prior(ard_num_dims=d)
    
    else:
        raise ValueError(
            f"Unknown kernel_type: {kernel_type!r}. Expected one of 'spherical', 'linear', 'ard_linear'."
        )


def initialize_model(
    X_train: Float[Tensor, "n d"],
    Y_train: Float[Tensor, "n 1"],
    kernel_type: str = "spherical",
) -> botorch.models.SingleTaskGP:
    """
    Initialize the model for Bayesian optimization.

    :param X_train: Training input points.
    :param Y_train: Training output points.
    :param kernel_type: Which covariance kernel to use: "spherical" (default),
        "linear", or "ard_linear".
    """
    d = X_train.size(-1)

    mean = gpytorch.means.ConstantMean()
    kernel = build_kernel(kernel_type, d)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior := LogNormalPrior(loc=-4.0, scale=1.0),
        noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
    )

    return botorch.models.SingleTaskGP(
        train_X=X_train,
        train_Y=Y_train,
        mean_module=mean,
        covar_module=kernel,
        likelihood=likelihood,
    )


def evaluate_y(
    test_function: botorch.test_functions.base.BaseTestProblem,
    X: Float[Tensor, "... d"],
) -> Float[Tensor, "..."]:
    r"""
    Evaluate a test function at a given (batch of) input(s) :math:`X`.

    :param test_function: The test function to evaluate.
    :param X: The input(s) at which to evaluate the function.
    """
    unnormalized_X = botorch.utils.transforms.unnormalize(X, test_function.bounds)
    Y = test_function(unnormalized_X)
    return Y


def fit_model(model) -> None:
    """
    Fits the GP to the training data.

    :param model: The model to fit.
    """
    model.train()
    botorch_optimizer = botorch.fit.fit_gpytorch_mll_scipy

    with ExitStack() as es:
        es.enter_context(gpytorch.settings.cholesky_max_tries(10))

        # log_prob=True to use the woodbury decomposition for scalability
        es.enter_context(gpytorch.settings.max_cholesky_size(float("inf")))
        es.enter_context(
            gpytorch.settings.fast_computations(log_prob=True, covar_root_decomposition=False, solves=False)
        )

        # Fit the model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        botorch.fit.fit_gpytorch_mll(mll, optimizer=botorch_optimizer)


def extract_ard_values(
    kernel: gpytorch.kernels.Kernel, d: int
) -> Tuple[Optional[list], Optional[str], bool]:
 
    sub_kernels = list(kernel.kernels) if hasattr(kernel, "kernels") else [kernel]

    for k in sub_kernels:
        for attr_name in ("variance", "lengthscale"):
            value = getattr(k, attr_name, None)
            if value is None:
                continue
            flat = value.detach().reshape(-1)
            if flat.numel() == d:
                return flat.cpu().tolist(), attr_name, True
            if flat.numel() == 1:
                return [flat.item()] * d, attr_name, False

    return None, None, False


def get_benchmark_name(config: DictConfig) -> str:

    name = OmegaConf.select(config, "benchmark.name", default=None)
    if name is not None:
        return str(name)
    try:
        from hydra.core.hydra_config import HydraConfig

        choice = HydraConfig.get().runtime.choices.get("benchmark")
        if choice is not None:
            return str(choice)
    except Exception:
        pass

    return "unknown_benchmark"


def run_optimization(config: DictConfig) -> dict:
    r"""
    Main function to run a BO loop.
    """
    # Resolve configuration
    OmegaConf.resolve(config)
    logging.info("\n" + OmegaConf.to_yaml(config))

    # Set seeds
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        np.random.seed(config.seed)

    # Dtype and device
    dtype = torch.float64
    device = torch.device(config.device) if torch.cuda.is_available() else torch.device("cpu")

    # Which GP kernel to use. 
    kernel_type: str = OmegaConf.select(config, "kernel", default="spherical")

    # Get benchmark
    test_function = hydra.utils.instantiate(config.benchmark.fn).to(dtype=dtype, device=device)

    # Lower and upper bound for model (normalized to be in [0, 1]^d hypercube)
    lb = torch.zeros(test_function.dim, dtype=dtype, device=device)
    ub = torch.ones(test_function.dim, dtype=dtype, device=device)
    bounds = torch.stack([lb, ub], dim=-2)
    n_tot = config.benchmark.n_tot
    d = test_function.dim
    n_init = config.benchmark.n_init

    # Construct tensors with x, y values
    Xs: Float[Tensor, "n_tot d"] = torch.empty((n_tot, d), dtype=dtype, device=device)
    Ys: Float[Tensor, "n_tot 1"] = torch.empty((n_tot, 1), dtype=dtype, device=device)

    # Get initial points
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=config.seed)
    Xs[:n_init] = sobol.draw(n=n_init).to(dtype=dtype, device=device)
    Ys[:n_init] = evaluate_y(test_function, Xs[:n_init]).unsqueeze(-1)

    # Diagnostics history. 
    history: dict = {
        "kernel": kernel_type,
        "seed": config.seed,
        "benchmark": get_benchmark_name(config),
        "d": d,
        "n_init": n_init,
        "n_tot": n_tot,
        "boundary_tol": TOL,
        "best_curve": [],
        "best_x": [],
        "boundary_pct": [],
        "per_dim_boundary_history": [],
        "per_dim_dist_history": [],
        "acquired_X": [],
        "ard_weights_history": [],
        "ard_param_name": [],
        "ard_is_per_dim": [],
        "posterior_slope_history": [],
    }

    with torch.no_grad():
        # Boundary diagnostics for every Sobol-initialized point.
        dist_init = torch.minimum(Xs[:n_init], 1.0 - Xs[:n_init])
        on_boundary_init = dist_init < TOL
        history["boundary_pct"].extend((100.0 * on_boundary_init.double().mean(dim=-1)).cpu().tolist())
        history["per_dim_boundary_history"].extend(on_boundary_init.cpu().tolist())
        history["per_dim_dist_history"].extend(dist_init.cpu().tolist())

        y_max = -float("inf")
        x_max = None
        for i in range(n_init):
            yi = Ys[i, 0].item()
            if yi > y_max:
                y_max = yi
                x_max = Xs[i].clone()
            history["best_curve"].append(y_max)
            history["best_x"].append(x_max.cpu().tolist())

    # Construct iterator for BO loop
    tqdm_log_list = ["y_max"]
    pbar = tqdm.tqdm(
        initial=n_init,
        total=n_tot,
        desc="BO loop",
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        disable=(not len(tqdm_log_list)),
    )

    ###
    # BO loop
    ###
    n = n_init
    n_prev = n_init
    while n_prev < n_tot:
        n = n_prev + 1

        X = Xs[:n_prev]
        Y = Ys[:n_prev]

        # Fit model
        model = initialize_model(X_train=X, Y_train=standardize(Y), kernel_type=kernel_type)
        fit_model(model)
        model.eval()

        # Kernel hyperparameters
        ard_values, ard_param_name, ard_is_per_dim = extract_ard_values(model.covar_module, d)

        # Maximize acquisition function to get next Xs
        acqf = LogEI(model, standardize(Y).max())
        options = {
            "raw_samples": 512,
            "num_restarts": 4,
            "retry_on_optimization_warning": False,
            "options": {
                "nonnegative": False,
                "sample_around_best": True,
                "sample_around_best_sigma": 0.1,
                "maxiter": 300,
                "batch_limit": 64,
            },
        }
        Xs[n_prev:n], _ = optimize_acqf(
            acqf,
            bounds=bounds,
            q=1,
            gen_candidates=gen_candidates_scipy,
            **options,
        )

        # Observe next Ys
        Ys[n_prev:n] = evaluate_y(test_function, Xs[n_prev:n]).unsqueeze(-1)

        with torch.no_grad():
            x_new = Xs[n_prev:n]  # shape (1, d)
            y_new = Ys[n_prev:n, 0]  # shape (1,)

            # See if we have a new best observation
            y_curr = y_new.max().item()
            if y_curr > y_max:
                y_max = y_curr
                x_max = x_new[y_new.argmax().item()].clone()

            # Boundary-seeking diagnostic (same TOL as the Sobol phase above)
            dist_new = torch.minimum(x_new, 1.0 - x_new)
            on_boundary_new = dist_new < TOL

            # Update progress bar
            iter_stats = OrderedDict(
                y_max=y_max,
                y_curr=y_curr,
            )
            pbar.set_postfix(**{stat: iter_stats.get(stat, None) for stat in tqdm_log_list})

            # Record this iteration's diagnostics
            history["best_curve"].append(y_max)
            history["best_x"].append(x_max.cpu().tolist())
            history["boundary_pct"].append(100.0 * on_boundary_new.double().mean().item())
            history["per_dim_boundary_history"].append(on_boundary_new.squeeze(0).cpu().tolist())
            history["per_dim_dist_history"].append(dist_new.squeeze(0).cpu().tolist())
            history["acquired_X"].append(x_new.squeeze(0).cpu().tolist())
            history["ard_weights_history"].append(ard_values)
            history["ard_param_name"].append(ard_param_name)
            history["ard_is_per_dim"].append(ard_is_per_dim)

        x_ref = x_new.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            posterior_mean = model.posterior(x_ref).mean.squeeze()
            (slope_grad,) = torch.autograd.grad(posterior_mean, x_ref)
        history["posterior_slope_history"].append(slope_grad.squeeze(0).cpu().tolist())

        n_prev = n
        pbar.update(1)

    pbar.close()

    logging.info(f"Best observation: {y_max}")

    history["X"] = Xs.cpu().tolist()
    history["Y"] = Ys[:, 0].cpu().tolist()

    try:
        base_dir = Path(hydra.utils.get_original_cwd())
    except ValueError:
        base_dir = Path.cwd()
    results_dir = base_dir / "results" / history["benchmark"]
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"results_{kernel_type}_seed{config.seed}.json"
    with open(results_path, "w") as f:
        json.dump(history, f)
    logging.info(f"Saved iteration history to {results_path}")

    return history
