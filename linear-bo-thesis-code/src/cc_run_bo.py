import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import gpytorch
import botorch
import csv
from contextlib import ExitStack
from gpytorch.constraints import GreaterThan
from gpytorch.priors import LogNormalPrior
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement as LogEI
from botorch.generation import gen_candidates_scipy
from botorch.optim import optimize_acqf
import causalchamber.lab as lab
from optimize import build_kernel  # reuse your own kernel definitions directly

# ============================================================
# Configuration -- change these three things as needed
# ============================================================
KERNEL_TYPE = 'linear'  # or 'spherical/linear'
SEED = 4
TARGET_MODE = 'center'  # or 'boundary/center' -- run capture_target_boundary.py / capture_target_center.py first

CHAMBER_ID = 'lt-demo-ch4lu'
CONFIG = 'led_matrix'
CREDENTIALS_FILE = '.credentials'

N_INIT = 30
N_BO = 470

TARGETS_DIR = os.path.join('cc', 'targets')
RUN_DIR = os.path.join('cc', 'results', f'{KERNEL_TYPE}_{TARGET_MODE}_seed{SEED}')
os.makedirs(RUN_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(RUN_DIR, 'results.csv')

# ============================================================
# Setup
# ============================================================
torch.manual_seed(SEED)

led_names = [f'{c}_{i}' for i in range(37) for c in ['red', 'green', 'blue']]
d = len(led_names)  # 111


target_image = np.load(os.path.join(TARGETS_DIR, f'cc_target_{TARGET_MODE}.npy')).astype(np.float64)

chamber = lab.Chamber(chamber_id=CHAMBER_ID, config=CONFIG, credentials_file=CREDENTIALS_FILE)


def evaluate(x_norm):

    x_raw = (x_norm.numpy() * 255).clip(0, 255)
    x_raw = np.round(x_raw).astype(int)  # the LED variables are integer-only on the server side

    batch = chamber.new_batch()
    for name, val in zip(led_names, x_raw):
        batch.set(name, int(val))
    batch.measure(n=1)
    df, [image] = batch.submit()

    l2_dist = float(np.linalg.norm(image.astype(np.float64) - target_image))

    return -l2_dist, x_raw, image


def calculate_candidate(train_X, train_Y):

    covar = build_kernel(kernel_type=KERNEL_TYPE, d=d)
    mean = gpytorch.means.ConstantMean()
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior := LogNormalPrior(loc=-4.0, scale=1.0),
        noise_constraint=GreaterThan(1e-4, initial_value=noise_prior.mode),
    )
    gp = SingleTaskGP(
        train_X, standardize(train_Y),
        mean_module=mean, covar_module=covar, likelihood=likelihood,
    )

    gp.train()
    botorch_optimizer = botorch.fit.fit_gpytorch_mll_scipy
    with ExitStack() as es:
        es.enter_context(gpytorch.settings.cholesky_max_tries(10))
        es.enter_context(gpytorch.settings.max_cholesky_size(float("inf")))
        es.enter_context(gpytorch.settings.fast_computations(
            log_prob=True, covar_root_decomposition=False, solves=False
        ))
        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        fit_gpytorch_mll(mll, optimizer=botorch_optimizer)
    gp.eval()

    bounds = torch.stack([torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64)])
    acqf = LogEI(gp, standardize(train_Y).max())
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
    candidate, _ = optimize_acqf(
        acqf, bounds=bounds, q=1, gen_candidates=gen_candidates_scipy, **options,
    )
    return candidate.squeeze()


# ============================================================
# Main loop
# ============================================================
def run():
    fieldnames = ['iteration', 'phase', 'objective', 'best_so_far', 'x_raw']
    csvfile = open(OUTPUT_CSV, 'w', newline='')  # overwrite -- each subfolder is one run, always fresh
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    X_list, Y_list = [], []
    best_so_far = -np.inf
    best_x_raw = None
    best_image = None

    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=SEED)
    for i in range(N_INIT):
        x_norm = sobol.draw(1).squeeze().to(torch.float64)
        y, x_raw, image = evaluate(x_norm)
        X_list.append(x_norm)
        Y_list.append(y)
        if y > best_so_far:
            best_so_far = y
            best_x_raw = x_raw
            best_image = image

        writer.writerow({
            'iteration': i, 'phase': 'sobol', 'objective': y,
            'best_so_far': best_so_far, 'x_raw': x_raw.tolist(),
        })
        csvfile.flush()
        print(f"[Sobol {i}] objective={y:.3f}  best_so_far={best_so_far:.3f}")

    for i in range(N_BO):
        train_X = torch.stack(X_list)
        train_Y = torch.tensor(Y_list, dtype=torch.float64).unsqueeze(-1)
        candidate = calculate_candidate(train_X, train_Y)

        y, x_raw, image = evaluate(candidate)
        X_list.append(candidate)
        Y_list.append(y)
        if y > best_so_far:
            best_so_far = y
            best_x_raw = x_raw
            best_image = image

        writer.writerow({
            'iteration': N_INIT + i, 'phase': 'bo', 'objective': y,
            'best_so_far': best_so_far, 'x_raw': x_raw.tolist(),
        })
        csvfile.flush()
        print(f"[BO {i}] objective={y:.3f}  best_so_far={best_so_far:.3f}")

    csvfile.close()

    np.save(os.path.join(RUN_DIR, 'best_image.npy'), best_image)
    np.save(os.path.join(RUN_DIR, 'target_image.npy'), target_image.astype(np.uint8))

    print(f"\nDone. Everything for this run saved in '{RUN_DIR}/'.")
    print(f"Best objective found: {best_so_far:.3f}")
    print(f"Best LED values (raw, 0-255):\n{best_x_raw}")


if __name__ == "__main__":
    run()
