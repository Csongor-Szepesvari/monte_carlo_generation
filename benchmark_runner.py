"""
Benchmark runner for comparing methods to estimate the sum of top-k order statistics.

New benchmarking dimensions to explore:
- Vary the number of categories beyond 4 (e.g., 4, 8, 16, 32, ...)
- Vary the number of admittees (top-k) across: 100, 250, 500, 750, 1000, 2500
- Measure the time required to find the optimal allocation under each setting
  for multiple estimators (approximate NN and Monte Carlo baselines).

This suite supports two modes:
1) Fixed 4-category scenarios from benchmark_config.enumerate_runs (legacy mode)
2) Variable-category scenarios (this work), controlled via CLI flags to sweep
   category counts and admittee targets, and to benchmark optimization runtimes.

Features:
- Iterates scenarios (legacy or variable-category mode)
- Generates a partition of total_n into N groups (biased or unbiased)
- Benchmarks:
    * Adaptive Monte Carlo (4-cat legacy or generic for N categories)
    * Neural model prediction using scripted or trained model
    * Optional: Train a neural model (once or per scenario) and record training + inference times
    * Optional: Optimize allocation (randomized search) and time entire optimization loop
- Logs per-scenario CSV rows including method, timings (training vs inference vs optimization),
  and errors vs a high-fidelity reference if requested.

Assumption: The number of admittees equals k (top-k). Total sample budget used to
generate candidate allocations (n_parts) is set via a multiplier of k and can be
adjusted with --sample-budget-multiplier.

Usage example:
    # Legacy 4-cat scenarios + NN inference
    python benchmark_runner.py --methods mc,nn_predict --output benchmarks.csv --limit 50 --use-biased-partitions

    # Variable categories + optimization timing
    python benchmark_runner.py --variable-categories --category-counts 4,8,16 --admittees 100,250,500 \
        --methods nn_opt,mc_opt --opt-num-candidates 200 --sample-budget-multiplier 4 --output benchmarks/benchmarks_varcat.csv

Notes:
- Training time is recorded separately from inference time. When reusing a pre-trained model,
  training_time_s is 0 and training_scope is "skipped".
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional
from itertools import product

import numpy as np
import pandas as pd
import torch

from benchmark_config import DEFAULT_GRID, enumerate_runs
from monte_carlo_generation import (
    stars_and_bars_partition,
    biased_stars_and_bars_partition,
    monte_carlo_adaptive_estimate,
)


# ------------------------ variable-category scenario enum ---------------------

def enumerate_variable_category_scenarios(
    category_counts: List[int],
    admittees_list: List[int],
    epsilon_values: List[float],
    seeds: List[int],
    regime_names: List[str],
):
    # Define a few simple regimes programmatically for N categories
    def build_regime(name: str, num_cats: int):
        if name == "balanced":
            mu = [0.0] * num_cats
            sigma = [1.0] * num_cats
        elif name == "mean_skew":
            mu = [1.0] + [0.0] * (num_cats - 1)
            sigma = [1.0] * num_cats
        elif name == "variance_skew":
            mu = [0.0] * num_cats
            sigma = [2.0] + [1.0] * (num_cats - 1)
        else:
            # default balanced
            mu = [0.0] * num_cats
            sigma = [1.0] * num_cats
        return mu, sigma

    for num_cats, k, eps, seed, reg in product(category_counts, admittees_list, epsilon_values, seeds, regime_names):
        mu, sigma = build_regime(reg, num_cats)
        yield {
            "regime_name": reg,
            "mu": mu,
            "sigma": sigma,
            "k": int(k),
            "epsilon": float(eps),
            "seed": int(seed),
        }


# ------------------------------ utility helpers ------------------------------

def time_function(callable_fn):
    start = time.time()
    result = callable_fn()
    elapsed = time.time() - start
    return result, float(elapsed)


def ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# -------------------------- reference / error metrics -------------------------

def high_fidelity_reference(mu: List[float], sigma: List[float], n_parts: List[int], k: int, epsilon: float) -> float:
    # Use a very small epsilon as a high-fidelity Monte Carlo reference
    if len(mu) == 4:
        topk_means, _ = monte_carlo_adaptive_estimate(mu, sigma, n_parts, k, epsilon=min(0.002, epsilon / 5))
        return float(topk_means[k])
    # Generic fallback for N categories
    est, _ = adaptive_mc_generic(mu, sigma, n_parts, k, epsilon=min(0.002, epsilon / 5))
    return float(est)


def compute_error_metrics(estimate: float, reference: Optional[float]) -> Dict[str, Optional[float]]:
    if reference is None:
        return {"abs_error": None, "rel_error": None}
    abs_err = float(abs(estimate - reference))
    rel_err = float(abs_err / reference) if reference != 0 else None
    return {"abs_error": abs_err, "rel_error": rel_err}


# ------------------------------ neural methods -------------------------------

def load_scripted_model(scripted_path: str) -> Optional[torch.jit.ScriptModule]:
    if os.path.exists(scripted_path):
        model = torch.jit.load(scripted_path)
        model.eval()
        return model
    return None


def nn_predict(mu: List[float], sigma: List[float], n_parts: List[int], k: int, model: torch.nn.Module) -> float:
    total_n = int(np.sum(n_parts))
    x_np = np.array([[mu[i], sigma[i], n_parts[i]] for i in range(4)], dtype=np.float32)
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)  # [1,4,3]
    k_ratio = torch.tensor([[k / total_n if total_n > 0 else 0.0]], dtype=torch.float32)
    with torch.no_grad():
        y = model(x, k_ratio).item()
    return float(y)


from typing import Tuple


def nn_train_and_return_model(
    csv_path: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    patience: int = 5,
) -> Tuple[torch.nn.Module, float]:
    from top_k_predictor import (
        DeepSetsTopKModel,
        TopKDataset,
        train as train_one_epoch,
        evaluate as evaluate_model,
    )
    from torch.utils.data import DataLoader
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepSetsTopKModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    loss_fn = nn.MSELoss()

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_loader = DataLoader(TopKDataset(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TopKDataset(val_df), batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    start_time = time.time()

    for _ in range(epochs):
        train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, *_ = evaluate_model(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    training_time_s = float(time.time() - start_time)
    model.eval()
    return model, training_time_s


# --------------------------- Generic adaptive MC (N) --------------------------

from typing import Tuple


def adaptive_mc_generic(
    mu: List[float],
    sigma: List[float],
    n_parts: List[int],
    k: int,
    epsilon: float = 0.01,
    max_rep: int = 10000,
    seed: Optional[int] = None,
) -> Tuple[float, int]:
    rng = np.random.default_rng(seed)

    def one_replication() -> float:
        all_samples = []
        for mean, std, n in zip(mu, sigma, n_parts):
            draws = int(n)
            if draws > 0:
                samples = mean + std * rng.standard_normal(draws)
                all_samples.append(samples)
        if not all_samples:
            return 0.0
        combined = np.concatenate(all_samples)
        top_k = np.sort(combined)[-k:] if len(combined) >= k else combined
        return float(np.sum(top_k))

    R = 10
    values = np.array([one_replication() for _ in range(R)], dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if R > 1 else 0.0
    se = std / np.sqrt(R) if R > 0 else np.inf
    rel_error = se / mean if mean != 0 else np.inf

    while rel_error > epsilon and R < max_rep:
        more = np.array([one_replication() for _ in range(R)], dtype=np.float64)
        values = np.concatenate([values, more])
        R = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if R > 1 else 0.0
        se = std / np.sqrt(R) if R > 0 else np.inf
        rel_error = se / mean if mean != 0 else np.inf

    return float(values.mean()), int(R)


# ------------------------------ MC baseline run ------------------------------

def run_mc_adaptive(mu: List[float], sigma: List[float], n_parts: List[int], k: int, epsilon: float) -> Dict[str, float]:
    def _call():
        topk_means, num_reps = monte_carlo_adaptive_estimate(mu, sigma, n_parts, k, epsilon=epsilon)
        return float(topk_means[k]), int(num_reps)

    # Use legacy 4-cat adaptive MC when possible; otherwise generic N-cat version
    if len(mu) == 4:
        (estimate, num_reps), elapsed = time_function(_call)
        return {"estimate": estimate, "runtime_s": elapsed, "num_reps": num_reps}

    def _call_generic():
        return adaptive_mc_generic(mu, sigma, n_parts, k, epsilon=epsilon, seed=None)

    (estimate, num_reps), elapsed = time_function(_call_generic)
    return {"estimate": estimate, "runtime_s": elapsed, "num_reps": num_reps}


# -------------------------------- main runner --------------------------------

def run_benchmarks(args) -> None:
    rng = np.random.default_rng(args.seed)
    results: List[Dict] = []

    # Optionally load a scripted model once for nn_predict
    scripted_model = None
    if "nn_predict" in args.methods:
        scripted_model = load_scripted_model(args.scripted_model_path)
        if scripted_model is None:
            print(f"Warning: scripted model not found at {args.scripted_model_path}; nn_predict will be skipped.")

    # Optionally train a model once for reuse across scenarios
    global_model = None
    global_training_time_s = 0.0
    if args.train_model_once and ("nn_predict" in args.methods or "nn_train_predict" in args.methods):
        if not os.path.exists(args.training_csv):
            raise FileNotFoundError(f"Training CSV not found: {args.training_csv}")
        global_model, global_training_time_s = nn_train_and_return_model(
            csv_path=args.training_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
        )

    # Determine scenario iterator: legacy or variable categories
    if args.variable_categories:
        scenarios = enumerate_variable_category_scenarios(
            category_counts=args.category_counts,
            admittees_list=args.admittees,
            epsilon_values=args.epsilon_values,
            seeds=[args.seed],
            regime_names=args.regimes,
        )
    else:
        scenarios = enumerate_runs()

    for idx, scenario in enumerate(scenarios):
        if args.limit is not None and idx >= args.limit:
            break

        mu = scenario["mu"]
        sigma = scenario["sigma"]
        k = scenario["k"]
        epsilon = scenario["epsilon"]

        # Decide total sample budget for allocations (n_parts)
        total_n = scenario.get(
            "total_n",
            int(round(args.sample_budget_multiplier * k)) if args.variable_categories else scenario["total_n"],
        )

        # Partition selection
        use_biased = args.use_biased_partitions and (rng.random() < 0.5)
        if use_biased:
            n_parts = biased_stars_and_bars_partition(
                total_n,
                k=len(mu),
                rng=rng,
                bias_factor=scenario.get("bias_factor", 2.0),
            )
            partition_type = "biased"
        else:
            n_parts = stars_and_bars_partition(total_n, k=len(mu), rng=rng)
            partition_type = "unbiased"

        # Reference (optional)
        reference_value = None
        if args.compute_reference:
            reference_value = high_fidelity_reference(mu, sigma, n_parts, k, epsilon)

        # Adaptive MC baseline
        if "mc" in args.methods:
            mc_out = run_mc_adaptive(mu, sigma, n_parts, k, epsilon)
            mc_row = {
                **{"method": "mc_adaptive"},
                **scenario,
                "partition_type": partition_type,
                **{f"n{i+1}": int(n_parts[i]) for i in range(len(n_parts))},
                "estimate": mc_out["estimate"],
                "runtime_s": mc_out["runtime_s"],
                "num_reps": mc_out["num_reps"],
                "training_time_s": 0.0,
                "training_scope": "none",
            }
            mc_row.update(compute_error_metrics(mc_row["estimate"], reference_value))
            results.append(mc_row)

        # NN predict-only (reuse scripted or globally trained model)
        if "nn_predict" in args.methods and (scripted_model is not None or global_model is not None):
            active_model = global_model if global_model is not None else scripted_model

            def _predict_call():
                return nn_predict(mu, sigma, n_parts, k, active_model)

            estimate, infer_time = time_function(_predict_call)
            nn_row = {
                **{"method": "nn_predict"},
                **scenario,
                "partition_type": partition_type,
                **{f"n{i+1}": int(n_parts[i]) for i in range(len(n_parts))},
                "estimate": float(estimate),
                "runtime_s": float(infer_time),
                "num_reps": None,
                "training_time_s": float(global_training_time_s if global_model is not None else 0.0),
                "training_scope": "global" if global_model is not None else "skipped",
            }
            nn_row.update(compute_error_metrics(nn_row["estimate"], reference_value))
            results.append(nn_row)

        # NN train+predict per scenario
        if "nn_train_predict" in args.methods:
            if not os.path.exists(args.training_csv):
                raise FileNotFoundError(f"Training CSV not found: {args.training_csv}")

            def _train():
                return nn_train_and_return_model(
                    csv_path=args.training_csv,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    patience=args.patience,
                )

            (model_s, train_time_s), _ = time_function(_train)

            def _predict_after_train():
                return nn_predict(mu, sigma, n_parts, k, model_s)

            estimate, infer_time = time_function(_predict_after_train)
            row = {
                **{"method": "nn_train_predict"},
                **scenario,
                "partition_type": partition_type,
                **{f"n{i+1}": int(n_parts[i]) for i in range(len(n_parts))},
                "estimate": float(estimate),
                "runtime_s": float(infer_time),
                "num_reps": None,
                "training_time_s": float(train_time_s),
                "training_scope": "per_scenario",
            }
            row.update(compute_error_metrics(row["estimate"], reference_value))
            results.append(row)

        # Optimization over allocations: time to find the best allocation
        if "nn_opt" in args.methods or "mc_opt" in args.methods:
            num_cands = int(args.opt_num_candidates)
            # Draw candidate allocations via unbiased partitions
            cand_parts = [stars_and_bars_partition(total_n, k=len(mu), rng=rng) for _ in range(num_cands)]

            if "nn_opt" in args.methods and (scripted_model is not None or global_model is not None):
                active_model = global_model if global_model is not None else scripted_model

                def _optimize_nn():
                    best_val = -float("inf")
                    for cand in cand_parts:
                        val = nn_predict(mu, sigma, cand, k, active_model)
                        if val > best_val:
                            best_val = val
                    return best_val

                best_estimate, opt_time = time_function(_optimize_nn)
                nn_opt_row = {
                    **{"method": "nn_opt"},
                    **scenario,
                    "partition_type": "opt_random",
                    "estimate": float(best_estimate),
                    "runtime_s": float(opt_time),
                    "num_reps": None,
                    "training_time_s": float(global_training_time_s if global_model is not None else 0.0),
                    "training_scope": "global" if global_model is not None else "skipped",
                    "opt_candidates": num_cands,
                }
                nn_opt_row.update(compute_error_metrics(nn_opt_row["estimate"], reference_value))
                results.append(nn_opt_row)

            if "mc_opt" in args.methods:
                def _optimize_mc():
                    best_val = -float("inf")
                    for cand in cand_parts:
                        est, _ = adaptive_mc_generic(mu, sigma, cand, k, epsilon=epsilon)
                        if est > best_val:
                            best_val = est
                    return best_val

                best_estimate, opt_time = time_function(_optimize_mc)
                mc_opt_row = {
                    **{"method": "mc_opt"},
                    **scenario,
                    "partition_type": "opt_random",
                    "estimate": float(best_estimate),
                    "runtime_s": float(opt_time),
                    "num_reps": None,
                    "training_time_s": 0.0,
                    "training_scope": "none",
                    "opt_candidates": num_cands,
                }
                mc_opt_row.update(compute_error_metrics(mc_opt_row["estimate"], reference_value))
                results.append(mc_opt_row)

        # Progress
        if (idx + 1) % max(1, (args.limit or 100) // 10) == 0:
            print(f"Progress: {idx+1} scenarios processed")

    # Save results
    ensure_dir(args.output)
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Saved benchmark results to {args.output} with {len(results)} rows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runner for top-k estimators")
    parser.add_argument("--methods", type=str, default="mc,nn_predict",
                        help="Comma-separated methods: mc,nn_predict,nn_train_predict")
    parser.add_argument("--output", type=str, default="benchmarks/benchmarks.csv")
    parser.add_argument("--limit", type=int, default=50, help="Limit number of scenarios (for quick runs)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-biased-partitions", action="store_true")
    parser.add_argument("--compute-reference", action="store_true", help="Compute a high-fidelity reference via tight MC")

    # NN options
    parser.add_argument("--scripted-model-path", type=str, default="runs/topk_experiment/best_model_biased.pt.jit")
    parser.add_argument("--training-csv", type=str, default="mixed_monte_carlo_topk.csv")
    parser.add_argument("--train-model-once", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)

    # Variable-category mode controls
    parser.add_argument("--variable-categories", action="store_true", help="Enable variable-category scenarios")
    parser.add_argument("--category-counts", type=str, default="4,8,16", help="Comma-separated category counts")
    parser.add_argument("--admittees", type=str, default="100,250,500,750,1000,2500", help="Comma-separated admittee (k) targets")
    parser.add_argument("--epsilon-values", type=str, default="0.05,0.02,0.01", help="Comma-separated epsilons for MC targets")
    parser.add_argument("--regimes", type=str, default="balanced,mean_skew,variance_skew", help="Comma-separated regime names")
    parser.add_argument("--sample-budget-multiplier", type=float, default=4.0, help="Total_n = multiplier * k in variable-category mode")

    # Optimization controls
    parser.add_argument("--opt-num-candidates", type=int, default=200, help="Number of random candidate allocations to evaluate in optimization")

    args = parser.parse_args()
    args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if args.variable_categories:
        args.category_counts = [int(x) for x in args.category_counts.split(",") if x.strip()]
        args.admittees = [int(x) for x in args.admittees.split(",") if x.strip()]
        args.epsilon_values = [float(x) for x in args.epsilon_values.split(",") if x.strip()]
        args.regimes = [x.strip() for x in args.regimes.split(",") if x.strip()]
    return args


if __name__ == "__main__":
    run_benchmarks(parse_args())


