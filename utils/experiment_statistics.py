"""Seed-level uncertainty: fixed data split, never pixels as independent trials."""
import itertools
import numpy as np


def bootstrap_mean(values, *, repeats=20000, seed=20260929):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need finite one-dimensional observations")
    if repeats < 1000:
        raise ValueError("Use at least 1000 bootstrap resamples")
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def paired_seed_comparison(reference, comparator, *, repeats=20000, seed=20260929):
    if set(reference) != set(comparator) or len(reference) < 2:
        raise ValueError("Paired comparison requires the same complete set of at least two seeds")
    seeds = sorted(reference)
    differences = np.array([reference[s] - comparator[s] for s in seeds], dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Non-finite seed metric")
    observed = abs(differences.mean())
    n = len(seeds)
    if n <= 16:
        signs = np.array(list(itertools.product((-1., 1.), repeat=n)))
        permuted = np.abs((signs * differences).mean(1))
        p = float(np.mean(permuted >= observed - 1e-12))
        method = "exact_two_sided_paired_sign_flip"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1., 1.], size=(repeats, n))
        p = float((1 + np.sum(np.abs((signs * differences).mean(1)) >= observed - 1e-12)) / (repeats + 1))
        method = "monte_carlo_two_sided_paired_sign_flip_plus_one"
    return {"n": n, "seeds": seeds, "differences_reference_minus_comparator": differences.tolist(),
            "mean_difference": float(differences.mean()), "sample_std_difference": float(differences.std(ddof=1)),
            "marginal_bootstrap_ci95": bootstrap_mean(differences, repeats=repeats, seed=seed),
            "p_value": p, "test": method,
            "assumption": "Under the null, within-seed differences are sign-exchangeable; independent seed blocks on one fixed split."}


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("p values must lie in [0,1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty_like(p)
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, (len(p) - rank) * p[index])
        adjusted[index] = min(previous, 1.)
    return adjusted.tolist()
