#!/usr/bin/env python3
"""Conservation-aware training budget tracker.

Implements γ+H=C (productive + wasted = total compute) for GPU training budgets.
Tracks productive GPU hours (γ), wasted hours from overfitting/restart (H),
and total budget (C). Provides spectral feedback via eigenvalue decomposition
of the loss landscape to detect overfitting early.

Usage:
    from si_integration.training_budget import TrainingBudget

    budget = TrainingBudget(total_gpu_hours=8.0, gamma_target=0.80)
    budget.record_run(gpu_hours=1.5, productive_ratio=0.85, losses=[2.1, 1.8, 1.6])
    budget.can_afford(estimate=2.0)  # True/False
    budget.report()
"""

import json
import math
from pathlib import Path


class TrainingBudget:
    """Track GPU training budget under conservation law γ+H=C."""

    def __init__(self, total_gpu_hours, gamma_target=0.80):
        self.total = total_gpu_hours  # C: total budget
        self.gamma_target = gamma_target  # target productive ratio
        self.gamma = 0.0  # productive GPU hours
        self.h = 0.0  # wasted GPU hours (overfit, restarts, bad hyperparams)
        self.runs = []

    @property
    def consumed(self):
        """Total consumed hours (γ + H)."""
        return self.gamma + self.h

    @property
    def remaining(self):
        """Remaining budget (C - γ - H)."""
        return self.total - self.consumed

    @property
    def productive_ratio(self):
        """γ / (γ + H), or 0 if no runs yet."""
        if self.consumed == 0:
            return 0.0
        return self.gamma / self.consumed

    def can_afford(self, estimate):
        """Check if a proposed run fits the remaining budget."""
        return estimate <= self.remaining

    def estimate_run(self, depth, dataset_size, epochs=1):
        """Rough GPU-hour estimate for a training run.
        
        Heuristic: ~0.001 GPU-hours per 1M tokens per layer per epoch.
        Calibrated for single H100. Adjust as needed.
        """
        tokens = dataset_size * 128  # assume ~128 tokens per sample
        token_millions = tokens / 1_000_000
        return token_millions * depth * epochs * 0.001

    def record_run(self, gpu_hours, productive_ratio, losses=None):
        """Record a completed training run.

        Args:
            gpu_hours: Wall-clock GPU hours consumed.
            productive_ratio: Fraction of compute that was productive (0-1).
                              Low ratio = overfitting, bad configs, restarts.
            losses: Optional list of per-step/epoch loss values for spectral analysis.
        """
        gamma_run = gpu_hours * productive_ratio
        h_run = gpu_hours * (1 - productive_ratio)
        self.gamma += gamma_run
        self.h += h_run

        run_record = {
            "gpu_hours": gpu_hours,
            "gamma": gamma_run,
            "h": h_run,
            "productive_ratio": productive_ratio,
        }

        if losses:
            spectral = self._spectral_feedback(losses)
            run_record["spectral"] = spectral

        self.runs.append(run_record)

        # Alert if H is growing faster than expected
        if self.productive_ratio < self.gamma_target - 0.10:
            self._alert_overfit(run_record)

        return run_record

    def _spectral_feedback(self, losses):
        """Eigenvalue decomposition of loss landscape proxy.

        Uses the autocorrelation matrix of consecutive loss differences
        to estimate the spectral structure. Large eigenvalue spread
        indicates a rough loss landscape (potential overfitting).
        """
        if len(losses) < 3:
            return {"status": "insufficient_data"}

        # Compute loss differences (proxy for gradient direction changes)
        diffs = [losses[i+1] - losses[i] for i in range(len(losses) - 1)]
        n = len(diffs)

        # Simple 2x2 autocorrelation matrix of diffs
        # [ E[x_t^2]    E[x_t * x_{t+1}] ]
        # [ E[x_t*x_{t+1}]  E[x_{t+1}^2]  ]
        mean_diff = sum(diffs) / n
        centered = [d - mean_diff for d in diffs]

        if n < 2:
            return {"status": "insufficient_data"}

        e_xx = sum(c * c for c in centered) / n
        e_xy = sum(centered[i] * centered[i+1] for i in range(n-1)) / (n - 1)

        # Eigenvalues of [[e_xx, e_xy], [e_xy, e_xx]]
        # λ = e_xx ± e_xy
        lambda_max = e_xx + abs(e_xy)
        lambda_min = abs(e_xx - abs(e_xy))
        condition_number = lambda_max / lambda_min if lambda_min > 1e-10 else float("inf")

        status = "healthy"
        if condition_number > 100:
            status = "rough_landscape"
        elif condition_number > 50:
            status = "warning"

        return {
            "lambda_max": round(lambda_max, 6),
            "lambda_min": round(lambda_min, 6),
            "condition_number": round(condition_number, 2),
            "status": status,
        }

    def _alert_overfit(self, run):
        """Emit an overfitting alert."""
        print(f"⚠️  OVERFIT ALERT: productive ratio {run['productive_ratio']:.0%} "
              f"below target {self.gamma_target:.0%}")
        print(f"   H (wasted): {self.h:.2f}h / γ (productive): {self.gamma:.2f}h "
              f"/ C (total): {self.total:.2f}h")
        if "spectral" in run and isinstance(run["spectral"], dict):
            print(f"   Spectral: {run['spectral'].get('status', 'unknown')} "
                  f"(κ={run['spectral'].get('condition_number', '?')})")

    def report(self):
        """Print a budget summary."""
        print(f"\n{'='*50}")
        print(f"  Training Budget: γ + H = C")
        print(f"{'='*50}")
        print(f"  γ (productive):  {self.gamma:7.2f}h  ({self.productive_ratio:.0%})")
        print(f"  H (wasted):      {self.h:7.2f}h  ({1-self.productive_ratio:.0%})")
        print(f"  C (total):       {self.total:7.2f}h")
        print(f"  Remaining:       {self.remaining:7.2f}h")
        print(f"  Target γ ratio:  {self.gamma_target:.0%}")
        print(f"  Runs completed:  {len(self.runs)}")
        print(f"{'='*50}")

    def save(self, path):
        """Save budget state to JSON."""
        state = {
            "total": self.total,
            "gamma_target": self.gamma_target,
            "gamma": self.gamma,
            "h": self.h,
            "runs": self.runs,
        }
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path):
        """Load budget state from JSON."""
        state = json.loads(Path(path).read_text())
        budget = cls(state["total"], state["gamma_target"])
        budget.gamma = state["gamma"]
        budget.h = state["h"]
        budget.runs = state["runs"]
        return budget
