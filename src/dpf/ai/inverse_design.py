from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from dpf.ai.surrogate import DPFSurrogate

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from scipy.optimize import differential_evolution

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)


@dataclass
class InverseResult:
    """
    Result from inverse design optimization.

    Args:
        best_params: Optimal parameter configuration found
        best_score: Objective function value at optimum (lower is better)
        all_trials: List of (params, score) tuples for all trials
        n_trials: Total number of trials evaluated
    """

    best_params: dict[str, float] = field(default_factory=dict)
    best_score: float = float("inf")
    all_trials: list[tuple[dict[str, float], float]] = field(default_factory=list)
    n_trials: int = 0


class InverseDesigner:
    """
    Inverse design optimization using DPF surrogate model.

    Finds DPF device configurations that achieve target plasma performance metrics
    (max density, temperature, etc.) subject to operational constraints.

    Args:
        surrogate: Trained DPFSurrogate model for predictions
        parameter_ranges: Dict mapping parameter names to (min, max) bounds
    """

    def __init__(
        self,
        surrogate: DPFSurrogate,
        parameter_ranges: dict[str, tuple[float, float]],
    ) -> None:
        self.surrogate = surrogate
        self.parameter_ranges = parameter_ranges

        logger.info(
            f"InverseDesigner initialized with {len(parameter_ranges)} parameters"
        )

    def find_config(
        self,
        targets: dict[str, float],
        constraints: dict[str, float] | None = None,
        method: str = "bayesian",
        n_trials: int = 100,
        seed: int = 42,
    ) -> InverseResult:
        """
        Find device configuration matching target metrics.

        Args:
            targets: Dict of target values for metrics (e.g., {"max_rho": 1e-3})
            constraints: Optional dict of maximum constraint values
                         (e.g., {"max_Te": 100.0})
            method: Optimization method ("bayesian" or "evolutionary")
            n_trials: Number of optimization trials
            seed: Random seed for reproducibility

        Returns:
            InverseResult with optimal parameters and optimization history

        Raises:
            ValueError: If method is invalid
        """
        logger.info(
            f"Starting inverse design with method={method}, n_trials={n_trials}"
        )
        logger.info(f"Targets: {targets}")
        if constraints:
            logger.info(f"Constraints: {constraints}")

        if method == "bayesian":
            return self._bayesian_search(targets, constraints, n_trials, seed)
        elif method == "evolutionary":
            return self._evolutionary_search(targets, constraints, n_trials, seed)
        else:
            raise ValueError(
                f"Invalid method: {method}. Choose 'bayesian' or 'evolutionary'"
            )

    def _objective(
        self,
        params: dict[str, float],
        targets: dict[str, float],
        constraints: dict[str, float] | None,
    ) -> float:
        """
        Objective function for optimization.

        Args:
            params: Parameter configuration to evaluate
            targets: Target metric values
            constraints: Maximum constraint values

        Returns:
            Scalar objective value (lower is better)
        """
        # Run surrogate prediction
        try:
            configs = [params]
            results = self.surrogate.parameter_sweep(configs, n_steps=50)

            if not results or "error" in results[0]:
                logger.warning(f"Surrogate prediction failed for params: {params}")
                return 1e10  # Large penalty for failed predictions

            predicted = results[0]

        except Exception as e:
            logger.error(f"Objective evaluation error: {e}")
            return 1e10

        # Compute target matching score
        score = 0.0
        for metric, target_val in targets.items():
            if metric not in predicted:
                logger.warning(f"Metric {metric} not in predictions")
                score += 1e6  # Large penalty for missing metrics
                continue

            pred_val = predicted[metric]
            # Normalized absolute error
            error = abs(pred_val - target_val) / max(abs(target_val), 1e-10)
            score += error

        # Add constraint violations
        if constraints:
            for metric, max_val in constraints.items():
                if metric not in predicted:
                    continue

                pred_val = predicted[metric]
                if pred_val > max_val:
                    # Quadratic penalty for constraint violation
                    violation = (pred_val - max_val) / max(abs(max_val), 1e-10)
                    score += 10.0 * violation**2

        return score

    def _bayesian_search(
        self,
        targets: dict[str, float],
        constraints: dict[str, float] | None,
        n_trials: int,
        seed: int,
    ) -> InverseResult:
        """
        Bayesian optimization using Optuna.

        Args:
            targets: Target metric values
            constraints: Maximum constraint values
            n_trials: Number of optimization trials
            seed: Random seed

        Returns:
            InverseResult with optimization results

        Raises:
            ImportError: If optuna is not available
        """
        if not HAS_OPTUNA:
            raise ImportError(
                "Optuna required for Bayesian search. Install with: pip install optuna"
            )

        # Suppress optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )

        # Define optuna objective
        def optuna_objective(trial: optuna.Trial) -> float:
            params = {}
            for param_name, (low, high) in self.parameter_ranges.items():
                params[param_name] = trial.suggest_float(param_name, low, high)

            return self._objective(params, targets, constraints)

        # Run optimization
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)

        # Collect results
        result = InverseResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_trials=[
                (trial.params, trial.value)
                for trial in study.trials
                if trial.value is not None
            ],
            n_trials=len(study.trials),
        )

        logger.info(f"Bayesian optimization complete: best_score={result.best_score}")
        logger.info(f"Best parameters: {result.best_params}")

        return result

    def _evolutionary_search(
        self,
        targets: dict[str, float],
        constraints: dict[str, float] | None,
        n_trials: int,
        seed: int,
    ) -> InverseResult:
        """
        Evolutionary optimization using scipy differential_evolution.

        Args:
            targets: Target metric values
            constraints: Maximum constraint values
            n_trials: Number of optimization trials (converted to maxiter)
            seed: Random seed

        Returns:
            InverseResult with optimization results

        Raises:
            ImportError: If scipy is not available
        """
        if not HAS_SCIPY:
            raise ImportError(
                "SciPy required for evolutionary search. "
                "Install with: pip install scipy"
            )

        # Build bounds array in consistent order
        param_names = sorted(self.parameter_ranges.keys())
        bounds = [self.parameter_ranges[name] for name in param_names]

        # Track all evaluations
        all_trials = []

        def scipy_objective(x: np.ndarray) -> float:
            params = dict(zip(param_names, x, strict=True))
            score = self._objective(params, targets, constraints)
            all_trials.append((params.copy(), score))
            return score

        # Run optimization
        result = differential_evolution(
            scipy_objective,
            bounds,
            maxiter=n_trials // 15,  # DE iterations (popsize=15 by default)
            seed=seed,
            disp=False,
            polish=True,
        )

        # Extract best parameters
        best_params = dict(zip(param_names, result.x, strict=True))

        inverse_result = InverseResult(
            best_params=best_params,
            best_score=result.fun,
            all_trials=all_trials,
            n_trials=len(all_trials),
        )

        logger.info(
            f"Evolutionary optimization complete: best_score={inverse_result.best_score}"
        )
        logger.info(f"Best parameters: {inverse_result.best_params}")

        return inverse_result
