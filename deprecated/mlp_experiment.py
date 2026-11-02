"""Reproducible PyTorch experiments for approximating a 2-D Gaussian PDF."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import shutil
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MU = np.array([0.0, 0.0], dtype=np.float64)
COVARIANCE = np.array([[0.3, -0.5], [-0.5, 2.0]], dtype=np.float64)
X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)
DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def gaussian_pdf(x: np.ndarray) -> np.ndarray:
    """Evaluate the specified bivariate Gaussian at rows of ``x``."""
    values = np.asarray(x, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("x must have shape (n_samples, 2)")
    centered = values - MU
    inverse = np.linalg.inv(COVARIANCE)
    exponent = -0.5 * np.einsum("ni,ij,nj->n", centered, inverse, centered)
    normalizer = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(COVARIANCE)))
    return normalizer * np.exp(exponent)


def generate_dataset(n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate uniform inputs in the assignment domain and exact PDF labels."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(seed)
    x = np.column_stack(
        [
            rng.uniform(*X1_RANGE, size=n_samples),
            rng.uniform(*X2_RANGE, size=n_samples),
        ]
    )
    return x.astype(np.float32), gaussian_pdf(x).astype(np.float32).reshape(-1, 1)


@dataclass(frozen=True)
class DatasetBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def create_datasets(
    train_size: int = 2000,
    validation_size: int = 400,
    test_size: int = 100,
    seed: int = DEFAULT_SEED,
) -> DatasetBundle:
    """Create independent, deterministic splits using disjoint RNG seeds."""
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("all split sizes must be positive")
    train_x, train_y = generate_dataset(train_size, seed)
    validation_x, validation_y = generate_dataset(validation_size, seed + 1)
    test_x, test_y = generate_dataset(test_size, seed + 2)
    return DatasetBundle(train_x, train_y, validation_x, validation_y, test_x, test_y)


def save_dataset(bundle: DatasetBundle, output_dir: Path) -> None:
    """Save all splits as CSV files with explicit split labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, x, y in (
        ("train", bundle.train_x, bundle.train_y),
        ("validation", bundle.validation_x, bundle.validation_y),
        ("test", bundle.test_x, bundle.test_y),
    ):
        with (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["x1", "x2", "y"])
            writer.writerows(np.column_stack([x, y]))


def save_dataset_summary(bundle: DatasetBundle, output_path: Path) -> None:
    """Save split sizes and min/max/mean/std values for x1, x2, and y."""
    rows: list[dict[str, float | int | str]] = []
    for name, x, y in (
        ("train", bundle.train_x, bundle.train_y),
        ("validation", bundle.validation_x, bundle.validation_y),
        ("test", bundle.test_x, bundle.test_y),
    ):
        values = np.column_stack([x, y])
        for index, column in enumerate(("x1", "x2", "y")):
            rows.append(
                {
                    "split": name,
                    "samples": len(x),
                    "variable": column,
                    "min": float(values[:, index].min()),
                    "max": float(values[:, index].max()),
                    "mean": float(values[:, index].mean()),
                    "std": float(values[:, index].std()),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MLP(nn.Module):
    """Configurable fully connected regressor with two inputs and one output."""

    def __init__(
        self,
        hidden_width: int = 10,
        hidden_depth: int = 1,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if hidden_width <= 0 or hidden_depth <= 0:
            raise ValueError("hidden_width and hidden_depth must be positive")
        activations: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        if activation not in activations:
            raise ValueError(f"activation must be one of {sorted(activations)}")
        layers: list[nn.Module] = [nn.Linear(2, hidden_width)]
        for _ in range(hidden_depth):
            layers.append(activations[activation]())
            if len(layers) < 2 * hidden_depth:
                layers.append(nn.Linear(hidden_width, hidden_width))
        layers.append(nn.Linear(hidden_width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class TrainConfig:
    hidden_width: int = 10
    hidden_depth: int = 1
    activation: str = "tanh"
    batch_size: int = 64
    epochs: int = 300
    learning_rate: float = 1e-3
    patience: int = 40
    seed: int = DEFAULT_SEED
    device: str = "cpu"


@dataclass
class TrainResult:
    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_loss: float
    training_seconds: float
    model: MLP


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_model(bundle: DatasetBundle, config: TrainConfig) -> TrainResult:
    """Train with explicit train/eval phases and restore the best validation checkpoint."""
    if config.epochs <= 0 or config.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    set_seed(config.seed)
    device = torch.device(config.device)
    model = MLP(config.hidden_width, config.hidden_depth, config.activation).to(device)
    train_loader = _loader(
        bundle.train_x, bundle.train_y, config.batch_size, True, config.seed
    )
    validation_loader = _loader(
        bundle.validation_x, bundle.validation_y, config.batch_size, False, config.seed
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(inputs)
            train_count += len(inputs)

        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                validation_total += criterion(model(inputs), targets).item() * len(inputs)
                validation_count += len(inputs)
        train_loss = train_total / train_count
        validation_loss = validation_total / validation_count
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_validation,
        training_seconds=time.perf_counter() - started,
        model=model,
    )


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, device: str = "cpu") -> dict[str, float]:
    """Compute MSE, MAE, and RMSE on an untouched split."""
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x).to(device)).cpu().numpy()
    error = prediction - y
    mse = float(np.mean(error**2))
    return {"mse": mse, "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(mse))}


def save_history(history: Iterable[dict[str, float | int]], path: Path) -> None:
    rows = list(history)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(rows)


def load_history(path: Path) -> list[dict[str, float | int]]:
    """Load a saved loss history for aggregate convergence plots."""
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "validation_loss": float(row["validation_loss"]),
            }
            for row in csv.DictReader(file)
        ]


def plot_dataset(bundle: DatasetBundle, output: Path) -> None:
    x1 = np.linspace(*X1_RANGE, 60)
    x2 = np.linspace(*X2_RANGE, 60)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    grid = np.column_stack([grid_x1.ravel(), grid_x2.ravel()])
    surface = gaussian_pdf(grid).reshape(grid_x1.shape)
    figure = plt.figure(figsize=(14, 6))
    scatter_axis = figure.add_subplot(121, projection="3d")
    scatter_axis.scatter(
        bundle.train_x[:, 0], bundle.train_x[:, 1], bundle.train_y[:, 0], s=5, alpha=0.35
    )
    scatter_axis.set(xlabel="x1", ylabel="x2", zlabel="Gaussian y", title="Training data")
    surface_axis = figure.add_subplot(122, projection="3d")
    surface_axis.plot_surface(grid_x1, grid_x2, surface, cmap="viridis", linewidth=0)
    surface_axis.set(xlabel="x1", ylabel="x2", zlabel="Gaussian y", title="True Gaussian surface")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_distributions(bundle: DatasetBundle, output: Path) -> None:
    """Plot the empirical distributions of x1, x2, and y for all splits."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, index, label in zip(axes, (0, 1, 0), ("x1", "x2", "y")):
        for name, x, y in (
            ("train", bundle.train_x, bundle.train_y),
            ("validation", bundle.validation_x, bundle.validation_y),
            ("test", bundle.test_x, bundle.test_y),
        ):
            values = np.column_stack([x, y])[:, index]
            axis.hist(values, bins=25, alpha=0.45, density=True, label=name)
        axis.set_xlabel(label)
        axis.set_ylabel("Density")
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle("Empirical distributions by split")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_history(history: Sequence[dict[str, float | int]], output: Path, title: str) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(epochs, [row["train_loss"] for row in history], label="train")
    axis.plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axis.set(xlabel="Epoch", ylabel="MSE loss", title=title)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_depth_convergence(
    histories: dict[int, Sequence[dict[str, float | int]]], output: Path
) -> None:
    """Compare validation-loss convergence for all depth configurations."""
    figure, axis = plt.subplots(figsize=(8, 5))
    for depth, history in sorted(histories.items()):
        axis.plot(
            [row["epoch"] for row in history],
            [row["validation_loss"] for row in history],
            label=f"{depth} hidden layer(s)",
        )
    axis.set(xlabel="Epoch", ylabel="Validation MSE", title="Experiment C: convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_predictions(model: nn.Module, bundle: DatasetBundle, output: Path, device: str) -> None:
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(bundle.test_x).to(device)).cpu().numpy()[:, 0]
    actual = bundle.test_y[:, 0]
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(actual, prediction, alpha=0.8)
    limits = [float(min(actual.min(), prediction.min())), float(max(actual.max(), prediction.max()))]
    axis.plot(limits, limits, "k--", label="perfect prediction")
    axis.set(xlabel="True y", ylabel="Predicted y", title="Test predictions")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_surface_comparison(
    model: nn.Module, output: Path, device: str, title: str = "True and approximated surfaces"
) -> None:
    """Compare the exact Gaussian surface with the selected model surface."""
    x1 = np.linspace(*X1_RANGE, 60)
    x2 = np.linspace(*X2_RANGE, 60)
    grid_x1, grid_x2 = np.meshgrid(x1, x2)
    grid = np.column_stack([grid_x1.ravel(), grid_x2.ravel()]).astype(np.float32)
    true_surface = gaussian_pdf(grid).reshape(grid_x1.shape)
    model.eval()
    with torch.no_grad():
        predicted_surface = (
            model(torch.from_numpy(grid).to(device)).cpu().numpy().reshape(grid_x1.shape)
        )
    figure = plt.figure(figsize=(14, 6))
    true_axis = figure.add_subplot(121, projection="3d")
    true_axis.plot_surface(grid_x1, grid_x2, true_surface, cmap="viridis", linewidth=0)
    true_axis.set(title="True Gaussian", xlabel="x1", ylabel="x2", zlabel="y")
    predicted_axis = figure.add_subplot(122, projection="3d")
    predicted_axis.plot_surface(grid_x1, grid_x2, predicted_surface, cmap="viridis", linewidth=0)
    predicted_axis.set(title="MLP approximation", xlabel="x1", ylabel="x2", zlabel="y")
    figure.suptitle(title)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_experiment_summaries(
    rows: Sequence[dict[str, float | int | str]], output_dir: Path
) -> None:
    """Write mean/std summary tables and the required A/B/C comparison plots."""
    groups: dict[str, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        groups[row["experiment"].split("_")[0]].append(row)
    summary_rows: list[dict[str, float | int | str]] = []
    for group_name, group_rows in groups.items():
        dimensions = {
            "A": ("train_size", "Training samples"),
            "B": ("hidden_width", "Hidden units"),
            "C": ("hidden_depth", "Hidden layers"),
        }
        x_key, x_label = dimensions[group_name]
        by_x: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
        for row in group_rows:
            by_x[int(row[x_key])].append(row)
        x_values = sorted(by_x)
        means = [
            float(np.mean([float(row["validation_mse"]) for row in by_x[x]])) for x in x_values
        ]
        stds = [
            float(np.std([float(row["validation_mse"]) for row in by_x[x]])) for x in x_values
        ]
        mae_means = [
            float(np.mean([float(row["validation_mae"]) for row in by_x[x]])) for x in x_values
        ]
        time_means = [
            float(np.mean([float(row["training_seconds"]) for row in by_x[x]])) for x in x_values
        ]
        for x, mse_mean, mse_std, mae_mean, time_mean in zip(
            x_values, means, stds, mae_means, time_means
        ):
            summary_rows.append(
                {
                    "experiment_group": group_name,
                    "variable": x_key,
                    "value": x,
                    "mse_mean": mse_mean,
                    "mse_std": mse_std,
                    "mae_mean": mae_mean,
                    "training_seconds_mean": time_mean,
                    "runs": len(by_x[x]),
                }
            )
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.errorbar(x_values, means, yerr=stds, marker="o", capsize=4, label="Test MSE")
        axis.set(xlabel=x_label, ylabel="Test MSE", title=f"Experiment {group_name}: {x_label}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "plots" / f"experiment_{group_name.lower()}_mse.png", dpi=160)
        plt.close(figure)
        if group_name == "C":
            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot(x_values, time_means, marker="o")
            axis.set(xlabel=x_label, ylabel="Training time (seconds)", title="Experiment C: training time")
            axis.grid(alpha=0.25)
            figure.tight_layout()
            figure.savefig(output_dir / "plots" / "experiment_c_time.png", dpi=160)
            plt.close(figure)
    with (output_dir / "summary_results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def run_one(
    name: str,
    bundle: DatasetBundle,
    config: TrainConfig,
    output_dir: Path,
) -> dict[str, float | int | str]:
    result = train_model(bundle, config)
    validation_metrics = evaluate(
        result.model, bundle.validation_x, bundle.validation_y, config.device
    )
    checkpoint = output_dir / "checkpoints" / f"{name}.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": result.model.state_dict(),
            "config": asdict(config),
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "history": result.history,
        },
        checkpoint,
    )
    save_history(result.history, output_dir / "logs" / f"{name}_history.csv")
    plot_history(result.history, output_dir / "plots" / f"{name}_loss.png", name)
    plot_predictions(result.model, bundle, output_dir / "plots" / f"{name}_predictions.png", config.device)
    return {
        "experiment": name,
        "train_size": len(bundle.train_x),
        "hidden_width": config.hidden_width,
        "hidden_depth": config.hidden_depth,
        "seed": config.seed,
        "best_epoch": result.best_epoch,
        "validation_mse": result.best_validation_loss,
        "training_seconds": result.training_seconds,
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
    }


def run_experiments(output_dir: Path, seeds: Sequence[int], epochs: int, device: str) -> None:
    """Run the required A (data), B (width), and C (depth) comparisons."""
    max_train = 2000
    base = create_datasets(max_train, seed=DEFAULT_SEED)
    save_dataset(base, output_dir / "data")
    save_dataset_summary(base, output_dir / "data" / "summary.csv")
    plot_dataset(base, output_dir / "plots" / "dataset.png")
    plot_distributions(base, output_dir / "plots" / "distributions.png")
    rows: list[dict[str, float | int | str]] = []
    depth_histories: dict[int, Sequence[dict[str, float | int]]] = {}
    for seed in seeds:
        common = dict(epochs=epochs, seed=seed, device=device)
        for size in [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]:
            bundle = DatasetBundle(
                base.train_x[:size], base.train_y[:size],
                base.validation_x, base.validation_y, base.test_x, base.test_y
            )
            rows.append(run_one(f"A_size_{size}_seed_{seed}", bundle, TrainConfig(**common), output_dir))
        standard = DatasetBundle(
            base.train_x[:1200], base.train_y[:1200],
            base.validation_x, base.validation_y, base.test_x, base.test_y
        )
        for width in range(1, 11):
            rows.append(run_one(
                f"B_width_{width}_seed_{seed}", standard,
                TrainConfig(hidden_width=width, **common), output_dir
            ))
        for depth in range(1, 5):
            row = run_one(
                f"C_depth_{depth}_seed_{seed}", standard,
                TrainConfig(hidden_depth=depth, **common), output_dir
            )
            rows.append(row)
            if seed == seeds[0]:
                depth_histories[depth] = load_history(
                    output_dir / "logs" / f"C_depth_{depth}_seed_{seed}_history.csv"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_experiment_summaries(rows, output_dir)
    plot_depth_convergence(depth_histories, output_dir / "plots" / "experiment_c_convergence.png")
    baseline = train_model(
        DatasetBundle(
            base.train_x[:1200], base.train_y[:1200],
            base.validation_x, base.validation_y, base.test_x, base.test_y
        ),
        TrainConfig(seed=seeds[0], epochs=epochs, device=device),
    )
    plot_surface_comparison(
        baseline.model,
        output_dir / "plots" / "baseline_surface_comparison.png",
        device,
        "Baseline MLP surface approximation",
    )
    baseline_checkpoint = output_dir / "checkpoints" / "selected_baseline.pth"
    torch.save(
        {
            "model_state": baseline.model.state_dict(),
            "config": asdict(TrainConfig(seed=seeds[0], epochs=epochs, device=device)),
            "best_epoch": baseline.best_epoch,
            "best_validation_loss": baseline.best_validation_loss,
            "history": baseline.history,
            "selection_rule": "lowest validation MSE; test reserved for final evaluation",
            "test_metrics": evaluate(baseline.model, base.test_x, base.test_y, device),
        },
        baseline_checkpoint,
    )
    selected_row = min(rows, key=lambda row: float(row["validation_mse"]))
    selected_checkpoint = output_dir / "checkpoints" / f"{selected_row['experiment']}.pth"
    shutil.copyfile(selected_checkpoint, output_dir / "checkpoints" / "selected_by_validation.pth")
    selected_config = TrainConfig(
        hidden_width=int(selected_row["hidden_width"]),
        hidden_depth=int(selected_row["hidden_depth"]),
        seed=int(selected_row["seed"]),
        epochs=epochs,
        device=device,
    )
    selected_model = MLP(
        selected_config.hidden_width, selected_config.hidden_depth, selected_config.activation
    ).to(device)
    selected_payload = torch.load(
        output_dir / "checkpoints" / "selected_by_validation.pth", map_location=device
    )
    selected_model.load_state_dict(selected_payload["model_state"])
    final_test_metrics = evaluate(selected_model, base.test_x, base.test_y, device)
    (output_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "selection_rule": "minimum validation_mse",
                "selected_experiment": selected_row["experiment"],
                "selected_validation_mse": selected_row["validation_mse"],
                "final_test_metrics": final_test_metrics,
                "note": "The test split was not used to select this configuration.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "configuration.json").write_text(
        json.dumps(
            {
                "data_seed": DEFAULT_SEED,
                "experiment_seeds": list(seeds),
                "validation_size": len(base.validation_x),
                "test_size": len(base.test_x),
                "batch_size": TrainConfig.batch_size,
                "epochs": epochs,
                "optimizer": "Adam",
                "learning_rate": TrainConfig.learning_rate,
                "activation": TrainConfig.activation,
                "early_stopping_patience": TrainConfig.patience,
                "device": device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    run_experiments(args.output_dir, args.seeds, args.epochs, args.device)


if __name__ == "__main__":
    main()
