"""PCA dimensionality reduction, rotated PCA candidates, and sweep experiments."""

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.signal_metrics import compute_all_metrics


def varimax(
    loadings: np.ndarray,
    gamma: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an orthogonal Varimax rotation for a loading matrix.

    Args:
        loadings: Array of shape (n_features, n_components).
        gamma: Varimax gamma parameter. ``1.0`` is classical Varimax.
        max_iter: Maximum number of rotation iterations.
        tol: Relative objective improvement tolerance.

    Returns:
        rotated_loadings: Rotated loading matrix.
        rotation: Orthogonal rotation matrix applied in component space.
    """
    n_features, n_components = loadings.shape
    rotation = np.eye(n_components, dtype=np.float64)
    objective = 0.0

    for _ in range(max_iter):
        lam = loadings @ rotation
        u, s, vh = np.linalg.svd(
            loadings.T
            @ (lam**3 - (gamma / n_features) * lam @ np.diag(np.sum(lam**2, axis=0))),
            full_matrices=False,
        )
        rotation = u @ vh
        new_objective = float(np.sum(s))
        if objective > 0.0 and new_objective <= objective * (1.0 + tol):
            break
        objective = new_objective

    # Fix sign ambiguity so the largest loading on each axis is positive.
    for j in range(rotation.shape[1]):
        rotated_col = loadings @ rotation[:, j]
        max_idx = int(np.argmax(np.abs(rotated_col)))
        if rotated_col[max_idx] < 0.0:
            rotation[:, j] *= -1.0

    rotated = loadings @ rotation
    return rotated.astype(np.float32), rotation.astype(np.float32)


@dataclass
class RotatedPCATransformer:
    """Pipeline-like PCA object for orthogonally rotated component axes."""

    scaler: StandardScaler
    base_pca: PCA
    rotation_matrix_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    name: str = "varimax"
    named_steps: dict = field(init=False, repr=False)

    def __post_init__(self):
        self.n_components = int(self.components_.shape[0])
        self.named_steps = {
            "scaler": self.scaler,
            "pca": self,
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project full latent vectors into the rotated control space."""
        X_scaled = self.scaler.transform(X)
        scores = self.base_pca.transform(X_scaled)
        return scores @ self.rotation_matrix_

    def inverse_transform(self, C: np.ndarray) -> np.ndarray:
        """Map rotated control coordinates back to full latent vectors."""
        squeeze = False
        if C.ndim == 1:
            C = C.reshape(1, -1)
            squeeze = True

        base_scores = C @ self.rotation_matrix_.T
        X_scaled = self.base_pca.inverse_transform(base_scores)
        X = self.scaler.inverse_transform(X_scaled)

        if squeeze:
            X = X.squeeze(0)
        return X


def get_pca_step(pipe) -> PCA | RotatedPCATransformer:
    """Return the PCA-like step from a baseline or rotated pipeline object."""
    if hasattr(pipe, "named_steps") and "pca" in pipe.named_steps:
        return pipe.named_steps["pca"]
    if hasattr(pipe, "components_"):
        return pipe
    raise TypeError("Unsupported PCA container: expected Pipeline-like object")


def get_n_components(pipe) -> int:
    """Return the number of control axes for a PCA-like object."""
    step = get_pca_step(pipe)
    if hasattr(step, "n_components"):
        return int(step.n_components)
    if hasattr(step, "components_"):
        return int(step.components_.shape[0])
    raise AttributeError("Could not determine n_components for PCA-like object")


def get_component_matrix(pipe) -> np.ndarray:
    """Return component vectors in the standardized latent space."""
    step = get_pca_step(pipe)
    if not hasattr(step, "components_"):
        raise AttributeError("PCA-like object is missing components_")
    return np.asarray(step.components_, dtype=np.float32)


def get_explained_variance_ratio(pipe) -> np.ndarray:
    """Return per-axis explained variance ratios for a PCA-like object."""
    step = get_pca_step(pipe)
    if not hasattr(step, "explained_variance_ratio_"):
        raise AttributeError("PCA-like object is missing explained_variance_ratio_")
    return np.asarray(step.explained_variance_ratio_, dtype=np.float32)


# ---------------------------------------------------------------------------
# Step 2: Fit PCA pipeline
# ---------------------------------------------------------------------------

def fit_pca_pipeline(
    Z: np.ndarray,
    n_components: int = 8,
    save_dir: str | None = None,
) -> tuple[Pipeline, np.ndarray]:
    """Standardize Z and fit PCA to obtain control dimensions.

    Args:
        Z: Latent vectors, shape (N, latent_dim).
        n_components: Number of PCA components (control dimensions).
        save_dir: If provided, saves pca_pipe.pkl and Z_pca.npy here.

    Returns:
        pipe: sklearn Pipeline (StandardScaler → PCA).
        Z_pca: Transformed data, shape (N, n_components).
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
    ])

    Z_pca = pipe.fit_transform(Z)

    # Report
    pca: PCA = pipe.named_steps["pca"]
    evr = pca.explained_variance_ratio_

    print(f"\nPCA Results ({Z.shape[1]}D → {n_components}D):")
    print("-" * 50)
    cumulative = 0.0
    for i, v in enumerate(evr):
        cumulative += v
        print(f"  PC{i+1}: {v:.4f} ({v:.2%})  cumulative: {cumulative:.2%}")
    print("-" * 50)
    print(f"  Total explained variance: {cumulative:.2%}")
    print(f"  Z_pca shape: {Z_pca.shape}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        pipe_path = save_dir / "pca_pipe.pkl"
        zpca_path = save_dir / "Z_pca.npy"

        with open(pipe_path, "wb") as f:
            pickle.dump(pipe, f)
        np.save(zpca_path, Z_pca)

        print(f"\n  Saved: {pipe_path}")
        print(f"  Saved: {zpca_path}")

    return pipe, Z_pca


def fit_rotated_pca_pipeline(
    base_pipe: Pipeline,
    Z: np.ndarray | None = None,
    Z_pca: np.ndarray | None = None,
    save_dir: str | None = None,
    prefix: str = "varimax",
) -> tuple[RotatedPCATransformer, np.ndarray]:
    """Rotate fitted PCA axes inside the existing PCA subspace.

    Args:
        base_pipe: Existing fitted PCA pipeline (StandardScaler -> PCA).
        Z: Optional full latent matrix. Used only if ``Z_pca`` is not supplied.
        Z_pca: Optional baseline PCA scores.
        save_dir: If provided, saves ``{prefix}_pipe.pkl`` and ``Z_{prefix}.npy``.
        prefix: Artifact prefix for the rotated candidate.

    Returns:
        rotated_pipe: Orthogonally rotated PCA-like transformer.
        Z_rot: Rotated control coordinates for the training set.
    """
    scaler = base_pipe.named_steps["scaler"]
    base_pca = base_pipe.named_steps["pca"]

    if Z_pca is None:
        if Z is None:
            raise ValueError("Either Z or Z_pca must be provided for rotated PCA fitting")
        Z_pca = base_pipe.transform(Z)

    loadings = base_pca.components_.T * np.sqrt(base_pca.explained_variance_)
    rotated_loadings, rotation = varimax(loadings)
    rotated_variance = np.sum(rotated_loadings**2, axis=0)

    order = np.argsort(rotated_variance)[::-1]
    rotation = rotation[:, order]
    rotated_loadings = rotated_loadings[:, order]
    rotated_variance = rotated_variance[order]

    total_variance = float(
        np.sum(base_pca.explained_variance_) / max(np.sum(base_pca.explained_variance_ratio_), 1e-12)
    )
    rotated_ratio = rotated_variance / total_variance
    rotated_components = rotation.T @ base_pca.components_
    Z_rot = np.asarray(Z_pca, dtype=np.float32) @ rotation

    rotated_pipe = RotatedPCATransformer(
        scaler=scaler,
        base_pca=base_pca,
        rotation_matrix_=rotation.astype(np.float32),
        components_=rotated_components.astype(np.float32),
        explained_variance_=rotated_variance.astype(np.float32),
        explained_variance_ratio_=rotated_ratio.astype(np.float32),
        name=prefix,
    )

    print(f"\nRotated PCA Results ({prefix}, {Z_rot.shape[1]} axes):")
    print("-" * 50)
    cumulative = 0.0
    for i, value in enumerate(rotated_ratio):
        cumulative += float(value)
        print(f"  Axis{i+1}: {value:.4f} ({value:.2%})  cumulative: {cumulative:.2%}")
    print("-" * 50)
    print(f"  Total explained variance: {cumulative:.2%}")
    print(f"  Rotated score shape: {Z_rot.shape}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        pipe_path = save_dir / f"{prefix}_pipe.pkl"
        score_path = save_dir / f"Z_{prefix}.npy"

        with open(pipe_path, "wb") as f:
            pickle.dump(rotated_pipe, f)
        np.save(score_path, Z_rot)

        print(f"\n  Saved: {pipe_path}")
        print(f"  Saved: {score_path}")

    return rotated_pipe, Z_rot


# ---------------------------------------------------------------------------
# Step 3: Control → Latent reconstruction
# ---------------------------------------------------------------------------

def control_to_latent(pipe: Pipeline, c: np.ndarray) -> np.ndarray:
    """Convert a PCA control vector back to a full latent vector.

    Args:
        pipe: Fitted sklearn Pipeline (StandardScaler → PCA).
        c: Control vector, shape (n_components,) or (B, n_components).

    Returns:
        z: Reconstructed latent vector, shape (latent_dim,) or (B, latent_dim).
    """
    squeeze = False
    if c.ndim == 1:
        c = c.reshape(1, -1)
        squeeze = True

    z = pipe.inverse_transform(c)

    if squeeze and z.ndim > 1:
        z = z.squeeze(0)
    return z


# ---------------------------------------------------------------------------
# Unified single-axis sweep
# ---------------------------------------------------------------------------

def sweep_axis(
    pipe,
    model,
    device: torch.device,
    axis: int = 0,
    sweep_range: tuple[float, float] = (-2.0, 2.0),
    n_steps: int = 9,
    T: int = 4000,
    sr: int = 8000,
    reference: np.ndarray | None = None,
    with_metrics: bool = False,
) -> dict:
    """Sweep one PCA control axis around a reference point.

    Args:
        pipe: Fitted PCA pipeline.
        model: Model with .decode() method.
        device: Torch device.
        axis: Which PC axis to sweep (0-indexed).
        sweep_range: (min, max) values for the swept axis.
        n_steps: Number of sweep steps.
        T: Signal length for decoder.
        sr: Sample rate (used only when with_metrics=True).
        reference: Base control vector; None → zeros.
        with_metrics: If True, compute signal metrics at each step.

    Returns:
        Dict with 'axis', 'values', 'signals', 'latents',
        and optionally 'metrics' (list of metric dicts).
    """
    n_components = get_n_components(pipe)
    values = np.linspace(sweep_range[0], sweep_range[1], n_steps)
    ref = reference if reference is not None else np.zeros(n_components, dtype=np.float32)

    signals, latents, metrics_list = [], [], []

    model.eval()
    with torch.no_grad():
        for val in values:
            c = ref.copy()
            c[axis] = val

            z_np = control_to_latent(pipe, c)
            z_t = torch.from_numpy(z_np).float().unsqueeze(0).to(device)
            sig = model.decode(z_t, target_len=T).squeeze().cpu().numpy()

            latents.append(z_np)
            signals.append(sig)
            if with_metrics:
                metrics_list.append(compute_all_metrics(sig, sr=sr))

    result = {
        "axis": axis,
        "values": values if not with_metrics else values.tolist(),
        "signals": np.stack(signals),
        "latents": np.stack(latents),
    }
    if with_metrics:
        result["metrics"] = metrics_list
    return result


def plot_sweep(sweep_result: dict, sr: int = 8000, save_path: str | None = None):
    """Visualize the single-axis sweep results."""
    import matplotlib.pyplot as plt

    values = sweep_result["values"]
    signals = sweep_result["signals"]
    axis = sweep_result["axis"]
    n = len(values)

    fig, axes = plt.subplots(n, 1, figsize=(14, 1.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (val, sig) in enumerate(zip(values, signals)):
        t = np.arange(len(sig)) / sr
        axes[i].plot(t, sig, linewidth=0.5)
        axes[i].set_ylabel(f"PC{axis+1}={val:+.1f}", fontsize=9)
        axes[i].set_ylim(-3.5, 3.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Single-axis sweep: PC{axis+1} from {values[0]:.1f} to {values[-1]:.1f}", fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def play_sweep(sweep_result: dict, sr: int = 8000):
    """Play audio for each step of the sweep (Jupyter/Colab only)."""
    from IPython.display import Audio, display

    values = sweep_result["values"]
    signals = sweep_result["signals"]
    axis = sweep_result["axis"]

    for val, sig in zip(values, signals):
        sig_norm = sig / (np.max(np.abs(sig)) + 1e-8)
        sig_norm = np.clip(sig_norm, -1.0, 1.0)
        print(f"PC{axis+1} = {val:+.2f}  |  max={np.max(np.abs(sig)):.4f}")
        display(Audio(sig_norm, rate=sr))
