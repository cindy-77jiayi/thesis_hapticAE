"""Steps 2–4: PCA dimensionality reduction, control-to-latent mapping, and sweep experiments."""

import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

    if squeeze:
        z = z.squeeze(0)
    return z


# ---------------------------------------------------------------------------
# Step 4: Single-axis sweep
# ---------------------------------------------------------------------------

def single_axis_sweep(
    pipe: Pipeline,
    model,
    device: torch.device,
    axis: int = 0,
    sweep_range: tuple[float, float] = (-2.0, 2.0),
    n_steps: int = 9,
    T: int = 4000,
) -> dict:
    """Sweep one PCA control axis while holding others at zero.

    Args:
        pipe: Fitted PCA pipeline.
        model: Trained VAE model (must have .decode() method).
        device: Torch device.
        axis: Which PC axis to sweep (0-indexed, 0 = PC1).
        sweep_range: (min, max) values for the swept axis.
        n_steps: Number of sweep steps.
        T: Signal length for decoder.

    Returns:
        Dict with 'values' (sweep values), 'signals' (generated waveforms),
        and 'latents' (corresponding latent vectors).
    """
    n_components = pipe.named_steps["pca"].n_components
    values = np.linspace(sweep_range[0], sweep_range[1], n_steps)

    signals = []
    latents = []

    model.eval()
    with torch.no_grad():
        for val in values:
            c = np.zeros(n_components, dtype=np.float32)
            c[axis] = val

            z_np = control_to_latent(pipe, c)
            z_t = torch.from_numpy(z_np).float().unsqueeze(0).to(device)

            x_hat = model.decode(z_t, target_len=T)
            sig = x_hat.squeeze().cpu().numpy()

            latents.append(z_np)
            signals.append(sig)

    return {
        "axis": axis,
        "values": values,
        "signals": np.stack(signals),    # (n_steps, T)
        "latents": np.stack(latents),    # (n_steps, latent_dim)
    }


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
        from pathlib import Path
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
