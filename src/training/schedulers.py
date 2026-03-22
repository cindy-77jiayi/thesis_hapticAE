"""KL annealing schedules for VAE training."""


def cyclical_beta_schedule(
    epoch: int,
    total_epochs: int,
    n_cycles: int = 4,
    ratio: float = 0.5,
    beta_max: float = 0.0001,
) -> float:
    """Cyclical annealing: beta rises from 0 to beta_max periodically.

    Args:
        epoch: Current epoch number.
        total_epochs: Total number of training epochs.
        n_cycles: Number of annealing cycles.
        ratio: Fraction of each cycle spent rising.
        beta_max: Maximum beta value.
    """
    cycle_length = total_epochs / n_cycles
    cycle_pos = epoch % cycle_length

    if cycle_pos / cycle_length < ratio:
        return beta_max * (cycle_pos / (cycle_length * ratio))
    else:
        return beta_max


def monotonic_beta_schedule(
    epoch: int,
    warmup_epochs: int = 40,
    beta_start: float = 0.0,
    beta_max: float = 0.0001,
) -> float:
    """Monotonic KL warmup: beta increases once and then stays at beta_max.

    Args:
        epoch: Current epoch number (1-based).
        warmup_epochs: Number of epochs used to ramp beta from beta_start to beta_max.
        beta_start: Initial beta value at epoch 1.
        beta_max: Final beta value after warmup.
    """
    if warmup_epochs <= 1:
        return beta_max

    progress = min(max((epoch - 1) / (warmup_epochs - 1), 0.0), 1.0)
    return beta_start + progress * (beta_max - beta_start)
