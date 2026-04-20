"""Generate reproducible uint8 random-noise vibration sequences.

The defaults target the Arduino RTP playback convention used for testing:
100 samples replayed every 10 ms, for a 1 second vibration.
"""

import argparse
import json
import random
from pathlib import Path


DEFAULT_OUTPUT = Path("data/random_noise_vibrations.json")
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_SAMPLE_INTERVAL_MS = 10
DEFAULT_SEED = 20260420

RANDOM_NOISE_SLOTS = [
    ("success", 1),
    ("success", 2),
    ("success", 3),
    ("error", 16),
    ("error", 17),
    ("error", 18),
    ("notification", 31),
    ("notification", 32),
    ("notification", 33),
    ("loading", 46),
    ("loading", 47),
    ("loading", 48),
]


def make_sequence(rng: random.Random, sample_count: int) -> list[int]:
    """Return a discrete uniform uint8 white-noise sequence."""
    return [rng.randrange(256) for _ in range(sample_count)]


def build_payload(
    *,
    seed: int,
    sample_count: int,
    sample_interval_ms: int,
) -> dict:
    rng = random.Random(seed)
    sample_rate_hz = 1000 / sample_interval_ms
    duration_seconds = sample_count * sample_interval_ms / 1000

    vibrations = []
    for index, (block, anonymous_stimulus_id) in enumerate(RANDOM_NOISE_SLOTS, start=1):
        vibrations.append(
            {
                "id": f"random_noise_{index:02d}",
                "block": block,
                "anonymous_stimulus_id": anonymous_stimulus_id,
                "arduino_symbol": f"STIMULUS_{anonymous_stimulus_id:02d}",
                "sequence": make_sequence(rng, sample_count),
            }
        )

    return {
        "version": "random_noise_uint8_v1",
        "description": "Twelve reproducible random-noise control vibrations for comparison testing.",
        "waveform_format": "uint8",
        "noise_model": "discrete_uniform_uint8",
        "seed": seed,
        "sample_count": sample_count,
        "sample_interval_ms": sample_interval_ms,
        "sample_rate_hz": sample_rate_hz,
        "duration_seconds": duration_seconds,
        "value_range": [0, 255],
        "vibrations": vibrations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random-noise uint8 vibration sequences as JSON."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sample-interval-ms", type=int, default=DEFAULT_SAMPLE_INTERVAL_MS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.sample_interval_ms <= 0:
        raise ValueError("--sample-interval-ms must be positive")

    payload = build_payload(
        seed=args.seed,
        sample_count=args.sample_count,
        sample_interval_ms=args.sample_interval_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(payload['vibrations'])} vibrations to {args.output}")
    print(
        f"{payload['sample_count']} samples each, "
        f"{payload['sample_interval_ms']} ms interval, "
        f"{payload['duration_seconds']:.3f} s duration"
    )


if __name__ == "__main__":
    main()
