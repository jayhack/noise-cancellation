from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .env import BlocketLeagueEnv, WorldConfig


STATE_NAMES = (
    "player_x",
    "player_y",
    "player_vx",
    "player_vy",
    "puck_x",
    "puck_y",
    "puck_vx",
    "puck_vy",
    "score",
    "reset_timer",
)


def make_clip(
    seed: int,
    *,
    context_frames: int = 4,
    future_frames: int = 8,
    image_size: int = 64,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    env = BlocketLeagueEnv(seed=seed, config=WorldConfig(image_size=image_size))

    previous_action = 0
    for _ in range(int(rng.integers(10, 80))):
        if rng.random() < 0.34:
            previous_action = env.policy_action(exploration=0.18)
        env.step(previous_action)

    frame_count = context_frames + future_frames
    frames: list[np.ndarray] = [env.render()]
    states: list[np.ndarray] = [env.state.vector()]
    actions: list[int] = []
    events: list[int] = []
    event_ids = {"coast": 0, "thrust": 1, "impact": 2, "wall": 3, "goal": 4, "kickoff": 5}

    for _ in range(frame_count - 1):
        if rng.random() < 0.31:
            previous_action = env.policy_action(exploration=0.16)
        env.step(previous_action)
        actions.append(previous_action)
        frames.append(env.render())
        states.append(env.state.vector())
        events.append(event_ids.get(env.state.last_event, 0))

    frame_array = np.stack(frames)
    state_array = np.stack(states)
    action_array = np.asarray(actions[context_frames - 1 :], dtype=np.int64)
    event_array = np.asarray(events[context_frames - 1 :], dtype=np.int64)
    return {
        "frames": frame_array,
        "context": frame_array[:context_frames],
        "target": frame_array[context_frames:],
        "actions": action_array,
        "state": state_array[context_frames:],
        "events": event_array,
    }


class ClipDataset:
    """A deterministic map-style PyTorch dataset with no stored video corpus."""

    def __init__(
        self,
        samples: int,
        *,
        seed: int = 0,
        context_frames: int = 4,
        future_frames: int = 8,
        image_size: int = 64,
    ) -> None:
        self.samples = samples
        self.seed = seed
        self.context_frames = context_frames
        self.future_frames = future_frames
        self.image_size = image_size

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch

        clip = make_clip(
            self.seed + index * 9_973,
            context_frames=self.context_frames,
            future_frames=self.future_frames,
            image_size=self.image_size,
        )

        def video_tensor(value: np.ndarray) -> object:
            tensor = torch.from_numpy(value.copy()).permute(0, 3, 1, 2).float()
            return tensor.div(127.5).sub(1.0)

        return {
            "context": video_tensor(clip["context"]),
            "target": video_tensor(clip["target"]),
            "actions": torch.from_numpy(clip["actions"].copy()).long(),
            "state": torch.from_numpy(clip["state"].copy()).float(),
            "events": torch.from_numpy(clip["events"].copy()).long(),
        }


def export_dataset(
    output: Path,
    *,
    samples: int,
    seed: int,
    context_frames: int,
    future_frames: int,
    image_size: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": np.asarray(1),
        "seed": np.asarray(seed),
        "context_frames": np.asarray(context_frames),
        "future_frames": np.asarray(future_frames),
        "image_size": np.asarray(image_size),
        "state_names": np.asarray(STATE_NAMES),
    }
    np.savez_compressed(output / "manifest.npz", **manifest)
    for index in range(samples):
        clip = make_clip(
            seed + index * 9_973,
            context_frames=context_frames,
            future_frames=future_frames,
            image_size=image_size,
        )
        np.savez_compressed(output / f"clip-{index:06d}.npz", **clip)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deterministic Blocket League clips")
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--future-frames", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()
    export_dataset(
        args.output,
        samples=args.samples,
        seed=args.seed,
        context_frames=args.context_frames,
        future_frames=args.future_frames,
        image_size=args.image_size,
    )


if __name__ == "__main__":
    main()
