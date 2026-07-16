from __future__ import annotations

import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from blocket_league.codec import RepresentationCodec, RepresentationCodecConfig, fake_backbone_outputs
from blocket_league.data import make_clip, make_passive_clip, passive_kickoff_state
from blocket_league.direct_model import DirectLatentTransformer, DirectWorldModelConfig
from blocket_league.latent_model import CausalLatentDiT, FlowMatchingSchedule, LatentWorldModelConfig
from blocket_league.latent_probe import (
    _fit_downstream_velocity_lens,
    binary_auc,
    derive_probe_targets,
    fit_ridge_probe,
)
from blocket_league.metrics import trajectory_metrics
from blocket_league.model import DiffusionSchedule, VideoDiT, VideoDiTConfig
from blocket_league.pixel_direct_model import DirectPixelTransformer, PixelDirectConfig
from blocket_league.trajectory_assets import sample_autoregressive
from blocket_league.train import (
    TrainConfig,
    _rollout_context_probability,
    initialize_from_checkpoint,
)
from blocket_league.train_codec import codec_loss
from blocket_league.train_direct import direct_training_loss, rollout_latents


class BlocketLeagueModelTests(unittest.TestCase):
    def test_direct_pixel_transformer_predicts_palette_pixels(self) -> None:
        config = PixelDirectConfig(
            image_size=16,
            patch_size=4,
            history_frames=3,
            pixel_embedding_size=4,
            hidden_size=16,
            depth=1,
            heads=4,
        )
        model = DirectPixelTransformer(config)
        frames = torch.randint(0, config.palette_size, (2, 3, 16, 16))
        logits, hidden = model(frames, return_hidden=True)
        self.assertEqual(logits.shape, (2, 3, config.palette_size, 16, 16))
        self.assertEqual(hidden[-1].shape, (2, 3, 16, 16))
        self.assertEqual(model.next_frame(frames).shape, (2, 16, 16))
        self.assertFalse(any("action" in name for name, _ in model.named_parameters()))
        direction = torch.randn(config.hidden_size)
        mask = torch.zeros(2, 3, 16)
        mask[:, -1, 5] = 1
        edited = model(
            frames,
            intervention_block=0,
            intervention=direction,
            intervention_mask=mask,
        )
        baseline = model(frames)
        self.assertFalse(torch.equal(baseline, edited))

    def test_passive_clips_have_motion_without_action_labels(self) -> None:
        clip = make_passive_clip(17, context_frames=4, future_frames=8, image_size=32)
        self.assertNotIn("actions", clip)
        self.assertEqual(clip["frames"].shape, (12, 32, 32, 3))
        self.assertEqual(clip["state"].shape, (8, 10))
        displacement = abs(clip["state"][-1, :8] - clip["state"][0, :8]).sum()
        self.assertGreater(displacement, 0.01)

    def test_goal_centered_passive_clips_restart_with_momentum(self) -> None:
        clip = make_passive_clip(
            17,
            context_frames=1,
            future_frames=23,
            image_size=32,
            goal_centered=True,
        )
        kickoff_indices = torch.from_numpy((clip["events"] == 5).nonzero()[0])
        self.assertGreater(len(kickoff_indices), 0)
        kickoff = int(kickoff_indices[0])
        player_speed = float(torch.from_numpy(clip["state"][kickoff, 2:4]).norm())
        puck_speed = float(torch.from_numpy(clip["state"][kickoff, 6:8]).norm())
        self.assertGreater(player_speed, 0.2)
        self.assertGreater(puck_speed, 0.18)
        score = int(clip["state"][kickoff, 8])
        expected = passive_kickoff_state(score)
        self.assertTrue(torch.allclose(torch.from_numpy(clip["state"][kickoff, :2]), torch.from_numpy(expected[0])))
        self.assertTrue(torch.allclose(torch.from_numpy(clip["state"][kickoff, 2:4]), torch.from_numpy(expected[1])))
        self.assertTrue(torch.allclose(torch.from_numpy(clip["state"][kickoff, 4:6]), torch.from_numpy(expected[2])))
        self.assertTrue(torch.allclose(torch.from_numpy(clip["state"][kickoff, 6:8]), torch.from_numpy(expected[3])))

    def test_direct_latent_transformer_predicts_and_rolls_out_in_one_pass(self) -> None:
        config = DirectWorldModelConfig(
            latent_dim=4,
            latent_grid_size=2,
            history_latents=3,
            hidden_size=16,
            depth=1,
            heads=4,
        )
        model = DirectLatentTransformer(config, torch.full((4,), 0.25))
        inputs = torch.randn(2, 3, 4, 2, 2)
        targets = torch.randn(2, 3, 4, 2, 2)
        actions = torch.randint(0, 9, (2, 3, 2))
        loss = direct_training_loss(
            model,
            inputs,
            targets,
            actions,
            corruption_std=0.1,
            late_token_weight=2.0,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        next_latent = model.next_latent(inputs, actions)
        self.assertEqual(next_latent.shape, (2, 4, 2, 2))
        generated = rollout_latents(
            model,
            inputs[:, :2],
            torch.zeros(2, 8, dtype=torch.long),
        )
        self.assertEqual(generated.shape, (2, 4, 4, 2, 2))
        self.assertTrue(torch.isfinite(generated).all())

    def test_downstream_velocity_lens_backpropagates_through_rendered_frames(self) -> None:
        class DummyBackbone(nn.Module):
            def forward(self, *, pixel_values: torch.Tensor, **_: object) -> object:
                pooled = torch.nn.functional.adaptive_avg_pool2d(pixel_values, (4, 4))
                patches = pooled.flatten(2).transpose(1, 2)
                patches = torch.cat((patches, patches, patches[:, :, :2]), dim=-1)
                tokens = torch.cat((torch.zeros(patches.shape[0], 1, 8), patches), dim=1)
                return fake_backbone_outputs(tuple(tokens for _ in range(13)))

        codec = RepresentationCodec(
            RepresentationCodecConfig(
                image_size=32,
                dino_input_size=56,
                dino_hidden_size=8,
                latent_dim=4,
                decoder_width=16,
                decoder_depth=1,
                decoder_heads=4,
                max_latent_frames=8,
            ),
            DummyBackbone(),
        ).eval()
        model = CausalLatentDiT(
            LatentWorldModelConfig(
                latent_dim=4,
                latent_grid_size=4,
                temporal_downsample=2,
                context_frames=2,
                future_frames=2,
                hidden_size=16,
                depth=1,
                heads=4,
                max_sequence_latents=3,
            )
        ).eval()
        torch.nn.init.normal_(model.output_projection.weight, std=0.05)
        model.requires_grad_(False)
        codec.requires_grad_(False)
        direction, sigma, protocol = _fit_downstream_velocity_lens(
            model,
            codec,
            torch.zeros(4),
            torch.ones(4),
            block_index=0,
            dataset_seed=13,
            dataset_offset=2,
            samples=2,
            batch_size=2,
            integration_steps=2,
        )
        self.assertEqual(tuple(direction.shape), (16, 16))
        self.assertTrue(torch.isfinite(direction).all())
        self.assertGreater(float(direction.norm()), 0.0)
        self.assertLessEqual(float(direction.norm()), 1.0001)
        self.assertGreater(sigma, 0.0)
        self.assertEqual(protocol["solverEvaluationsPerContext"], 2)

    def test_latent_probe_targets_and_linear_readout(self) -> None:
        states = torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.2, 0.1], [0.2, 0.3, 0.4, 0.0, 0.6, 0.3, 0.0, -0.5]],
                [[0.3, 0.4, -0.1, 0.2, 0.7, 0.8, 0.3, 0.4], [0.4, 0.4, 0.0, 0.2, 0.4, 0.8, 0.3, 0.4]],
            ]
        )
        events = torch.tensor([[1, 2], [1, 3]])
        continuous, binary = derive_probe_targets(states, events)
        self.assertEqual(tuple(continuous.shape), (2, 13))
        self.assertEqual(tuple(binary.shape), (2, 3))
        self.assertTrue(torch.equal(binary[:, 0], torch.ones(2)))
        self.assertAlmostEqual(float(continuous[0, 8]), 0.4, places=5)
        self.assertAlmostEqual(float(continuous[0, 9]), 0.5, places=5)
        self.assertAlmostEqual(binary_auc(torch.tensor([0, 0, 1, 1]), torch.tensor([0.1, 0.2, 0.8, 0.9])), 1.0)

        generator = torch.Generator().manual_seed(11)
        features = torch.randn(180, 9, generator=generator)
        weights = torch.randn(9, 3, generator=generator)
        targets = features @ weights + 0.01 * torch.randn(180, 3, generator=generator)
        fit = fit_ridge_probe(
            features,
            targets,
            train_count=108,
            validation_count=36,
            device=torch.device("cpu"),
        )
        self.assertGreater(float(fit.r2.mean()), 0.98)

    def test_representation_codec_compresses_space_and_time(self) -> None:
        class DummyBackbone(nn.Module):
            def forward(self, *, pixel_values: torch.Tensor, **_: object) -> object:
                pooled = torch.nn.functional.adaptive_avg_pool2d(pixel_values, (4, 4))
                patches = pooled.flatten(2).transpose(1, 2)
                patches = torch.cat((patches, patches, patches[:, :, :2]), dim=-1)
                tokens = torch.cat((torch.zeros(patches.shape[0], 1, 8), patches), dim=1)
                return fake_backbone_outputs(tuple(tokens * ((index + 1) / 13) for index in range(13)))

        config = RepresentationCodecConfig(
            image_size=32,
            dino_input_size=56,
            dino_hidden_size=8,
            latent_dim=4,
            decoder_width=16,
            decoder_depth=1,
            decoder_heads=4,
            max_latent_frames=8,
        )
        codec = RepresentationCodec(config, DummyBackbone())
        video = torch.randn(2, 6, 3, 32, 32).clamp(-1, 1)
        reconstruction, latents, features, logits = codec(video)
        self.assertEqual(features.shape, (2, 6, 8, 4, 4))
        self.assertEqual(latents.shape, (2, 3, 4, 4, 4))
        self.assertEqual(reconstruction.shape, video.shape)
        self.assertTrue(torch.isfinite(reconstruction).all())
        terms = codec_loss(
            codec,
            video,
            reconstruction,
            features,
            logits,
            feature_weight=1.0,
            categorical_weight=0.2,
            foreground_weight=1.0,
            puck_weight=1.0,
        )
        terms["loss_total"].backward()
        self.assertTrue(torch.isfinite(terms["feature_auto_weight"]))

    def test_causal_latent_flow_loss_and_sampler_preserve_shapes(self) -> None:
        config = LatentWorldModelConfig(
            latent_dim=4,
            latent_grid_size=2,
            temporal_downsample=2,
            context_frames=2,
            future_frames=4,
            hidden_size=16,
            depth=1,
            heads=4,
            max_sequence_latents=3,
        )
        model = CausalLatentDiT(config)
        schedule = FlowMatchingSchedule()
        context = torch.randn(2, 1, 4, 2, 2)
        target = torch.randn(2, 2, 4, 2, 2)
        actions = torch.zeros(2, 2, 2, dtype=torch.long)
        loss = schedule.training_loss(model, context, target, actions, late_frame_weight=2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        generated = schedule.sample_autoregressive(
            model,
            context,
            torch.zeros(2, 6, dtype=torch.long),
            rollout_frames=6,
            integration_steps=2,
        )
        self.assertEqual(generated.shape, (2, 3, 4, 2, 2))
        self.assertTrue(torch.isfinite(generated).all())

    def test_checkpoint_initialization_extends_frame_positions(self) -> None:
        source = VideoDiT(
            VideoDiTConfig(
                image_size=8,
                patch_size=4,
                context_frames=2,
                future_frames=2,
                hidden_size=16,
                depth=1,
                heads=4,
                attention_mode="factorized",
            )
        )
        target = VideoDiT(
            VideoDiTConfig(
                image_size=8,
                patch_size=4,
                context_frames=2,
                future_frames=4,
                hidden_size=16,
                depth=1,
                heads=4,
                attention_mode="factorized",
            )
        )
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "model": source.state_dict(),
                    "model_config": source.config.to_dict(),
                    "step": 17,
                },
                checkpoint,
            )
            metadata = initialize_from_checkpoint(target, checkpoint)

        self.assertEqual(metadata["checkpoint_step"], 17)
        self.assertEqual(metadata["extended_frame_positions"], 2)
        self.assertTrue(
            torch.equal(target.frame_position[:, :4], source.frame_position)
        )
        self.assertTrue(
            torch.equal(target.action_embedding.weight, source.action_embedding.weight)
        )

    def test_rollout_context_probability_warms_up_and_ramps(self) -> None:
        config = TrainConfig(
            rollout_context_fraction=0.4,
            rollout_context_start_step=100,
            rollout_context_ramp_steps=200,
        )
        self.assertEqual(_rollout_context_probability(99, config), 0.0)
        self.assertAlmostEqual(_rollout_context_probability(199, config), 0.2)
        self.assertAlmostEqual(_rollout_context_probability(400, config), 0.4)

    def test_autoregressive_sampler_refeeds_generated_context(self) -> None:
        class StubSchedule:
            def __init__(self) -> None:
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

            def sample(
                self,
                model: object,
                context: torch.Tensor,
                actions: torch.Tensor,
                **_: object,
            ) -> torch.Tensor:
                self.calls.append((context.clone(), actions.clone()))
                return torch.full(
                    (context.shape[0], model.config.future_frames, 3, 2, 2),
                    float(len(self.calls)),
                )

        schedule = StubSchedule()
        model = SimpleNamespace(config=SimpleNamespace(context_frames=2, future_frames=3))
        rollout = sample_autoregressive(
            schedule,  # type: ignore[arg-type]
            model,  # type: ignore[arg-type]
            torch.zeros(1, 2, 3, 2, 2),
            torch.arange(6).reshape(1, 6),
            rollout_frames=5,
            ddim_steps=2,
        )

        self.assertEqual(rollout.shape, (1, 5, 3, 2, 2))
        self.assertTrue(torch.equal(rollout[:, :3], torch.ones_like(rollout[:, :3])))
        self.assertTrue(torch.equal(rollout[:, 3:], torch.full_like(rollout[:, 3:], 2)))
        self.assertEqual(len(schedule.calls), 2)
        self.assertTrue(torch.equal(schedule.calls[0][1], torch.tensor([[0, 1, 2]])))
        self.assertTrue(torch.equal(schedule.calls[1][1], torch.tensor([[3, 4, 5]])))
        self.assertTrue(torch.equal(schedule.calls[1][0], torch.ones_like(schedule.calls[1][0])))

    def test_factorized_model_preserves_video_shape_and_hidden_layout(self) -> None:
        config = VideoDiTConfig(
            image_size=32,
            patch_size=4,
            context_frames=2,
            future_frames=3,
            hidden_size=32,
            depth=2,
            heads=4,
            attention_mode="factorized",
        )
        model = VideoDiT(config)
        result = model(
            torch.randn(2, 3, 3, 32, 32),
            torch.randn(2, 2, 3, 32, 32),
            torch.zeros(2, 3, dtype=torch.long),
            torch.full((2,), 99, dtype=torch.long),
            return_hidden=True,
        )
        prediction, hidden = result
        self.assertEqual(prediction.shape, (2, 3, 3, 32, 32))
        self.assertEqual(len(hidden), 2)
        self.assertEqual(hidden[-1].shape, (2, 5 * 64, 32))

    def test_rgb_trajectory_metric_recovers_rendered_entity_centers(self) -> None:
        clip = make_clip(42)
        target = torch.from_numpy(clip["target"].copy()).permute(0, 3, 1, 2).float()
        target = target.div(127.5).sub(1.0).unsqueeze(0)
        state = torch.from_numpy(clip["state"].copy()).float().unsqueeze(0)
        metrics = trajectory_metrics(target, state)
        self.assertLess(metrics["player_position_error_px"], 0.75)
        self.assertLess(metrics["puck_position_error_px"], 0.75)

    def test_weighted_diffusion_loss_is_finite(self) -> None:
        config = VideoDiTConfig(
            image_size=32,
            patch_size=8,
            context_frames=2,
            future_frames=2,
            hidden_size=32,
            depth=1,
            heads=4,
            attention_mode="factorized",
        )
        model = VideoDiT(config)
        schedule = DiffusionSchedule(prediction_type="x0")
        target = torch.randn(2, 2, 3, 32, 32).clamp(-1, 1)
        loss = schedule.training_loss(
            model,
            target,
            torch.randn(2, 2, 3, 32, 32).clamp(-1, 1),
            torch.zeros(2, 2, dtype=torch.long),
            foreground_weight=10,
            puck_weight=28,
            terminal_timestep_fraction=0.35,
            late_frame_weight=1.75,
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
