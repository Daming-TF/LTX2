from pathlib import Path
import sys
from typing import Any
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from ltx_trainer import logger
from ltx_trainer.timestep_samplers import TimestepSampler
from ltx_trainer.trainer import LtxvTrainer
from ltx_trainer.training_strategies.base_strategy import (
    ModelInputs, 
    DEFAULT_FPS
)
from ltx_core.model.transformer.modality import Modality

from .batch_bucket import AspectRatioBatchSampler, prepare_train_dataset, collate_fn


BUCKET_EXAMPLE = {"1.78-97":[512,288,97],"0.55-65":[352,352,65]}


class LtxTrainerDev(LtxvTrainer):
    def __init__(self, config):
        super().__init__(config)
    
    def _init_dataloader(self) -> None:
        """Override the dataloader initialization to add custom behavior for development."""
        dataset = load_dataset(
            "json", 
            data_files=self._config.data.training_jsonl_path, 
            split='train'
        )
        # just for debugging, add sample_idx for tracking samples if not exist
        if "sample_idx" not in dataset.column_names:
            dataset = dataset.add_column("sample_idx", list(range(len(dataset))))
        
        # prepare dataset
        ratio_nums = dataset.to_pandas().groupby('bucket').size().to_dict()
        data_sources = self._training_strategy.get_data_sources()
        dataset = prepare_train_dataset(dataset, self._accelerator, data_sources, BUCKET_EXAMPLE)
        
        # prepare batch sampler and dataloader
        generator=torch.Generator().manual_seed(self._config.seed)
        batch_sampler = AspectRatioBatchSampler(
            sampler=RandomSampler(dataset, generator=generator), 
            # sampler=SequentialSampler(dataset),
            dataset=dataset,
            batch_size=self._config.optimization.batch_size, aspect_ratios=BUCKET_EXAMPLE, drop_last=True,
            ratio_nums=ratio_nums, 
            valid_num=0,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_sampler=batch_sampler, 
            collate_fn=collate_fn
        )
        self._dataloader = self._accelerator.prepare(dataloader)

    def _t2V_prepare_training_inputs(self, batch: dict[str, Any], timestep_sampler: TimestepSampler,) -> dict[str, torch.Tensor]:
        """
        Override the T2V training input preparation to add custom behavior for development."""
        # Get pre-encoded latents - dataset provides uniform non-patchified format [B, C, F, H, W]
        latents = batch["v_latents"]
        video_latents = batch["a_latents"]

        # Get video dimensions (assume same for all batch elements)
        num_frames = batch["v_num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        # Patchify latents: [B, C, F, H, W] -> [B, seq_len, C]
        video_latents = self._training_strategy._video_patchifier.patchify(video_latents)

        # Handle FPS with backward compatibility
        fps = batch.get("v_fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(
                f"Different FPS values found in the batch. Found: {fps.tolist()}, using the first one: {fps[0].item()}"
            )
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        # Get text embeddings (already processed by embedding connectors in trainer)
        video_prompt_embeds = batch["video_prompt_embeds"]
        audio_prompt_embeds = batch["audio_prompt_embeds"]
        prompt_attention_mask = batch["prompt_attention_mask"]

        batch_size = video_latents.shape[0]
        video_seq_len = video_latents.shape[1]
        device = video_latents.device
        dtype = video_latents.dtype

        # Create conditioning mask (first frame conditioning)
        video_conditioning_mask = self._training_strategy._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=video_seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=self._training_strategy.config.first_frame_conditioning_p,
        )

        # Sample noise and sigmas
        sigmas = timestep_sampler.sample_for(video_latents)
        video_noise = torch.randn_like(video_latents)

        # Apply noise: noisy = (1 - sigma) * clean + sigma * noise
        sigmas_expanded = sigmas.view(-1, 1, 1)
        noisy_video = (1 - sigmas_expanded) * video_latents + sigmas_expanded * video_noise

        # For conditioning tokens, use clean latents
        conditioning_mask_expanded = video_conditioning_mask.unsqueeze(-1)
        noisy_video = torch.where(conditioning_mask_expanded, video_latents, noisy_video)

        # Compute video targets (velocity prediction)
        video_targets = video_noise - video_latents

        # Create per-token timesteps
        video_timesteps = self._training_strategy._create_per_token_timesteps(video_conditioning_mask, sigmas.squeeze())

        # Generate video positions using ltx_core's native implementation
        video_positions = self._training_strategy._get_video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        # Create video Modality
        video_modality = Modality(
            enabled=True,
            latent=noisy_video,
            timesteps=video_timesteps,
            positions=video_positions,
            context=video_prompt_embeds,
            context_mask=prompt_attention_mask,
        )

        # Video loss mask: True for tokens we want to compute loss on (non-conditioning tokens)
        video_loss_mask = ~video_conditioning_mask

        # Handle audio if enabled
        audio_modality = None
        audio_targets = None
        audio_loss_mask = None

        if self._training_strategy.config.with_audio:
            audio_modality, audio_targets, audio_loss_mask = self._training_strategy._prepare_audio_inputs(
                batch=batch,
                sigmas=sigmas,
                audio_prompt_embeds=audio_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        return ModelInputs(
            video=video_modality,
            audio=audio_modality,
            video_targets=video_targets,
            audio_targets=audio_targets,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )
        return 

    def _training_step(self, batch: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Override the training step to add custom behavior for development.
        Args:
            batch (dict[str, dict[str, Tensor]]): A batch of data containing video latents, conditions, and audio latents.
                v_latents: torch.Tensor, shape (B,C,F,V_H,V_W)=(-,128,-,-,-), video latents
                v_num_frames: int, number of frames in the video ((ori_F-1)//8+1)
                v_height: int, height of the video latents (ori_H/32)
                v_width: int, width of the video latents (ori_W/32)
                v_fps: float, frames per second of the video
                c_prompt_embeds: torch.Tensor, shape (B,S,C)=(-,1024,3840), condition prompt embeddings
                c_prompt_attention_mask: torch.Tensor, shape (B,S), condition prompt attention mask
                a_latents: torch.Tensor, shape (B,C,F,A_bins)=(-,8,-,16), audio latents
                a_num_time_steps: int, number of frame in the audio latents ((ori_duration/(16000//hop_length))//4)
                a_frequency_bins: int, number of frequency bins in the audio latents (16)
                a_duration: float, duration of the audio in seconds
        Returns:
            Tensor: The computed loss for the batch.
        """
        # Apply embedding connectors to transform pre-computed text embeddings
        video_embeds, audio_embeds, attention_mask = self._text_encoder._run_connectors(
            batch["c_prompt_embeds"], 
            batch["c_prompt_attention_mask"]
        )
        batch["video_prompt_embeds"] = video_embeds
        batch["audio_prompt_embeds"] = audio_embeds
        batch["prompt_attention_mask"] = attention_mask

        # Use strategy to prepare training inputs (returns ModelInputs with Modality objects)
        model_inputs = self._t2V_prepare_training_inputs(batch, self._timestep_sampler)

        # Run transformer forward pass with Modality-based interface
        video_pred, audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        # Use strategy to compute loss
        loss = self._training_strategy.compute_loss(video_pred, audio_pred, model_inputs)
        return loss