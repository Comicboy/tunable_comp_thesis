"""ACT with the stiffness as its own encoder token (policy type "act_ktoken").

ACTKToken subclasses LeRobot's ACT model (0.6.1) and changes only the transformer encoder's input:

    ACT:         [latent, state (48, incl. log10 K), environment state (3)]
    act_ktoken:  [latent, state without K (47), log10 K (1), environment state (3)]  (default)
                 [latent, state without K + environment state (47 + 3), log10 K (1)]  (env_state_in_state_token=True)

Each token is a linear projection of its inputs to dim_model plus a learned position embedding, as in ACT.
The forward pass is ACT.forward copied with only that part changed (Apache-2.0, The HuggingFace Inc. team).
The VAE encoder (training only) still gets the whole state, K included, as in ACT. ACTKTokenPolicy is
ACTPolicy with this model; select_action, temporal ensembling, the loss and saving are inherited.
"""

import einops
import torch
from torch import Tensor, nn

from lerobot.policies.act.modeling_act import ACT, ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from .configuration_act_ktoken import ACTKTokenConfig


class ACTKToken(ACT):
    """LeRobot's ACT with log10 K split off the state token into a token of its own."""

    def __init__(self, config: ACTKTokenConfig):
        super().__init__(config)
        d = config.dim_model
        n_state = config.robot_state_feature.shape[0]
        self.stiffness_index = config.stiffness_index % n_state
        self.merge_env = bool(config.env_state_feature) and config.env_state_in_state_token
        n_rest = n_state - 1 + (config.env_state_feature.shape[0] if self.merge_env else 0)
        # replaces ACT's state projection (48 -> d) by one for the state without K, with the cube position
        # if merged (50 or 47 -> d) ...
        self.encoder_robot_state_input_proj = nn.Linear(n_rest, d)
        # ... and adds one for K alone (1 -> d)
        self.encoder_stiffness_input_proj = nn.Linear(1, d)
        if self.merge_env:
            del self.encoder_env_state_input_proj  # ACT's own environment-state token is not used
        # 1-D tokens: latent, state, K, (environment state if kept separate)
        n_1d_tokens = 3 + (1 if config.env_state_feature and not self.merge_env else 0)
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, d)

    def split_state(self, state: Tensor) -> tuple[Tensor, Tensor]:
        """(B, n) normalised state -> (B, n - 1) without K, (B, 1) log10 K."""
        i = self.stiffness_index
        return torch.cat([state[:, :i], state[:, i + 1:]], dim=1), state[:, i:i + 1]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """ACT.forward of LeRobot 0.6.1, copied unchanged except for the encoder's state tokens (marked below).

        A forward pass through the Action Chunking Transformer (with optional VAE encoder).

        `batch` should have the following structure:
        {
            [robot_state_feature] (optional): (B, state_dim) batch of robot states.

            [image_features]: (B, n_cameras, C, H, W) batch of images.
                AND/OR
            [env_state_feature]: (B, env_dim) batch of environment states.

            [action_feature] (optional, only if training with VAE): (B, chunk_size, action dim) batch of actions.
        }

        Returns:
            (B, chunk_size, action_dim) batch of action sequences
            Tuple containing the latent PDF's parameters (mean, log(σ²)) both as (B, L) tensors where L is the
            latent dimension.
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # Prepare fixed positional embedding.
            # Note: detach() shouldn't be necessary but leaving it the same as the original code just in case.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)

            # Prepare key padding mask for the transformer encoder. We have 1 or 2 extra tokens at the start of the
            # sequence depending whether we use the input states or not (cls and robot state)
            # False means not a padding token.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        # State token (state without K, plus the cube position if merged) and the stiffness token.
        # <-- the only change to ACT.forward (down to the environment-state token)
        state_rest, stiffness = self.split_state(batch[OBS_STATE])
        if self.merge_env:
            state_rest = torch.cat([state_rest, batch[OBS_ENV_STATE]], dim=1)
        encoder_in_tokens.append(self.encoder_robot_state_input_proj(state_rest))
        encoder_in_tokens.append(self.encoder_stiffness_input_proj(stiffness))
        # Environment state token (only if kept separate).
        if self.config.env_state_feature and not self.merge_env:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        if self.config.image_features:
            # For a list of images, the H and W may vary but H*W is constant.
            # NOTE: If modifying this section, verify on MPS devices that
            # gradients remain stable (no explosions or NaNs).
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # Extend immediately instead of accumulating and concatenating
                # Convert to list to extend properly
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)

        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTKTokenPolicy(ACTPolicy):
    """ACTPolicy with the ACTKToken model; everything else inherited."""

    config_class = ACTKTokenConfig
    name = "act_ktoken"

    def __init__(self, config: ACTKTokenConfig, **kwargs):
        # ACTPolicy.__init__ with ACTKToken instead of ACT (calling it would build an unused ACT first)
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.model = ACTKToken(config)
        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        self.reset()