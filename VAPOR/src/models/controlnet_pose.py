import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from src.models.resnet import InflatedConv3d
from src.models.motion_module import zero_module
from src.models.unet_3d_blocks import (
    get_down_block,
    UNetMidBlock3DCrossAttn,
)

try:
    from diffusers.models.embeddings import TimestepEmbedding, Timesteps
except ImportError:
    from diffusers.models.resnet import TimestepEmbedding, Timesteps



class PoseConditioningEmbedding(nn.Module):

    def __init__(
        self,
        conditioning_channels: int = 3,
        conditioning_embedding_channels: int = 320,
        block_out_channels: Tuple[int, ...] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0],
            kernel_size=3, padding=1,
        )

        self.blocks = nn.ModuleList([])
        for i in range(len(block_out_channels) - 1):
            ch_in = block_out_channels[i]
            ch_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(ch_in, ch_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(ch_in, ch_out, kernel_size=3, padding=1, stride=2)
            )

        self.conv_out = InflatedConv3d(
            block_out_channels[-1], conditioning_embedding_channels,
            kernel_size=3, padding=1,
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        embedding = self.conv_out(embedding)
        return embedding



class ControlNetPose(nn.Module):

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def __init__(self, down_blocks_config, mid_block_config, conv_in_channels,
                 time_embed_dim, conditioning_block_out_channels=(16, 32, 64, 128)):
        super().__init__()

        first_out_ch = conv_in_channels

        self.controlnet_cond_embedding = PoseConditioningEmbedding(
            conditioning_channels=3,
            block_out_channels=conditioning_block_out_channels,
            conditioning_embedding_channels=first_out_ch,
        )

        self.time_proj = Timesteps(first_out_ch, True, 0)
        self.time_embedding = TimestepEmbedding(first_out_ch, time_embed_dim)

        self.conv_in = InflatedConv3d(first_out_ch, first_out_ch, kernel_size=3, padding=1)

        self.controlnet_down_blocks = nn.ModuleList([
            zero_module(InflatedConv3d(first_out_ch, first_out_ch, 1))
        ])

        self.down_blocks = nn.ModuleList()
        for cfg in down_blocks_config:
            block = get_down_block(**cfg)
            self.down_blocks.append(block)

            out_ch = cfg['out_channels']
            n_layers = cfg['num_layers']
            for _ in range(n_layers):
                self.controlnet_down_blocks.append(
                    zero_module(InflatedConv3d(out_ch, out_ch, 1))
                )
            if cfg.get('add_downsample', True):
                self.controlnet_down_blocks.append(
                    zero_module(InflatedConv3d(out_ch, out_ch, 1))
                )

        self.mid_block = UNetMidBlock3DCrossAttn(**mid_block_config)
        mid_ch = mid_block_config['in_channels']
        self.controlnet_mid_block = zero_module(InflatedConv3d(mid_ch, mid_ch, 1))

    def enable_gradient_checkpointing(self):
        for block in self.down_blocks:
            if hasattr(block, 'gradient_checkpointing'):
                block.gradient_checkpointing = True
        if hasattr(self.mid_block, 'gradient_checkpointing'):
            self.mid_block.gradient_checkpointing = True

    def forward(
        self,
        conditioning: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        conditioning_scale: float = 1.0,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        controlnet_cond = self.controlnet_cond_embedding(conditioning)

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long,
                                     device=conditioning.device)
        elif timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        t_emb = self.time_embedding(t_emb)

        hidden_states = self.conv_in(controlnet_cond)

        down_block_res_samples = [self.controlnet_down_blocks[0](hidden_states)]
        zero_idx = 1

        for down_block in self.down_blocks:
            has_attn = hasattr(down_block, 'attentions')
            if has_attn and encoder_hidden_states is not None:
                hidden_states, output_states = down_block(
                    hidden_states=hidden_states,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                hidden_states, output_states = down_block(
                    hidden_states=hidden_states,
                    temb=t_emb,
                )

            for state in output_states:
                down_block_res_samples.append(
                    self.controlnet_down_blocks[zero_idx](state)
                )
                zero_idx += 1

        has_mid_attn = hasattr(self.mid_block, 'attentions')
        if has_mid_attn and encoder_hidden_states is not None:
            hidden_states = self.mid_block(
                hidden_states,
                temb=t_emb,
                encoder_hidden_states=encoder_hidden_states,
            )
        else:
            hidden_states = self.mid_block(hidden_states, temb=t_emb)

        mid_res = self.controlnet_mid_block(hidden_states)

        down_block_res_samples = [s * conditioning_scale for s in down_block_res_samples]
        mid_res = mid_res * conditioning_scale

        return tuple(down_block_res_samples), mid_res

    @classmethod
    def from_unet(cls, unet, pretrained_unet_state_dict=None):
        config = unet.config

        block_out_channels   = list(config.block_out_channels)
        down_block_types     = list(config.down_block_types)
        layers_per_block     = config.layers_per_block
        cross_attention_dim  = config.cross_attention_dim
        norm_num_groups      = config.norm_num_groups
        norm_eps             = getattr(config, 'norm_eps', 1e-5)

        attention_head_dim   = config.attention_head_dim
        if isinstance(attention_head_dim, int):
            attention_head_dim = [attention_head_dim] * len(down_block_types)

        use_inflated_groupnorm         = getattr(config, 'use_inflated_groupnorm', True)
        unet_use_cross_frame_attention = getattr(config, 'unet_use_cross_frame_attention', None)
        unet_use_temporal_attention    = getattr(config, 'unet_use_temporal_attention', None)
        use_linear_projection          = getattr(config, 'use_linear_projection', False)
        upcast_attention               = getattr(config, 'upcast_attention', False)
        only_cross_attention           = getattr(config, 'only_cross_attention', False)
        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)
        downsample_padding             = getattr(config, 'downsample_padding', 1)
        resnet_time_scale_shift        = getattr(config, 'resnet_time_scale_shift', 'default')
        act_fn                         = getattr(config, 'act_fn', 'silu')

        first_out_ch    = block_out_channels[0]
        time_embed_dim  = first_out_ch * 4

        down_blocks_config = []
        output_channel = first_out_ch

        for i, (block_type, out_ch) in enumerate(zip(down_block_types, block_out_channels)):
            input_channel  = output_channel
            output_channel = out_ch
            is_final       = (i == len(block_out_channels) - 1)

            block_cfg = dict(
                down_block_type=block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=False,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
                unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                unet_use_temporal_attention=unet_use_temporal_attention,
                use_inflated_groupnorm=use_inflated_groupnorm,
                use_motion_module=False,
                motion_module_type=None,
                motion_module_kwargs=None,
            )
            down_blocks_config.append(block_cfg)

        mid_channel = block_out_channels[-1]
        mid_block_config = dict(
            in_channels=mid_channel,
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=1.0,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
            use_linear_projection=use_linear_projection,
            upcast_attention=upcast_attention,
            unet_use_cross_frame_attention=unet_use_cross_frame_attention,
            unet_use_temporal_attention=unet_use_temporal_attention,
            use_inflated_groupnorm=use_inflated_groupnorm,
            use_motion_module=False,
            motion_module_type=None,
            motion_module_kwargs=None,
        )

        controlnet = cls(
            down_blocks_config=down_blocks_config,
            mid_block_config=mid_block_config,
            conv_in_channels=first_out_ch,
            time_embed_dim=time_embed_dim,
        )

        if pretrained_unet_state_dict is not None:
            unet_sd = pretrained_unet_state_dict
        else:
            unet_sd = unet.state_dict()

        cn_sd = controlnet.state_dict()
        loaded = 0
        total  = len(cn_sd)

        for key in cn_sd:
            if key in unet_sd and cn_sd[key].shape == unet_sd[key].shape:
                cn_sd[key] = unet_sd[key].clone()
                loaded += 1

        controlnet.load_state_dict(cn_sd)

        total_params = sum(p.numel() for p in controlnet.parameters())
        block_summary = ", ".join(
            t.replace("Block3D", "").replace("CrossAttn", "XAttn")
            for t in down_block_types
        )
        print(f"  ControlNet: {loaded}/{total} weight tensors loaded from denoising UNet"
              f" (resnet + attention)")
        print(f"  ControlNet: {total_params:,} total params ({block_summary} + MidXAttn)")

        return controlnet

    def load_pose_guider_weights(self, pose_guider_state_dict):
        pg_sd = pose_guider_state_dict
        cond_sd = self.controlnet_cond_embedding.state_dict()
        loaded = 0
        for key in cond_sd:
            if key in pg_sd and cond_sd[key].shape == pg_sd[key].shape:
                cond_sd[key] = pg_sd[key].clone()
                loaded += 1
        self.controlnet_cond_embedding.load_state_dict(cond_sd)
        total = len(cond_sd)
        print(f"  PoseGuider → ControlNet cond embedding: {loaded}/{total} tensors loaded")
