# Copyright (c) 2023, Albert Gu, Tri Dao.
import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from einops import rearrange, repeat
from pydantic import validate_call

from zoology.mixers.mamba_ssm.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
try:
    from causal_conv1d import causal_conv1d_fn
except:
    assert 0, print(f"Need to install causal_conv1d: pip install causal_conv1d")
try:
    from zoology.mixers.mamba_ssm.selective_scan_interface import selective_scan_fn, mamba_inner_fn
except:
    assert 0, print(f"Need to install selective_scan_interface: pip install mamba_ssm")

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

import torch.nn.functional as F


class Mamba(nn.Module):

    @validate_call
    def __init__(
        self,
        d_model,
        d_state: int=16,
        d_conv:int=4,
        expand: int=2,
        dt_rank: str="auto",
        dt_min: float=0.001,
        dt_max: float=0.1,
        dt_init: str="random",
        dt_scale: float=1.0,
        dt_init_floor: float=1e-4,
        conv_bias: bool=True,
        bias: bool=False,
        use_fast_path: bool=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # print(f"mixer: {type(hidden_states)}")
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    None,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def state_size(self, sequence_length: int=2048):
        return 2 * self.d_model * self.d_state


class MambaBlock(nn.Module):
    def __init__(
        self, config, fused_add_norm=True, residual_in_fp32=True, norm_epsilon=1e-5, **factory_kwargs
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        config=config
        d_model = config.d_model
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        #self.mixer = config.sequence_mixer.instantiate(d_model=d_model, **factory_kwargs)
        self.mixer = Mamba(d_model, **factory_kwargs, **config.sequence_mixer[0].kwargs)
        from zoology.mixers.mamba_ssm.triton.layernorm import RMSNorm
        self.norm = RMSNorm(d_model, eps=norm_epsilon)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def MambaInit(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    """
    `MambaInit` 函数的作用是初始化神经网络模块的权重和偏置。它特别关注以下几点：

1. **线性层和嵌入层的初始化**：
   - 对于 `nn.Linear` 层，如果存在偏置且未标记为 `_no_reinit`，则将其初始化为零。
   - 对于 `nn.Embedding` 层，将其权重初始化为正态分布，标准差为 `initializer_range`。

2. **重新缩放预归一化残差**：
   - 如果 `rescale_prenorm_residual` 为 `True`，则根据 OpenAI GPT-2 论文中的方案重新初始化选定的权重。
   - 具体来说，对于 `out_proj.weight` 或 `fc2.weight`，使用 `kaiming_uniform_` 初始化，并按 `1/sqrt(2 * n_layer)` 缩放权重。

这个函数确保了模型在训练开始时的权重和偏置处于合理的初始状态，从而有助于模型的稳定训练和收敛。
    """
    
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            # print(f"{name=}")
            if "out_proj.weight" in name or "fc2.weight" in name:
                print(f"found in initialization phase - {name=}!")
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class AnotherMamba(nn.Module):
    def __init__(
            self,
            d_model,  # 模型的维度
            d_inner,  # 内部维度
            d_xb,  # 输入投影的维度
            d_state=16,  # 状态维度
            d_conv=4,  # 卷积核大小
            expand=2,  # 扩展因子
            dt_rank="auto",  # 时间步长的秩
            dt_min=0.001,  # 时间步长的最小值
            dt_max=0.1,  # 时间步长的最大值
            dt_init="random",  # 时间步长的初始化方式
            dt_scale=1.0,  # 时间步长的缩放因子
            dt_init_floor=1e-4,  # 时间步长初始化的下限
            repeat_kv_before_conv=True,  # 是否在卷积前重复键值
            conv_bias=True,  # 卷积层是否使用偏置
            proj_x_bias=True,  # 输入投影层是否使用偏置
            proj_z_bias=True,  # 输出投影层是否使用偏置
            out_proj_bias=True,  # 最终输出投影层是否使用偏置
            use_fast_path=True,  # 是否使用快速路径
            layer_idx=None,  # 层的索引
            device=None,  # 设备
            dtype=None,  # 数据类型
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_xb = d_xb
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_inner if d_inner is not None else int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.repeat_kv_before_conv = repeat_kv_before_conv

        self.in_proj_x = nn.Linear(self.d_model, self.d_xb, bias=proj_x_bias, **factory_kwargs)
        self.in_proj_z = nn.Linear(self.d_model, self.d_inner, bias=proj_z_bias, **factory_kwargs)

        if self.repeat_kv_before_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        else:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_xb,
                out_channels=self.d_xb,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_xb,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.num_B_head = self.d_xb // self.d_state
        self.num_C_head = self.d_inner // self.d_state
        self.num_groups = self.num_C_head // self.num_B_head

        # load using k weights
        self.B_proj = nn.Linear(self.d_model, self.d_xb, bias=False, **factory_kwargs)
        # load using q weights
        self.C_proj = nn.Linear(self.d_model, self.d_inner, bias=False, **factory_kwargs)

        self.dt_proj_down = nn.Linear(self.d_model, self.dt_rank, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=out_proj_bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        x = rearrange(self.in_proj_x(hidden_states), "b l d -> b d l")
        z = rearrange(self.in_proj_z(hidden_states), "b l d -> b d l")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        B = self.B_proj(hidden_states)  # B, L, H_inner
        C = self.C_proj(hidden_states)  # B, L, H

        B = rearrange(B, "b l (n_group dstate) -> b n_group l dstate", dstate=self.d_state)
        B = repeat_kv(B, self.num_groups)  # B, n_group, L, H
        B = rearrange(B, "b n_group l dstate -> b n_group dstate l").contiguous()
        C = rearrange(C, "b l (n_group dstate) -> b n_group dstate l", dstate=self.d_state).contiguous()

        dt = self.dt_proj_down(hidden_states)  # B, L, d_rank
        dt = self.dt_proj(dt)  # B, L, d_inner
        dt = rearrange(dt, "b l d -> b d l")  # B, d_inner, L

        if self.repeat_kv_before_conv:
            # b d l
            x = rearrange(x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.num_groups)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            # Update state (B D W)
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        if not self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) l -> b n_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.num_groups)
            x = rearrange(x, "b n_group l dstate -> b (n_group dstate) l")

        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state.unsqueeze(-2))
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        hidden_states_input = hidden_states.squeeze(1)

        # hidden_states_input shape: (B, H)
        x = self.in_proj_x(hidden_states_input)
        z = self.in_proj_z(hidden_states_input)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        B = self.B_proj(hidden_states_input)  # B, H_inner
        C = self.C_proj(hidden_states_input)  # B, H

        B = rearrange(B, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
        B = torch.repeat_interleave(B, dim=1, repeats=self.num_groups)

        C = rearrange(C, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state).contiguous()

        dt = self.dt_proj_down(hidden_states_input)  # B, d_rank
        dt = self.dt_proj(dt)  # B, d_inner

        if self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
            x = torch.repeat_interleave(x, dim=1, repeats=self.num_groups)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        # Conv step
        if causal_conv1d_update is None:
            # Update state (B D W)
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        if not self.repeat_kv_before_conv:
            x = rearrange(x, "b (n_group dstate) -> b n_group dstate", dstate=self.d_state)
            x = torch.repeat_interleave(x, dim=1, repeats=self.num_groups)
            x = rearrange(x, "b n_group dstate -> b (n_group dstate)")

        x = x.unsqueeze(-1)
        dt = dt.unsqueeze(-1)
        dt_bias = self.dt_proj.bias.unsqueeze(-1)
        A = A.unsqueeze(1)
        D = self.D.unsqueeze(-1)
        z = z.unsqueeze(-1)

        # SSM step
        assert selective_state_update is not None
        y = selective_state_update(
            ssm_state, x, dt, A, B, C, D, z=z, dt_bias=dt_bias, dt_softplus=True
        )

        y = y.squeeze(-1)
        out = self.out_proj(y)

        return out.unsqueeze(1), conv_state, ssm_state
