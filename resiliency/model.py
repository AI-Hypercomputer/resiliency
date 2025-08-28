"""Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional
from nemo.collections.llm import GemmaConfig2B
from nemo.collections.llm.gpt.model.llama import Llama31Config405B, Llama31Config70B, Llama31Config8B, Llama3Config
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from resiliency.callbacks import comm_overlap
import torch


class ParallelConfig(NamedTuple):
  tp: int = 1  # tensor parallelism
  pp: int = 1  # pipeline parallelism
  vp: Optional[int] = None  # virtual pipeline parallelism size
  cp: int = 1  # context parallelism


@dataclass
class Llama3Config36M(Llama3Config):
  rotary_base: int = 500_000
  seq_length: int = 8192
  num_layers: int = 12
  hidden_size: int = 768
  ffn_hidden_size: int = 2688
  num_attention_heads: int = 16


class ModelConfig:
  """Unified configuration class combining model architecture, parallelism, and training settings."""

  def __init__(self, model_name: str):
    self.model_name = model_name
    self._setup_configs()

  def _setup_configs(self):
    # Model architecture configurations
    self.model_configs = {
        "36M": Llama3Config36M(),
        "2B": GemmaConfig2B(),
        "8B": Llama31Config8B(seq_length=8192),
        "70B": Llama31Config70B(seq_length=8192),
        "70Bt": Llama31Config70B(num_layers=4, seq_length=8192),
        "405B": Llama31Config405B(seq_length=8192),
        "405B-A3U": Llama31Config405B(seq_length=8192),
        "405Bt": Llama31Config405B(num_layers=4, seq_length=8192),
    }

    self.model_configs["36MReplicatedOpt"] = self.model_configs["36M"]
    self.model_configs["70BReplicatedOpt"] = self.model_configs["70B"]
    self.model_configs["405BReplicatedOpt"] = self.model_configs["405B"]

    # Parallel configurations
    self.parallel_configs = {
        "36M": ParallelConfig(tp=2, pp=2, vp=3, cp=2),
        "2B": ParallelConfig(tp=1, pp=1, cp=2),
        "8B": ParallelConfig(tp=1, pp=1, cp=2),
        "70B": ParallelConfig(tp=4, pp=4, vp=4, cp=2),
        "70Bt": ParallelConfig(tp=4, pp=1, cp=2),
        "405B": ParallelConfig(tp=8, pp=18, vp=7, cp=1),
        "405B-A3U": ParallelConfig(tp=8, pp=9, vp=14, cp=2),
        "405Bt": ParallelConfig(tp=2, pp=2, vp=2, cp=2),
    }

    self.parallel_configs["36MReplicatedOpt"] = ParallelConfig(tp=4, pp=2, vp=3)
    self.parallel_configs["70BReplicatedOpt"] = ParallelConfig(tp=8, pp=4, vp=4)
    self.parallel_configs["405BReplicatedOpt"] = ParallelConfig(
        tp=8, pp=18, vp=7
    )

    # Communication overlap callback configurations
    self.comm_overlap_configs = {
        "36M": comm_overlap.MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
            align_param_gather=True,
        ),
        "8B": comm_overlap.MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
            align_param_gather=True,
        ),
        "70B": comm_overlap.MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
            align_param_gather=True,
        ),
        "70Bt": comm_overlap.MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
            align_param_gather=True,
        ),
        "405B": comm_overlap.MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            # 'overlap_param_gather_with_optimizer_step' is set automatically. Added here for user's knowledge
            overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
        ),
        "405B_A3U": comm_overlap.MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            # 'overlap_param_gather_with_optimizer_step' is set automatically. Added here for user's knowledge
            overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
        ),
        "405Bt": comm_overlap.MegatronCommOverlapCallback(
            tp_comm_overlap=True,
            tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
            defer_embedding_wgrad_compute=True,
            wgrad_deferral_limit=50,
            # 'overlap_param_gather_with_optimizer_step' is set automatically. Added here for user's knowledge
            overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing
        ),
    }

    self.comm_overlap_configs["36MReplicatedOpt"] = self.comm_overlap_configs[
        "36M"
    ]
    self.comm_overlap_configs["70BReplicatedOpt"] = self.comm_overlap_configs[
        "70B"
    ]
    self.comm_overlap_configs["405BReplicatedOpt"] = self.comm_overlap_configs[
        "405B"
    ]

  def _create_comm_overlap_callback(self):
    """Create communication overlap callback with standard configuration."""
    # from resiliency.callbacks import MegatronCommOverlapCallback
    from resiliency.callbacks import comm_overlap

    return comm_overlap.MegatronCommOverlapCallback(
        tp_comm_overlap=True,
        tp_comm_overlap_cfg=userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=50,
        overlap_param_gather_with_optimizer_step=False,
        align_param_gather=True,
    )

  @property
  def model_config(self):
    """Get model architecture configuration."""
    return self.model_configs[self.model_name]

  @property
  def parallel_config(self):
    """Get parallel training configuration."""
    return self.parallel_configs[self.model_name]

  @property
  def comm_overlap_callback(self):
    """Get communication overlap callback if available."""
    return self.comm_overlap_configs.get(self.model_name)

  def create_model(self):
    """Create model instance based on configuration."""
    from nemo.collections.llm.gpt.model.llama import LlamaModel
    from nemo.collections.llm import GemmaModel

    if self.model_name == "2B":
      return GemmaModel(config=self.model_config)
    return LlamaModel(config=self.model_config)


def get_model_config(model_name: str) -> ModelConfig:
  """Factory function to create ModelConfig instance."""
  return ModelConfig(model_name)
