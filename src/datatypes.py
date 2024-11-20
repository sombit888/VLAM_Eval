from typing import Dict, Optional, Tuple

import numpy as np
import torch
# from databib.dataclasses import Dataclass, dataclass
from transformers.modeling_outputs import ModelOutput
import torch
import transformers
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
# from barrel.components.types.model_types import  ModelTarget, PolicyControlPlan
# from barrel.components.types.train_types import TrainDataBatch
# from barrel.core.common.tensor_dataclass import TensorDataclass, dataclass as tensor_dataclass
# from barrel.pipes.vlams.types.vlm_types import LLMOutput


@dataclass
class RoboticsMetadata():
    """Metadata for a SINGLE example in the dataset"""

    image: torch.Tensor  # [B, C, H, W]; torch.uint8
    # prompt: np.ndarray  # [B]; dtype U<str_len>
    dataset_name: np.ndarray  # [B]; dtype U<str_len>
    # OpenVLA-style action of shape [B, 7] and normalized via [q1, q99]; always euler angles
    action: torch.Tensor


@dataclass
class RoboticsInput():
    # Preprocessed input images for different image encoders
    # Dict keys correspond to encoders, e.g. 'dino', 'siglip'
    images: Dict[str, torch.Tensor]  # [B, C, H, W]; torch.float32
    text_tokens: torch.Tensor  # [B, S]; torch.int64


@dataclass
class RoboticsOutput():
    translation: torch.Tensor  # [B, num_timesteps, 3]
    rotation: torch.Tensor  # [B, num_timesteps, 3] or [B, num_timesteps, 4] if quaternion
    gripper: torch.Tensor  # [B, num_timesteps, 1]
    llm_output: Any


@dataclass
class RoboticsTarget():
    target_tokens: torch.Tensor  # [B, S]; torch.int64
    translation: torch.Tensor  # [B, num_timesteps, 3]
    rotation: torch.Tensor  # [B, num_timesteps, 3] or [B, num_timesteps, 4] if quaternion
    gripper: torch.Tensor  # [B, num_timesteps, 1]


@dataclass

class RoboticsBatch():
    input: RoboticsInput
    target: RoboticsTarget
    attention_mask: torch.Tensor  # [B, S], torch.bool
    metadata: Optional[RoboticsMetadata]

    @property
    def batch_shape(self) -> Tuple[int, ...]:
        return tuple(self.input.text_tokens.shape[:1])

    @property
    def multimodal_indices(self) -> torch.Tensor:
        """
        Returns a torch.Tensor containing only the indices of the batch examples which are multimodal.
        Return shape is [B]
        """
        return torch.arange(self.batch_size, dtype=torch.int64, device=self.attention_mask.device)


@dataclass

class RoboticsControlPlan():
    translation_m: torch.Tensor  # [B, S, 3]
    rotmat: torch.Tensor  # [B, S, 9]
    gripper_prob: torch.Tensor  # [B, S, 1]

    def __post_init__(self):
        super().__post_init__()
        assert self.translation_m.ndim == 3, self.translation_m.shape
        assert self.rotmat.ndim == 3, self.rotmat.shape
        assert self.gripper_prob.ndim == 3, self.gripper_prob.shape
from typing import Dict, List, Optional, Tuple




@dataclass
class VLMInput():
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Dict[str, torch.Tensor]
    labels: torch.Tensor
    multimodal_indices: torch.Tensor
    inputs_embeds: Optional[torch.Tensor] = None
    past_key_values: Optional[List[torch.Tensor]] = None


@dataclass
class LLMOutput():
    """Fork of transformers.modeling_outputs.CausalLMOutputWithPast"""

    input_ids: torch.Tensor
    logits: torch.Tensor
    loss: Optional[torch.Tensor]
    past_key_values: List[Tuple[torch.Tensor]]
    hidden_states: List[torch.Tensor]

    @classmethod
    def from_transformers(
        cls, input_ids: torch.Tensor, llm_output: transformers.modeling_outputs.CausalLMOutputWithPast
    ) -> "LLMOutput":
        return LLMOutput(
            input_ids=input_ids,
            logits=llm_output.logits,
            loss=llm_output.loss,
            past_key_values=(
                list(llm_output.past_key_values) if llm_output.past_key_values is not None else []
            ),
            hidden_states=list(llm_output.hidden_states) if llm_output.hidden_states is not None else [],
        )


@dataclass
class VLMOutput():
    llm_output: LLMOutput
    vit_tokens: Optional[torch.Tensor]  # ViT output tokens
    attn_mask: torch.Tensor


@dataclass
class WaypointerInput():
    """Pack all information which different waypointer architectures might need"""

    vit_tokens: torch.Tensor  # ViT output tokens; [B, S, vit_token_size]
    llm_image_tokens: torch.Tensor  # Image tokens output by the LLM; [B, S, llm_token_size]
    control_tokens: torch.Tensor  # Control tokens at LLM output; [B, L, llm_token_size]
    llm_tokens: torch.Tensor  # Entire sequence output by LLM; [B, N, llm_token_size]
    attn_mask: Optional[torch.Tensor]  # For entire seq, masks PAD tokens; [B, N]

