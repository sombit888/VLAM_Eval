"""
configuration_prismatic.py

HuggingFace-style configuration definition for Prismatic VLMs, inheriting from `transformers.PretrainedConfig`.
Default configuration specifies `siglip-224px+7b`.
"""

from typing import Any, Dict, List, Optional
import transformers
from transformers import PretrainedConfig,AutoModel, AutoConfig
from transformers.models.auto import CONFIG_MAPPING
import numpy as np
from pydantic import BaseModel
# === Utilities for Mapping Prismatic names to HF names ===
# fmt: off
VISION_BACKBONE_TO_RESOLUTION: Dict[str, List[int]] = {
    "clip-vit-l": [224], "siglip-vit-so400m": [224], "dinov2-vit-l": [224], "in1k-vit-l": [224],

    "clip-vit-l-336px": [336],
    "siglip-vit-so400m-384px": [384],

    "dinoclip-vit-l-336px": [336, 336],
    "dinosiglip-vit-so-224px": [224, 224],
    "dinosiglip-vit-so-384px": [384, 384],
}
VISION_BACKBONE_TO_TIMM_ID: Dict[str, List[str]] = {
    "clip-vit-l": ["vit_large_patch14_clip_224.openai"],
    "clip-vit-l-336px": ["vit_large_patch14_clip_336.openai"],

    "dinov2-vit-l": ["vit_large_patch14_reg4_dinov2.lvd142m"],
    "in1k-vit-l": ["vit_large_patch16_224.augreg_in21k_ft_in1k"],

    "siglip-vit-so400m": ["vit_so400m_patch14_siglip_224"],
    "siglip-vit-so400m-384px": ["vit_so400m_patch14_siglip_384"],

    "dinoclip-vit-l-336px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_large_patch14_clip_336.openai"],
    "dinosiglip-vit-so-224px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
    "dinosiglip-vit-so-384px": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_384"],
}
TIMM_OVERRIDE_ACT_LAYER: Dict[str, List[Optional[str]]] = {
    "clip-vit-l": ["quick_gelu"], "clip-vit-l-336px": ["quick_gelu"],
    "dinov2-vit-l": [None], "in1k-vit-l": [None],
    "siglip-vit-so400m": [None], "siglip-vit-so400m-384px": [None],
    "dinoclip-vit-l-336px": [None, "quick_gelu"],
    "dinosiglip-vit-so-224px": [None, None], "dinosiglip-vit-so-384px": [None, None]
}

LLM_BACKBONE_TO_HF_PATH = {
    "llama2-7b-pure": "meta-llama/Llama-2-7b-hf", "llama2-13b-pure": "meta-llama/Llama-2-13b-hf",
    "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf", "llama2-13b-chat": "meta-llama/Llama-2-13b-chat-hf",

    "vicuna-v15-7b": "lmsys/vicuna-7b-v1.5", "vicuna-v15-13b": "lmsys/vicuna-13b-v1.5",

    "mistral-v0.1-7b-pure": "mistralai/Mistral-7B-v0.1",
    "mistral-v0.1-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.1",

    "phi-2-3b": "microsoft/phi-2",
}
LLM_BACKBONE_TO_HF_METACLASS = {
    "llama2-7b-pure": "llama", "llama2-13b-pure": "llama", "llama2-7b-chat": "llama", "llama2-13b-chat": "llama",
    "vicuna-v15-7b": "llama", "vicuna-v15-13b": "llama",

    "mistral-v0.1-7b-pure": "mistral", "mistral-v0.1-7b-instruct": "mistral",

    "phi-2-3b": "phi",
}

VALID_VISION_BACKBONES = set(VISION_BACKBONE_TO_RESOLUTION.keys())
VALID_LLM_BACKBONES = set(LLM_BACKBONE_TO_HF_PATH)
# fmt: on

class WaypointTokenizer:
    """
    Wraps base LLM/VLM tokenizer and overloads least used token as a control token

    NOTE: By default, assumes a BPE-style tokenizer akin to the LlamaTokenizer,
        where *the least used tokens* appear at the end of the vocabulary!

    TODO: Adding new token vs overloading? When I call `tokenizer.add_token()` vocab stays the same
    """
    model_type = "waypointer"
    is_composition: bool = True
    def __init__(self, tokenizer: transformers.PreTrainedTokenizerBase, num_tokens: int = 10) -> None:
        self.tokenizer = tokenizer
        self.num_tokens = num_tokens

    def __call__(self, *_) -> str:
        """Get the text token for control"""
        return self.tokenizer.decode(self.control_token_ids)

    @property
    def control_token_ids(self) -> np.ndarray:
        # Assumes we're overwriting the final tokens of the vocabulary (least used tokens)
        return np.arange(self.num_tokens) + int(self.tokenizer.vocab_size - self.num_tokens)

    @property
    def num_control_tokens(self) -> int:
        return self.num_tokens

class PrismaticConfig(PretrainedConfig):
    model_type: str = "prismatic"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "dinosiglip-vit-so-224px",
        llm_backbone_id: str = "llama2-7b-pure",
        arch_specifier: str = "no-align+gelu-mlp", ## TODO: check
        use_fused_vision_backbone: Optional[bool] = None, ## TODO: check
        image_resize_strategy: str = "letterbox",
        text_config: Optional[Dict[str, Any]] = None,
        llm_max_length: int = 2048,
        pad_token_id: int = 32000,
        pad_to_multiple_of: int = 64,
        output_projector_states: bool = False,
        **kwargs: str,
    ) -> None:
        if vision_backbone_id not in VALID_VISION_BACKBONES:
            raise ValueError(f"Vision backbone `{vision_backbone_id}` not in {VALID_VISION_BACKBONES = }")

        if llm_backbone_id not in VALID_LLM_BACKBONES:
            raise ValueError(f"LLM backbone `{llm_backbone_id}` not in {VALID_LLM_BACKBONES = }")

        # Set Prismatic Configuration Fields
        self.vision_backbone_id = vision_backbone_id
        self.llm_backbone_id = llm_backbone_id
        self.arch_specifier = arch_specifier
        self.output_projector_states = output_projector_states

        # [Contract] All vision backbone parameters are lists =>> supports fused backbones with different preprocessing
        self.use_fused_vision_backbone = (
            use_fused_vision_backbone
            if use_fused_vision_backbone is not None
            else any(self.vision_backbone_id.startswith(v) for v in ["dinoclip", "dinosiglip"])
        )

        self.timm_model_ids = VISION_BACKBONE_TO_TIMM_ID[self.vision_backbone_id]
        self.timm_override_act_layers = TIMM_OVERRIDE_ACT_LAYER[self.vision_backbone_id]
        self.image_sizes = VISION_BACKBONE_TO_RESOLUTION[self.vision_backbone_id]
        self.image_resize_strategy = image_resize_strategy

        self.hf_llm_id = LLM_BACKBONE_TO_HF_PATH[self.llm_backbone_id]
        self.llm_max_length = llm_max_length
        self.pad_token_id, self.pad_to_multiple_of = pad_token_id, pad_to_multiple_of

        # [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that specific naming!
        self.text_config = (
            CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]](**text_config)
            if text_config is not None
            else CONFIG_MAPPING[LLM_BACKBONE_TO_HF_METACLASS[self.llm_backbone_id]]()
        )

        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=pad_token_id, **kwargs)

# Here  we need trajectory_vla config, with
# prismatic_config fields and then the waypointer fields
class TrajectoryVLAConfig(PretrainedConfig):
    model_type: str = "trajectoryvla"

    def __init__(
        self,
        prismatic_config = {},
        token_size: int = 1024,  # Timestep token size
        cheat: bool = False,  # If True, cheat and use action tokens; Works only with OpenVLA checkpoint
        num_timesteps: int = 20,  # Number of prediction time steps
        rotation_components: int = 9,  # Number of rotation componens: euler -> 3, quaternion -> 4, rotmat -> 9
        num_timestep_tokens : int = 3,
        seperate_control_proj: bool = True,  # If True, project control components separately
        timestep_proj_config: Dict[str, Any] = {},
        token_proj_config: Dict[str, Any] = {},
        transformer_config: Dict[str, Any] = {},
        # prismatic_config: PrismaticConfig,
        # waypointer_config: Dict[str, Any],
        # **kwargs: str,
    ):

        super().__init__(**prismatic_config)
        # super().__init__()
        self.prismatic_config = PrismaticConfig(**prismatic_config)

        self.token_size = token_size
        self.cheat = cheat
        self.num_timesteps = num_timesteps
        self.rotation_components = rotation_components
        self.seperate_control_proj = seperate_control_proj
        self.timestep_proj_config = timestep_proj_config
        self.token_proj_config = token_proj_config
        self.transformer_config = transformer_config
        # self.num_timestep_tokens = num_timestep_tokens

    @property
    def control_components(self) -> int:
        # Number of control dimensions: 3 translation, N rotation, 1 gripper
        return 3 + self.rotation_components + 1

    @property
    def num_timestep_tokens(self) -> int:
        return self.timestep_proj_config['num_tokens']

class OpenVLAConfig(PrismaticConfig):
    model_type: str = "openvla"

    def __init__(
        self,
        norm_stats: Optional[Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = None,
        n_action_bins: int = 256,
        **kwargs: str,
    ) -> None:
        self.norm_stats, self.n_action_bins = norm_stats, n_action_bins

        super().__init__(**kwargs)