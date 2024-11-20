
import logging
from dataclasses import dataclass
from functools import partial,cached_property
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union, Type
# from barrel.components.nn.layers.nerf_pos_embed import NeRFPositionalEmbedding

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
from transformers.modeling_outputs import ModelOutput
import collections
import math
from src.configuration_utils import OpenVLAConfig, PrismaticConfig , TrajectoryVLAConfig, WaypointTokenizer
# from barrel.pipes.vlams.models.control.token_proj import TokenProjector
from src.datatypes import *
import os
from PIL import Image
from pathlib import Path
from torch.amp.autocast_mode import autocast  # Corrected import for latest PyTorch
from scipy.spatial.transform import Rotation as R
# import automodel
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

ht_token_path = Path(".hf_token")
HF_TOKEN  = ht_token_path.read_text().strip() if isinstance(ht_token_path, Path) else ht_token_path

# Get Logger
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"

        self.dino_featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=True,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.dino_featurizer.eval()

        self.embed_dim = self.dino_featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        # if self.use_fused_vision_backbone:
        self.siglip_featurizer = timm.create_model(
            timm_model_ids[1],
            pretrained=True,
            num_classes=0,
            img_size=image_sizes[1],
            act_layer=timm_override_act_layers[1],)

        self.siglip_featurizer.eval()

        self.dino_featurizer.forward = partial(
            self.dino_featurizer.forward_intermediates,
            indices=[len(self.dino_featurizer.blocks) - 2],
            return_prefix_tokens=False,
            norm=False,
            stop_early=True,
            output_fmt='NLC',
            intermediates_only=True,
        )
        self.siglip_featurizer.forward = partial(
            self.siglip_featurizer.forward_intermediates,
            indices=[len(self.siglip_featurizer.blocks) - 2],
            return_prefix_tokens=False,
            norm=False,
            stop_early=True,
            output_fmt='NLC',
            intermediates_only=True,
        )
        self.embed_dim += self.siglip_featurizer.embed_dim

    def forward(self, pixel_values) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        # img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        img = pixel_values['dino']
        img_fused = pixel_values['siglip']
        patches, patches_fused = self.dino_featurizer(img)[0], self.siglip_featurizer(img_fused)[0]

        return torch.cat([patches, patches_fused], dim=2)



class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.initial_projection_dim = vision_dim * 4
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(vision_dim, self.initial_projection_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(self.initial_projection_dim, llm_dim, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(llm_dim, llm_dim, bias=True),
        )

    def forward(self, fused_img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(fused_img_patches)

# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa

class LLMBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.llm : AutoModelForCausalLM
        self.tokenizer = self._create_tokenizer()

    def _create_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        # Load (Fast) Tokenizer
        print(f"Loading (Fast) Tokenizer via the AutoTokenizer API")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config['hf_model_id'],
            model_max_length=self.config['llm_max_length'],
            token=HF_TOKEN,
            padding_side="right",
        )

        # Validation =>> Our VLM logic currently operates under the assumption that the tokenization of a new input
        #                starts with a <BOS> token unless `add_special_tokens = False`; for these models, we empirically
        #                find that adding image patches *after* the BOS leads to much better performance.
        #
        # As a result we explicitly validate that a tokenizer conforms to the expected behavior; if you're reading this
        # line, it's probably because you're adding a new LLM with a different tokenizer behavior. If so, feel free to
        # override the `SPECIAL_CASES` set below, but make sure to make the appropriate changes in the `datasets.py`
        # and VLM `forward()` logic!
        SPECIAL_CASES = {
            # Phi-2 Tokenizer doesn't add any BOS tokens by default, and sets BOS == EOS == "<|endoftext|>"
            #   =>> We'll prepend BOS to first input (to play nicely with image token insertion logic; verified that
            #       this works well with base LLM generation.
            #   =>> Like Llama-2 Tokenizers -- we'll add a special PAD token for training purposes.
            "microsoft/phi-2",
        }
        if self.config['hf_model_id'] not in SPECIAL_CASES:
            # Note =>> this assert should hold for all Llama-derived tokenizers (`LlamaTokenizerFast` ==> includes Mistral!
            assert (
                tokenizer("Test 123", add_special_tokens=True).input_ids[0] == tokenizer.bos_token_id
            ) and (
                tokenizer("Test 123", add_special_tokens=False).input_ids[0] != tokenizer.bos_token_id
            ), f"Default Tokenizer of type `{type(tokenizer)}` does not automatically prefix inputs with BOS token!\n"

        return tokenizer

# @AutoModel.register(PrismaticConfig)
class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    # model_type: ClassVar[str] = "prismatic"
    config_class: PretrainedConfig = PrismaticConfig
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)
        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        # if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
        #     raise NotImplementedError(
        #         "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
        #         "if you urgently need support for latest TIMM versions."
        #     )

        # if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
        #     logger.warning(
        #         f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
        #         f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
        #         f"there might be inference-time regressions due to dependency changes. If in doubt, please"
        #         f"use the above versions."
        #     )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.llm_backbone = LLMBackbone({'hf_model_id': config.hf_llm_id, 'llm_max_length': config.llm_max_length, "pad_token_id" :32000,
        "pad_to_multiple_of" : 64,})

        # self.llm_backbone.llm = AutoModelForCausalLM.from_config(
        #     config.text_config, attn_implementation="flash_attention_2"
        # )
        self.llm_backbone.llm = AutoModelForCausalLM.from_pretrained(
                'meta-llama/Llama-2-7b-hf',
                token=HF_TOKEN,
                attn_implementation='flash_attention_2',
                # The following parameters are set to prevent `UserWarnings` from HF; we want greedy decoding!
                do_sample=False,
                temperature=1.0,
                use_cache=False,
                top_p=1.0, )

        self.llm_backbone.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm_backbone.llm.config.pad_token_id = self.llm_backbone.tokenizer.pad_token_id
        self.llm_backbone.llm.resize_token_embeddings(len(self.llm_backbone.tokenizer), pad_to_multiple_of=64)



        # self.llm_backbone.llm.config.pad_token_id = self.llm_backbone.tokenizer.pad_token_id
        # self.llm_backbone.llm.resize_token_embeddings(len(self.llm_backbone.tokenizer), pad_to_multiple_of=64)
        # self.resize_token_embeddings(32001,64)

        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.llm_backbone.llm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.llm_backbone.llm.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.llm_backbone.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.llm_backbone.llm.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.llm_backbone.llm.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.llm_backbone.llm.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.llm_backbone.llm.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    # def resize_token_embeddings(
    #     self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    # ) -> nn.Embedding:
    #     updated_embeddings = self.llm_backbone.llm.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

    #     # Update config/instance variables
    #     self.config.text_config.vocab_size = updated_embeddings.num_embeddings
    #     self.vocab_size = updated_embeddings.num_embeddings

    #     return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] ,
        attention_mask: Optional[torch.Tensor],
        # pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values: Dict[str, torch.Tensor] = {},
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.llm_backbone.llm(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.llm_backbone.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===

        elif (input_ids.shape[0] == pixel_values['dino'].shape[0]) or (inputs_embeds.shape[0] == pixel_values['dino'].shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)

            projected_patch_embeddings = self.projector(patch_features)  ## matches
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            # Dispatch to Language Model
            language_model_output = self.llm_backbone.llm(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output


        return (PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        ),patch_features,multimodal_attention_mask)

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class TokenProjectorConfig(PretrainedConfig):
    vit_tokens_layers: List[int] = []  # If empty, torch.nn.Identity
    llm_image_tokens_layers: List[int] = []  # If empty, torch.nn.Identity
    control_tokens_layers: List[int] = []  # If empty, torch.nn.Identity

    # image_tokens_mode:
    #   vit: use ViT tokens only
    #   llm: use LLM tokens only
    #   skip: skip connection between projector(ViT) and LLM with addition
    #   none: don't feed to TokenProjector
    image_tokens_mode: str

    def __post_init__(self):
        super().__post_init__()

        if self.image_tokens_mode == 'vit':
            assert len(self.vit_tokens_layers) > 0 or len(self.control_tokens_layers) > 0
        elif self.image_tokens_mode == 'llm':
            assert len(self.vit_tokens_layers) > 0 or len(self.control_tokens_layers) > 0
        elif self.image_tokens_mode == 'skip':
            assert len(self.vit_tokens_layers) > 0 or len(self.llm_image_tokens_layers) > 0
        elif self.image_tokens_mode == 'none':
            assert len(self.vit_tokens_layers) == 0
            assert len(self.llm_image_tokens_layers) == 0
        else:
            raise NotImplementedError(f"Unknown image tokens mode {self.image_tokens_mode}")

class TokenProjector(nn.Module):
    """Project and pack VLM output tokens"""

    def __init__(self, config):
        super().__init__()
        self.config = TokenProjectorConfig()
        self.config.vit_tokens_layers = config['vit_tokens_layers']
        self.config.llm_image_tokens_layers = config['llm_image_tokens_layers']
        self.config.control_tokens_layers = config['control_tokens_layers']
        self.config.image_tokens_mode = config['image_tokens_mode']

        self.vit_tokens_proj = self._make_token_proj_module(self.config.vit_tokens_layers)
        self.llm_image_tokens_proj = self._make_token_proj_module(self.config.llm_image_tokens_layers)
        self.control_tokens_proj = self._make_token_proj_module(self.config.control_tokens_layers)

    def forward(self, inputs: WaypointerInput) -> torch.Tensor:
        """
        Args:
            inputs: Contains VLM outputs
        Returns:
            torch.Tensor of shape [B, num_tokens, token_size] that always contains the control tokens
            and possibly the image tokens (prepended), depending on the configuration
        """

        vit_tokens = self.vit_tokens_proj(inputs.vit_tokens)
        control_tokens = self.control_tokens_proj(inputs.control_tokens)
        llm_image_tokens = self.llm_image_tokens_proj(inputs.llm_image_tokens)

        if self.config.image_tokens_mode == 'vit':
            output = torch.cat([vit_tokens, control_tokens], dim=1)  # [B, img + control, token_size]
        elif self.config.image_tokens_mode == 'llm':
            output = torch.cat([llm_image_tokens, control_tokens], dim=1)  # [B, img + control, token_size]
        elif self.config.image_tokens_mode == 'skip':
            image_tokens = llm_image_tokens + vit_tokens
            output = torch.cat([image_tokens, control_tokens], dim=1)  # [B, img + control, token_size]
        elif self.config.image_tokens_mode == 'none':
            output = control_tokens
        else:
            raise NotImplementedError(f"Unknown image tokens mode {self.config.image_tokens_mode}")

        return output

    def _make_token_proj_module(self, layer_sizes: List[int]) -> torch.nn.Module:
        if len(layer_sizes) == 0:
            return torch.nn.Identity()

        assert len(layer_sizes) > 1, "Need to provide input and output layer sizes at least"

        module = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    collections.OrderedDict(
                        {
                            'linear': torch.nn.Linear(layer_in_features, layer_out_features),
                            'act': torch.nn.ReLU(),
                            'norm': torch.nn.LayerNorm(layer_out_features),
                        }
                    )
                )
                for layer_in_features, layer_out_features in zip(layer_sizes[:-1], layer_sizes[1:])
            ]
        )
        return module

class NeRFPositionalEmbedding(torch.nn.Module):
    def __init__(self, proj_scale: int):
        """
        Args:
            proj_scale: Dimension size, same as L parameter in the NeRF paper
        """
        super().__init__()
        self.proj_scale = proj_scale

        freq = 2 ** torch.arange(self.proj_scale, dtype=torch.float32) * math.pi  # size: [L]

        self.register_buffer('freq', freq)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Maps values from R^N to a higher dimensional space R^(N2L)
        Args:
            inputs: torch.Tensor of shape [B, ..., N]; input values to be transformed
        Returns: torch.Tensor of shape [B, ..., N2L]; encoded input values
        """

        spectrum = self.freq.view(*[1] * inputs.ndim, -1) * inputs.unsqueeze(-1)  # [B, ..., N, L]
        encoding = torch.stack([torch.sin(spectrum), torch.cos(spectrum)], dim=-2)  # [B, ..., N, 2, L]
        encoding = encoding.view(inputs.shape[-1], -1)  # [B, ..., N2L]

        return encoding

class TimestepProjModuleConfig(PretrainedConfig):
    pos_embed_scale: int  # How much to scale timestep values when doing position embedding
    proj_layers: List[int]
    time_delta_sec: float = 0.25  # Time delta between two predictions
    num_tokens: int = 3  # Number of tokens per timestep; Currently 3 - translation, rotation, gripper


class TimestepProjModule(nn.Module):

    def __init__(self, config: TimestepProjModuleConfig, num_timesteps: int, token_size: int):
        """
        Args:
            num_timesteps: Number of control timesteps
            token_size: Single token size
        """
        super().__init__()
        self.config = TimestepProjModuleConfig()
        self.config.pos_embed_scale = config['pos_embed_scale']
        self.config.proj_layers = config['proj_layers']
        self.config.time_delta_sec = config['time_delta_sec']
        self.config.num_tokens = config['num_tokens']

        self.num_timesteps = num_timesteps
        self.token_size = token_size

        input_size = 2 * self.config.pos_embed_scale

        self.pos_embed = NeRFPositionalEmbedding(self.config.pos_embed_scale)

        # We output one token for translation, one for rotation and one for gripper state
        feature_size = self.config.num_tokens * self.token_size

        # Make MLP projection

        self.timestep_proj = self._make_timestep_proj(in_features=int(input_size), out_features=int(feature_size))

    def _make_timestep_proj(self, in_features: int, out_features: int) -> torch.nn.Module:
        layer_sizes = [in_features] + list(self.config.proj_layers) + [out_features]
        module = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    collections.OrderedDict(
                        {
                            'linear': torch.nn.Linear(layer_in_features, layer_out_features),
                            'act': torch.nn.ReLU(),
                            'norm': torch.nn.LayerNorm(layer_out_features),
                        }
                    )
                )
                for layer_in_features, layer_out_features in zip(layer_sizes[:-1], layer_sizes[1:])
            ]
        )
        return module

    def forward(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor of sequence of timestep tokens, shape [1, num_timesteps * num_tokens, token_size]
        """
        device = self.timestep_proj[0].linear.weight.device  # type: ignore[index]

        # Position encode timesteps
        time_deltas_norm = self.time_deltas_norm.view(1, self.num_timesteps)  # [1, num_timesteps]
        time_deltas_norm = time_deltas_norm.to(device=device)

        # Embed timesteps to intermediate dimension
        timesteps_embed = self.pos_embed(time_deltas_norm)  # [1, num_timesteps * 2 * L]
        timesteps_embed = timesteps_embed.view(self.num_timesteps, -1)  # [num_timesteps, 2 * L]

        # Project the timesteps via MLP to tokens
        timesteps_tokens = self.timestep_proj(timesteps_embed)  # [num_timesteps, token_size * 3]

        # Reshape MLP outputs into tokens
        timesteps_tokens = timesteps_tokens.view(  # [1, num_timesteps * 3, token_size]
            1, self.num_timesteps * self.config.num_tokens, self.token_size
        )

        return timesteps_tokens

    @cached_property
    def time_deltas_sec(self) -> torch.Tensor:
        return torch.arange(0, self.num_timesteps, 1, dtype=torch.float32) * self.config.time_delta_sec

    @cached_property
    def time_deltas_norm(self) -> torch.Tensor:
        # Normalize time deltas between [0, 1]. We are saving [-1, 0] interval for possible past supervision
        if self.time_deltas_sec.shape[0] == 1:
            # Can't divide by 0
            time_deltas_norm = self.time_deltas_sec
        else:
            time_deltas_norm = self.time_deltas_sec / self.time_deltas_sec.max()  # [num_timesteps]
        return time_deltas_norm.detach()


# class Waypointer(nn.Module):
# @AutoModel.register(TrajectoryVLAConfig)
class TrajectoryVLA(PrismaticForConditionalGeneration):


    config_class: PretrainedConfig = TrajectoryVLAConfig

    def __init__(self, config: TrajectoryVLAConfig) -> None:
        super().__init__(config.prismatic_config)

        self.control_tokenizer = WaypointTokenizer(self.llm_backbone.tokenizer)
        self.timestep_proj = TimestepProjModule(
            config.timestep_proj_config,
            num_timesteps=config.num_timesteps,
            token_size=config.token_size, )
        self.num_timesteps = config.num_timesteps
        self.token_proj = TokenProjector(config.token_proj_config)
        self.transformer = DETR(config.transformer_config)
        self.token_size = config.token_size
        self.rotation_components = config.rotation_components
        # if self.config.separate_control_proj:
        # Project translation, rotation and gripper separately. Each timestep is projected separately
        self.translation_proj = torch.nn.Sequential(
            torch.nn.Linear(in_features=config.token_size, out_features=config.token_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config.token_size // 2, out_features=3),
        )
        self.rotation_proj = torch.nn.Sequential(
            torch.nn.Linear(in_features=config.token_size, out_features=config.token_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(
                in_features=config.token_size // 2, out_features=config.rotation_components
            ),
        )

        self.gripper_proj = torch.nn.Sequential(
            torch.nn.Linear(in_features=config.token_size, out_features=config.token_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=config.token_size // 2, out_features=1),
        )

    def _pack_waypointer_input(self, input_ids: torch.Tensor, vlm_output: PrismaticCausalLMOutputWithPast,vit_tokens,fused_attention_mask) -> WaypointerInput:
        # Get the LLM output
        # assert vlm_output.llm_output.hidden_states is not None
        projected_tokens = vlm_output.hidden_states[-1]

        control_tokens = self._extract_control_tokens(input_ids, projected_tokens)  # type: ignore

        num_image_tokens = vit_tokens.shape[1]  # type: ignore[union-attr]
        # TODO: This assumes a specific position of image tokens in the sequence. Make general
        llm_image_tokens = projected_tokens[..., 1 : 1 + num_image_tokens, :]


        return WaypointerInput(
            vit_tokens=vit_tokens,
            llm_image_tokens=llm_image_tokens,
            control_tokens=control_tokens,
            llm_tokens=projected_tokens,
            attn_mask=fused_attention_mask,
        )

    def predict_tracks(self,inputs):

        vlm_output,vit_tokens,fused_attention_mask = super().forward(**inputs,output_hidden_states=True,output_attentions=True,return_dict=True)
        waypointer_input = self._pack_waypointer_input(inputs['input_ids'], vlm_output,vit_tokens,fused_attention_mask)
        waypoint_output = self._waypointer_forward(waypointer_input)
        translation, rotation, gripper = torch.split(
            waypoint_output, [3, self.rotation_components, 1], dim=-1 )
        translation, rotation, gripper = self.process_output(translation, rotation, gripper)
        return translation, rotation, gripper
    def process_output(self,translation,rotation,gripper):
        ## convert rotation from matrix to  euler angles
        euler_angles = []
        for matrix in rotation[0]:
            # Convert each rotation matrix to a Rotation object
            rotation_obj = R.from_matrix(matrix.view(3, 3).detach().cpu().float().numpy().squeeze())
            # Convert to Euler angles in radians with chosen convention, e.g., 'xyz'
            euler_angle = rotation_obj.as_euler('xyz', degrees=False)
            euler_angles.append(euler_angle)

        translation = translation.detach().cpu().float().numpy().squeeze()
        ## sigmoid and clip from 0-1
        gripper = np.round(torch.sigmoid(gripper).detach().cpu().float().numpy().squeeze())
        return translation,euler_angles,gripper

    def _extract_control_tokens(self, input_ids: torch.Tensor, output_tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract the action tokens from the LLM output sequence. Assumes the following order
            [image_tokens, language_tokens, action_tokens, padding]

        Args:
            input_ids: IDs of the tokens in text input sequence; shape [B, S]
            output_tokens: Token sequence output from LLM; shape [B, L, token_size]. Note the length is
                different from input_ids as it also contains image tokens
        Returns:
            torch.Tensor of shape [B, 7, token_size] containing only action tokens
        """

        assert input_ids.ndim == 2
        assert output_tokens.ndim == 3
        batch, in_seq_len, out_seq_len = *input_ids.shape, output_tokens.shape[1]

        device = input_ids.device

        num_control_tokens = self.control_tokenizer.num_control_tokens  # type: ignore[attr-defined]

        control_token_ids = torch.from_numpy(  # type: ignore[attr-defined]
            self.control_tokenizer.control_token_ids  # type: ignore[attr-defined]
        )
        control_token_ids = control_token_ids.to(dtype=input_ids.dtype, device=input_ids.device)
        is_control_token = torch.any(  # shape: [B, S]
            input_ids.unsqueeze(-1) == control_token_ids.view(1, 1, -1),
            dim=-1,
        )
        if not torch.all(mask := is_control_token.sum(dim=-1) == num_control_tokens):
            raise RuntimeError(
                f"Can't properly detect control tokens with ids {control_token_ids} of len="
                f"{len(control_token_ids)} in input_ids {input_ids}. Rows mask: {mask}"
            )

        # Pad is_control_tokens mask to the LLM output sequence size
        tokens_mask = torch.cat(  # shape: [B, L]
            [
                torch.zeros(batch, out_seq_len - in_seq_len, dtype=torch.bool, device=device),
                is_control_token.to(torch.bool),
            ],
            dim=1,
        )

        control_tokens = output_tokens[tokens_mask]  # shape: 1D tensor
        control_tokens = control_tokens.view(  # [B, num_control_tokens, token_size]
            batch, num_control_tokens, output_tokens.shape[-1]
        )

        return control_tokens

    def _waypointer_forward(self, inputs:WaypointerInput):

        timesteps_tokens = self.timestep_proj()  # [1, num_timesteps * 3, token_size]

        # Project and pack LLM tokens
        llm_tokens = self.token_proj(inputs)  # [B, num_tokens, token_size]

        # TODO: Pass inputs.attn_mask if you start using the LLM tokens
        output_tokens = self.transformer(  # [B, num_timesteps * 3, token_size]
            feature_tokens=llm_tokens, query_tokens=timesteps_tokens, attn_mask=None
        )

        output_tokens = output_tokens.view(  # [B, num_timesteps, 3 * token_size]
            -1, self.num_timesteps, 3 * self.token_size
        )

        # if self.config.separate_control_proj:
            # [B, num_timesteps, token_size] each
        translation_tokens, rotation_tokens, gripper_tokens = torch.split(
            output_tokens, [self.token_size] * 3, dim=-1
        )

        translation = self.translation_proj(translation_tokens)  # [B, num_timesteps, 3]
        rotation = self.rotation_proj(rotation_tokens)  # [B, num_timesteps, rotation_components]
        gripper = self.gripper_proj(gripper_tokens)  # [B, num_timesteps, 1]

        output = torch.cat(  # [B, num_timesteps, control_components]
            [translation, rotation, gripper], dim=-1
        )

        return output
    # def predict_waypoints(self,input_ids: Optional[torch.LongTensor] = None, **kwargs: str) -> np.ndarray:
    #     vlm_output = super().forward(
    #         inputs=input_ids,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=True,
    #         return_dict=return_dict,
    #     )


    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None and len(norm_stats) != 1:
            raise ValueError(
                f"Your model was trained on more than one dataset. "
                f"Please pass a `unnorm_key` from the following options to choose the statistics used for "
                f"de-normalizing actions: {norm_stats.keys()}"
            )

        # If None, grab the (singular) dataset in `norm_stats` to use as `unnorm_key`
        unnorm_key = unnorm_key if unnorm_key is not None else next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"The `unnorm_key` you chose ({unnorm_key = }) is not in the available statistics. "
                f"Please choose from: {norm_stats.keys()}"
            )

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

# from barrel.components.nn.modules import ConfigurableModule, ConfigurableModuleConfig
@dataclass
class LearnedPosEmbed1DConfig(PretrainedConfig):
    num_embeddings: int  # Number of embeddings, i.e. maximum sequence length (per dimension)
    embedding_dim: int  # Embedding dimension, usually same as token size
    # hyperparams: Dict[str, Any] = {}


class LearnedPosEmbed1D(nn.Module):
    """Learned 2D positional embeddings"""

    def __init__(self, config: LearnedPosEmbed1DConfig):
        super().__init__()
        self.config = config
        self.embed = torch.nn.Embedding(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            # **self.config.hyperparams,
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.embed.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: torch.Tensor of shape [B, S, token_size]
        Returns:
            torch.Tensor of shape [B, S, token_size]
        """
        assert tokens.ndim == 3, f"Expected tensor with 3 dims, but got {tokens.shape}"
        batch, seq_len = tokens.shape[:2]

        if seq_len > self.config.num_embeddings:
            raise ValueError(
                f"{self.__class__.__name__} got sequence of length {seq_len}, which "
                f"is longer than the max configured length {self.config.num_embeddings}"
            )

        indices = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)

        embed = self.embed(indices)  # shape: [S, embedding_dim]
        pos = embed.unsqueeze(0).repeat(batch, 1, 1)  # shape: [B, S, embedding_dim]

        return pos
@dataclass
class TransformerEncoderBlockConfig(PretrainedConfig):
    feature_size: int  # Number of input, intermediate and output features
    head_dim: int = 64  # Number of features that Q, K, V are each projected to in a single head
    num_heads: int = 16  # Number of parallel heads in MultiheadAttention
    norm: str = 'LayerNorm'  # Type of normalization to use. RMSNorm or LayerNorm
    activation: str = 'GELU'  # Type of activation for the MLP. Usually one of ReLU, SiLU, GELU

    def __post_init__(self):
        # super().__post_init__()
        assert self.norm in ['RMSNorm', 'LayerNorm'], self.norm
        assert hasattr(torch.nn, self.activation), f"No such module torch.nn.{self.activation}"

    @property
    def TorchNorm(self) -> Type[torch.nn.Module]:
        return getattr(torch.nn, self.norm)

    @property
    def TorchActivation(self) -> Type[torch.nn.Module]:
        return getattr(torch.nn, self.activation)

@dataclass
class TransformerDecoderBlockConfig(PretrainedConfig):
    feature_size: int  # Number of input, intermediate and output features
    head_dim: int = 64  # Number of features that Q, K, V are each projected to in a single head
    num_heads: int = 16  # Number of parallel heads in MultiheadAttention
    norm: str = 'LayerNorm'  # Type of normalization to use. RMSNorm or LayerNorm
    activation: str = 'GELU'  # Type of activation for the MLP. Usually one of ReLU, SiLU, GELU
    dropout: float = 0.0
    def __post_init__(self):
        # super().__post_init__()
        assert self.norm in ['RMSNorm', 'LayerNorm'], self.norm
        assert hasattr(torch.nn, self.activation), f"No such module torch.nn.{self.activation}"

    @property
    def TorchNorm(self) -> Type[torch.nn.Module]:
        return getattr(torch.nn, self.norm)

    @property
    def TorchActivation(self) -> Type[torch.nn.Module]:
        return getattr(torch.nn, self.activation)


@dataclass
class DETRConfig(PretrainedConfig):
    # Positional embedding configuration
    encoder_block_config : TransformerEncoderBlockConfig  # Config for each Transformer encoder block
    decoder_block_config : TransformerDecoderBlockConfig # Config for each Transformer decoder block
    pos_embed_config = {"num_embeddings" : 300, "token_size" : 1024}

    num_blocks: int =2


class DETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        # import pdb ; breakpoint()
        self.pos_embed_config = LearnedPosEmbed1DConfig(**config['pos_embed_config'])

        self.pos_embed_feature_tokens = LearnedPosEmbed1D(self.pos_embed_config)

        # Transformer encoder blocks
        self.encoder_block_config = TransformerEncoderBlockConfig(**config['encoder_block_config'])
        self.encoder_blocks = torch.nn.ModuleList(
         [TransformerEncoderBlock(self.encoder_block_config) for _ in range(config['num_blocks'])]
        )

        # Transformer decoder blocks
        self.decoder_block_config = TransformerDecoderBlockConfig(**config['decoder_block_config'])
        self.decoder_blocks = torch.nn.ModuleList(
            [TransformerDecoderBlock(self.decoder_block_config) for _ in range(config['num_blocks'])]
        )

    def forward(
        self,
        feature_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            feature_tokens: Encoded tokens of the model input, e.g. from LLM or conv encoder,
                shape [B, S, token_size]
            query_tokens: Learnable query tokens on the input, shape [1, N, token_size]
            attn_mask: Attention mask of shape [B, S] that prevents attention to tokens in `feature_tokens`
        Returns:
            torch.Tensor with of shape [B, N, token_size] corresponding to encoded query_tokens
        """
        batch_size = feature_tokens.shape[0]
        # Expand query_tokens to batch size
        query_tokens = query_tokens.expand(batch_size, *query_tokens.shape[1:])  # [B, N, token_size]
        output_tokens = query_tokens

        # Compute position embeddings for LLM tokens
        pos_embed_feature_tokens = self.pos_embed_feature_tokens(feature_tokens)  # [B, S, token_size]

        # Apply Transformer encoder and decoder blocks
        for encoder_block in self.encoder_blocks:
            feature_tokens = encoder_block(tokens=feature_tokens, pos_embed=pos_embed_feature_tokens)

        for decoder_block in self.decoder_blocks:
            output_tokens = decoder_block(  # [B, N, token_size]
                learnable_queries=query_tokens,
                query_tokens=output_tokens,
                key_tokens=feature_tokens + pos_embed_feature_tokens,
                value_tokens=feature_tokens,
                key_padding_mask=attn_mask,
            )

        return output_tokens

    @property
    def fsdp_wrap_policy(self) -> Callable:
        # Wrap each transformer block
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerEncoderBlock, TransformerDecoderBlock},
        )



class TransformerEncoderBlock(nn.Module):
    """
    Implementation of DERT TransformerEncoderBlock: https://arxiv.org/pdf/2005.12872
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = self.config.num_heads * self.config.head_dim

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=self.config.num_heads,
            batch_first=True,
        )

        self.attn_norm = self.config.TorchNorm(self.config.feature_size)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.config.feature_size, self.config.feature_size, bias=False),
            self.config.TorchActivation(),
            torch.nn.Linear(self.config.feature_size, self.config.feature_size, bias=False),
        )
        self.output_norm = self.config.TorchNorm(self.config.feature_size)

    def forward(self, tokens: torch.Tensor, pos_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: torch.Tensor of shape [B, S, features]. Input tokens or previous encoder output
            pos_embed: torch.Tensor of same shape as tokens
        Return:
            torch.Tensor of shape [B, S, features]
        """

        residual = tokens

        query = tokens + pos_embed
        key = query
        value = tokens

        x, _ = self.self_attn(query=query, key=key, value=value)  # shape: [B, S, features]

        x = x + residual  # Skip connection
        x = self.attn_norm(x)  # shape: [B, S, features]

        residual = x

        x = self.mlp(x)  # shape: [B, S, features]

        x = x + residual  # Skip connection
        x = self.output_norm(x)  # shape: [B, S, features]

        return x


class TransformerDecoderBlock(nn.Module):
    """
    Implementation of DETR TransformerDecoderBlock: https://arxiv.org/pdf/2005.12872
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = self.config.num_heads * self.config.head_dim

        # self.input_norm = self.config.TorchNorm(self.config.feature_size)

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=self.config.num_heads,
            batch_first=True,
        )

        self.cross_attn = torch.nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=self.config.num_heads,
            batch_first=True,
        )

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.config.feature_size, self.config.feature_size, bias=False),
            self.config.TorchActivation(),
            torch.nn.Dropout(self.config.dropout),
            torch.nn.Linear(self.config.feature_size, self.config.feature_size, bias=False),
        )

        self.norm_1 = self.config.TorchNorm(self.config.feature_size)
        self.norm_2 = self.config.TorchNorm(self.config.feature_size)
        self.norm_3 = self.config.TorchNorm(self.config.feature_size)

        self.dropout_1 = torch.nn.Dropout(self.config.dropout)
        self.dropout_2 = torch.nn.Dropout(self.config.dropout)
        self.dropout_3 = torch.nn.Dropout(self.config.dropout)


    def forward(
        self,
        learnable_queries: torch.Tensor,
        query_tokens: torch.Tensor,
        key_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            learnable_queries: torch.Tensor of shape [B, L, features]. The learnable object queries.
                Added to `query_tokens` for self and cross attention.
            query_tokens: torch.Tensor of shape [B, L, features]. The same as `learnable_queries`
                in the first transformer block and the output of the previous transformer block afterwards
            key_tokens: torch.Tensor of shape [B, S, features]. Already contain positional embedding
            value_tokens: torch.Tensor of shape [B, S, features]. Same as key tokens, but without pos embed
            key_padding_mask: torch.Tensor of shape [B, S]. Optional mask for key_tokens and value_tokens in
                cross attention
        Return:
            torch.Tensor of shape [B, L, features]
        """

        # NOTE: We might need masking for self_attn

        residual = query_tokens

        query = key = query_tokens + learnable_queries  # shape: [B, L, features]; Add learnable tokens
        value = query_tokens

        # Self-attention -> Dropout -> Skip connection -> Norm
        x, _ = self.self_attn(query=query, key=key, value=value)  # shape: [B, L, features]
        x = residual + self.dropout_1(x)  # Skip connection
        x = self.norm_1(x)  # shape: [B, L, features]

        residual = x

        query = x + learnable_queries  # Add learnable tokens
        key = key_tokens  # NOTE: Positional embeddings added to key_tokens from the outside
        value = value_tokens

        # Cross-attention -> Dropout -> Skip connection -> Norm
        x, _ = self.cross_attn(  # shape: [B, L, features]
            query=query, key=key, value=value, key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout_2(x)  # Skip connection
        x = self.norm_2(x)  # shape: [B, L, features]

        residual = x

        # MLP -> Dropout -> Skip connection -> Norm
        x = self.mlp(x)
        x = residual + self.dropout_3(x)  # Skip connection
        x = self.norm_3(x)

        return x
