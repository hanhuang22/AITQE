import logging
from typing import List, Optional
from transformers.configuration_utils import PretrainedConfig
from transformers import Qwen2ForCausalLM
from transformers import SiglipImageProcessor, SiglipVisionModel, SiglipVisionConfig
import torch
import torch.nn as nn
import math
from transformers import PreTrainedModel
from data_processor import MMInferenceProcessor
logger = logging.getLogger(__name__)


class MMConfig(PretrainedConfig):
    model_type = "mm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=4096,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        language_model_name_or_path=None,
        vision_model_name_or_path=None,
        mm_projector_type="mlp",
        image_aspect_ratio="square",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.language_model_name_or_path = language_model_name_or_path
        self.vision_model_name_or_path = vision_model_name_or_path
        self.mm_projector_type = mm_projector_type
        self.image_aspect_ratio = image_aspect_ratio

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def set_additional_attributes(self, additional_attributes):
        if additional_attributes and isinstance(additional_attributes, dict):
            for attr, value in additional_attributes.items():
                print(attr, value)
                setattr(self, attr, value)

    def _rope_scaling_validation(self):
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


class MMForCausalLM(PreTrainedModel):
    config_class = MMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(self, config, load_checkpoint=False):
        super().__init__(config)
        self.use_varlen = False
        assert self.config.image_aspect_ratio in {"square"}
        logger.info('#'*100)
        logger.info(f'Utilize {self.config.image_aspect_ratio}')
        logger.info('#'*100)
        self.left_padding = not self.training
        if load_checkpoint:
            self.model = Qwen2ForCausalLM.from_pretrained(self.config.language_model_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        else:
            self.model = Qwen2ForCausalLM._from_config(self.config, attn_implementation="flash_attention_2")
        self._no_split_modules = self.model._no_split_modules

        self.get_vision_model(config, load_checkpoint)

    def get_vision_model(self, config, load_checkpoint):
        self.image_processor = SiglipImageProcessor.from_pretrained(self.config.vision_model_name_or_path)
        if load_checkpoint:
            self.vision_model = SiglipVisionModel.from_pretrained(self.config.vision_model_name_or_path)
            self.vision_config = self.vision_model.config
        else:
            self.vision_config = SiglipVisionConfig.from_pretrained(self.config.vision_model_name_or_path)
            self.vision_model = SiglipVisionModel._from_config(self.vision_config)
        self.vision_model._no_split_modules = []

        size = self.image_processor.size
        if not isinstance(size, int):
            if "height" in size:
                assert size["height"] == size["width"], "need to be square, but image_processor's size is not"
                size = size["height"]
            elif "shortest_edge" in size:
                size = size["shortest_edge"]
        self.image_size = size
        
        self.patch_size = self.config.pesudo_patch_size
        modules = [nn.Linear(self.config.visual_hidden_dim, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size, bias=False)]

        if self.config.mm_projector_type == "mlp":
            self.mm_projector = nn.Sequential(*modules)
            self.num_queries = math.floor((self.config.pesudo_img_size / (self.config.pesudo_patch_size )))**2
            self.feature_map_size = int(self.config.pesudo_img_size / (self.config.pesudo_patch_size))

        
        embed_std = 1 / torch.sqrt(torch.tensor(config.hidden_size, dtype=torch.float32))
        self.image_newline = nn.Parameter(torch.randn(config.hidden_size) * embed_std.to(self.model.dtype))


    def bind_processor(self, tokenizer, device=None, config=None, **kwargs):
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        return MMInferenceProcessor(
            tokenizer=tokenizer,
            image_processor=self.image_processor,
            dtype=self.model.dtype,
            device=device if device else self.model.device,
            config=config,
            **kwargs
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        batch_image_patch: Optional[torch.Tensor] = None,
        batch_image_size: Optional[torch.Tensor] = None,
        batch_mm_obj_type: Optional[str] = None,
        batch_mm_obj_image_aspect_ratio: Optional[str] = None,
        img_attention_mask_list: Optional[List[torch.FloatTensor]] = None,
        return_dict: Optional[bool] = None,
    ):

        if inputs_embeds is None:
            (
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                cur_len_list,
            ) = self._merge_input_ids_with_image_features(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                labels,
                images,
                batch_image_patch,
                batch_image_size,
                batch_mm_obj_type,
                batch_mm_obj_image_aspect_ratio,
                img_attention_mask_list=img_attention_mask_list if img_attention_mask_list is not None else None,
                varlen=self.use_varlen,
                left_padding=self.left_padding
            )
        kwargs = {}
        if cur_len_list is not None:
            kwargs["seqlens"] = cur_len_list
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def mm_resolution_merge(self, image_features, batch_image_patch, batch_image_size, batch_mm_obj_type, batch_mm_obj_image_aspect_ratio, img_attention_mask_list=None):
        total_img_num = 0
        result_image_features = []
        result_image_tokens = []
        for i, (image_patch, image_size, mm_obj_type, mm_obj_image_aspect_ratio) in enumerate(zip(batch_image_patch, batch_image_size, batch_mm_obj_type, batch_mm_obj_image_aspect_ratio)):
            image_size = tuple(x.item() for x in image_size.cpu())
            image_patch = tuple(x.item() for x in image_patch.cpu())
            cur_img_num = image_patch[0] * image_patch[1]
            cur_image_features = image_features[total_img_num:total_img_num+cur_img_num]
            total_img_num += cur_img_num
            if 'square' in mm_obj_image_aspect_ratio:
                if mm_obj_type == 'I':
                    cur_image_features = self.mm_projector(cur_image_features).to(image_features.device)

                    if self.image_newline == None:
                        cur_image_features = cur_image_features[0]
                    else:
                        cur_image_features = torch.cat((cur_image_features[0], self.image_newline[None][0:0]))
            result_image_features.append(cur_image_features)
            result_image_tokens.append(cur_image_features.shape[0])
        result_image_features = torch.cat(result_image_features, dim=0)

        result_image_tokens = torch.tensor(result_image_tokens)
        return result_image_features, result_image_tokens
    
    def _merge_input_ids_with_image_features(
        self, input_ids, attention_mask, position_ids, past_key_values, labels, images, batch_image_patch, batch_image_size, 
        batch_mm_obj_type, batch_mm_obj_image_aspect_ratio,
        varlen=False, left_padding=True, img_attention_mask_list=None, 
    ):

        if input_ids.shape[1] == 1 and images is None:
            target_shape = past_key_values[-1][-1].shape[-2] + 1
            attention_mask = torch.cat((attention_mask, torch.ones(
                (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )), dim=1)
            position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            if not self.training:
                attention_mask = None
            return input_ids, attention_mask, position_ids, past_key_values, None, labels, None
        image_features = self.vision_model(images, output_hidden_states=True).hidden_states[-2]
        if not "siglip" in self.config.vision_model_name_or_path.lower():
            image_features = image_features[:, 1:]
        image_features, batch_image_tokens = self.mm_resolution_merge(image_features, batch_image_patch, batch_image_size, batch_mm_obj_type, batch_mm_obj_image_aspect_ratio)

        combine_mask_with_pad = input_ids.clone()
        combine_mask_with_pad = torch.clamp(combine_mask_with_pad, min=0)
        combine_embedding = self.model.model.embed_tokens(combine_mask_with_pad)
        special_image_token_mask = input_ids == -1
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        image_feature_idx = torch.nonzero(special_image_token_mask)
        if num_special_image_tokens.sum() > 0:
            combine_embedding.index_put_([image_feature_idx[:, 0], image_feature_idx[:, 1]], image_features)
        else:
            combine_embedding[0] = torch.cat([combine_embedding[0], image_features[0:0]])
        final_embedding = combine_embedding

        batch_size, final_token_length = input_ids.shape
        final_attention_mask = torch.ones(batch_size, final_token_length, dtype=torch.bool, device=input_ids.device)
        final_attention_mask[(input_ids == self.config.pad_token_index)] = 0

        token_length_list = torch.sum((final_attention_mask != self.config.pad_token_index), dim=-1)
        final_labels = labels

        nb_image_pad = final_token_length - token_length_list
        if not left_padding:
            nb_image_pad.zero_()

        return None, final_attention_mask, None, None, final_embedding, final_labels, None

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        images = kwargs.pop("images", None)
        batch_image_patch = kwargs.pop("batch_image_patch", None)
        batch_image_size = kwargs.pop("batch_image_size", None)
        batch_mm_obj_type = kwargs.pop("batch_mm_obj_type", None)
        batch_mm_obj_image_aspect_ratio = kwargs.pop("batch_mm_obj_image_aspect_ratio", None)
        img_attention_mask_list = kwargs.pop("img_attention_mask_list", None)

        if past_key_values:
            input_ids = input_ids[:, -1:]
            images = None
            batch_image_patch = None
            batch_image_size = None
            batch_mm_obj_type = None
            batch_mm_obj_image_aspect_ratio = None
            img_attention_mask_list = None

        model_inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "past_key_values": past_key_values,
                        "use_cache": kwargs.get("use_cache"),
                        "images": images,
                        "batch_image_patch": batch_image_patch,
                        "batch_image_size": batch_image_size,
                        "batch_mm_obj_type": batch_mm_obj_type,
                        "batch_mm_obj_image_aspect_ratio": batch_mm_obj_image_aspect_ratio,
                        "img_attention_mask_list": img_attention_mask_list,
                    }

        return model_inputs
