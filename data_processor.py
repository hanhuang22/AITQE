from dataclasses import dataclass
import io
import json
import logging
import os
import re
import requests
from typing import Dict, Sequence

from PIL import Image
import torch
from transformers.modeling_outputs import ModelOutput

NOLOSS_START = "<noloss_start>"
NOLOSS_END = "<noloss_end>"

logger = logging.getLogger(__name__)

class PuretextError(Exception):
    pass

def can_be_json_loaded(s):
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False

def parse_file_type_from_str(file_type):
    return_type = ""
    if file_type.lower() == "i":
        return_type = "I"

    assert return_type in {"I"}, f"get return_type {return_type}"
    return return_type

@dataclass
class MMDataCollector():
    tokenizer: None
    config: None

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, instances: Sequence[Dict], has_labels=True) -> Dict[str, torch.Tensor]:
        input_ids = tuple([instance["input_ids"] for instance in instances])
        for input_id in input_ids:
            assert self.config.pad_token_index not in input_id, f"self.config.pad_token_index in input_id, {self.config.pad_token_index}, {input_id}"
        pad_dict = self.tokenizer.pad({"input_ids": input_ids})
        input_ids = pad_dict["input_ids"]
        img_patch_list = [instance["img_patch_list"] for instance in instances]
        img_size_list = [instance["img_size_list"] for instance in instances]
        mm_obj_type_list = [instance["mm_obj_type_list"] for instance in instances]
        mm_obj_image_aspect_ratio_list = [instance["mm_obj_image_aspect_ratio_list"] for instance in instances]
        untext_token_mask = [torch.tensor(instance["untext_token_mask"]) for instance in instances]
        untext_token_mask_attnmask = [torch.ones(len(instance["untext_token_mask"])) for instance in instances]
        untext_token_mask = torch.nn.utils.rnn.pad_sequence(untext_token_mask, batch_first=True, padding_value=self.config.pad_token_index)
        untext_token_mask_attnmask = torch.nn.utils.rnn.pad_sequence(untext_token_mask_attnmask, batch_first=True, padding_value=0)
        bs = len(instances)
        _, c, h, w = instances[0]["images"].shape
        batch_images_nums = sum([instance["images"].shape[0] for instance in instances])
        batch_image_patch_list = []
        batch_image_size_list = []
        batch_mm_obj_type_list = []
        batch_mm_obj_image_aspect_ratio_list = []
        batch_img_attention_mask_list = []
        if batch_images_nums > 0:
            batch_images = [instance["images"] for instance in instances]
            batch_images = torch.cat(batch_images, dim=0)
            for i in range(bs):
                batch_image_patch_list.extend(img_patch_list[i])
                batch_image_size_list.extend(img_size_list[i])
                batch_mm_obj_type_list.extend(mm_obj_type_list[i])
                batch_mm_obj_image_aspect_ratio_list.extend(mm_obj_image_aspect_ratio_list[i])
            assert (input_ids == self.config.image_token_index).sum() == len(batch_image_size_list), "images nums not match"
            assert len(batch_image_patch_list) == len(batch_image_size_list), "images nums not match"
        else:
            assert batch_image_patch_list == [], f"batch_image_patch_list is {batch_image_patch_list}, but images is zero"
            batch_images = torch.zeros(1, c, h, w)
            batch_image_patch_list = [(1, 1)]
            batch_image_size_list = [(self.config.pesudo_img_size, self.config.pesudo_img_size)]
            batch_mm_obj_type_list = ["I"]
            batch_mm_obj_image_aspect_ratio_list = ["square"]
            assert (input_ids == self.config.image_token_index).sum() == 0, "images nums not match"
        batch_image_patch_list = torch.tensor(batch_image_patch_list)
        batch_image_size_list = torch.tensor(batch_image_size_list)

        if has_labels:
            untext_token_mask_label = [torch.tensor(instance["untext_token_mask_label"]) for instance in instances]
            untext_token_mask_label = torch.nn.utils.rnn.pad_sequence(untext_token_mask_label, batch_first=True, padding_value=self.config.IGNORE_INDEX)
            labels = tuple([instance["labels"] for instance in instances])
            labels = self.tokenizer.pad({"input_ids": labels}, return_attention_mask=False)["input_ids"]
            assert self.config.pad_token_index == self.tokenizer.pad_token_id, f"self.config.pad_token_index ({self.config.pad_token_index}) not equal to self.tokenizer.pad_token_id ({self.tokenizer.pad_token_id})"
            labels[labels == self.config.pad_token_index] = self.config.IGNORE_INDEX
            labels = labels[:, :self.tokenizer.model_max_length]
            assert self.config.pad_token_index not in labels, f"self.config.pad_token_index in labels, {self.config.pad_token_index}, {labels}"
            assert labels.shape == input_ids.shape, f"input_ids.shape not equal to labels.shape, {labels.shape} {input_ids.shape}"
        else:
            labels = None
            untext_token_mask_label = None

        return dict(
            input_ids=untext_token_mask, # (B, L), torch.int64
            labels=untext_token_mask_label, # (B, L), torch.int64
            attention_mask=untext_token_mask_attnmask.bool(), # (B, L), torch.bool
            images=batch_images, # (N, C, H, W), torch.float32
            batch_image_patch=batch_image_patch_list, # (N), torch.int64
            batch_image_size=batch_image_size_list, # (N, 2), torch.int64
            batch_mm_obj_type=batch_mm_obj_type_list, # (N), str
            batch_mm_obj_image_aspect_ratio=batch_mm_obj_image_aspect_ratio_list, # (N), str
            img_attention_mask_list=batch_img_attention_mask_list
        )

@dataclass
class MMDataProcessor(object):
    config: None
    tokenizer: None
    image_processor: None
    image_root_path: str = ""
    image_aspect_ratio: str = ""
    image_start_token: str = "<img_start>"
    image_end_token: str = "<img_end>"
    image_pad_token: str = "<reserved_113>"
    has_label: bool = False

    def __init__(
        self,
        tokenizer,
        image_processor=None,
        image_root_path="",
        has_label=False,
        config=None,
        question_loss=False,
        text_filter_domain=[],
        is_infra=False,
        target_format='file',
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.add_bos_token = False
        self.add_eos_token = False
        try:
            if self.tokenizer.add_bos_token:
                self.add_bos_token = True
            if self.tokenizer.add_eos_token:
                self.add_eos_token = True
        except:
            pass
        assert not self.add_bos_token, "do not support self.add_bos_token now"
        assert not self.add_eos_token, "do not support self.add_eos_token = True"

        self.image_processor = image_processor
        self.has_label = has_label
        image_tile_tokens = config.image_tile_tokens
        self.image_tile_tokens = image_tile_tokens
        self.raw_image_tile_tokens = config.image_tile_tokens
        self.kernel = self.config.pooling_kernel
        assert self.image_tile_tokens > 0
        size = self.config.pesudo_img_size
        self.image_size = size
        self.patch_size = self.config.pesudo_patch_size
        crop_size = self.image_processor.crop_size
        if not isinstance(crop_size, int):
            if "height" in crop_size:
                assert crop_size["height"] == crop_size["width"], "need to be square, but image_processor's size is not"
                crop_size = crop_size["height"]
            else:
                crop_size = crop_size["shortest_edge"]
        self.crop_size = crop_size
        self.image_aspect_ratio == config.image_aspect_ratio
        self.default_image_aspect_ratio = config.image_aspect_ratio
        self.mm_file_local_root_path = image_root_path
        self.has_label = has_label
        self.question_loss = question_loss
        self.text_filter_domain = text_filter_domain
        self.is_infra = is_infra

        if self.has_label:
            assert not self.is_infra, f"has_label: {self.has_label} and is_infra: {self.is_infra} conflict"
        if self.is_infra:
            assert not self.has_label, f"has_label: {self.has_label} and is_infra: {self.is_infra} conflict"
        self.split_special_tokens = self.has_label

        self.image_pattern = getattr(self.config, 'image_pattern', '<image>')

        self.target_format = target_format

    def apply_pattern_on_mmobj(self, s):
        target_format = self.target_format
        assert target_format in ['file'], f'incorrect target format {target_format}, should be file'
        mm_obj_pattern = re.compile(f"{self.image_start_token}(.*?){self.image_end_token}")
        mm_obj_str_list = mm_obj_pattern.findall(s)
        return_s = s
        for mm_obj_idx, mm_obj_str in enumerate(mm_obj_str_list):
            try:
                assert can_be_json_loaded(mm_obj_str), f"{mm_obj_str} not JSON"
            except:
                continue
            mm_obj_json = json.loads(mm_obj_str)
            file_type = parse_file_type_from_str(mm_obj_json.get("file_type", "image"))  
            if file_type=='I':
                image_str = self.image_pattern.format(mm_obj_idx) if '{}' in self.image_pattern else self.image_pattern
                image_str = image_str.replace('<image>', f'{self.image_start_token}{mm_obj_str}{self.image_end_token}')
                replace_str = f"{NOLOSS_START}{image_str}{NOLOSS_END}"
                return_s = return_s.replace(f"{self.image_start_token}{mm_obj_str}{self.image_end_token}", replace_str)
            else:
                continue  
        return return_s     

    def add_newline_from_end_token(self, s):
        tag_escaped = re.escape(self.image_end_token)
        s = re.sub(f"{tag_escaped} +", f"{tag_escaped}", s)
        s = re.sub(f"{tag_escaped}(?![ \n])", f"{tag_escaped}\n", s)
        return s

    def get_mm_obj_byte(self, source, path, mm_obj_json):
        mm_obj_byte = None
        if source in {"path", "local", "localpath"}:
            if os.path.exists(os.path.join(self.mm_file_local_root_path, path)):
                mm_obj_byte = open(os.path.join(self.mm_file_local_root_path, path), "rb").read()
            elif os.path.exists(path):
                mm_obj_byte = open(path, "rb").read()
        return mm_obj_byte

    def __call__(
        self,
        raw_input_obj: str,
        raw_input_has_conversation: bool = False,
    ):
        assert isinstance(raw_input_obj, str), f"input object {raw_input_obj} should be string, but get {type(raw_input_obj)}"
        raw_input_str = raw_input_obj

        assert isinstance(raw_input_str, str), f"input object {raw_input_str} should be string, but get {type(raw_input_str)}"
        raw_input_str = self.apply_pattern_on_mmobj(raw_input_str)
        raw_input_str = self.add_newline_from_end_token(raw_input_str)
        mm_obj_pattern = re.compile(f"{self.image_start_token}(.*?){self.image_end_token}")
        mm_obj_str_list = mm_obj_pattern.findall(raw_input_str)


        ought_number_mm_obj = len(mm_obj_str_list)
        mm_meta_info = {"line": raw_input_obj, "ought_number_mm_obj": ought_number_mm_obj, "wrong_mm_obj": [], "error_info": []}        
        mm_obj_list = []
        mm_obj_type_list = []
        mm_obj_image_aspect_ratio_list = []

        for mm_obj_str in mm_obj_str_list:
            try:
                assert mm_obj_str not in mm_meta_info["wrong_mm_obj"], "mm_obj_str has been failed before"
                assert can_be_json_loaded(mm_obj_str), f"{mm_obj_str} not JSON"
                mm_obj_json = json.loads(mm_obj_str)

                assert "path" in mm_obj_json, f"{mm_obj_json} no path"
                path = mm_obj_json.get("path")
                source = mm_obj_json.get("source", "path")
                assert source in {"path", "local", "localpath"}, f"source should be in path/local/localpath, but get {source}"
                file_type = parse_file_type_from_str(mm_obj_json.get("file_type", "image"))
                aspect_ratio = mm_obj_json.get("aspect_ratio", self.default_image_aspect_ratio)
                assert aspect_ratio in {"square"}, f"aspect_ratio should be in the set, but get {aspect_ratio}"

                mm_obj_byte = self.get_mm_obj_byte(source, path, mm_obj_json)

                if mm_obj_byte == None:
                    mm_obj_list.append(None)
                else:
                    if file_type == "I":
                        image = Image.open(io.BytesIO(mm_obj_byte)).convert("RGB")
                        if self.is_infra or (image.size[0] > 32 and image.size[1] > 32):
                            mm_obj_list.append(image)
                        else:
                            mm_obj_list.append(None)
                if mm_obj_list[-1] != None:
                    mm_obj_type_list.append(file_type)
                    mm_obj_image_aspect_ratio_list.append(aspect_ratio)
                else:
                    mm_obj_type_list.append(None)
                    mm_obj_image_aspect_ratio_list.append(None)
            except Exception as e:
                print(e)
                mm_meta_info["wrong_mm_obj"].append(mm_obj_str)
                mm_meta_info["error_info"].append(e)
                raw_input_str = raw_input_str.replace(self.image_start_token + mm_obj_str + self.image_end_token, "")

        real_mm_obj_list = [mm_obj for mm_obj in mm_obj_list if mm_obj is not None]
        real_mm_obj_type_list = [mm_obj_type for mm_obj_type in mm_obj_type_list if mm_obj_type is not None]
        real_mm_obj_image_aspect_ratio_list = [mm_obj_image_aspect_ratio for mm_obj_image_aspect_ratio in mm_obj_image_aspect_ratio_list if mm_obj_image_aspect_ratio is not None]
        real_number_mm_obj = len(real_mm_obj_list)
        assert real_number_mm_obj == len(real_mm_obj_type_list), f"{real_number_mm_obj}, {len(real_mm_obj_type_list)} not equal"
        assert real_number_mm_obj == len(real_mm_obj_image_aspect_ratio_list), f"{real_number_mm_obj}, {len(real_mm_obj_image_aspect_ratio_list)} not equal"
        if real_number_mm_obj == ought_number_mm_obj:
            assert len(mm_meta_info["wrong_mm_obj"]) == 0, f"len mm_meta_info should be 0, {json.dumps(raw_input_obj, ensure_ascii=False)}"
            mm_meta_info = {}
        else:
            mm_meta_info["real_number_mm_obj"] = real_number_mm_obj
            if not self.is_infra:
                assert real_number_mm_obj != 0, f"real_number_mm_obj should not be zero, when there ought be {ought_number_mm_obj} images, {json.dumps(raw_input_obj, ensure_ascii=False)}"


        img_list = []
        img_patch_list = []
        img_size_list = []
        img_attention_mask_list = [] 

        for mm_obj, mm_obj_type, mm_obj_image_aspect_ratio in zip(real_mm_obj_list, real_mm_obj_type_list, real_mm_obj_image_aspect_ratio_list):
            if mm_obj_image_aspect_ratio == "square":
                if mm_obj_type in ["I"]:
                    img_tensor = self.image_processor.preprocess(mm_obj, return_tensors="pt")["pixel_values"]
                    img_list.append(img_tensor)
                    img_patch_list.append((img_tensor.shape[0], 1))
                    if mm_obj_type == "I":
                        img_size_list.append(mm_obj.size[-2:])
        assert len(img_list) == len(img_patch_list), f"{len(img_list)}, {len(img_patch_list)} not equal"
        assert len(img_list) == len(img_size_list), f"{len(img_list)}, {len(img_size_list)} not equal"


        if not self.is_infra:
            raw_input_str = re.sub(r"\s*\.\.\.$", "", raw_input_str).strip()
        parts = re.split(f"({NOLOSS_START}.*?{NOLOSS_END})", raw_input_str, flags=re.DOTALL)
        assert len(parts) == 2 * real_number_mm_obj + 1, f"number of image_pad_token is wrong, {parts}, {real_number_mm_obj}"
        input_ids = []
        labels = []
        for part_text in parts:
            if part_text != "":
                if re.match(f"{NOLOSS_START}.*?{NOLOSS_END}", part_text, flags=re.DOTALL):
                    parts_in_noloss = re.split(f"({self.image_start_token}.*?{self.image_end_token})", part_text.replace(NOLOSS_START, '').replace(NOLOSS_END, ''))
                    for part_noloss in parts_in_noloss:
                        if re.match(f"{self.image_start_token}.*?{self.image_end_token}", part_noloss):
                            input_ids.append(self.config.image_token_index)
                            labels.append(self.config.IGNORE_INDEX)
                        else:
                            input_ids_part_noloss = self.tokenizer(part_noloss, split_special_tokens=False).input_ids
                            input_ids.extend(input_ids_part_noloss)
                            labels.extend([self.config.IGNORE_INDEX] * len(input_ids_part_noloss))
                else:
                    input_ids_part_text = self.tokenizer(part_text, split_special_tokens=self.split_special_tokens).input_ids
                    input_ids.extend(input_ids_part_text)
                    labels.extend(input_ids_part_text)
        assert input_ids.count(self.config.image_token_index) == real_number_mm_obj, f"number of image_pad_token {input_ids.count(self.config.pad_token_index)} should match real number of image {len(real_number_mm_obj)}"
        assert len(input_ids) == len(labels), "length unmatching between input_ids and labels"
        if self.split_special_tokens:
            assert self.config.eos_token_index not in input_ids, f"{self.config.eos_token_index} in text"
            assert self.config.pad_token_index not in input_ids, f"{self.config.pad_token_index} in text"
        input_ids = torch.tensor(input_ids)


        if self.has_label:
            labels = torch.tensor(labels)
        else:
            labels = None
            untext_token_mask_label = None

        if not self.is_infra:
            input_ids = torch.tensor(input_ids.tolist() + [self.config.eos_token_index])
            labels = torch.tensor(labels.tolist() + [self.config.eos_token_index])

        untext_token_mask = []
        if not self.is_infra:
            untext_token_mask_label = []
        tokens_per_image_list_for_packing = []
        image_idx = 0
        for i in range(len(input_ids)):
            if input_ids[i] == self.config.image_token_index:
                if self.config.mm_projector_type == "mlp":
                    assert self.kernel == 1, "MLP only has kernel 1"
                tokens_length = self.image_tile_tokens
                untext_token_mask.extend([-1] * tokens_length)
                if not self.is_infra:
                    untext_token_mask_label.extend([self.config.IGNORE_INDEX] * tokens_length)
                tokens_per_image_list_for_packing.append(tokens_length)
                image_idx += 1
            else:
                untext_token_mask.append(input_ids[i].item())
                if not self.is_infra:
                    untext_token_mask_label.append(labels[i].item())
        if not self.is_infra:
            return dict(
                input_ids=input_ids, # (L), torch.int64
                labels=labels, # (L), torch.int64
                untext_token_mask=untext_token_mask, # List[int]
                untext_token_mask_label=untext_token_mask_label, # List[int]
                images=img_list, # List (C, H, W), torch.float32
                img_patch_list=img_patch_list, # List[int]
                img_size_list=img_size_list, # List[Tuple(int, int)]
                mm_obj_type_list=real_mm_obj_type_list,  # List[Str]
                mm_obj_image_aspect_ratio_list=real_mm_obj_image_aspect_ratio_list, # List[Str]
                tokens_per_image_list_for_packing=tokens_per_image_list_for_packing, # List[ing]
                image_meta_info=mm_meta_info if mm_meta_info != {} else None,
                img_attention_mask_list=img_attention_mask_list

            )
        elif self.is_infra:
            if len(img_list):
                images = torch.concatenate(img_list)
                if len(img_attention_mask_list) > 0:
                    img_attention_mask_list = torch.concatenate(img_attention_mask_list)
            else:
                images = torch.zeros(0, 3, self.crop_size, self.crop_size)

            return dict(input_ids=input_ids, # (L), torch.int64
                        labels=labels, # (L), torch.int64
                        untext_token_mask=untext_token_mask, # List[int]
                        untext_token_mask_label=untext_token_mask_label, # List[int]
                        images=images, # (N_i, C, H, W), torch.float32
                        img_patch_list=img_patch_list, # List[int]
                        img_size_list=img_size_list, # List[Tuple(int, int)]
                        mm_obj_type_list=real_mm_obj_type_list,  # List[Str]
                        mm_obj_image_aspect_ratio_list=real_mm_obj_image_aspect_ratio_list, # List[Str]
                        img_attention_mask_list=img_attention_mask_list
                        ) 


class MMInferenceProcessor:
    def __init__(self, tokenizer, image_processor, dtype, device, config=None, **kwargs):
        self.data_processor = MMDataProcessor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            has_label=False,
            config=config,
            is_infra=True,
            **kwargs
        )
        self.data_collector = MMDataCollector(tokenizer, config)
        self.dtype = dtype
        self.device = device

    def __call__(self, example):
        if isinstance(example, list):
            return_dict_list = [self.data_processor(obj) for obj in example]
            ret = self.data_collector(return_dict_list, has_labels=False)
            new_ret = ModelOutput()
            for key, value in ret.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype in (torch.float16, torch.float32, torch.bfloat16):
                        new_ret[key] = value.to(dtype=self.dtype, device=self.device)
                    else:
                        new_ret[key] = value.to(device=self.device)
                else:
                    new_ret[key] = value
            return new_ret
        elif isinstance(example, str):
            example = [example]
            return self.__call__(example)
