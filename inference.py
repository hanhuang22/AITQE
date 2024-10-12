import argparse
import torch
from modeling_mm import MMForCausalLM
import json


def move_tensors_to_cuda(data_dict, gpu_id):
    if torch.cuda.is_available():
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(f'cuda:{gpu_id}')
    else:
        print("CUDA is not available.")
    return data_dict

def process_prompt(image_path: str, text: str):
    image_dict = {
        "path": image_path,
        "source": "localpath",
        "file_type": 'I'
    }
    question = f"Evaluate the caption: {text}\nif score is low, give a short description of the image.\nRespond using a dictionary structure."
    return f"<img_start>{json.dumps(image_dict)}<img_end>\n{question}"


def eval_single_node(model, tokenizer, image_path, text, output_all=False, gpu_id=0):
    processor = model.bind_processor(tokenizer,
                                     config=model.config)
    prompt = process_prompt(image_path, text)
    proc_ret = processor(prompt)
    proc_ret = move_tensors_to_cuda(proc_ret, gpu_id)
    with torch.inference_mode():
        if output_all:
            output_ids = model.generate(**proc_ret,
                                    do_sample=False,
                                    num_beams=1,
                                    max_new_tokens=384,
                                    use_cache=True)
        else:
            output_ids = model.generate(**proc_ret,
                                    do_sample=False,
                                    stop_strings=["<Overall>"],
                                    tokenizer=tokenizer,
                                    num_beams=1,
                                    max_new_tokens=384,
                                    use_cache=True)
    outputs = tokenizer.batch_decode(
        output_ids[:, proc_ret["input_ids"].shape[1]:],
        skip_special_tokens=True)[0].strip()
    if output_all:
        return outputs
    else:
        return outputs[:-1] + "}"


def load_model(model_path, gpu_id):
    model = MMForCausalLM.from_pretrained(model_path,
                                            device_map=f'cuda:{gpu_id}',
                                            torch_dtype=torch.bfloat16).eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        truncation_side="left",
        use_fast=False,
        trust_remote_code=True,
    )
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--model_path", type=str, default="path/to/aitqe")
    parser.add_argument("--image_path", type=str, default="./figs/test.png")
    parser.add_argument("--caption", type=str, default="Some random text to the image like this is a test")
    parser.add_argument("--output_all", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, args.gpu_id)
    output = eval_single_node(model, tokenizer, args.image_path, args.caption, args.output_all, args.gpu_id)
    print("="*100)
    print("[Input]")
    print(f"Image:   {args.image_path}")
    print(f"Caption: {args.caption}")
    print("="*100)
    print("[Output]")
    print(output)


if __name__ == "__main__":
    main()