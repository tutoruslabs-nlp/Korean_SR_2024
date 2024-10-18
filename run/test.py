
import argparse
import json
import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import CustomDataset


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, required=True, help="input filename")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
# fmt: on

response_map = {
    'sentence_2_candidate_1': 'sentence_2_candidate_1',
    'sentence_2_candidate_2': 'sentence_2_candidate_2',
    'default': 'sentence_2_candidate_1'
}

def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=args.device,
    )
    model.eval()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    input_file = args.input
    dataset = CustomDataset(input_file, tokenizer)

    with open(input_file, "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]
        outputs = model.generate(
            inp.to(args.device).unsqueeze(0),
            max_new_tokens=128,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

        response = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
        print(f'result: {response}')
        if '### Response:' in response:
            response = response.split('### Response:')[1]
        print(f'response: {response}')
        if response in response_map:
            response = response_map[response]
        else:
            response = response_map['default']
        result[idx]["output"] = response

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))
