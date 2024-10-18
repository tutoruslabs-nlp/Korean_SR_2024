
import json

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, fname, tokenizer):
        IGNORE_INDEX=-100
        self.inp = []
        self.label = []

        PROMPT = '''You are a helpful AI assistant. Please answer the user's questions kindly. 당신은 유능한 AI 어시스턴트 입니다. 사용자의 질문에 대해 친절하게 답변해주세요.'''

        with open(fname, "r") as f:
            data = json.load(f)

        def make_chat(inp):
            instruction = '''
            Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
            ### Instruction:
            context로 주어지는 문장1과 문장3 사이에 올 수 있는 올바른 문장을 선택하시오. 생성 기준을 꼼꼼히 읽고 이해하는 것이 중요합니다.
            생성 기준:
            1 - 당신은 후보 중에서 문장1과 문장3의 중간에 올 수 있는 문장을 선택하는 챗봇입니다.
            2 - 후보1이 맞으면 '### Response:sentence_2_candidate_1'을 생성하고, 후보2가 맞으면 '### Response:sentence_2_candidate_2'를 생성하시오.
            3 - 출력은 '### Response:sentence_2_candidate_1'과 '### Response:sentence_2_candidate_2' 중에서 1개만 생성하시오.
            '''

            context = f"### context:\n"
            context += f"문장1: {inp['sentence_1']}\n"
            context += f"문장3: {inp['sentence_3']}"

            # question = f"[Question]\n위 Context로 주어진 문장1과 문장3 사이에 올 수 있는 올바른 문장을 아래의 후보 중에서 선택하시오. 후보1이 맞으면 '### Response:1'이라고 생성하고, 후보2가 맞으면 '### Response:2'라고 생성하시오."

            candidate = f"### 후보:\n"
            candidate += f"후보1: {inp['sentence_2_candidate_1']}\n"
            candidate += f"후보2: {inp['sentence_2_candidate_2']}"

            chat = instruction + "\n\n" + context + "\n\n" + candidate + '\n\n### Response:\n'

            return chat
        
        for example in data:
            chat = make_chat(example["input"])
            message = [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": chat},
            ]
     
            source = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            if "output" in example:
                target = example["output"]
            else:
                target = ""
            target = f'### Response:{target}'
            if target != "":
                target += tokenizer.eos_token
            
            target = tokenizer(target,
                      return_attention_mask=False,
                      add_special_tokens=False,
                      return_tensors="pt")
            target["input_ids"] = target["input_ids"].type(torch.int64)

            input_ids = torch.concat((source[0], target["input_ids"][0]))
            labels = torch.concat((torch.LongTensor([IGNORE_INDEX] * source[0].shape[0]), target["input_ids"][0]))
            self.inp.append(input_ids)
            self.label.append(labels)

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        return self.inp[idx]


class DataCollatorForSupervisedDataset(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
