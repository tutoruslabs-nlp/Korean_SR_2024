# 연결 문장 추론 Baseline
본 리포지토리는 '2024년 국립국어원 인공지능의 한국어 능력 평가' 상시 과제 중 '연결 문장 추론'에 대한 베이스라인 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.  

학습 및 추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.  

|Model|Accuracy(%)|
|:---|---:|
|MLP-KTLim/llama-3-Korean-Bllossom-8B (without SFT)|61.5|
|MLP-KTLim/llama-3-Korean-Bllossom-8B (with SFT)|95.0|

## 리포지토리 구조 (Repository Structure)
```
# 학습에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
├── test.py
└── train.py

# 학습에 사용될 함수들을 보관하는 디렉토리
src
└── data.py
```

## 데이터 형태 (Data Format)
```
[
    {
        "id": "nikluge-2024-연결문장추론-valid-000001",
        "input": {
            "sentence_1": "할아버지께서 병으로 아주 편찮으셨다.",
            "sentence_3": "나는 할아버지의 장례식 때 많이 울었다.",
            "sentence_2_candidate_1": "그렇게 할머니는 하늘의 별이 되셨다.",
            "sentence_2_candidate_2": "그렇게 할아버지는 하늘의 별이 되셨다."
        },
        "output": "sentence_2_candidate_2"
    }
]
```

## 실행 방법 (How to Run)
### 학습 (Train)
```
CUDA_VISIBLE_DEVICES=0 python -m run.train \
    --trainset resource/data/train_data.jsonl \
    --devset resource/data/valid_data.jsonl \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --epoch 1 \
    --lr 2e-5 \
    --batch_size 4 \
    --warmup_steps 20 \
    --gradient_accumulation_steps 16 \
    --save_dir ./models/e1
```

### 추론 (Inference)
```
python -m run.test \
    --input resource/data/test_data.jsonl \
    --output result.json \
    --model_id MLP-KTLim/llama-3-Korean-Bllossom-8B \
    --device cuda:0
```


## Reference
huggingface/transformers (https://github.com/huggingface/transformers)  
Bllossome (https://huggingface.co/MLP-KTLim/llama-3-Korean-Bllossom-8B)  
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
