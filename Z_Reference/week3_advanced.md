# Week3_advanced

## 1. 어떤 Task를 선택하셨나요?

### 1) MNLI
Premise와 Hypothesis 간의 중립 / 포함 / 모순 관계를 분류하는 자연어 추론(NLI) 문제입니다.  
- [과제 링크](https://github.com/Habonit/sparta_coding_ai/blob/main/week3_advanced_mlni.ipynb)

### 2) Translation
영어(EN)에서 프랑스어(FR)로 번역하는 시퀀스-투-시퀀스(Seq2Seq) 문제입니다.  
- [과제 링크](https://github.com/Habonit/sparta_coding_ai/blob/main/week3_advanced_translation.ipynb)

### 3) Ner
개체명 인식을 하는 문제 입니다.
- [과제 링크](https://github.com/Habonit/sparta_coding_ai/blob/main/week3_advanced_ner.ipynb)
---

## 2.모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태는 어떻게 되나요?

### 1) MNLI
- **모델 구조**: DistilBERT 기반의 텍스트 분류 모델  
- **특징**: 경량화된 인코더 모델인 DistilBERT를 사용하며, 분류 개수(3개 클래스)에 맞춰 Text Classifier를 추가  
- **입력**: Premise와 Hypothesis 두 텍스트  
- **출력**: 중립 / 포함 / 모순 중 하나의 클래스 라벨  

### 2) Translation
- **모델 구조**: T5 기반의 인코더-디코더 모델  
- **특징**: 번역 문제를 처리하기 위해 T5를 사용  
- **입력**: 영어 문장 (EN)  
- **출력**: 번역된 프랑스어 문장 (FR)  

### 3) Ner
- **모델 구조**: DistilBERT 기반의 텍스트 분류 모델   
- **특징**: 경량화된 인코더 모델인 DistilBERT를 사용하며, 분류 개수(17개 클래스)에 맞춰 Text Classifier를 추가 
- **입력**: 토큰화 된 단어어 
- **출력**: 토큰에 태그된 Bio Tag

---

## 3. 데이터 입력 형태

### 1) MNLI
- **입력 데이터 예시**:

input premise and hypothesis
Instead, you're still leading with Jacob Weisberg on Clinton's African apology (Sorry Excuse) and Cullen Murphy's discourse on lying (The Lie of the Land).
Clinton gave an apology to Africa.

input_ids shape
(400,)

attention_mask shape
(400,)

label
0

### 2) Translation
- **입력 데이터 형태**:

Source 데이터 형태
['translate English to French: Hi.',
'translate English to French: Run!',
'translate English to French: Run!',
'translate English to French: Who?']

Target 데이터 형태
['Salut!', 'Cours\u202f!', 'Courez\u202f!', 'Qui ?']

Model input 데이터 형태
{'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1]],
'input_ids': [[13959, 1566, 12, 2379, 10, 2018, 5, 1],
            [13959, 1566, 12, 2379, 10, 7113, 55, 1],
            [13959, 1566, 12, 2379, 10, 7113, 55, 1],
            [13959, 1566, 12, 2379, 10, 2645, 58, 1]],
'labels': [[25801, 55, 1],
            [13579, 7, 3, 55, 1],
            [13579, 457, 3, 55, 1],
            [6590, 3, 58, 1]]}

### 3) Ner
- **입력 데이터 형태**:

DatasetDict({
    train: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 37543
    })
    val: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 5611
    })
    test: Dataset({
        features: ['id', 'tokens', 'ner_tags', 'input_ids', 'attention_mask', 'labels'],
        num_rows: 4795
    })
})

## 4. Fine-tuning 결과

### 1) MNLI
![MNLI - Loss Curve](https://github.com/user-attachments/assets/cc1a0c93-3a4c-46ce-a1bc-49f5d5e8bb54)  
*Loss Curve*  

![MNLI - Result](https://github.com/user-attachments/assets/6b2f869e-19f4-4638-aa27-29c07f22d387)  
*Result Chart*  

- 일반 30% 대에서 54% 정도의 accuracy와 F1 score를 기록했습니다.  
- Zero-shot 설정에서 55%의 accuracy와 F1 score를 확인했습니다.  

### 2) Translation
![Translation - Loss Curve](https://github.com/user-attachments/assets/32c8f8dc-e4a3-4628-b136-2cda1f31e8ea)  
*Loss Curve*  

![Translation - Result](https://github.com/user-attachments/assets/0cd54766-14b6-4c86-9cd8-d2d7ccc117da)  
*Result Chart*  

- T5 모델을 사용하여 초기 BLEU 점수 0.42에서 0.50까지 성능을 향상시킨 것을 확인했습니다.

### 3) NER
![Translation - Loss Curve](https://github.com/user-attachments/assets/ab12fbc6-bcad-4dd0-8bba-664302a698ec)  
*Loss Curve*  

![Translation - Result](https://github.com/user-attachments/assets/6439926b-38ed-4a9b-8017-f3dd9238b5cc)  
*Result Chart*  

- acc 14%에서 94%, f1은 1%에서 64%까지 향상되었습니다.


## 5. 사용 Metric

### 1) MNLI
- **Accuracy**: 모델이 올바르게 분류한 샘플의 비율  
- **F1 Score (weighted)**: 클래스 불균형을 고려한 조화 평균  

### 2) Translation
- **BLEU Score**: 번역된 문장이 원문과 얼마나 유사한지 평가하는 점수 

### 3) Ner
- **Accuracy**: 모델이 올바르게 분류한 샘플의 비율  
- **F1 Score (weighted)**: 클래스 불균형을 고려한 조화 평균  

## 6. 한계 및 어려웠던 점

### 1) MNLI
1. **분류군 개수 설정 오류**: 초반에 분류군 개수를 잘못 설정하여, 코드가 동작은 했지만 accuracy가 오르지 않는 문제가 발생. 이러한 개념적 오류를 발견하고 해결하는 데 어려움이 있었습니다.
2. **Hugging Face Trainer 전환**: Naive PyTorch 코드를 Hugging Face Trainer로 전환하는 과정에서, `tokenize` 함수와 `data collator` 함수 간의 관계를 이해하는 데 시간이 걸렸습니다.

### 2) Translation
1. **Seq2Seq 구조 이해**: 모델이 T5로 바뀌면서 Seq2Seq 구조를 확장시켜 생각하는 것이 어려웠습니다.
2. **훈련 시간**: T5 모델의 파라미터 수가 많아 훈련 시간이 오래 걸려, 실험에 따른 개선을 확인하는 과정이 더디게 진행되었습니다.
3. **Fine-tuning 전략**: 인코더-디코더 구조에서 어느 레이어를 동결하고 Fine-tuning할지 결정하는 데 참고할 자료가 부족하여, 실험을 통해 결정해야 했습니다.
4. **Hugging Face Trainer 전환**: Naive PyTorch 코드에서 Hugging Face Trainer로 전환하면서, `tokenize` 함수와 `data collator`의 관계를 이해하는 데 시간이 소요되었습니다.

### 3) Ner
1. **전처리** :데이터가 문장이 아니라 토큰화된 문장의 리스트로 들어가면서 전처리를 이해하기가 까다로웠습니다.