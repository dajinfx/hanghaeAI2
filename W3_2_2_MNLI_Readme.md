## Q1) 어떤 task를 선택하셨나요?

> NER, MNLI, 기계 번역 셋 중 하나를 선택

###### **Re**: **MNLI task 를 선택했습니다.**

## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?

> 모델의 입력과 출력 형태 또는 shape을 정확하게 기술

###### Re: 구현한 모델은 DistilBert 와 Transformer 를 택하였습니다.

> **DistilBert:**

**Embeddings Layer:**

입력 형태 : 

word_embeddings:  Vocabulary  len: 30522, Hidden size: 768 

position_embeddings: 시퀀스 max len: 512, Hidden size: 768 

Dropout: 0.1

출력 형태 : 

(batch_size, seq_len, hidden_size) :  (64, 512,768)

**Classifier**: 

입력: 768 (hidden_size) 

출력: 3 (3개의 분류 클래스)

> DistilBert:
> ![image](https://github.com/user-attachments/assets/04f48f12-e342-4bc1-a5f9-95e33244d954)

> **Transformer:**

입력 형태 :
word_embeddings:  Vocabulary  len: 30522, Hidden size: 768
Position embeddings: (512,768)

**Self-Attention:**

* Query: `(batch_size, in_features, out_features)` -> (64, 768, 768)
* Key: `(batch_size, in_features, out_features)`-> (64, 768, 768)
* Value: `(batch_size, in_features, out_features)`-> (64, 768, 768)

**FFN:**

* Layer1: Hidden size 확장 (768 → 3072).
* Layer2: Hidden size 축소 (3072 → 768).

**출력 형태** : `(in_features, num_classes)`: `(768, 3)`

![image](https://github.com/user-attachments/assets/92cb4c82-7cb3-4ad3-b8a2-63b9b9adff17)


## Q3) 어떤 pre-trained 모델을 활용하셨나요?

> PyTorch에서 위에서 정한 task에 맞는 pre-trained 모델을 선정
> **Re**: DistilBert 모델을 활용했습니다.

## Q4) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요?

> 비교 metric은 loss curve, accuracy, 또는 test data에 대한 generalization 성능 등을 활용.
> ![image](https://github.com/user-attachments/assets/e96b2c90-bfbf-4b5e-99dc-8362d014a330)
> ![image](https://github.com/user-attachments/assets/b887131f-ece2-402a-bab8-e2329596f6ee)
> ![image](https://github.com/user-attachments/assets/0338e3c1-98e0-422b-adf6-16c2d0c85526)

## Conclusion

* distilbert pretaining 모델을 통한 학습은 속도가 빠르고 정상적인 결과가 보였습니다.
* distilbert 결과에서는 정확도가 천천히 올라가는 모습이지만 20번의 학습차례에서 55% 가까이의 정확도가 나옵니다.
* transformer 를 통한 학습은 속도가 느리고, 정확도가 학습기간 정확도 계선되지 못했습니다.
* distilbert 모델의 성능에서 더욱 빨르고 효과가 더 좋았음을 보여줍니다.



