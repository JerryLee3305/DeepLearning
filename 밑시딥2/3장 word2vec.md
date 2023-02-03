# 밑시딥 2
## 한빛 미디어
### 3장 word2vec

- 추론 기반 기법인 word2vec

- 통계기반 기법 문제점 : 대규모 말뭉치 다룰 시 문제 발생 (차원 수가 너무 큼)
    - 추론 기반 기법에서는 신경망 이용할 시 미니배치로 학습 일반적
    - 추론 기반 기법은 여러 머신과 여러 GPU 병렬 계산 가능해 학습 속도를 높일 수 있음

- 신경망 단어 처리 대표적 : 원핫 벡터 변환 -> 완전연결 계층 통과 후 변환

#### 단순한 word2vec
- CBOW(continuous bag-of-words) 모델 : 맥락으로부터 타깃을 추측하는 용도의 신경망
    - 맥락에 포함시킬 단어가 N개라면 입력층도 N개
    - 은닉층의 뉴런은 입력층이 여러개 일시 전체를 '평균'
    - 출력층은 뉴런 하나하나가 각각의 단어에 대응, 값이 높을 수록 대응 단어 출현 확률 높아짐

2개의 입력층이 있다고 가정할 시
1. CBOW 모델 가장 앞단 2개의 MatMul 계층이 존재
2. 두 계층 출력을 더함
3. 1/2 곱해 평균을 만듦
4. 은닉층 뉴런이 되는데 또 다른 MatMul 계층 적용 되어 점수 출력
5. Softmax
6. Cross Entropy Error
7. Loss 손실이 나오면 이것을 통해 학습을 진행

- Softmax와 Cross Entropy Error 계층을 하나의 계층인 softmax with Loss 라는 하나의 계층으로 구현
- word2vec (특히 skip-gram 모델) 에서는 입력층의 가중치만 이용하는 게 대중적인 선택
    - GloVe 에서는 두 가중치를 더했을 때 좋은 결과가 나옴


```python
from common.util import create_contexts_target, convert_one_hot, MatMul, SoftmaxWithLoss

contexts, target = create_contexts_target(corpus, window_size = 1)
vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V,H = vocab_size, hidden_size

        # 가중치 초기화 작업
        W_in = 0.01 * np.random.randn(V,H).astype('f')
        W_out = 0.01 * np.random.randn(H,V).astype('f')

        # 계층 생성, 입력층 2개
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기 모아 매개변수와 기울기를 리스트에 넣음
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # 분산 표현
        self.word_vecs = W_in
    # 맥락(contexts), 타겟(target) 입력받아 손실 반환
    def forward(self, contexts, target): #contexts 3차원 넘파이 배열 가정
        h0 = self.in_layer0.forward(contexts[:, 0]) #미니 배치 수
        h1 = self.in_layer1.forward(contexts[:, 1]) # 윈도우 크기
        # 2번째 차원은 원핫 벡터

        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
```

- 학습 코드 구현
```python
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, covert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = # 아무 텍스트나 입력 받기
corpus, word_to_id, id_to_word = preprocess(text) # 텍스트 전처리 작업

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot() # 학습 거듭될수록 손실 줄어드는 지 확인

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

```

동시확률 = P(A,B),   사후확률 = P(A|B)

#### skip-gram
- 입력층은 하나이지만 출력층은 맥락의 수만큼 존재
    - 개별적으로 손실 구하고 모두 더해 최종 손실로 사용

- P(Wt-1, Wt+1 | Wt)
    - Wt가 주어졌을 때 Wt-1과 Wt+1 이 동시에 일어날 확률
    - skip-gram에서는 맥락 단어들 사이 관련성 없다고 가정

- 단어 분산 표현 정밀도 면에서 skip-gram모델 결과가 더 좋음
- 말뭉치 커질수록 저빈도 단어나 유추 문제 성능 뛰어남

but
- 학습 속도면에서는 CBOW 모델이 더 빠름

#### 정리
- 추론 기반 기반은 추측하는 것이 목적이며 단어의 분산 표현 얻을 수 있음
- word2vec은 추론 기반 기법이며, 단순한 2층 신경망 모델
- word2vec은 skip-gram 모델과 CBOW 모델 제공
- CBOW 모델은 여러 단어로부터 하나의 단어 추측
- 반대로 skip-gram은 하나의 단어로부터 다수의 단어 추측
- word2vec은 가중치 다시 학습할 수 있으며, 분산 표현 갱신이나 새로운 단어 추가 효율적 수행
