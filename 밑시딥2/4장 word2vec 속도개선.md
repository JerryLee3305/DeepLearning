# 밑시딥 2
## 한빛 미디어
### 4장 word2vec 속도개선

말뭉치에 포함된 어휘 수 많아지면 계산량도 커짐 => 개선
1. Embedding : 입력층의 원핫 표현 많으면 상당한 메모리 차지하기에 도입하여 해결
2. 네거티브 샘플링 (손실함수) : 은닉층 이후 가중치 곱과 softmax 계층 계산량 증가로 도입하여 해결

#### Embedding 계층
```python
class Embedding:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        dW[self.idx] = dout # 이렇게 하면 중복 문제 발생

        np.add.at(dW, self.idx, dout) # 더해주기를 통해 중복 방지
        # np.add.at(A,idx,B) B를 A의 idx번째 행에 더해준다
        return None
```
- 행렬의 특정 행을 추출하는 것이기에 전체를 곱할 필요가 없다는 생각에서 만들어짐

```python
class EmbeddingDot:
    def __init__(self,W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis = 1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0],1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
```

#### 네거티브 샘플링
은닉층 이후 병목 해소를 위해 사용
- 은닉층의 뉴런과 가중치 행렬의 곱
- Softmax 계층의 계산

- 다중분류 -> 이진분류로 하여 가벼운 계산으로
    - 시그모이드 함수와 교차 엔트로피 오차 사용

오차가 앞 계층으로 갈 수록, 오차가 크면 크게 학습하고 작으면 작게 학습

- 긍정적인 것만 학습을 하였으나 부정적인 것을 학습하지 않았기에 모든 부정적인 예를 하는 것이 아닌 부정적인 예 몇개만 선택해서 샘플링

- 샘플링 방법 : 단어 출현 횟루를 구해 확률 분포로 나타내어 확률 분포대로 단어 샘플링 => 자주 등장하는 단어 선택될 가능성 높음
    - 출현 확률이 낮은 단어를 버리지 않기위해 0.75를 제곱해줌
```python
sampler = UnigramSampler(corpus, power, sample_size)
target = np.array([1,3,0])
negative_sample = sampler.get_negative_sample(target) # 긍정적인 예를 넣어서 부정적인 것 샘플링

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power = 0.75, sample_size = 5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size +1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size +1)]
        self.params, self.gards = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 긍정적인 예에 대한 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype = np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적인 예에 대한 순전파
        negative_label = np.zeros(batch_size, dtype = np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:,i]
            score = self.embed_dot_layers[1+i].forward(h, negative_target)
            loss += self.loss_layers[1+i].forward(score, negative_label)
        return loss

    def backward(self, dout = 1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh

# 개선 CBOW
class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        W_in = 0.01* np.random.randn(V,H).astpye('f')
        W_out = 0.01*np.random.randn(V,H).astype('f')

        self.in_layers = []
        for i in range(2*window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power = 0.75, sample_size = 5)

        layers = self.in_layer + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:,i])
        h *= 1/len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss
    def backward(self, dout = 1):
        dout = self.ns_loss.backward(dout)
        dout *= 1/len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

```

- 전이학습 : 한 분야에서 배운 지식을 다른 분야에도 적용
- 평가 척도 : 유사성, 유추 문제 활용한 평가


#### 정리
- Embedding 계층은 단어의 분산 표현 담고 있으며, 순전파 시 지정 단어 ID 벡터 추출
- word2vec은 어휘 수 증가에 비례하여 계산량도 증가하므로, 근사치로 계산하는 빠른 기법 사용
- 네거티브 샘플링은 부정적인 예를 몇개 샘플링 하는 기법으로 다중분류를 이중분류처럼 취급
- word2vec로 얻은 단어의 분산 표현에는 단어의 의미가 녹아들어 있으며, 비슷한 맥락에서 사용되는 단어는 단어 벡터 공간에 가까이 위치
- word2vec 단어 분산 표현 이용 시 유추 문제를 벡터 덧셈과 뺄셈으로 풀 수 있음
- word2vec 전이 학습 측면에서 특히 중요