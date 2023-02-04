# 밑시딥 2
## 한빛 미디어
### 5장 RNN

- 피드포워드 (feed forward) : 흐름이 단방향인 신경망 

구성이 단순하여 구조 이해하기 쉽고, 많은 문제에 응용 가능

but! 시계열 데이터 잘 다루지 못함 (시계열 데이터 성질 충분히 학습 불가능)

- 언어 모델 : 단어 나열에 확률 부여
    - 특정 단어 시퀀스에 대해, 시쿼스가 일어날 가능성이 어느정도인지를 확률로 평가

동시 확률 : 여러 사건이 동시에 일어날 확률 = 사후 확률의 총곱

CBOW 모델의 은닉층에서는 맥락의 단어 순서가 무시되기에 등장한 것이 RNN

#### RNN (Recurrent Neural Network)
순환하는 신경망
- Recursive Neural Netword(재귀신경망) : 트리 구조 데이터 처리하기 위한 신경망 (순환 신경망이랑 다름)

순환하기에 과거의 정보를 기억하는 동시에 최신 데이터 갱신

ht = tanh(h(t-1) * Wh+ Xt*Wx+ b)
- 가중치 2개가 있음
    - 입력 x를 출력 h로 변환하기 위한 가중치 Wx
    - RNN 출력을 시각 출력으로 변환하기 위한 가중치 Wh
- RNN은 h 라는 '상태'를 가지고 있으며 위 식의 형태로 갱신됨

- BPTT(backpropagation through time) : 시간 방향으로 펼친 신경망의 오차역전파법
    - 시계열 데이터 시간 크기 커지는 것 비례해 BPTT 소비 컴퓨팅 자원도 증가
    - 시간 크기 커지면 역전파 시 기울기 불안정

- Truncated BPTT : 적당 지점에서 잘라내 작은 신경망 여러개로 만들어 오차역전파법 수행
    - 역전파의 연결만 끊어야됨
    - 순전파 연결은 반드시 그대로 유지되어야 함
    - 각각의 블록 단위로 미래 블록과 독립적으로 오차역전파법 완결

```python
class RNN:
    def __init__(self, Wx, Wh,b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeeros_like(b)]
        self.cache = None # 역전파 계산 시 사용하는 중간 데이터 담을 변수
    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1-h_next**2)
        db = np.sum(dt, axis = 0)
        dWh = np.matmul(h_prev.T, dt)
        dh_prev = np.matmul(dt, Wh.T)
        dWx = np.matmul(x.T, dt)
        dx = np.matmul(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev

```

- TimeRNN : T개의 RNN 계층으로 구성
    - RNN 계층의 은닉 상태 h를 변수로 유지 (인계받는 용도)
    - 이 은닉 상태를 TimeRNN 계층에서 관리 => RNN 계층 사이 은닉 상태를 인계 받지 않아도 됨 --- 이 기능을 stateful 이라는 인수로 조정할 수 있도록 함

```python
class TimeRNN:
    def __init__(self, Wx, Wh, b, statefull = False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        # h는 forward 마지막 RNN 계층 은닉상태 저장
        # dh는 backward 하나 앞 블록의 은닉상태 기울기
        self.h, self.dh = None, None
        self.stateful = stateful

    def set_state(self, h):
        self.h = h
    
    def reset_state(self):
        self.h = None

    def forward(self, xs): # xs는 T개 시계열 데이터를 하나로 모은 것
        Wx, Wh, b = self.params
        # 미니배치 크기 N, 입력 벡터 차원 수 D
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N,T,H), dtype = 'f')
        if not self.stateful or self.h is None:
            self.h = np.zeors((N,H), dtype = 'f') # 영행렬로 초기화
        
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:,t,:],self.h)
            hs[:,t,:] = self.h
            self.layers.append(layer)
        return hs
# RNN 계층의 순전파에서는 출력이 2개로 분기 됨 => 역전파에서는 각 기울기가 합산됨 (dht + dhnext)
    def bacwkard(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N,T,D), dtype = 'f')
        dh = 0
        grads = [0,0,0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:,t,:]+dh) #합산 기울기
            dxs[:,t,:] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs
```
#### 시계열 데이터 처리 계층 구현
- RNNLM

    1. 첫 번째 층은 Embedding 계층 - 단어 ID를 단어 벡터로 변환
    2. RNN 계층 - 은닉 상태를 다음층으로 출력함과 동시에 다음 시각의 RNN 계층으로 출력
    3. 위로 출력한 은닉 상태는 Affine 계층을 거쳐 Softmax 계층으로 전해짐
    - 지금까지 입력된 단어를 기억하고 그것을 바탕으로 다음 단어 예측 <= RNN 계층에서 과거의 정보를 인코딩해 저장

- Time 계층
    - 손실함수로 Softmax with Loss를 사용하는데 x 데이터는 '점수', t 데이터는 정답 레이블 => 합산후 평균한 값이 최종 손실

```python
class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V,D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')

        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H,V) / np.sqrt(H)).astype('f') # Xavier 초깃값
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful = True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss
    
    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.rnn_layer.reset_state() # 신경망 상태 초기화
```

- 퍼플렉서티 (perplexity) : 언어모델 예측 성능 평가 척도
    - 확률의 역수로 계산
    - 작을수록 좋은 것
    - 직관적으로는 선택사항의 수
    - L = -1/N * 시그마 시그마 t_nk * log y_nk      (t_n은 원핫 벡터로 정답 레이블, t_nk 는 데이터의 k번째 값 , y_nk 는 확률분포)
    - perplexity = e ^ L

```python
jump = (corpus_size -1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        batch_x = np.empty((batch_size, time_size), dtype = 'i')
        batch_t = np.empty((batch_size, time_size), dtype = 'i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset+ time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1
        
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count +=1

ppl = np.exp(total_loss/ loss_count) # 에폭마다 퍼플렉서티 평가
ppl_list.append(float(ppl))
total_loss, loss_count = 0,0
```
#### 정리
- RNN은 순환 경로가 있고, 이를 통해 내부에 '은닉 상태' 기억 가능
- RNN 순환 경로 펼침으로써 다수 RNN 계층이 연결된 신경망으로 해석 가능, 보통 오차역전파법으로 학습 (BPTT)
- 긴 시계열 데이터 학습 시 데이터를 적당 길이씩 모음 (블록)
- 블록 단위로 BPTT 학습 (Truncated BPTT)
- Truncated BPTT 에서는 역전파의 연결만 끊음 (순전파 연결 유지 위해서는 데이터 '순차적' 입력)
- 언어 모델은 단어 시퀀스를 확률로 해석
- RNN 계층 이용한 조건부 언어 모델은 그때까지 등장한 모든 단어의 정보 기억할 수 있음