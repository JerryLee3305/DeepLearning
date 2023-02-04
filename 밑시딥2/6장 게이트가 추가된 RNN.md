# 밑시딥 2
## 한빛 미디어
### 6장 게이트가 추가된 RNN

#### RNN 문제
- 시계열 데이터의 장기 의존 관계 학습 여러움
    - 이유 : BPTT에서 기울기 소실 or 폭발 발생

    - RNN 계층에서는 Matmul , +, tanh 연산을 통과함
        - 역전파 시 + 연산은 기울기 변화를 시키지 않음
        - tanh의 미분 값은 1- y^2 의 형태 => 0으로부터 멀어질수록 작아짐, 기울기가 tanh 노드 지날 때마다 계속 값 작아짐
        - Matmul 에서는 매번 똑같은 가중치인 Wh가 사용 됨
            - 만일 Wh 가 1보다 크면 지수적으로 증가 (기울기 폭발), 1보다 작으면 지수적으로 감소 (기울기 소실)
            - Wh가 스칼라가 아닌 행렬인 경우, '특이값'이 척도로 사용됨 (데이터가 얼마나 퍼져있는지)
            - 특이값의 최댓값이 1보다 크면 지수적으로 증가할 가능성이 높다고 예측 (반드시 그렇다는 것은 아님)


- 기울기 폭발 대책
    - 기울기 클리핑
```python
dW1 = np.random.rand(3,3) * 10
dW2 = np.random.rand(3,3) * 10
grads = [dW1, dW2]
max_norm = 5.0

def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad **2)
    total_norm = np.sqrt(total_norm) # L2 노름

    rate = max_nrom / (total_norm + 1e-6) # 분모 0 안되게 방지
    if rate <1: # threshod
        for grad in grads:
            grad *= rate
```

#### 기울기 소실과 LSTM
- LSTM 계층의 인터페이스에는 c 경로가 존재 (기억 셀 or 셀) => 기억 메커니즘
    - 특징 : 자기 자신으로만 (LSTM 계층 내에서만) 주고 받음
    - 외부에서는 보이지 않아 존재 자체 생각할 필요 없음

    - 갱신된 c_t 를 사용해 h_t를 계산 => h_t = tanh(c_t)

- output 게이트 (출력 게이트) : 다음 은닉 상태 h_t 출력을 담당하는 게이트
    - h_t = o 와 tanh(c_t) 의 '아다마르 곱' (원소별 곱)
        - o = 시그모이드(x_t*W_x + h_(t-1) * W_h + b)
    
    - 게이트에서는 시그모이드 함수가, 실질적 정보를 지닌 데이터에는 tanh 함수가 활성화 함수로 사용됨

- forget 게이트 (망각 게이트) : c_(t-1) 기억 중 불필요한 기억을 잊게 해줌
    - c_t = f 와 c_(t-1) 아다마르 곱
        - f= 시그모이드(x_t*W_x + h_(t-1) * W_h + b)

- 새로운 기억 셀 : 새로 기억해야 할 정보를 기억 셀에 추가
    - g= 시그모이드(x_t*W_x + h_(t-1) * W_h + b)

- input 게이트 : g의 각 원소가 새로 추가되는 정보로써 가치가 얼마나 큰지 판단
    - i= 시그모이드(x_t*W_x + h_(t-1) * W_h + b)

- 행렬의 곱을 사용하는 것이 아닌 아다마르 곱 (원소별 곱)을 사용하기에 매번 새로운 게이트 값을 이용하기에 곱셈 효과가 누적되지 않아 기울기 소실이 일어나기 힘듦

- 위 4개의 게이트의 f, g, i, o 의 가중치 및 편향을 하나로 모아 아핀 변환을 단 1회로 계산
    - slice 노드를 통해 4개의 결과를 꺼내고 균등하게 4 조각으로 나눠 꺼냄

```python
class LSTM:
    def __init__(self, Wx, Wh, b):
        self.parmas = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None # 순전파 때 중간 결과를 보관했다가 역전파 계산에 사용하려는 용도

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        
        # 아핀 변환을 우선적으로 진행
        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b
        # 아핀 변환 결과를 4등분
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
# 4등분한 것을 역전파로 하나로 만들고 싶을 때는 np.hstack 활용

class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful = False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N,T,H), dtype = 'f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype = 'f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype = 'f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:,t,:], self.h, self.c)
            hs[:,t,:] = self.h

            self.layers.append(layer)
        
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype = 'f')
        dh, dc = 0, 0

        grads = [0,0,0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:,t,:] + dh, dc)
            dxs[:,t,:] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs
    
    def set_state(self, h, c = None):
        self.h, self.c = h,c
    
    def reset_state(self):
        self.h, self.c = None, None

# Time LSTM
class Rnnlm:
    def __init__(self, vocab_size = 10000, wordvec_size = 100, hidden_size = 100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V,D) / 100).astype('f')
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4*H).astype('f')
        affine_W = (rn(H,V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful = True),
            TimeAffine(affine_W, affine_b)
        ]

        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout = 1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.lstm_layer.reset_state()
```

- 테스트 데이터로 평가 시 perplexity 를 이용하는 데, 이때 모델 상태 재설정 후 평가를 수행한다는 점!
- RNNLM 으로 정확한 모델을 만들고자 한다면 LSTM 계층을 깊게 쌓기
    - 층을 깊게 쌓으면 `과적합` 문제 발생 가능

- 과적합 억제 방법
    1. 훈련데이터 양 늘리기
    2. 모델 복잡도 줄이기
    3. 모델 복잡도에 페널티 부여 : `정규화`
    4. 드롭아웃

- 드롭아웃을 RNNLM 계층에 넣을 시 시계열 방향 삽입보다는 깊이 방향으로 삽입이 좋음 (시계열 방향 삽입 시 시간 흐름에 비례해 노이즈 축적되어 정보를 잃을 가능성 큼)
    - 최근엔 `변형 드롭아웃`을 제안하여 시간 방향으로 적용하는데 성공
        - 같은 계층의 드롭아웃끼리 마스크(통과 / 차단 을 결정하는 binary 형태의 무작위 패턴)를 공유하여 마스크가 '고정'

- 가중치를 공유하면 학습 매개변수 수 줄어들고 정확도 향상
```python
self.layers = [
    TimeEmbedding(embed_W),
    TimeDropout(dropout_ratio), #일반적인 드롭아웃
    TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful = True),
    TimeDropout(dropout_ratio),
    TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful = True), # 계층을 더 깊게
    TimeDropout(dropout_ratio),
    TimeAffine(embed_W.T, affine_b) # 가중치 공유
]
```
#### 정리
- RNN 학습시 기울기 소실과 폭발 문제 가능
- 기울기 폭발시 `기울기 클리핑`, 기울기 소실시 (LSTM or GRU) 효과적
- LSTM에는 input, forget, output, 새로운 게이트 있음
- 게이트에는 전용 가중치 존재, 시그모이드 함수 사용
- 언어 모델 개선에는 LSTM 계층 다층화, 드롭아웃, 가중치 공유 등