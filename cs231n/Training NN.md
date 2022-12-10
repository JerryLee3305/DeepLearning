# cs231n
## 5강

1. train on ImageNet 
2. finetune network on your own data
- 전체 초기화, 아래쪽만 새롭게

3. if small dataset => fix all weights, retain only the classifier
- 이미 학습이 된 imagenet에서 마지막 부분만 가져다 쓰기

그렇기에 모델을 처음부터 바로 학습 시킬 필요는 없다.

- Mini-batch SGD
    1. sample a batch of data
    2. forward prop it through the graph, get loss
    3. backprop to calculate the gradient
    4. update the parameter using the gradient

#### Activation functions
    - Sigmoid : 문제점
         1. vanishing gradient (0과 1사이이며 x가 크거나 작을 경우 local gradient가 0이 되어버림) 포화지점(saturated regime)
         2. sigmoid outputs are not zero-centered (x는 모두 양수로 생각할 시 w는 모두 음수이거나 양수이어야 함 => zigzag path 그래서 매우 느려짐)
         3. exp() is a bit compute expensive (컴퓨터 성능 저하) 

    - tanh(x) (=hyperbolic tangent) : -1~1, 가운데가 0임
        - still kill gradients when saturated (vanish gradient)

    - ReLU (Rectified Linear Unit) : f(x) = max(0,x)
        - not zero-centered
        - x가 0보다 작을 때는 0이기에 기울기가 0이 된다. => gradient가 죽어버림, not defined(vanish gradient)
        - dead ReLU will never activate => never update (weight 초기화시, alpha가 너무 큰 경우, 10% 정도, learning rate가 너무 클 경우)

    - Leaky ReLU : max(0.01x, x)
        - will not die gradient

    - Exponential Linear Units (ELU) : 알파(exp(x)-1) when x<0 
    
    - maxout : 파라미터 두개를 가지기에 연산을 두배 사용

LSTM 에서는 여전히 sigmoid 사용



#### Weight Initialization
모든 가중치가 0이면 모두 동일한 연산 수행

1. small random numbers : W = 0.01*np.random.rand(d,h)
    - 네트워크 커지면 문제가 생김
    - all activations become zero
    - 1이나 -1이 되면 gradient will be all zero

2. Xavier initalization
    - np.random.rand(fan_in, fan_out)/np.sqrt(fan_id)
    - 잘 적용됨
    - 문제는 ReLU를 쓸 때 문제 생김
        - np.random.rand(fan_in, fan_out)/np.sqrt(fan_id`/2`) 2를 나누니까 ReLU 잘 작동함

3. Batch Normalization
    - 전체적으로 적용하는게 아니라 배치에 대해 평균과 분산을 통해 정규화
    - Fc와 tanh 사이에 구성
    - normalize -> squash the range if it want to (학습을 통해 조정할 수 있도록)
    - improve gradient flow, allow higher learning rate, reduce the strong dependence on initialization, regularization (do not need dropout)

#### Babysitting the Learning Process
1. preprocess the data : zero-centered data (np.mean(x, axis = 0)), PCA, Whitening(인접 픽셀과 중복성을 줄여줌)
    - image에서는 zero-cented 만 주로 사용

2. choose the architecture
    - 몇개의 layer를 둘 것인지, class 수, regularization 변화 check
    - take first 20 example => turn off regularization(reg = 0.0) => use simple vanilla 'sgd' ===> very small loss and train accuracy 1.0 에 나와야 overfitting 이 일어나야 됨 안일어나면 문제가 된 것임

#### Hyperparameter Optimization
1. Cross-validation : coasrse => fine
```python
max_count = 100
for count in xrange(max_count):
    reg = 10**uniform(-5,5) # 이렇게 10** 으로 해야 좋음
    lr = 10**uniform(-3,-6) # 이런 방식이 random search
```
- random search vs grid search : grid search 는 등간격이라서 좋은 정보를 못 찾을 수 있다. 하이퍼파라미터 설정할 때 random search 사용하는 것을 추천

- monitor the loss curve : 시간이 지나면 loss 가 감소한 경우 bad initialization
- monitor the accuracy :
    - big gap = overfitting
    - no gap = increase model capacity
- monitor the ratio of weight updates/ weight magnitudes
```python
param_scale = np.linalg.norm(W.ravel())
update = -learning_rate*dW # simple SGD update
update_scale = np.linalg.norm(update.ravel())
W += update # actual update
print update_scale/param_scale
```

#### Summary
- Activation Functions (use ReLU)
- data preprocessing (images : subtract mean)
- weight initialization (use Xavier init)
- Batch Normalization (use)
- Babysitting the Learning process
- Hyperparameter Optimization (random sample hyperparams, in log when appropriate)
___
6강에 이어서