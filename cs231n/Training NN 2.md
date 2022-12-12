# cs231n
## 6강

- SGD very slow progress along flat direction, jitter along steep one (지그제그)

#### 1st order optimization methods

1. Momentum update
- x의 값을 직접 업데이트 함
- v (속도) = mu *v - learning_rate*dx ==> x+= v
- mu = 0.5,0.9,0.99 로 사용함
- 처음엔 overshooting the target but overall getting to the minimum much faster

2. Nesterov Momentum update (NAG : Nesterov Accelerated Gradient)
- 모멘텀보다 더 빠르고 정확하게 목표지점으로 감
- lookahead gradient step (모멘텀 방향을 생각해 끝나는 점에서 시작한다고 생각함)
- 호환성이 떨어짐 => Phi를 도입해서 치환
- vanilla update
```python
v_prev = v
v = mu*v - learning_rate*dx
x += -mu*v_prev + (1+mu)*v
```
3. AdaGrad update
- per-parameter adaptive learning rate method (cache += dx**2 를 도입해 파라미터 별로 다른 lr을 대입)
```
chae += dx**2
x+= -learning_rate*dx / (np.sqrt(cache)+1e-7)
```
- 경사에 연연하지 않음
- lr이 종료되는 문제 발생

4. RMSProp update
```
cache = decay_rate *cache + (1-decay_rate)*dx**2
x += -learning_rate *dx / (np.sqrt(cache)+1e-7)
```
- Ada의 단점인 lr이 없애지는 것을 해결함

5. Adam update
- RMSProp과 momentum의 결합
```
m = beta1*m + (1-beta)*dx #momentum
v = beta2*v + (1-beta2)*(dx**2) #RMSProp
mb = m/(1-beta1**t) # bias correction
vb = v/(1-beta2**t)
x += -learning_rate*m / (np.sqrt(v)+1e-7)
```

___

- Learning_rate 결정법 (처음엔 매우 큰 수로 했다가 점점 decay 시키기)
1. step decay : lr을 일정간격으로 decay
2. exponential decay 를 현식적으로 많이 사용

#### 2nd order optimization method
- 경사뿐만 아니라 Hessian을 통해 곡면이 어떻게 생겼는지 알 수 있음 (바로 최적점으로 이동 할 수 있음)
=> lr 필요 없어짐

- 장점 : convergence가 매우 빠름, lr이 필요 없음
- 단점 : deep neural net에서는 사용 안함 => 엄청난 행렬을 가지며 inverse만들기 힘들어서 현실적 불가능

- BGFS가 제일 유명 (메모리 사용하기에 큰 네트워크에선 현실적 불가능)

- L-BFGS (메모리를 사용 안함, 모두 noise 제거하고 사용해야함, full batch에선 잘 작동, mini batch에서는 잘 작동안함)

#### Ensemble
1. train multiple independent models
2. at test time average their result

- 2% extra performance => 필수적으로 사용
- linear하게 속도 감소

3. parameter 앙상블도 성능 향상을 가져옴
4. checkpoint에서도 성능 향상 가저옴

#### dropout
- randomly set some neurons to zero in the forward pass (랜덤하게 일부 노드 0으로 설정)

왜 좋은지
1. forces the network to have a redundant representation
=> 하나의 노드가 다른 노드것을 같이 봄

2. dropout is training a large ensemble of models

- testime에는 dropout를 사용안해야 좋음
- training할 때에도 testime에 적용할려면 scale 해줘야 함
- at testime all neurons are active always -> we must scale the activations so that for eath neurons : output at test time = expected output at training time


#### Convolutional Neural Networks
은 다음 7강에서
