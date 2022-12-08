# cs231n
## 3강 Loss functions and Optimization

### Loss fn
1. SVM - Hinge Loss
(xi, yi) where xi is image, yi is label

score vector s = f(xi, W)

- SVM Loss form : Li = 시그마(max(0, sj-syi +1))
    - sj 는 잘못된 레이블 스코어, 잘 분류된 스코어, 1은 safety margin
    - 1이 더해졌기 때문에 본인의 값을 넣으면 Loss가 1이 되기 때문에 모두 1씩 증가 됨
- 최종 Loss 값 : L = 1/N * 시그마(Li)

- Li 중 max 값에 ^2을 한 squared hinge loss 도 존재함 => non linear
- 최저 값은 0, 최대 값은 무한대
- W를 초기화 해줄 때 score 값이 0 이된다. 그렇게 되면 Loss는 Class -1 이 된다 => score가 0이 되기 때문에 Li 들이 모두 1 씩을 가지게 되므로 본인 class 제외 하면 L은  # of class -1 이 된다.

> 0으로 만드는 Weight 값인 W가 unique하지 않다!
- L = 1/N시그마(Li) + 람다R(W)
    - lambda = regularization strength (hyperparameter)
    - a way of trading off training loss and generalization loss on test set
    - training error 는 안좋아지지만 test error 는 좋아짐
    - Weight가 모든 값을 고려하기를 원함 (diffuse over everthing)

```python
def Li_vectorized(x,y,W):
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] +1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i
```

2. Softmax Classifier (Multinomial Logistic Regression)
- scores = unnoramlized log probabilities of the classes, s = f(xi;W)
    - want to maximize the log likelihood or to minimize negative log likelihood of the correct class
- Li = -log(P(Y=yi|X = xi)) => Cross-Entropy Loss

- exp 해준 다음 => normalize => -log()
- Loss 최소 0, 최대 무한대
- W를 초기화 할 때 score 값이 0이라면 Loss의 값은 -log(1/class) #sanity check


> svm은 둔감하다. softmax는 예민하다(모든 값들을 확인해 변화함)

### Optimization
1. random search 는 좋지 않는 성능을 가짐
2. follow the slope : 1차원에선 numerical gradient, 다차원은 gradient vector
- lim(h->0) (f(x+h)-f(x))/h  이것을 통해 업데이트 => 근사치, 평가 매우 느림

- analytic gradient : exact, fast => gradient check

3. Mini-batch Gradient Descent : only use small portion of training set to compute the gradient
- common size는 32/64/128 (CPU/GPU 환경에 맞게끔)
- Loss over mini-batches goes down over time 근데 지그제그로 왔다갔다 함
- very high learning rate => diverge / explode (위로 치솟음)
- low learning rate => slow convergence (오랜 시간 걸림)
- high learning rate => local minimum

=> decay (높게 설정했다가 낮게 설정)

- 딥러닝에서는 피처 추출을 해줄 필요가 없음
___
4강에 이어서
