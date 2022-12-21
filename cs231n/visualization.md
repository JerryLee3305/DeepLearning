# cs231n
## 9강
### visualizations, adversarial examples

1. pool5 에 있는 뉴런이 어떤 부분을 특화 했는 지 시각화
2. gabor filter - 특정 방향성 외곽선 텍스처 분석에 사용 (conv1 에 사용) only interpretable on the first layer => 다른 것도 할 수 는 있는지만 해석이 용이하지는 않음 (하나의 filter에 대응)
3. fc7 layer - code들을 시각화 (t-SNE) 
4. occlusion experiments (은닉, 은폐한 부분을 0으로된 정사각형으로 만듦) => 확률적으로 그림을 그려 확률이 떨어진 곳을 찾아 분류를 진행

#### visualizing activations
1. deconvolution-based approach
    1. feed image into net
    2. pick a layer, set the gradient there to be all zero except for 1 for some neuron of interest
    3. backprop to image => `guided backpropagation` instead (ReLU의 경우 음수의 값을 0으로 놓고 나머지로 backward pass를 진행하는데 여기서 guided는 음수 값 0 놓고 backward pass 이후 음수 나온 값도 0으로 놔서 양수의 값만 가져올 수도 있도록 함) == conv6, conv9


2. optimization-based approach
    - 이미지를 parameter로 업데이트
    - 특성 클래스를 가진 스코어를 최대화로 하기
    1. zero 이미지를 네트워크에 넣어 forward
    2. 관심 대상인 클래스의 스코어 벡터의 그래디언트를 1로 backprop to image (나머지는 0임)
    3. 이미지에 대한 업데이트를 약간 수행
    4. 3번에 된 것을 다시 네트워크에 넣어 forward
    5. 3번 다시 수행

    - gradient on data has three channels => maximize => 1차원으로 히트맵
    - 영향력의 강도, 세기, 척도
    
    - L2 사용하는 것이 아니라 Blur 를 사용하면 조금 더 선명하게 보임
    - activation 을 1로 설정하는 것임

3. Neural Style
    - extract content target (low activation을 저장)
    - extract style target (pair-wise statistic 에서 gram matrices => 공분산 행렬)
    - 두개의 loss를 최소화
- can we use this to fool ConvNets? => yes! adversarial examples (우리 눈에는 동일하게 보이지만 실제로는 다르게 분류) linear nature 성질 때문