# cs231n
### 7강

#### CNN
- `convolve` the filter with the image i.e. slide over the image spatially, computing dot products

- 32(height)x32(width)x3(depth) image => 5x5x3 filter    => depth는 같음 , filter가 5x5x3 이므로 75번의 연산을 진행해 하나의 값이 나옴 => 총 28x28x1 activation map을 생성 (하나의 필터는 하나의 activation map을 생성)

- if we had 6 filters = get nex image of size 28x28x6 (re -representation)

- 32x32x3 =>(convolve, ReLU, 6 5x5x3) => 28x28x6 => (convolve, ReLU, 10 5x5x6) depth가 같아야하므로 6을 넣는것임 => 24x24x10

- 7x7 input assume 3x3 filter => 5x5 output (stride를 1)
- 7x7 input assume 3x3 filter => 3x3 output (stride를 2)
- 7x7 input assume 3x3 filter => doesn't fit (stride를 2)

- output size : (N-F)/s +1

- padding을 쓰면 size를 보정해줌 일반적으로 padding = (F-1)/2 개의 zero-padding을 시켜주면 됨

- n = (n+2p-f)/s +1

quiz
- input : 32x32x3, 10 5x5 filter with stride 1, pad 2
- output =>? (32+2*2-5)/1 +1 = 32 ===> filter의 개수가 10이므로 32x32x10이 나옴
- number of parameter =>? (5*5*3 +1) = 76 1은 bias => 10개의 filter이므로 76*10=760

- 동일한 depth 내에 있는 뉴런들은 동일한 weight를 가지게 된다. (parameter sharing)
- 별개의 activation map에 속해있는 동일한 뉴런들은 같은 local 를 쳐다보게 되지만 다른 weight를 가지게 됨


- pooling layer (파라미터는 없음)
    - makes the representations smaller and more manageable
    - operates over each activation map independently

- max pooling 을 많이 사용
    - input : 4x4 => 2x2 filter and stride 2=> output (n-f)/s +1 => 2x2에 가장 큰 값들만 가져옴
    - 정보를 손실하면서 invariance
    - depth는 보존됨, 주로 f=2, s=2 를 사용해서 1/2을 해줌

#### AlexNet
- first use of ReLU
- used Norm layers (not common anymore)
- heavy data augmentation
- dropout 0.5
- batch size 128
- SGD Momentum 0.9

#### ZFNet
- AlexNet에서 filter의 크기는 작게하면서 filter의 수는 늘림

#### VGGNet
- only 3x3 CONV stride1, pad 1 and 2x2 MAX POOL stride 2
- total memory = 24*4 bytes = 93MB / image(only forward ~ *2 for bwd)
- total params = 138M parameters

#### GoogleNet
- Inception module
- avg pool => parameter 수를 줄임

#### ResNet
- layer가 많아짐 = > error rate가 커져야 하는데 아님
- skip connection이 있음
- every CONV layer에서 batch normalization 사용
- lr:0.1, divied by 10 when validation error plateaus 
- no dropout used (batch normalization을 하면 사용 안해도 됨)

