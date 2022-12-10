# cs231n
## 4강

e.g) f(x,y,z) = (x+y)*z 

순차적으로 구하면 forward pass (FP)

gradient를 역으로 구하는 것을 backward pass, backpropagation

z는 쉽게 구할 수 있으나 x,y는 chain rule를 이용해서 구해야함

df/dq (global gradient) * dq/dy (local gradient) 를 이용

local gradient 는 forward pass에서 구해서 메모리에 저장이 된다.

global gradient 는 backward pass로만 구할 수 있다.

sigmoid function = 1/(1+e^(-x))
ds/dx = (1-s)s  #s는 local gradient 값을 넣어주면 됨

max gate는 큰 값을 1 작은 값을 0으로 해서 넣어주는 것임

jacobian matrix 는 input과 output 사이에 있는 matrix로 만일 input이 2048x1 이고 출력도 2048x1 이라면 자코비안도 2048x2048 형태를 가지게 된다.

이 모양은 Identity matrix와 형태는 비슷하면서 1과 0이 섞여 있어 sparse structure

mini batch로 100개씩 들어오면 형태는 204800x204800 형태가 된다.

sigmoid, tanh, ReLU, Leaky ReLU, Maxout, ELU

신경망 구조

input layer - hidden layer - output layer

- 2 layer NN, 1 hidden layer NN

input layer - hidden layer- hidden layer - output layer

- 3 layer NN, 2 hidden layer NN

hidden 이 많을 수록 more neurouns = more papacity 분류를 더 잘함

데이터가 오버피팅이 일어나지 않도록 하려면 네트워크를 작게 만드는 것이 아닌 regularization 을 높여줘야한다. (람다)
___
5강에 이어서