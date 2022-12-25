# cs224n - 2021
## Lecture 3 - Backprop and Neural Networks

#### Named Entity Recognition(NER)
- simple NER - classify each word in its context window using binary logistic classification

- SGD에서 미분 값을 계산하기 위해서는 직접 계산 및 backpropagation algorithm 을 이용

#### compute gradient by hand
- matrix calculus (행렬 미적분)
- gradient is vector of partial derivatives with respect to each input

- Jacobian Matrix - m outputs and n inputs
![image](https://user-images.githubusercontent.com/108413432/209464126-fd824a28-450b-46b1-873f-aedcffabc993.png)

- chain rule
  - one variable ftns : multiply derivative
  - multiple variables at once : multiply Jacobians

![image](https://user-images.githubusercontent.com/108413432/209464221-4a53da3e-a498-400c-b614-88acb0f9f60d.png)
  - Jacobian으로 계산하는 방법

![image](https://user-images.githubusercontent.com/108413432/209464315-9b9e1019-8304-49c3-9cf9-5e08b13e25bb.png)
1. break up equation : Wx +b 를 z로 바꿔서 계산을 좀 더 편하게 해준다.
2. apply the chain rule : ![image](https://user-images.githubusercontent.com/108413432/209464347-8d44b710-0211-4812-8b7d-df39541bb16c.png)
3. Jacobians
  - uTh 에 대해 u 미분 값은 hT 이 된다
  - f(z)의 미분은 diag(f'(z)) 가 된다.
  - uT * f'(z) 이 되게 된다

4. Re-using compute => s를 W로 미분 (마지막 b 미분을 W 미분으로 비꾸면 같아짐)
  - 1xnxm matrix를 가짐
  - instead pure math => use shape convention
    - 델타를 이용해 답을 구하기
![image](https://user-images.githubusercontent.com/108413432/209464525-80d9d141-a45a-4bba-8f9c-6437dcb39175.png)

5. deriving local input gradient in backprop

- 미분을 할 때 모양
  1. use Jacobian form as much as possible (미분이 편하기 때문에) , reshape to follow the shape convention at the end (마지막에 모양 같게 하기)
    - row vector가 나오므로 마지막에 Transpose 를 이용해 make column vector
  2. always follow the shape convention (계속 해야하기에 조금 복잡)


#### Backpropagation
- take derivative and use the chain rule (위에서 했던 것을 이용한 것임)
![image](https://user-images.githubusercontent.com/108413432/209464835-7de4b271-4f32-4f18-948f-d7286d59b15b.png)
  - 위 검정 글씨로 진행 되는게 forward propagation (순차적으로 진행해서 계산)
  - 밑 파란 글씨가 backpropagation (1부터 시작해서 기울기를 구하는거)
    - chain rule 이용해서 구함

- complexity of f-prop and b-prop is same

- 잘 되고 있는지 check 위해서 numeric gradient 이용해 approximate 사용하여 확인 가능

Backward pass : apply chain rule to compute gradient
