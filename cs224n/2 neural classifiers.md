# cs224n - 2021
## Lecture 2 - Neural Classifiers

- Gradient
  - 미분 값을 구하는 것은 very expensive to compute
  - sol : SGD (일부 데이터만 이용해 샘플링 계산) => 빠르게 계산 가능

- use two vector => easier optimization
  - Skip-gram(SG) : predict context words given center word (중심 단어를 통해 문장 단어 예측)
    - more natural
    - with negative sampling (try to maximize the same objective)
  - Continuous Bag of Words(CBOW) : predict center word form context words (주변 단어 통해 중심 단어 예측)
  
- co-occurrence matrix
  - windows vs full documnet
  - simple count => increase in size, high dimensional, sparsity issue
  - low-dimensinal vectors (더 선호)
    - 주로 25~1000 dimension
    - Singular Value Decomposition (특이값 분해) 사용해서 차원 감소 => 실제로 잘 작동안함 (fixed 시켜주면 됨)

- count based vs direct prediction

![image](https://user-images.githubusercontent.com/108413432/209461858-01ec7ad4-3161-4626-bdc5-8c69a4a4abcc.png)

- Glove
  - 위 둘을 합침
  - 훈련 빠르고 scalable (큰 것에서도), 작은 벡터나 코퍼스에서도 좋은 성능

![image](https://user-images.githubusercontent.com/108413432/209461937-1ebd7bfa-f294-4e75-a5d1-9972401653c4.png)
  - wi, wj 모두 linear => log-bilinear model

- evaluate word vectors
![image](https://user-images.githubusercontent.com/108413432/209461968-24dac361-484d-4468-b9e5-4b813bb72d1c.png)
  - Glove word vectors evaluation이 높은 점수를 보여줌
  - more data help, Wikipedia 가 도움을 많이 줌
  - word2vec 은 구글은 뉴스에서만 가져옴
  - 최적은 300vector 값임

- word amibuity : 단어들이 여러개 뜻을 가짐
  - 클러스터링을 이용해 각각 다른 단어 벡터를 만들어 주도록 하기
    - 의미 확장이기에 의미 구분은 힘듦
  
  - weighted sum in word embedding like word2vec => 고차원에서 sparse 표현
