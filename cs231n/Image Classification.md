# cs231n
## 2강 Image Classification Pipeline

challenges
- view point, illumination, deformation (형태 변형), Occlusion(은닉), Background Clutter, Intraclass variation

1. Nearest Neighbor Classifier (주로 학습용으로 사용)
- remember all training images and their labels -> predict the label of the most similar training image

- L1 distance (Manhatten) = 시그마|d1-d2| 절댓값 합 (np.sum(np.abs(self.X - x[i,:])), axis = 1) 

> Quiz

how does the classification speed depend on the size of the training data?
- linearly increase

- test time performance is usually much more important in practice
- CNN에서는 expensive training, cheap test evaluation

- L2 distance (Euclidean) = 루트(시그마(d1-d2)^2) 루트 제곱
    - K-NN : 좀 더 부드럽게 분류 수행함

> Quiz

what is the accuracy of the nearest neighbor classifier on the training data, when using the Euclidean distance?
- 100% => 동일한 이미지 비교니까 100% 정확도

what is the accuracy of the k-nearest neighbor classifier on the training data?
- 상황에 따라 다름

1. what is the best distance to use?
2. what is the best balue of k to use
3. how do we set the hyperparameters?
=> problem- dependent, try them all out and see what works best

바로 테스트 셋에 적용해보며 하이퍼 파라미터 적용해서는 안됨
- validation data 를 생성 => CV 활용

KNN은 이미지에 절대 사용 안함
1. test time에 성능이 안좋음
2. distance가 정확한 예측을 하지 못함 (unintutive)

### Linear Classification
- Parametirc approach (NN은 nonparametric)

이미지 내 모든 픽셀 값을 곱하여 처리한 것에 대한 합 (just a weighted sum of all the pixel values in the image)

각각 다른 공간적 위치에 있는 컬러를 파악 (counting colors at different spatial position)

컬러 정반대의 것 구분 힘듦, gray 색상 이미지 성능 안좋음, 색상이 동일한 경우 좋지 않음

score function으로 분류 => loss function을 이용해 score를 loss 계산해야함

___
3강에 이어서