# cs231n
### 8강

- localization : single object
- object detection : multiple object
- segmentation : 형상대로 땀 multiple object

- classification : C class
    - input : image
    - output : class label
    - evaluation metric : accuracy

- localization
    - input : image
    - output : box in the image (x,y,w,h)
    - evaluation : Intersection over Union(IOU)

ImageNet = Classification + Localization


### Localization
1. regression + classification
- input image (only one object, simpler than detection) -> NN (Neural Net)
    - output : Box coordinates (x,y,w,h)
    - correct output : Box coordinates
    - Loss : L2 distance

- regression을 after cov layers (Overfeat, VGG) 나 after last FC layers (DeepPose, R-CNN) 에 붙임
- 정해진 수의 object를 찾는 것은 multiple이여도 잘함
- very simple

2. sliding Window
- classification + regression
- fully connect layers into convolution
- Overfeat
    - bounding box 와 score를 merge

#### Object detechtion
1. detection as regression => 이미지의 갯수에 따라서 output의 size가 달라지고 많은 수가 필요해져서 적당하지 않음 => YOLO

2. HOG => DPM
    - need to test many positions and scales and use a computationally demanding classifier (CNN) => only look tiny subset of possible position (region proposal)
        - find blobby (class - agnostic), 클래스와 상관없이 blob으로 박스를 침
        - selective search, EdgeBoxes
3. R-CNN
    - 2000개의 RoI를 추출 (다른 크기, 다른 위치) => warped => ConvNet => Bounding-Box reg, SVM

    - train for ImageNet
    - fine-tune model for detection
    - extract features (crop + Warp) =? forward pass
    - train on binary SVM per class
    - bbox regression (cached region feature) 보정 역할

- Evaluation : mAP (mean average precision) : 0.5가 넘으면 맞다고 판단

- 단점 : slow at test-time , svm과 regressors are post-hoc (바로바로 반응 못함) , complex multistage training => Fast R-CNN (CNN을 먼저 돌린다음 region을 추출) 
    - share computation (속도 빨라짐), train whole system end-to-end (효율적이됨), image map에서 feature map으로 projection
    - but dont include region proposal => just make CNN do region proposal => Faster R-CNN
        - Region Proposal Network(RPN) after the last CNN
        - anchor box (슬라이딩 윈도우마다 각각 다른 크기의 비율) translation invariant (feature map에서 image map으로 투영)
        - 총 4개의 loss (RPN classification, RPN regression, Fast R-CNN classification, Fast R-CNN regression)

4. YOLO (You Only Look Once)
- detection as regression

- 7x7 grid => regression from image to 7x7x(5*B+C) tensor
- faster than faster R-CNN but not as good

