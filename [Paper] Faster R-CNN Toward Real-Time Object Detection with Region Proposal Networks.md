# 논문 리뷰 - Faster R-CNN : Toward Real-Time Object Detection with Region Proposal Networks

# 1. Introduction

- 기존 방식(rpn)의 문제점
    - R-CNN
        - reqion proposal : 물체가 속할 수 있는 후보군을 제안함
        - 모든 proposal이 CNN을 거쳐야함 → 연산량 매우 많음
        - selective search(유사한 픽셀들끼리의 그룹화)의 느린 속도 : 약 2초정도 걸림
    - Fast R-CNN
        - 이미지가 한 번의 CNN만 거침 : 하나의 feature map 생성
        - ROI pooling
            - region of interest 영역 → max pooling : (r,c,h,w)의 튜플 형태(사각형 좌표)
        - cpu 연산 → gpu연산을 도입하자
- 새로운 방법
    - RPN
    - region based detector와 generating region proposal을 함께 수행 가능
    - anchor 박스 도입

# 3. Faster R-CNN

- 두 개의 module로 구성되어 있음
    1. deep fully conv network : regions propose에 사용(RPN)
    2. Fast R-CNN detector : 1에서 제안된 영역을 사용
- attention 사용 → RPN module은 Fast R-CNN이 어디를 볼지를 알려줌

<img width="311" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-29_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 02 29" src="https://github.com/user-attachments/assets/a9569dd1-3d54-423f-bde9-1ad3413b53f0">

## 3-1. Region Proposal Networks(RPN)

![653d9cf9-8426-4622-b0c2-b515896e4ef2](https://github.com/user-attachments/assets/1518e78d-7221-4701-8c2e-962b56a0e02c)



- input : image
- output : set of rectangular object proposals with scores
    - output 출력 전에는 이동하는 변화량이 output으로 나오고 regression을 통해 anchor box 생성
- fully convolutional network
- regression, classification 수행

### Anchor Box

- anchor box를 만들기 위해 sliding window 기법 사용
- 각 sliding window 위치의 중심마다 최대 k개의 anchor box 생성
    - reg layer → 각 anchor box 네 귀퉁이 좌표 : 4k개 출력 생성
    - cls layer → yes or no : 2k개 출력 생성
    - feature map 크기가 W*H라면 anchor box 개수 = W*H*k : (그 중에서 가장자리가 feature map 벗어나는 anchor는 제거)
    - 논문에서는 3 scales, 3 ratios 총 9개 생성
    - translation invariant를 가짐 : 객체가 이미지 내에서 이동하더라도 같은 객체로 인식 : (CNN의 특성때문인가?)

<img width="600" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-31_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12 50 08" src="https://github.com/user-attachments/assets/9ff79090-307c-4ef7-a164-451b6e360472">


(a). image를 변형하여 여러개의 scale에서 각 변형된 이미지마다 feature map을 구함 → 시간 많이걸림

(b). image에서 하나의 feature map을 구하지만 다양한 filter로 pooling 수행 → (a)와 함께 자주 사용됨 

(c). image당 하나의 feature map, 하나의 filter로 다수의 anchor box 생성

### Loss function

<img width="400" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-31_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_10 00 37" src="https://github.com/user-attachments/assets/44af2e67-b881-4ef7-ac13-8ce318bd8dc2">


- binary class label 수행 (being object or not)
    - anchor 중 ground truth box(실제 경계 박스)와 IoU(Intersection-over-Union)-얼마나 많이 겹치는가, 가장 큰 anchor
    - anchor 중 ground truth box와 IoU가 0.7이상인 anchor
        - 위 조건 만족 : positive label
        - IoU 0.3 미만 : negative label
        - 나머지 : 훈련 제외
- L_cls / L_reg : cls, reg에서의 loss function
- p_i / p*_i : anchor box가 객체일 확률 / positive면 1, negative면 0
- t_i / t*_i : 예측 경계 박스의 4가지 좌표 / 실제 경계 박스의 4가지 좌표
- N_cls / N_reg : 정규분포

<img width="420" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-31_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_11 46 31" src="https://github.com/user-attachments/assets/b6161dcd-b07f-47b0-be3a-f14cab7258b4">


- 박스의 좌표값

### Training RPNs

- back propagation + SGD로 end-to-end 학습
- negative가 dominant함(객체보다는 background가 더 많으니깐?)
- 그렇기 때문에 positive label과 negative label을 1:1로 무작위로 sampling함.
- new layer는 zero mean, standard deviation 0.01 Gaussian distribution으로 초기화

## 3-2. Sharing Features for RPN and Fast R-CNN(진짜 뭔소리?)

- 4 Step Alternating Training
    1. RPN만 훈련함
    2. 1에서 훈련한 RPN으로 만들어진 proposal을 통해 Fast R-CNN을 훈련함
        
        → 두 개의 network는 conv layer를 공유하지 않음(뭔소리?)
        
    3. conv layer를 공유하며 RPN 훈련 초기화를 위해 Fast R-CNN 사용(RPN만 fine tunning)
        
        → conv layer 공유함
        
    4.  conv layer를 고정한 채로 Fast R-CNN을 fine tunning
- conv를 rpn, r-cnn에 맞게 학습 시키고 학습시킨 conv로 rpn, r-cnn을 학습시키자!

## 3-3. Implementation Details

<img width="467" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-31_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_12 28 21" src="https://github.com/user-attachments/assets/ef792d8d-688c-404e-a575-8730de4c6fa3">


- anchor보다 큰 객체에 대해서도 잘 인식하는 것을 확인할 수 있음(ex. boat)
- training 중에는 이러한 cross-boundary anchor들은 무시하고 훈련함 : loss에 기여 안함
- test 중에는 포함
- 중복되는 proposal들을 줄이기 위해 NMS(비 최댓값 억제) 채택

# 4. Experiments

![b3a9c1e3-d98a-40b9-b91b-914870693d9f](https://github.com/user-attachments/assets/fa965bad-bae7-453e-9d14-86e220964316)


- SS / EB / Faster R-CNN 성능비교와 conv layer shared, unshared 성능 비교

<img width="465" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-31_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 21 33" src="https://github.com/user-attachments/assets/1ef8582a-c351-4bcd-9764-9542fbdfa27d">


- ZF 대신 VGG 사용 성능 비교

<img width="596" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-31_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 25 13" src="https://github.com/user-attachments/assets/f4078f51-d2b6-42c8-a22a-e959216a1720">


- SS, ZF, VGG 성능 비교 : VGG가 ZF보다는 성능은 좋으나  속도는 느린것을 확인할 수 있음

<img width="368" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-31_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 28 32" src="https://github.com/user-attachments/assets/7e286ca2-e4f1-4864-bed9-711fdcf7700d">


- anchor parameter별 성능 비교

<img width="310" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-31_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 32 34" src="https://github.com/user-attachments/assets/e3167f48-96a6-4671-a0bb-b7800ef9bde6">


- lamda별 성능 비교 → 유의미한 parameter는 아닌듯

<img width="533" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-07-31_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_1 34 50" src="https://github.com/user-attachments/assets/47914636-a7cc-4206-b1a6-67c5b636df5e">


- proposal 개수 별 recall비교 → 적은 수의 proposal일수록 Faster R-CNN이 재현율이 좋은 것을 확인할 수 있음
