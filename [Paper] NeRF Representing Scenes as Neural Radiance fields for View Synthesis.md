# 이윤영 - NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

<img width="524" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-10_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3 45 58" src="https://github.com/user-attachments/assets/e23576d9-da15-44f8-9e45-d7e319a70868">


# 1. Introduction

- NeRF : view synthesis 문제를 해결하는 새로운 방식
- 객체의 3d 모델을 생성하는 기술이 아니라, 객체를 바라보는 모든 장면을 생성하는 Novel view synthesis 기술
    
    → 실제로 촬영하지 않은 각도에서의 view를 만들어냄
    

**방법론**

- 5D parameters($x, y, z$, $\theta$, $\phi$)
- 각 지점 ($x, y, z$)에서 방향($\theta, \phi$)으로 방출되는 복사율을 출력
- 각 지점에서의 밀도(density)는 광선이 지점을 통과할 때 누적되는 복사율을 조절하는 미분 불투명도(differential opacity)처럼 작용됨

**→ 뭔소리?**

---

### 복사율

- 특정 위치에서 특정 방향으로 얼마나 많은 빛이 방출되냐
- 즉, 특정 위치에서 카메라가 어떤 방향으로 바라볼 때 어떤 색이 보이냐를 예측

### 밀도(density)

- 특정 위치의 불투명도를 나타냄 → 해당 위치를 통과하는 빛의 양에 영향
    - 밀도 높을수록 빛이 많이 흡수됨 → 더 불투명하게 보임
    - 밀도 낮을수록 빛이 쉽게 통과됨 → 더 투명하게 보임

### 미분 불투명도

- 밀도가 위치에 따라 미세하게 변할 수 있음
    
    → 빛이 특정 위치를 통과할 때 얼마나 많이 흡수되거나 방출되는지를 조정가능
    

---

- 놀라운 것은 CNN이 아닌 **MLP만을 사용**함..!

input : 단일 5D 좌표

output : 단일 볼륨 밀도, view-dependent RGB

**NeRF 렌더링 과정**

1. 카메라 ray를 장면에 통과시켜 3D points들의 set을 생성
2. 포인트들과 해당하는 2D 시점 방향을 NN 신경망의 input으로 사용 
    
    → output으로 RGB, 밀도 출력
    
3. 고전적인 볼륨 렌더링 기법을 사용해 이 색상과 밀도를 누적하여 2D이미지 생성 → 미분가능
    - loss : 관찰된 이미지와 렌더링된 뷰 사이의 오차

<img width="724" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-10_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3 18 26" src="https://github.com/user-attachments/assets/b121e650-6031-4525-9fe6-d8c27c3c565d">


- 다만 이러한 MLP만으로는 충분한 성능을 낼 수는 없어서 두 가지 방식을 제시함
    - positional encoding : MLP가 high frequency 함수 표현 위함
    - hierarchical sampling : high frequnecy representation을 적절히 샘플링하기 위함
- NeRF의 기술적 기여를 이 논문에서는 아래와 같이 설명함
    1. 복잡한 기하학+재질을 가진 연속적 장면을 5D Neural radiance field로 표현 → MLP 네트워크로 매개변수화
    2. 고전적인 볼륨렌더링 + 미분 렌더링
    3. positional encoding + hierarchical sampling으로 최적화 가능

# 2. Related Work

1. **Neural 3D shape representation**
    - 이 기존 방식은 3D ground truth가 필요함
    - 기하학적 복잡도가 낮은 단순한 형상에만 적용됨
    
    → NeRF : 5D radiance field를 인코딩하는 NN을 최적화 : 더 복잡한 장면의 새로운 뷰를 고해상도로 렌더링 가능
    
2. **View synthesize and Image based randering**
    - 메쉬 기반 표현법
        - local minimum에 빠지거나 loss의 gradient불량 문제
        - 최적화 전에 고정된 형태의 템플릿 메쉬가 제공되어야함
    - 부피 기반 표현
        - 고해상도 이미지를 생성할때 이산 샘플링으로 인한 시간 및 공간 복잡도가 높아짐
    
    → NeRF : deep fully-connected NN의 파라미터 내에 연속적인 부피를 인코딩 : 기존의 부피 기반 접근법보다 훨씬 고해상도 렌더링 + 샘플링된 부피 표현이 요구하는 저장 공간의 일부만 필요로 함.
    

# 3. Neural Radiance Field Scene Representation

![image](https://github.com/user-attachments/assets/63bd00ff-3ce9-4785-ae95-602b7289b000)


input : $x=(x, y, z)$ : 3차원 위치 + $(\theta + \phi)$ : 2차원 시점 방향 → 총 5차원 벡터값 함수

- 실제로는 2차원 방향을 3차원 직교 단위 벡터 $d$로 표현
- $(x,d)$

output : $c = (r, g, b)$ : 방출 색상, $\sigma$ : 볼륨밀도(투명도의 역수)

$F_\Theta : (x, d)$ → $(c, \sigma)$     ($\Theta$ : 가중치)

- multi-view에서 일관성을 유지하도록 하기 위해, 네트워크가 볼륨밀도를 위치 함수 $x$로만 예측하도록 제한함
- 색상은 위치와 시점 방향의 함수로 예측할 수 있도록 함

![image 1](https://github.com/user-attachments/assets/0717e19f-2682-4d4e-a9db-61eda176901f)


→ 위 두가지 방식을 적용하기 위해 $F_\Theta$는 

1. 먼저 입력 3D좌표 $x$를 8개의 fully-connected layer(each layer : 256 channels, ReLU)로 처리 : $\sigma$와 256차원의 feature vector 출력
2. 시점 방향  $d$ 와 feature vector concat : 추가적인 fully-connected layer(128 channels, ReLU)로 처리 : view-dependent RGB 출력

- 이러한 방식이 non-Lambertian effect를 어떻게 표현하는지 확인 가능

**non-Lambertian effect**

- 람베르트 표면 : 이상적인 확산 방사면 → 빛이 어떤 방향에서 들어와도 모든 방향으로 동일하게 반사
- 비-람베르트 표면 : 실제 세계의 표면 → 빛이 일정하지 않은 방식으로 반사
    - 반사광 : 빛이 특정 방향으로 강하게 반사됨
    - 입사각과 관찰자의 시점에 따라 반사되는 빛의 양이 달라짐

<img width="530" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-10_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4 31 51" src="https://github.com/user-attachments/assets/1bb984e1-b2fd-4126-b3a2-6f1b6ef4e7a3">


- 배 표면과 물의 표면에서 시점에 따른 각기 다른 반사광을 확인할 수 있음

# 4. Volume Rendering with Radiance Field

- 예상 색상 $C_n$

<img width="419" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-10_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 00 17" src="https://github.com/user-attachments/assets/8d3518cf-86c1-4dbc-83f4-d82dcca1e44f">


- $\sigma(x)$ : 광선이 위치 $x$에서 미소한 입자에 의해 종료될 확률의 미분값
- $r(t) = o + td$
- $T(t)$ : $t_n$에서 $t$까지 광선이 다른 입자와 충돌하지 않고 이동할 확률
    
    → 누적 투과율
    

위 적분을 계산하기 위해 구적법 사용

- 기존의 deterministic quadrature(특정 위치에서 미리 정해진 일정한 샘플 위치를 따라 값을 계산→ 고정된 이산 위치) 은 표현의 해상도를 제한
- 그래서 stratified sampling 사용
    - 연속적인 위치에서 샘플 추출 가능(샘플링 위치가 고정되지 않고 매번 달라짐)
    - 적은 샘플 수로도 더 부드럽고 연속적으로 표현 가능 → MLP에 적합

<img width="279" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-10_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 17 02" src="https://github.com/user-attachments/assets/be03b764-6214-4021-857c-964bff595e47">



- 구간 $[t_n, t_f]$를 $N$개의 균등하게 나누어진 구간으로 나누고, 각 구간에서 무작위로 하나의 샘플을 추출

<img width="377" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-10_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 19 14" src="https://github.com/user-attachments/assets/625f96b9-06bf-47c5-8574-1ac6af0aa594">


- 위 식으로 $C(r)$추정
- $\delta_i = t_{i+1} - t_i$ : 인접한 샘플들 사이의 거리

# 5. Optimizing Neural Radiance Field

<img width="466" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-10-11_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 18 25" src="https://github.com/user-attachments/assets/dc57fdf6-7244-4291-a4be-0140999b58dd">


- 기본적인 MLP만으로는 좋은 성능을 뽑아내긴 어려움
- 그래서 두가지 방법 더 추가 (positional encoding, hierarchical volume sampling)

## 5-1. Positional Encoding

- neural network : 낮은 frequency의 함수를 학습하는 경향이 있음
    
    → $F_\Theta$가 $xyz\theta\phi$입력 좌표를 직접 다루면 색상, 기하학의 high frequency 표현하지 못함
    
     : input을 neural net에 전달하기 전에 고차원 공간에 매핑으로 해결
    
- $F_\Theta$ = $F'_\Theta \circ \gamma$
    - $\gamma$는 $\R$에서 $\R^{2L}$로 고차원 공간으로 매핑하는 함수
    - $F'_\Theta$는 기존의 MLP

<img width="366" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-03_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 15 16" src="https://github.com/user-attachments/assets/34ac6bc4-f8d1-48d7-b08b-c0ffd8185492">


- 순서 정보를 포함시키기 위해 사용하는 transformer의 positional encoding과는 달리 MLP가 high frequency 함수를 더 쉽게 근사할 수 있도록 함

## 5-2. Hierarchical Volume Sampling

- 기존 전략 : neural radience field net을 카메라 광선의 각 경로를 따라 N개의 쿼리 지점에서 조밀하게 평가 → 비효율적 : 렌더링된 이미지에 기여하지 않는 빈 공간과 가려진 영역도 반복적으로 샘플링됨
- 최종 렌더링에 미치는 예상 효과에 따라 샘플을 비례적으로 할당
- coarse network, fine network 두개로 나눠 동시에 최적화
    - $N_c$개의 위치를 샘플링하고, coarse network를 여기서 평가
    - coarse network 출력 바탕으로 각 광선의 더 중요한 부분에 샘플이 편중되도록 위치를 샘플링
        
<img width="275" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-03_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 36 09" src="https://github.com/user-attachments/assets/28200bc3-84f5-40c2-8074-55748c15abd0">

        
        - $c_i$는 광선에 따른 모든 샘플 색상
    - 가중치들을 정규화하면 PDF를 생성할 수 있는데, 이 분포에서 inverse transform sampling을 거침
        
        (대충  중요한 부분에 더 많이 샘플링하는 전략인듯)
        
        → 두 번째 샘플 세트 $N_f$개를 샘플링
        
    - 첫 번째 샘플, 두 번째 샘플 합집합에서 fine network를 평가
    - $N_c+N_f$개의 샘플을 사용해 최종 렌더링 된 색상을 계산

## 5-3. Implementation Details

- 각 scene마다 별도의 continuous volume representation network를 최적화
- data : 각 해당 scene의 RGB image dataset + 그에 대응하는 카메라 위치 + 내부 파라미터 + 장면 경계
- 각 최적화 반복마다 dataset의 모든 픽셀에서 무작위로 카메라 광선 배치를 샘플링한 후, $N_c$와 $N_c+N_f$를 차례로 샘플을 쿼리

<img width="281" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-03_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_5 54 21" src="https://github.com/user-attachments/assets/0b27c101-7862-4713-b6e1-56cd70d50e14">


- coarse network, fine network 각각에서 실제 색상과 렌더링된 색상의 차이를 확인
- 각 배치에는 4096개의 광선을 사용
- $N_c = 64$, $N_f = 128$

<img width="469" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-03_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 00 41" src="https://github.com/user-attachments/assets/f202ebb0-3a01-4df2-8063-e72bea644974">


# 6. Results

## 6-1. Dataset

- 합성 이미지
- 실제 이미지

## 6-2. Comparisons

## 6-3. Discussion

<img width="483" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-03_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 02 41" src="https://github.com/user-attachments/assets/f63d02b3-e28b-4aea-ac02-39ca5949ee95">


<img width="488" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-03_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 02 58" src="https://github.com/user-attachments/assets/50f0359d-b5aa-4ce8-94b3-8aad56b30466">


## 6-4. Ablation Studies

<img width="471" alt="%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-03_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_6 12 59" src="https://github.com/user-attachments/assets/8fceacf5-a0f3-4270-9ae5-eace88977e11">
