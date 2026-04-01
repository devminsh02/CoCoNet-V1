# Objectness-Guided Contour Closure Reasoning for Camouflaged Object Detection
## 코드 기준 전체 설명서 (기호/정의 일관성 보강 버전)

이 문서는 현재 구현된 `COCO_V1` 코드베이스를 기준으로,  
추상 기호 `\phi`, `\psi`, `h`, `g` 같은 묶음 함수 대신 실제 모듈 이름, 레이어 순서, 출력 이름을 직접 적는 방식으로 전체 구조를 설명한다.

특히 다음 원칙을 따른다.

1. 실제 코드에 있는 `ConvBNReLU`, `MLP`, `ResNet50Backbone`, `SimpleFPNNeck`, `GlobalObjectnessBranch`, `ContourClosureBranch`, `GatedFusion`, `RefinementDecoder`를 그대로 드러낸다.
2. Branch A의 `uncertainty`는 독립 head가 아니라 fine logits의 sigmoid entropy에서 파생된 map이라는 점을 명시한다.
3. Branch A signal은 **raw signal**과 **used signal**을 구분한다.
4. Branch B에서 사용하는 `f_0`, `r`, `f_B`, `c`를 정의한 뒤 사용한다.
5. Graph 파트에서는 token feature, token coordinate, edge set, valid edge mask를 먼저 정의한 뒤 message passing 식을 적는다.
6. Fusion에서는 gate symbol `\gamma`를 먼저 정의한 뒤 최종 결합 식을 적는다.
7. Loss도 “이름만 Dice/topology”처럼 모호하게 쓰지 않고, 현재 코드에서 실제로 계산하는 식과 코드 의미를 적는다.

> GitHub 렌더링 호환성을 위해 display math는 `$$ ... $$` 형식으로 적고, 수식이 들어간 문장은 bold/italic 강조 대신 일반 문장으로 적었다.

---

## 0. 프로젝트 구조와 역할

현재 코드 기준 주요 디렉터리 역할은 다음과 같다.

- `main.py`  
  진입점. `prepare-splits`, `sanity-model`, `train`, `eval` 네 커맨드를 제공한다.
- `datasets/`  
  split 생성, record 구성, train/eval dataset, transform, target 생성 담당
- `models/`  
  backbone, neck, Branch A/B, fusion, decoder, 전체 조립(`cod_model.py`) 담당
- `losses/`  
  최종 mask/aux/boundary/affinity/topology loss 묶음
- `engine/trainer.py`  
  학습 루프, checkpoint, CSV/TensorBoard 로그, validation 관리
- `engine/evaluator.py`  
  metric 계산, prediction 저장, branch 시각화 저장
- `utils/visualization.py`  
  README에 적힌 시각화 산출물 생성

---

## 1. 표기와 기본 block

### 1.1 텐서 표기

입력 이미지를 다음처럼 둔다.

$$
x \in \mathbb{R}^{B \times 3 \times H \times W}
$$

여기서

- `B`: batch size
- `H, W`: 입력 이미지 해상도

현재 기본 학습/평가 해상도는 보통 `352 × 352`이다.

---

### 1.2 `ConvBNReLU`

`utils/common.py`의 `ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=None)`는 아래 순서다.

1. `nn.Conv2d`
2. `nn.BatchNorm2d`
3. `nn.ReLU(inplace=True)`

수식으로 쓰면

$$
\mathrm{ConvBNReLU}(x)
=
\mathrm{ReLU}\big(\mathrm{BN}(\mathrm{Conv}(x))\big)
$$

padding을 지정하지 않으면 `kernel_size // 2`가 들어가므로, `kernel_size=3`에서는 spatial size를 유지한다.

---

### 1.3 `MLP`

`utils/common.py`의 `MLP(in_dim, hidden_dim, out_dim, dropout)`는 아래 순서다.

1. `nn.Linear(in_dim, hidden_dim)`
2. `nn.ReLU(inplace=True)`
3. `nn.Dropout(dropout)`
4. `nn.Linear(hidden_dim, out_dim)`

수식으로는

$$
\mathrm{MLP}(x)
=
W_2\Big(\mathrm{Dropout}\big(\mathrm{ReLU}(W_1 x)\big)\Big)
$$

이다.

---

### 1.4 `BCEWithLogits`

이 문서에서 `\mathrm{BCEWithLogits}(z, y)`는 `nn.BCEWithLogitsLoss`를 뜻한다.

여기서

- `z`: sigmoid를 적용하기 전의 logit
- `y \in \{0,1\}`: binary target

이다.

한 픽셀 기준 식은 다음과 같다.

$$
\mathrm{BCEWithLogits}(z,y)
=
-\,y\log\sigma(z) - (1-y)\log(1-\sigma(z))
$$

현재 코드에서는 이 연산을 map 전체에 대해 평균하는 형태로 사용한다.

---

### 1.5 자주 쓰는 기호

이 문서에서는 필요한 최소 기호만 쓴다.

- `σ(·)`: sigmoid
- `U(a \to b)`: `a`를 `b`의 spatial size에 bilinear upsampling
- `[a,b,c]`: channel concat
- `⊙`: element-wise multiplication
- `A_G(·)`: `G × G` adaptive average pooling
- 기본 token grid 크기: `G = 16`
- token 수: `N = 16 × 16 = 256`

---

## 2. Backbone과 Neck

### 2.1 Backbone: `ResNet50Backbone`

현재 backbone은 `models/backbones/resnet50_backbone.py`의 `ResNet50Backbone`이다.

내부 구성:

- `stem = conv1 + bn1 + relu`
- `maxpool`
- `layer1`
- `layer2`
- `layer3`
- `layer4`

출력은

$$
(c_2, c_3, c_4, c_5) = \mathrm{ResNet50Backbone}(x)
$$

이다.

shape는 대략

- `c_2 ∈ R^{B × 256 × H/4 × W/4}`
- `c_3 ∈ R^{B × 512 × H/8 × W/8}`
- `c_4 ∈ R^{B × 1024 × H/16 × W/16}`
- `c_5 ∈ R^{B × 2048 × H/32 × W/32}`

이다.

---

### 2.2 Neck: `SimpleFPNNeck`

현재 neck은 `models/necks/simple_fpn_neck.py`의 `SimpleFPNNeck`이다.

구성은 다음과 같다.

#### lateral projection
- `lateral_c2 = Conv2d(256 → 256, kernel=1)`
- `lateral_c3 = Conv2d(512 → 256, kernel=1)`
- `lateral_c4 = Conv2d(1024 → 256, kernel=1)`
- `lateral_c5 = Conv2d(2048 → 256, kernel=1)`

#### smoothing
- `smooth_p2 = ConvBNReLU(256 → 256, kernel=3)`
- `smooth_p3 = ConvBNReLU(256 → 256, kernel=3)`
- `smooth_p4 = ConvBNReLU(256 → 256, kernel=3)`
- `smooth_p5 = ConvBNReLU(256 → 256, kernel=3)`

실제 계산은

$$
p_5 = \mathrm{smooth\_p5}(\mathrm{lateral\_c5}(c_5))
$$

$$
p_4 = \mathrm{smooth\_p4}(\mathrm{lateral\_c4}(c_4) + U(p_5 \to c_4))
$$

$$
p_3 = \mathrm{smooth\_p3}(\mathrm{lateral\_c3}(c_3) + U(p_4 \to c_3))
$$

$$
p_2 = \mathrm{smooth\_p2}(\mathrm{lateral\_c2}(c_2) + U(p_3 \to c_2))
$$

이다.

출력은

$$
(p_2, p_3, p_4, p_5)
$$

이며, 네 feature 모두 채널 수는 256이다.

---

## 3. Branch A: Global / Objectness Branch

Branch A는 `models/branches/branch_a/global_objectness_branch.py`의 `GlobalObjectnessBranch`다.

출력 이름은 코드 기준으로 다음을 포함한다.

- `coarse_logits`
- `fine_logits`
- `objectness_logits`
- `boundary_prior_logits`
- `objectness_map`
- `uncertainty_map`
- `boundary_prior`
- `a_feats`
- `coarse_feat`

중요한 점은 uncertainty가 독립 conv head가 아니라 `fine_logits`에서 파생된다는 것이다.

---

### 3.1 Branch A fused feature `f_A`

먼저 `p_3`, `p_4`, `p_5`를 모두 `p_2` 해상도로 올린다.

$$
\tilde p_3 = U(p_3 \to p_2), \quad
\tilde p_4 = U(p_4 \to p_2), \quad
\tilde p_5 = U(p_5 \to p_2)
$$

그리고 concat 후 `pyramid_fuse`를 통과시킨다.

`pyramid_fuse`의 실제 구성:

1. `ConvBNReLU(1024 → 256)`
2. `ConvBNReLU(256 → 256)`

즉

$$
f_A
=
\mathrm{ConvBNReLU}_{256\to256}
\Big(
\mathrm{ConvBNReLU}_{1024\to256}
([p_2,\tilde p_3,\tilde p_4,\tilde p_5])
\Big)
$$

이다.

shape:

$$
f_A \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
$$

---

### 3.2 Coarse branch

coarse segmentation은 `p_4` 기반으로 계산한다.

먼저

$$
f_{\mathrm{coarse}} = \mathrm{ConvBNReLU}_{256\to256}(p_4)
$$

를 만든다.

그 다음 `CoarseHead`를 적용한다.

`CoarseHead`의 실제 구성:
1. `ConvBNReLU(256 → 256)`
2. `Conv2d(256 → 1, kernel=1)`

즉

$$
z_{\mathrm{coarse}}^{(low)}
=
\mathrm{Conv2d}_{256\to1}
\Big(
\mathrm{ConvBNReLU}_{256\to256}(f_{\mathrm{coarse}})
\Big)
$$

그 다음 이를 `p_2` 해상도로 업샘플한다.

$$
z_{\mathrm{coarse}} = U(z_{\mathrm{coarse}}^{(low)} \to p_2)
$$

---

### 3.3 Fine branch

fine segmentation은 `f_A`에서 직접 계산한다.

`FineHead`의 실제 구성:
1. `ConvBNReLU(256 → 256)`
2. `ConvBNReLU(256 → 256)`
3. `Conv2d(256 → 1, kernel=1)`

즉

$$
z_{\mathrm{fine}}
=
\mathrm{Conv2d}_{256\to1}
\Big(
\mathrm{ConvBNReLU}_{256\to256}
(
\mathrm{ConvBNReLU}_{256\to256}(f_A)
)
\Big)
$$

---

### 3.4 Objectness branch

objectness는 `f_A`에서 바로 1×1 conv로 만든다.

$$
z_{\mathrm{obj}} = \mathrm{Conv2d}_{256\to1}(f_A)
$$

$$
o = \sigma(z_{\mathrm{obj}})
$$

현재 코드에서 objectness head는 1×1 conv 단일층이다.

---

### 3.5 Boundary prior branch

boundary prior는 `BoundaryPriorHead`에서 계산한다.

`BoundaryPriorHead`의 실제 구성:
1. `ConvBNReLU(256 → 256)`
2. `Conv2d(256 → 1, kernel=1)`

즉

$$
z_{\mathrm{bp}}
=
\mathrm{Conv2d}_{256\to1}
\Big(
\mathrm{ConvBNReLU}_{256\to256}(f_A)
\Big)
$$

$$
b = \sigma(z_{\mathrm{bp}})
$$

---

### 3.6 Uncertainty는 독립 head가 아니다

현재 코드에서는 uncertainty를 위한 별도 `nn.Conv2d` head가 없다.

먼저 fine logit을

$$
z = z_{\mathrm{fine}}
$$

라고 둔다.

그 다음 sigmoid probability를

$$
p = \sigma(z)
$$

라고 둔다.

여기서
- `z`: fine branch가 출력한 fine segmentation logit map
- `p`: `z`에 sigmoid를 적용한 fine probability map
- `\epsilon`: 수치 안정성을 위한 작은 양수 (`10^{-6}` 수준)
- `\bar p`: `p`를 `[\epsilon, 1-\epsilon]` 범위로 clamp한 값
- `u`: 최종 uncertainty map

이다.

즉 numerical clip은

$$
\bar p = \mathrm{clip}(p, \epsilon, 1-\epsilon)
$$

이고, uncertainty는 sigmoid entropy로 계산된다.

$$
u
=
-\Big(
\bar p \log \bar p
+
(1-\bar p)\log(1-\bar p)
\Big)
$$

즉 uncertainty는 **fine logits에서 계산한 sigmoid entropy map**이다.

---

### 3.7 Branch A signal의 의미

이 문서에서 **signal**이라는 말은, Branch A가 생성해서 이후 Branch B와 Fusion에 guidance로 전달하는 **1채널 spatial map**을 뜻한다.

현재 signal은 세 개다.

- `o = objectness_map`
- `u = uncertainty_map`
- `b = boundary_prior`

즉 signal은 feature tensor(`a_feats`)와 구분되는 개념이다.

- `a_feats`: 256채널 feature representation
- `signal`: 1채널 guidance map

이다.

### raw signal vs used signal

현재 `cod_model.py`에서는 Branch A signal을 두 단계로 구분한다.

1. **raw signal**  
   Branch A가 원래 예측한 값
2. **used signal**  
   signal switch를 통과한 뒤, 실제로 downstream에 전달되는 값

즉 예를 들어 objectness에 대해서는

$$
o^* =
\begin{cases}
o, & \text{if objectness switch = True}\\
0, & \text{if objectness switch = False}
\end{cases}
$$

uncertainty와 boundary prior도 같은 방식이다.

$$
u^* =
\begin{cases}
u, & \text{if uncertainty switch = True}\\
0, & \text{if uncertainty switch = False}
\end{cases}
$$

$$
b^* =
\begin{cases}
b, & \text{if boundary prior switch = True}\\
0, & \text{if boundary prior switch = False}
\end{cases}
$$

즉 ablation은 raw signal을 없애는 것이 아니라, **downstream으로 실제 전달되는 used signal만 끄는 방식**이다.

---

## 4. Branch B: 입력, ROI gating, feature, tokenization

Branch B는 `models/branches/branch_b/contour_closure_branch.py`에 구현되어 있으며,  
현재 코드에서는 다음 순서로 동작한다.

1. low-level / high-level feature 결합
2. Branch A signal로부터 ROI mask 생성
3. ROI mask로 Branch B feature 강조
4. boundary candidate 생성
5. tokenization
6. graph reasoning
7. closure map 생성
8. refined boundary 생성
9. optional affinity edge score 생성

문서에서는 위 순서를 그대로 따라가며,  
각 단계에서 쓰는 `f_0`, `r`, `f_B`, `c`를 먼저 정의한 뒤 다음 연산에 사용한다.

---

### 4.1 Branch B 입력

Branch B는 다음 다섯 입력을 받는다.

- `p_2`: neck의 low-level feature
- `p_4`: neck의 high-level feature
- `o^*`: used objectness map
- `u^*`: used uncertainty map
- `b^*`: used boundary prior

각 shape는 다음과 같다.

$$
p_2 \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
$$

$$
p_4 \in \mathbb{R}^{B \times 256 \times H/16 \times W/16}
$$

$$
o^*,u^*,b^* \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
$$

---

### 4.2 High-level feature upsampling과 base fused feature `f_0`

먼저 high-level feature `p_4`를 `p_2` 해상도로 bilinear upsampling 한다.

$$
\tilde p_4 = U(p_4 \to p_2)
$$

즉,

$$
\tilde p_4 \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
$$

그 다음 `p_2`와 `\tilde p_4`를 channel concat한다.

$$
[p_2,\tilde p_4] \in \mathbb{R}^{B \times 512 \times H/4 \times W/4}
$$

이 concat feature를 `feature_fuse`에 넣는다.  
현재 `feature_fuse`의 실제 구성은 다음과 같다.

1. `ConvBNReLU(512 → 256, kernel=3)`
2. `ConvBNReLU(256 → 256, kernel=3)`

따라서 Branch B의 base fused feature `f_0`는 다음처럼 정의한다.

$$
f_0
=
\mathrm{ConvBNReLU}_{256\to256}
\Big(
\mathrm{ConvBNReLU}_{512\to256}
([p_2,\tilde p_4])
\Big)
$$

shape는

$$
f_0 \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
$$

이다.

여기서 `f_0`는 아직 ROI gating이 적용되기 전의,  
Branch B가 사용할 기본 spatial feature라고 해석하면 된다.

---

### 4.3 ROI gating과 ROI mask `r`

Branch B는 Branch A signal을 이용해 **soft ROI mask**를 만든다.  
입력 signal은 다음 세 개다.

- `o^* = used_objectness_map`
- `u^* = used_uncertainty_map`
- `b^* = used_boundary_prior`

세 map을 channel 방향으로 concat하면 ROI gating 입력이 된다.

$$
s = [o^*,u^*,b^*]
\in
\mathbb{R}^{B \times 3 \times H/4 \times W/4}
$$

현재 `SoftROIGating`의 실제 구성은 다음과 같다.

1. `ConvBNReLU(3 → 64, kernel=3)`
2. `ConvBNReLU(64 → 64, kernel=3)`
3. `Conv2d(64 → 1, kernel=1)`
4. `Sigmoid`

따라서 ROI mask `r`은 다음처럼 정의한다.

$$
r
=
\sigma
\Big(
\mathrm{Conv2d}_{64\to1}
(
\mathrm{ConvBNReLU}_{64\to64}
(
\mathrm{ConvBNReLU}_{3\to64}(s)
)
)
\Big)
$$

shape는

$$
r \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
$$

이다.

여기서 `r`은 hard binary crop mask가 아니라,  
각 위치를 얼마나 강조할지 나타내는 **soft ROI confidence map**이다.

---

### 4.4 ROI-enhanced Branch B feature `f_B`

이제 ROI mask `r`을 base fused feature `f_0`에 적용한다.

현재 코드의 실제 연산은

$$
f_B = f_0 \odot (1 + r)
$$

이다.

shape는

$$
f_B \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
$$

이다.

이 식의 의미는 다음과 같다.

- `r = 0`이면  
  `f_B = f_0` 이므로 원래 feature를 그대로 유지한다.
- `r > 0`이면  
  해당 위치의 feature magnitude가 증가한다.

즉 ROI gating은 “ROI 밖 feature를 제거”하는 hard masking이 아니라,  
**기본 feature는 유지하고 ROI 내부를 추가로 증폭하는 soft spatial modulation**이다.

---

### 4.5 Boundary candidate map `c`

ROI-enhanced feature `f_B`에서 먼저 boundary candidate를 만든다.

현재 `BoundaryCandidateHead`의 실제 구성은 다음과 같다.

1. `ConvBNReLU(256 → 256, kernel=3)`
2. `Conv2d(256 → 1, kernel=1)`

즉 boundary candidate logits는

$$
z_{\mathrm{cand}}
=
\mathrm{Conv2d}_{256\to1}
\Big(
\mathrm{ConvBNReLU}_{256\to256}(f_B)
\Big)
$$

이고, 이를 sigmoid에 통과시킨 boundary candidate probability map을 `c`라고 둔다.

$$
c = \sigma(z_{\mathrm{cand}})
$$

shape는

$$
c \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
$$

이다.

여기서 `c`는 “최종 refined boundary”가 아니라,  
**tokenization과 이후 graph reasoning에 쓰일 넓은 의미의 boundary candidate map**이다.

---

### 4.6 Fragment Tokenization

Fragment Tokenization은 다음 세 입력을 사용한다.

- `f_B`: ROI-enhanced Branch B feature
- `c`: boundary candidate probability map
- `r`: ROI mask

즉 tokenization 입력은 이미 앞 절에서 정의된

$$
f_B \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
$$

$$
c \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
$$

$$
r \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
$$

이다.

현재 `FragmentTokenizer`는 이 세 개에 대해 모두 `16×16` adaptive average pooling을 적용한다.

$$
\bar f = A_{16}(f_B)
$$

$$
\bar c = A_{16}(c)
$$

$$
\bar r = A_{16}(r)
$$

shape는 각각

$$
\bar f \in \mathbb{R}^{B \times 256 \times 16 \times 16}
$$

$$
\bar c,\bar r \in \mathbb{R}^{B \times 1 \times 16 \times 16}
$$

이다.

그 다음 token score를 다음처럼 정의한다.

$$
s_{\mathrm{tok}} = \bar c \odot \bar r
$$

즉 어떤 grid cell이 token으로 중요해지려면,
- boundary candidate가 높고
- ROI confidence도 높아야 한다.

feature token 자체는 `\bar f`를 flatten 후 transpose해서 만든다.

$$
T = \mathrm{Flatten}(\bar f)
\in
\mathbb{R}^{B \times N \times C}
$$

여기서
- `N = 16 \times 16 = 256`: token 수
- `C = 256`: token channel dim

이다.

더 명시적으로 쓰면, batch `b`의 i번째 token feature를

$$
T_i \in \mathbb{R}^{256}
$$

라고 둔다.

그리고 valid token mask는 다음 기준으로 만든다.

$$
m_i = \mathbf{1}(s_{\mathrm{tok},i} > 0.05)
$$

좌표는 learned embedding이 아니라 **고정된 16×16 정규화 grid coordinate**다.  
배치 `b`의 i번째 token 좌표를

$$
q_i \in [0,1]^2
$$

라고 둔다.

전체 좌표 집합은

$$
Q = \{q_i\}_{i=1}^{N}
$$

이다.

즉 tokenization 단계는 단순히 `f_B`만 쓰는 것이 아니라,  
`f_B`로 token feature를 만들고, `c`와 `r`를 함께 써서 어떤 token이 graph reasoning에 실질적으로 참여할지를 정한다.

---

## 5. Graph construction과 message passing

### 5.1 Graph construction

Graph는 `FragmentTokenizer`가 만든 정규화 좌표 집합

$$
Q = \{q_i\}_{i=1}^{N}, \qquad q_i \in [0,1]^2
$$

위에서 구성한다.

현재 코드에서 graph는 k-nearest neighbor graph이며, 기본 `k = 8`이다.

따라서 각 token `i`에 대해 좌표 `q_i`와 가장 가까운 `k`개의 이웃 좌표를 찾는다.  
이 과정을 좌표 집합 `Q` 전체에 대해 적용한 것이 graph construction이다.

이를 기호로 쓰면, edge set은

$$
E = \{(i,j) \mid q_j \in \mathrm{kNN}(q_i; Q, k)\}
$$

이다.

여기서
- `Q`: 모든 token coordinate 집합
- `q_i`: i번째 token coordinate
- `\mathrm{kNN}(q_i; Q, k)`: 좌표 집합 `Q` 안에서 `q_i`와 가장 가까운 `k`개 좌표의 인덱스 집합

이다.

즉 `\mathrm{kNN}(q_i; Q, k)`라고 써야 맞고,  
갑자기 `\mathrm{kNN}(q,k)`처럼 쓰면 안 된다.

현재 구현에서 graph는 **좌표 기반 고정 KNN graph**이며, feature similarity 기반 graph가 아니다.

---

### 5.2 Token feature와 token coordinate의 정의

Graph message passing에서 사용하는 변수는 다음 두 종류다.

#### token feature
각 token의 feature vector:

$$
T_i \in \mathbb{R}^{256}
$$

전체를 모으면

$$
T = \{T_i\}_{i=1}^{N}
$$

또는 batch tensor 표기로는

$$
T \in \mathbb{R}^{B \times N \times 256}
$$

이다.

#### token coordinate
각 token의 2D normalized coordinate:

$$
q_i \in [0,1]^2
$$

전체 좌표 집합은

$$
Q = \{q_i\}_{i=1}^{N}
$$

이고, batch tensor 표기로는

$$
Q \in \mathbb{R}^{B \times N \times 2}
$$

이다.

즉 message passing은 feature `T_i`와 좌표 `q_i`를 함께 사용한다.

---

### 5.3 Valid edge mask

모든 KNN edge를 다 쓰는 것이 아니라, tokenization 단계에서 만든 valid token mask를 이용해 **valid edge mask**를 만든다.

앞에서 token valid mask는

$$
m_i = \mathbf{1}(s_{\mathrm{tok},i} > 0.05)
$$

였다.

이제 edge `(i,j)`가 valid인지 여부를 다음처럼 정의한다.

$$
m_{ij}^{\mathrm{edge}} = m_i \land m_j
$$

즉 source token과 destination token이 모두 valid일 때만 edge를 살린다.

현재 코드의 `edge_valid_mask`는 바로 이

$$
m_{ij}^{\mathrm{edge}}
$$

에 해당한다.

문서에서 단순히 “valid edge mask”라고만 쓰면 안 되고,  
적어도 어떤 edge `(i,j)`의 validity를 나타내는 mask인지 명시해야 한다.

---

### 5.4 GraphMessagePassingLayer

현재 `GraphReasoner`는 기본적으로 `num_layers = 2`개의 `GraphMessagePassingLayer`를 쌓는다.

각 layer는 다음 두 단계로 동작한다.

#### (a) edge message 계산

각 valid edge `(i,j)`에 대해 source feature, destination feature, 좌표 차이를 concat한다.

$$
[T_i, T_j, q_j - q_i] \in \mathbb{R}^{514}
$$

현재 message MLP의 실제 구조는

$$
\mathrm{MLP}_{514 \to 256 \to 256}
$$

이다.

따라서 edge message는

$$
m_{i \to j}
=
\mathrm{MLP}_{514\to256\to256}
([T_i, T_j, q_j - q_i])
$$

이다.

#### (b) message aggregation

destination token `j`에 들어오는 valid message를 모두 합한다.

$$
a_j = \sum_{i:(i,j)\in E,\; m_{ij}^{\mathrm{edge}}=1} m_{i\to j}
$$

#### (c) token update

현재 update MLP의 구조는

$$
\mathrm{MLP}_{512 \to 256 \to 256}
$$

이다.

입력은 destination token feature와 aggregated message concat이다.

$$
[T_j, a_j] \in \mathbb{R}^{512}
$$

따라서 updated token은

$$
T'_j
=
\mathrm{LayerNorm}
\Big(
T_j + \mathrm{MLP}_{512\to256\to256}([T_j, a_j])
\Big)
$$

이다.

여기서 `LayerNorm`은 token channel dimension(256)에 대해 적용된다.

---

### 5.5 GraphReasoner 전체 출력

두 layer를 지난 뒤의 최종 token feature를

$$
T^{(L)} = \{T_i^{(L)}\}_{i=1}^{N}
$$

이라고 둔다.

이 최종 token feature가 다음 단계의
- closure head
- affinity head

입력이 된다.

---

## 6. Closure head, boundary refinement, affinity head

### 6.1 Closure head

`ClosureHead`의 실제 구성:
1. `Linear(256 → 256)`
2. `ReLU`
3. `Linear(256 → 1)`

따라서 token 단위 closure logits는

$$
z_{\mathrm{cls}}^{\mathrm{tok}}
=
\mathrm{Linear}_{256\to1}
(
\mathrm{ReLU}
(
\mathrm{Linear}_{256\to256}(T^{(L)})
)
)
$$

이다.

이를 `16×16`으로 reshape하면

$$
z_{\mathrm{cls}}^{\mathrm{map}} \in \mathbb{R}^{B \times 1 \times 16 \times 16}
$$

이고, 이를 Branch B feature 해상도로 업샘플한다.

$$
\tilde z_{\mathrm{cls}} = U(z_{\mathrm{cls}}^{\mathrm{map}} \to f_B)
$$

---

### 6.2 Boundary refinement

`boundary_refine`의 실제 구성:
1. `ConvBNReLU(257 → 256)`
2. `Conv2d(256 → 1, kernel=1)`

즉 `f_B`와 `\tilde z_{\mathrm{cls}}`를 concat해서 refined boundary logits를 만든다.

$$
z_{\mathrm{bdry}}
=
\mathrm{Conv2d}_{256\to1}
\Big(
\mathrm{ConvBNReLU}_{257\to256}
([f_B,\tilde z_{\mathrm{cls}}])
\Big)
$$

---

### 6.3 Affinity head

`AffinityHead`는 edge `(i,j)`마다 affinity score를 예측한다.

입력은 message calculation과 동일하게

$$
[T_i^{(L)}, T_j^{(L)}, q_j - q_i] \in \mathbb{R}^{514}
$$

이다.

현재 구조는

$$
\mathrm{MLP}_{514 \to 256 \to 1}
$$

이다.

따라서 affinity edge logit은

$$
z_{\mathrm{aff}}^{(i,j)}
=
\mathrm{MLP}_{514\to256\to1}
([T_i^{(L)}, T_j^{(L)}, q_j - q_i])
$$

이다.

즉 현재 코드에는 affinity head가 실제로 존재하고, edge-level logits를 낸다.  
다만 loss 연결 여부는 별도 실험/구현 상태에 따라 달라질 수 있다.

---

## 7. Fusion

Fusion은 `models/fusion/gated_fusion.py`다.

### 7.1 B projection

먼저 Branch B feature를 `1×1 ConvBNReLU`로 projection한다.

$$
\hat f_B = \mathrm{ConvBNReLU}_{256\to256,\,1\times1}(f_B)
$$

---

### 7.2 Gate 입력

gate는 다음을 concat해서 만든다.

- `f_A` : 256채널
- `\hat f_B` : 256채널
- `z_{\mathrm{bdry}}` : 1채널
- `\tilde z_{\mathrm{cls}}` : 1채널
- `u^*` : 1채널

총 515채널이다.

즉 입력은

$$
[f_A, \hat f_B, z_{\mathrm{bdry}}, \tilde z_{\mathrm{cls}}, u^*]
\in
\mathbb{R}^{B \times 515 \times H/4 \times W/4}
$$

---

### 7.3 Fusion gate `\gamma`

현재 gate network의 실제 구성은

1. `ConvBNReLU(515 → 256)`
2. `Conv2d(256 → 256, kernel=1)`
3. `Sigmoid`

이다.

따라서 fusion gate `\gamma`를 다음처럼 정의한다.

$$
\gamma
=
\sigma
\Big(
\mathrm{Conv2d}_{256\to256}
(
\mathrm{ConvBNReLU}_{515\to256}
([f_A,\hat f_B,z_{\mathrm{bdry}},\tilde z_{\mathrm{cls}},u^*])
)
\Big)
$$

shape는

$$
\gamma \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
$$

이다.

중요: `\gamma`는 1채널이 아니라 256채널 spatial gate다.

---

### 7.4 최종 fusion

최종 fused feature는

$$
f_{\mathrm{fused}} = f_A + \gamma \odot \hat f_B
$$

이다.

즉 A를 본체로 두고, B를 채널별/위치별로 주입하는 residual-style gated fusion 구조다.

---

## 8. Final decoder

`RefinementDecoder`의 실제 구성:
1. `ConvBNReLU(256 → 256)`
2. `ConvBNReLU(256 → 256)`
3. `Conv2d(256 → 1, kernel=1)`

즉

$$
z_{\mathrm{final}}
=
\mathrm{Conv2d}_{256\to1}
(
\mathrm{ConvBNReLU}_{256\to256}
(
\mathrm{ConvBNReLU}_{256\to256}(f_{\mathrm{fused}})
)
)
$$

최종 확률은

$$
\hat y = \sigma(z_{\mathrm{final}})
$$

이다.

---

## 9. GT boundary 생성: `y_{\mathrm{bdry}}`

현재 코드는 외부 `GT_Edge` 파일을 직접 쓰지 않고, object mask `y`에서 boundary target을 직접 생성한다.

여기서

$$
y \in \{0,1\}^{H \times W}
$$

는 binary object mask다.

- `y=1`: foreground object
- `y=0`: background

`dilate(y)`는 morphology dilation을 적용한 mask이고, object를 바깥쪽으로 확장한 결과다.

`erode(y)`는 morphology erosion을 적용한 mask이고, object를 안쪽으로 줄인 결과다.

따라서

$$
\mathrm{dilate}(y) - \mathrm{erode}(y)
$$

는 object 경계 주변에서만 양수가 되는 **boundary band**를 만든다.

현재 코드의 boundary target은 이를 indicator로 이진화한 값이다.

$$
y_{\mathrm{bdry}} = \mathbf{1}(\mathrm{dilate}(y) - \mathrm{erode}(y) > 0)
$$

여기서 `\mathbf{1}(\cdot)`는 indicator function이다.  
조건이 참이면 1, 거짓이면 0을 출력한다.

즉 `y_{\mathrm{bdry}}`는 1-pixel contour line 그 자체라기보다는,  
**경계 주변의 얇은 supervision band**라고 이해하는 것이 맞다.

---

## 10. Loss

현재 코드 기준 최종 loss는 다음 항들의 합이다.

- final segmentation loss
- auxiliary segmentation loss
- boundary loss
- optional affinity loss
- optional topology loss

---

### 10.1 Final segmentation loss

`losses/seg_loss.py`의 `BCEDiceLoss`라는 이름은 구현과 다르다.  
실제로는 **BCE + IoU surrogate**다.

즉

$$
L_{\mathrm{seg}}
=
\mathrm{BCEWithLogits}(z_{\mathrm{final}}, y)
+
\Big(1 - \frac{|\sigma(z_{\mathrm{final}}) \cap y| + 1}{|\sigma(z_{\mathrm{final}}) \cup y| + 1}\Big)
$$

이다.

---

### 10.2 Auxiliary segmentation loss

auxiliary supervision은 현재 코드에서 다음 출력들에 걸린다.

- `coarse_logits`
- `fine_logits`
- `objectness_logits`
- `boundary_prior_logits`

즉 문서에서는 “aux branch들에 BCE/IoU 계열 supervision”이라고 쓰는 게 맞다.

---

### 10.3 Boundary loss

boundary loss는 `boundary_logits`와 `y_{\mathrm{bdry}}` 사이의 binary loss다.  
구현상 candidate boundary에도 optional supervision이 들어갈 수 있다.

---

### 10.4 Affinity loss

full-option 실험에서는 edge-level affinity target이 정의되면,

$$
L_{\mathrm{aff}}
=
\mathrm{BCEWithLogits}(z_{\mathrm{aff}}, y_{\mathrm{aff}})
$$

형태로 연결할 수 있다.

현재 코드 버전에 따라 placeholder일 수도 있고, full-option 버전에서는 실제로 활성화될 수도 있다.

---

### 10.5 Topology loss

현재 topology loss라는 이름은 강한 persistent homology 기반 loss를 뜻하는 것은 아니다.  
구현 버전에 따라
- boundary continuity surrogate
- smoothness / connectivity surrogate
- clDice-style surrogate
중 하나로 들어갈 수 있다.

즉 문서에서는 “topology-aware regularizer”라고 쓰되,  
현재 구현이 어떤 surrogate인지 함께 써야 한다.

---

## 11. Branch A/B 전체 흐름 요약

전체 흐름은 다음과 같이 정리할 수 있다.

1. `ResNet50Backbone`이 `(c_2,c_3,c_4,c_5)`를 생성
2. `SimpleFPNNeck`가 `(p_2,p_3,p_4,p_5)`를 생성
3. Branch A가
   - `coarse_logits`
   - `fine_logits`
   - `objectness_map`
   - `uncertainty_map`
   - `boundary_prior`
   - `a_feats`
   를 생성
4. signal switch를 거쳐 `o^*,u^*,b^*` 생성
5. Branch B가
   - `f_0`
   - `r`
   - `f_B`
   - `c`
   - token set `T`
   - coordinate set `Q`
   - graph edge set `E`
   - `closure_logits`
   - `boundary_logits`
   를 생성
6. Fusion이 `f_A`, `f_B`, `boundary`, `closure`, `u^*`를 받아 gate `\gamma` 생성
7. `f_{\mathrm{fused}} = f_A + \gamma \odot \hat f_B`
8. `RefinementDecoder`가 최종 `z_{\mathrm{final}}` 생성

---

## 12. 문서에서 절대 빼먹지 말아야 하는 정의 요약

문서 처음 또는 각 절 시작에서 반드시 먼저 정의해야 하는 것:

- `z`: fine logits
- `p`: sigmoid probability
- `\epsilon`: numerical stability constant
- `\bar p`: clipped probability
- `u`: uncertainty map
- `o,u,b`: Branch A raw signals
- `o^*,u^*,b^*`: Branch A used signals
- `f_0`: Branch B base fused feature
- `r`: ROI mask
- `f_B`: ROI-enhanced Branch B feature
- `c`: boundary candidate probability map
- `T_i`: i번째 token feature
- `q_i`: i번째 token coordinate
- `Q = \{q_i\}`: coordinate set
- `E`: graph edge set
- `m_i`: token valid mask
- `m_{ij}^{\mathrm{edge}}`: valid edge mask
- `\gamma`: fusion gate

이 정의를 먼저 주지 않고 식에서 바로 쓰면 문서가 끊겨 보인다.

---

## 13. 현재 코드의 해석상 주의점

1. `BCEDiceLoss`라는 이름은 실제 구현을 완전히 설명하지 않는다. 구현은 BCE + IoU surrogate에 가깝다.
2. uncertainty는 별도 head가 아니라 fine logits entropy다.
3. objectness는 작은 head가 아니라 1×1 conv 단일층이다.
4. fusion gate `\gamma`는 1채널이 아니라 256채널 spatial gate다.
5. 시각화 heatmap은 보통 sample-wise normalization이기 때문에 절대값 비교가 아니라 샘플 내부 상대 분포를 보는 용도다.
6. edge/instance annotation path가 config에 있어도, 현재 버전의 기본 supervision은 mask 기반 내부 생성 boundary를 쓴다.

---

## 14. 가장 간단한 요약 문단

현재 구현은 `ResNet50Backbone + SimpleFPNNeck` 위에 `GlobalObjectnessBranch`와 `ContourClosureBranch`를 병렬로 두고, Branch A가 coarse/fine/objectness/boundary prior 및 uncertainty-derived signal을 생성한 뒤, Branch B가 low/high feature와 used signal을 입력으로 받아 ROI-gated boundary candidate, graph-based closure reasoning, refined boundary를 계산하도록 구성되어 있다. Branch B의 token은 `16×16` grid tokenization으로 생성되며, 각 token feature `T_i`와 token coordinate `q_i`를 이용해 좌표 기반 KNN graph를 만들고, `GraphMessagePassingLayer`를 통해 message passing을 수행한다. 이후 closure map과 refined boundary를 만든 뒤, `GatedFusion`이 Branch A feature와 Branch B feature를 256채널 spatial gate `\gamma`로 결합하고, `RefinementDecoder`가 최종 camouflage mask logits를 생성한다.

