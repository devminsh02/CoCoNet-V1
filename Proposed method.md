# Branch A / Branch B 동작 원리 정리

## 코드 기준 수식화 + 역할 + Tensor 흐름 + Ablation 의미

이 문서는 현재 `COCO_V1` 코드 구조를 기준으로, `Branch A`와 `Branch B`가 무엇을 입력받고, 어떤 중간 표현을 만들고, 어떻게 최종 출력으로 이어지는지를 **실제 모듈 이름과 실제 연산 순서** 기준으로 다시 정리한 것이다.

중요한 원칙은 다음과 같다.

- `\phi`, `\psi`, `h`, `g` 같은 추상 함수 기호로 뭉뚱그려 적지 않는다.
- 실제 코드에 존재하는 block 이름과 레이어 순서를 직접 적는다.
- Branch A의 `uncertainty`는 별도 head가 아니라 `fine_logits`의 sigmoid entropy에서 계산된다는 점을 명시한다.
- Branch A raw signal과 downstream에 실제로 전달되는 used signal을 분리해서 적는다.
- Branch B의 affinity head 존재 여부와 affinity loss 연결 여부를 따로 구분해 적는다.

---

# 0. 전체 큰 흐름

모델 전체를 가장 압축해서 쓰면 다음과 같다.

```math
x
\;\xrightarrow{\text{ResNet50Backbone + SimpleFPNNeck}}\;
(p_2,p_3,p_4,p_5)
\;\xrightarrow{\text{GlobalObjectnessBranch}}\;
(z_{\mathrm{coarse}}, z_{\mathrm{fine}}, z_{\mathrm{obj}}, z_{\mathrm{bp}}, o, u, b, f_A)
\;\xrightarrow{\text{signal switch}}\;
(o^\*,u^\*,b^\*)
\;\xrightarrow{\text{ContourClosureBranch}}\;
(r, z_{\mathrm{cand}}, z_{\mathrm{cls}}, z_{\mathrm{bdry}}, z_{\mathrm{aff}}, f_B)
\;\xrightarrow{\text{GatedFusion}}\;
f_{\mathrm{fused}}
\;\xrightarrow{\text{RefinementDecoder}}\;
z_{\mathrm{final}}
```

각 기호는 다음 의미를 가진다.

- `x`: 입력 이미지
- `p2, p3, p4, p5`: FPN multi-scale feature
- `z_coarse`: Branch A coarse segmentation logits
- `z_fine`: Branch A fine segmentation logits
- `z_obj`: objectness logits
- `z_bp`: boundary prior logits
- `o = sigmoid(z_obj)`: objectness map
- `u`: uncertainty map
- `b = sigmoid(z_bp)`: boundary prior map
- `o*, u*, b*`: Branch B/Fusion에 실제로 전달되는 used signal
- `r`: ROI mask
- `z_cand`: boundary candidate logits
- `z_cls`: closure logits / closure map
- `z_bdry`: Branch B refined boundary logits
- `z_aff`: edge-level affinity logits
- `f_A`: Branch A feature
- `f_B`: Branch B feature
- `f_fused`: fusion 결과 feature
- `z_final`: 최종 camouflage segmentation logits

---

# 1. 표기와 기본 연산

이 문서에서는 아래 표기를 쓴다.

- `σ(·)`: sigmoid
- `U(·)`: bilinear upsampling
- `[a,b,c]`: channel concat
- `⊙`: element-wise multiplication
- `A_G(·)`: adaptive average pooling to `G × G`
- 기본 token grid는 `G = 16`
- 따라서 token 수는 `N = G^2 = 256`

이 문서는 추상 함수 기호 대신 실제 코드 block 이름을 직접 사용한다.

## 1.1 `ConvBNReLU`

`utils/common.py`의 `ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=None)`는 실제로 아래 순서다.

1. `nn.Conv2d`
2. `nn.BatchNorm2d`
3. `nn.ReLU(inplace=True)`

수식으로 쓰면:

```math
\mathrm{ConvBNReLU}(x)
=
\mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}(x)))
```

padding을 따로 주지 않으면 `kernel_size // 2`가 들어간다.

## 1.2 `MLP`

`utils/common.py`의 `MLP(in_dim, hidden_dim, out_dim, dropout=0.0)`는 다음 순서다.

1. `Linear(in_dim → hidden_dim)`
2. `ReLU(inplace=True)`
3. `Dropout(dropout)`
4. `Linear(hidden_dim → out_dim)`

수식으로 쓰면:

```math
\mathrm{MLP}(x)
=
W_2(\mathrm{Dropout}(\mathrm{ReLU}(W_1 x)))
```

## 1.3 `sigmoid_entropy_from_logits`

현재 uncertainty map은 별도 head가 아니라 `utils/common.py`의 `sigmoid_entropy_from_logits(logits)`로 계산된다.

```math
p = \sigma(z)
```

```math
\bar p = \mathrm{clip}(p, \epsilon, 1-\epsilon)
```

```math
u = -\big(\bar p\log \bar p + (1-\bar p)\log(1-\bar p)\big)
```

즉 uncertainty는 **fine logits에서 유도된 entropy map**이다.

---

# 2. Backbone + FPN

입력 이미지를 다음처럼 둔다.

```math
x \in \mathbb{R}^{B \times 3 \times H \times W}
```

## 2.1 Backbone: `ResNet50Backbone`

현재 backbone은 `models/backbones/resnet50_backbone.py`의 `ResNet50Backbone`이다.

실제 구성은 다음과 같다.

- `stem = conv1 + bn1 + relu`
- `maxpool`
- `layer1`
- `layer2`
- `layer3`
- `layer4`

forward는 아래 출력을 만든다.

```math
(c_2, c_3, c_4, c_5) = \mathrm{ResNet50Backbone}(x)
```

shape는 다음처럼 본다.

- `c2 ∈ R^{B × 256 × H/4 × W/4}`
- `c3 ∈ R^{B × 512 × H/8 × W/8}`
- `c4 ∈ R^{B × 1024 × H/16 × W/16}`
- `c5 ∈ R^{B × 2048 × H/32 × W/32}`

## 2.2 Neck: `SimpleFPNNeck`

현재 neck은 `models/necks/simple_fpn_neck.py`의 `SimpleFPNNeck`이다.

실제 구성:

### lateral projection
- `lateral_c2 = Conv2d(256 → 256, kernel=1)`
- `lateral_c3 = Conv2d(512 → 256, kernel=1)`
- `lateral_c4 = Conv2d(1024 → 256, kernel=1)`
- `lateral_c5 = Conv2d(2048 → 256, kernel=1)`

### smoothing block
- `smooth_p2 = ConvBNReLU(256 → 256, kernel=3)`
- `smooth_p3 = ConvBNReLU(256 → 256, kernel=3)`
- `smooth_p4 = ConvBNReLU(256 → 256, kernel=3)`
- `smooth_p5 = ConvBNReLU(256 → 256, kernel=3)`

실제 계산은 다음과 같다.

```math
p_5 = \mathrm{smooth\_p5}(\mathrm{lateral\_c5}(c_5))
```

```math
p_4 = \mathrm{smooth\_p4}(\mathrm{lateral\_c4}(c_4) + U(p_5 \to c_4))
```

```math
p_3 = \mathrm{smooth\_p3}(\mathrm{lateral\_c3}(c_3) + U(p_4 \to c_3))
```

```math
p_2 = \mathrm{smooth\_p2}(\mathrm{lateral\_c2}(c_2) + U(p_3 \to c_2))
```

최종 출력은 `(p2,p3,p4,p5)`이고, 네 level 모두 256채널이다.

---

# 3. Branch A: Global / Objectness Branch

Branch A는 `models/branches/branch_a/global_objectness_branch.py`의 `GlobalObjectnessBranch`다.

Branch A의 역할은 다음과 같다.

1. coarse segmentation logits 생성
2. fine segmentation logits 생성
3. objectness logits / map 생성
4. boundary prior logits / map 생성
5. fine logits에서 uncertainty map 생성
6. Branch B와 Fusion이 사용할 semantic prior 제공

## 3.1 Branch A의 입력

입력은 FPN 출력 전체다.

```math
(p_2,p_3,p_4,p_5)
```

## 3.2 Branch A의 핵심 feature `f_A`

먼저 `p3, p4, p5`를 모두 `p2` 해상도로 올린다.

```math
\tilde p_3 = U(p_3 \to p_2), \qquad
\tilde p_4 = U(p_4 \to p_2), \qquad
\tilde p_5 = U(p_5 \to p_2)
```

그 다음 concat 후 `pyramid_fuse`를 통과시킨다.

`pyramid_fuse`의 실제 구성은 다음과 같다.

1. `ConvBNReLU(1024 → 256)`
2. `ConvBNReLU(256 → 256)`

즉:

```math
f_A = \mathrm{ConvBNReLU}_{256\to256}(\mathrm{ConvBNReLU}_{1024\to256}([p_2,\tilde p_3,\tilde p_4,\tilde p_5]))
```

shape:

```math
f_A \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
```

## 3.3 Coarse Head

Coarse path는 `p4` 기반이다.

먼저 `coarse_adapter = ConvBNReLU(256 → 256)`:

```math
f_{\mathrm{coarse}} = \mathrm{ConvBNReLU}_{256\to256}(p_4)
```

그 다음 `CoarseHead` 적용:

- `ConvBNReLU(256 → 256)`
- `Conv2d(256 → 1, kernel=1)`

즉:

```math
z_{\mathrm{coarse}}^{(low)} = \mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(f_{\mathrm{coarse}}))
```

마지막으로 `p2` 해상도로 올린다.

```math
z_{\mathrm{coarse}} = U(z_{\mathrm{coarse}}^{(low)} \to p_2)
```

## 3.4 Fine Head

Fine path는 `f_A`에서 바로 계산한다.

`FineHead` 실제 구성:

1. `ConvBNReLU(256 → 256)`
2. `ConvBNReLU(256 → 256)`
3. `Conv2d(256 → 1, kernel=1)`

즉:

```math
z_{\mathrm{fine}} = \mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(\mathrm{ConvBNReLU}_{256\to256}(f_A)))
```

## 3.5 Objectness Head

Objectness는 별도 작은 CNN block 없이 `f_A`에 바로 `1×1 conv`를 적용한다.

```math
z_{\mathrm{obj}} = \mathrm{Conv2d}_{256\to1}(f_A)
```

```math
o = \sigma(z_{\mathrm{obj}})
```

## 3.6 Boundary Prior Head

`BoundaryPriorHead` 실제 구성:

1. `ConvBNReLU(256 → 256)`
2. `Conv2d(256 → 1, kernel=1)`

즉:

```math
z_{\mathrm{bp}} = \mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(f_A))
```

```math
b = \sigma(z_{\mathrm{bp}})
```

## 3.7 Uncertainty는 별도 head가 아니다

현재 uncertainty를 위한 `Conv2d` head는 없다.

```math
u = \mathrm{sigmoid\_entropy\_from\_logits}(z_{\mathrm{fine}})
```

즉 uncertainty는 **fine logits에서 파생**된다.

## 3.8 Branch A 전체 출력

실제 코드 관점에서 Branch A는 다음을 만든다.

```math
(z_{\mathrm{coarse}}, z_{\mathrm{fine}}, z_{\mathrm{obj}}, z_{\mathrm{bp}}, o, u, b, f_A, f_{\mathrm{coarse}})
```

실제 key 이름 기준으로는 다음이 중요하다.

- `coarse_logits`
- `fine_logits`
- `objectness_logits`
- `boundary_prior_logits`
- `objectness_map`
- `uncertainty_map`
- `boundary_prior`
- `a_feats`
- `coarse_feat`

---

# 4. Branch A의 Supervision과 Loss

## 4.1 기본 segmentation loss 형태

현재 코드에서 `BCEDiceLoss`라는 이름은 구현과 다르다. 실제 계산은 **BCE + IoU surrogate**다.

즉 final segmentation loss는 개념적으로 다음과 같다.

```math
\mathcal{L}_{\mathrm{seg}}(z,y)
=
\mathrm{BCEWithLogits}(z,y)
+
\mathcal{L}_{\mathrm{IoU}}(\sigma(z),y)
```

## 4.2 Branch A auxiliary loss

Branch A auxiliary loss는 다음 항을 사용한다.

```math
\mathcal{L}_A
=
\lambda_c \mathcal{L}_{\mathrm{seg}}(z_{\mathrm{coarse}}, y)
+
\lambda_f \mathcal{L}_{\mathrm{seg}}(z_{\mathrm{fine}}, y)
+
\lambda_o \mathcal{L}_{\mathrm{seg}}(z_{\mathrm{obj}}, y)
+
\lambda_b \mathcal{L}_{\mathrm{seg}}(z_{\mathrm{bp}}, y_{\mathrm{bdry}})
```

즉,

- coarse: mask GT supervision
- fine: mask GT supervision
- objectness: mask GT supervision
- boundary prior: boundary GT supervision

중요한 점은 `uncertainty_map`에는 직접 supervision이 없다는 것이다.

## 4.3 Boundary target 생성

현재 프로젝트는 외부 edge GT를 기본 supervision으로 사용하지 않고, mask에서 boundary를 내부 생성한다.

개념적으로는 다음과 같다.

```math
y_{\mathrm{bdry}} = \mathbf{1}(\mathrm{dilate}(y) - \mathrm{erode}(y) > 0)
```

즉 morphology-based boundary band를 target으로 사용한다.

---

# 5. Raw Signal vs Used Signal

이 구조의 중요한 실험 축은 Branch A signal ablation이다.

Branch A가 만든 raw signal을 다음처럼 둔다.

```math
(o,u,b)
```

하지만 Branch B와 Fusion에 실제로 들어가는 것은 raw가 아니라 **used signal**이다.

```math
o^\* = \alpha_o o, \qquad
u^\* = \alpha_u u, \qquad
b^\* = \alpha_b b
```

여기서

```math
\alpha_o, \alpha_u, \alpha_b \in \{0,1\}
```

이다.

즉,

- objectness ablation: `α_o = 0`
- uncertainty ablation: `α_u = 0`
- boundary prior ablation: `α_b = 0`

중요한 점:

- raw signal `o,u,b`는 계속 계산된다.
- Branch A head도 계속 학습된다.
- 단지 Branch B/Fusion이 실제로 쓰는 used signal만 0이 된다.

이 구조는 다음 두 질문을 분리해서 볼 수 있게 해 준다.

1. 그 head가 의미 있는 map을 만들어내는가?
2. 그 map을 downstream이 실제로 사용했을 때 성능 기여가 있는가?

---

# 6. Branch B: Contour Closure Branch

Branch B는 `models/branches/branch_b/contour_closure_branch.py`의 `ContourClosureBranch`다.

이 브랜치의 역할은 semantic segmentation을 한 번 더 하는 것이 아니라,

- Branch A가 알려준 중요 위치에서
- 경계 fragment를 더 뚜렷하게 만들고
- token graph reasoning을 통해 closure cue를 만들고
- refined boundary를 생성한 뒤
- 이를 Fusion을 통해 최종 segmentation에 반영하는 것

이다.

## 6.1 Branch B의 입력

### Feature 입력

- low-level feature: `p2`
- high-level feature: `p4`

### Signal 입력

- `used_objectness_map = o*`
- `used_uncertainty_map = u*`
- `used_boundary_prior = b*`

즉 Branch B는 실제로 다음을 입력으로 받는다.

```math
(p_2, p_4, o^\*, u^\*, b^\*)
```

## 6.2 Base Feature 생성

먼저 `p4`를 `p2` 해상도로 올린다.

```math
\tilde p_4 = U(p_4 \to p_2)
```

그 다음 concat 후 `feature_fuse`를 통과시킨다.

`feature_fuse`의 실제 구성:

1. `ConvBNReLU(512 → 256)`
2. `ConvBNReLU(256 → 256)`

즉:

```math
f_0 = \mathrm{ConvBNReLU}_{256\to256}(\mathrm{ConvBNReLU}_{512\to256}([p_2, \tilde p_4]))
```

## 6.3 ROI Gating

ROI gating은 `SoftROIGating`이 담당한다.

실제 구성:

1. `ConvBNReLU(3 → 64)`
2. `ConvBNReLU(64 → 64)`
3. `Conv2d(64 → 1, kernel=1)`
4. `Sigmoid`

입력은 `[o*,u*,b*]`다.

즉:

```math
r = \sigma(\mathrm{Conv2d}_{64\to1}(\mathrm{ConvBNReLU}_{64\to64}(\mathrm{ConvBNReLU}_{3\to64}([o^\*,u^\*,b^\*]))))
```

## 6.4 ROI로 Feature 강조

Branch B feature는 ROI mask로 강조된다.

```math
f_B = f_0 \odot (1 + r)
```

즉 ROI가 큰 위치는 feature magnitude가 더 커진다.

## 6.5 Boundary Candidate Head

`BoundaryCandidateHead` 실제 구성:

1. `ConvBNReLU(256 → 256)`
2. `Conv2d(256 → 1, kernel=1)`

즉:

```math
z_{\mathrm{cand}} = \mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(f_B))
```

```math
c = \sigma(z_{\mathrm{cand}})
```

## 6.6 Fragment Tokenization

`FragmentTokenizer`는 다음 세 가지를 `16×16`으로 adaptive average pooling 한다.

- `f_B`
- `c`
- `r`

즉:

```math
\bar f = A_{16}(f_B), \qquad
\bar c = A_{16}(c), \qquad
\bar r = A_{16}(r)
```

이제 token score를 만든다.

```math
s = \bar c \odot \bar r
```

그 다음 feature token을 flatten해서 만든다.

```math
T = \mathrm{Flatten}(\bar f) \in \mathbb{R}^{B \times N \times C}
```

여기서

- `N = 256`
- `C = 256`

이므로

```math
T \in \mathbb{R}^{B \times 256 \times 256}
```

## 6.7 Valid Token Mask

모든 token을 reasoning에 쓰지 않는다.

```math
m_i = \mathbf{1}(s_i > \tau)
```

기본 threshold는

```math
\tau = 0.05
```

이다.

즉 Branch B는 **ROI 안에 있고, boundary candidate도 충분히 높은 token만** reasoning에 사용한다.

## 6.8 Token 좌표

각 token에는 정규화된 regular grid 좌표가 붙는다.

```math
q_i \in [0,1]^2
```

이 좌표는 learned coordinate가 아니라 regular `16×16` grid에서 생성된다.

## 6.9 Graph Construction

Branch B는 token 좌표 기반 KNN graph를 만든다.

```math
E = \mathrm{KNN}(q, k)
```

기본 이웃 수는

```math
k = 8
```

이다.

중요한 점은 이 graph가 **feature similarity 기반 graph가 아니라 spatial KNN graph**라는 것이다.

## 6.10 Graph Message Passing

각 edge `(i,j)`에 대해 message를 만든다.

message MLP 입력은 다음 세 가지다.

- `T_i` : 256차원
- `T_j` : 256차원
- `q_j - q_i` : 2차원

즉 입력 차원은 총 514다.

실제 message block은

```math
m_{i\to j} = \mathrm{MLP}_{514\to256\to256}([T_i, T_j, q_j - q_i])
```

그 다음 destination 쪽으로 들어오는 message를 합친다.

```math
a_j = \sum_{i \in \mathcal{N}(j)} m_{i\to j}
```

update block은

```math
T'_j = \mathrm{LayerNorm}(T_j + \mathrm{MLP}_{512\to256\to256}([T_j, a_j]))
```

이다.

보통 layer 수는 2다.

## 6.11 Valid Edge Mask

모든 edge를 다 쓰지는 않는다.

```math
e_{ij}^{\mathrm{valid}} = m_i \land m_j
```

즉 valid token끼리 이어진 edge만 실제 reasoning에 사용된다.

## 6.12 Closure Prediction

Graph reasoning이 끝난 token에서 closure logits를 만든다.

`ClosureHead` 실제 구성:

1. `Linear(256 → 256)`
2. `ReLU`
3. `Linear(256 → 1)`

즉:

```math
z_{\mathrm{cls}}^{\mathrm{tok}} = \mathrm{Linear}_{256\to1}(\mathrm{ReLU}(\mathrm{Linear}_{256\to256}(T^{(L)})))
```

이를 다시 `16×16` map으로 reshape한다.

```math
z_{\mathrm{cls}}^{\mathrm{map}} = \mathrm{Reshape}(z_{\mathrm{cls}}^{\mathrm{tok}}) \in \mathbb{R}^{B \times 1 \times 16 \times 16}
```

그 다음 feature resolution로 업샘플한다.

```math
\tilde z_{\mathrm{cls}} = U(z_{\mathrm{cls}}^{\mathrm{map}} \to f_B)
```

이 closure map은 끊긴 fragment들을 더 닫힌 contour처럼 정리하도록 돕는 learned structural cue다.

## 6.13 Boundary Refinement

최종 boundary logits는 `f_B`와 closure map을 concat해서 만든다.

`boundary_refine` 실제 구성:

1. `ConvBNReLU(257 → 256)`
2. `Conv2d(256 → 1, kernel=1)`

즉:

```math
z_{\mathrm{bdry}} = \mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{257\to256}([f_B, \tilde z_{\mathrm{cls}}]))
```

이는 단순 boundary candidate가 아니라,

- ROI 정보
- feature 강조
- boundary candidate
- token graph reasoning
- closure map

을 모두 반영한 refined boundary다.

## 6.14 Affinity Head

Branch B에는 affinity head도 존재한다.

입력은 edge마다 다음 세 가지다.

- `T_i`
- `T_j`
- `q_j - q_i`

즉:

```math
z_{\mathrm{aff}}^{(i,j)} = \mathrm{MLP}_{514\to256\to1}([T_i, T_j, q_j - q_i])
```

현재 v1 문맥에서는 affinity 구조는 존재하지만, 버전에 따라 loss 연결 여부는 별도 확인이 필요하다.

## 6.15 Branch B 전체를 하나의 함수로 쓰면

```math
\mathrm{BranchB}(p_2,p_4,o^\*,u^\*,b^\*)
\mapsto
(r, z_{\mathrm{cand}}, z_{\mathrm{cls}}, z_{\mathrm{bdry}}, z_{\mathrm{aff}}, f_B)
```

Branch B를 한 문장으로 요약하면 다음과 같다.

> Branch A가 알려준 중요 위치 안에서 경계 fragment를 뽑고, token graph로 그 관계를 reasoning하여, 더 닫힌 contour에 가까운 refined boundary를 만드는 브랜치다.

---

# 7. Fusion: Branch A와 Branch B를 어떻게 합치는가

이 모델은 Branch A와 Branch B를 단순 평균하거나 concat만 하는 구조가 아니다. A를 본체로 두고 B를 gate로 얹는 구조다.

## 7.1 B Feature Projection

먼저 Branch B feature를 projection한다.

`GatedFusion` 안의 `b_proj`는 `ConvBNReLU(256 → 256, kernel=1, padding=0)`이다.

```math
\hat f_B = \mathrm{ConvBNReLU}_{256\to256,1\times1}(f_B)
```

## 7.2 Fusion Gate 계산

Fusion gate는 다음 입력을 본다.

- `f_A` : 256채널
- `\hat f_B` : 256채널
- `z_bdry` upsample : 1채널
- `\tilde z_cls` upsample : 1채널
- `u*` upsample : 1채널

총 515채널이다.

게이트 네트워크 실제 구성:

1. `ConvBNReLU(515 → 256)`
2. `Conv2d(256 → 256, kernel=1)`
3. `Sigmoid`

즉:

```math
\gamma = \sigma(\mathrm{Conv2d}_{256\to256}(\mathrm{ConvBNReLU}_{515\to256}([f_A,\hat f_B,U(z_{\mathrm{bdry}}),U(\tilde z_{\mathrm{cls}}),U(u^\*)])))
```

중요한 점: `γ`는 1채널이 아니라 **256채널 spatial gate**다.

```math
\gamma \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
```

## 7.3 최종 Fusion

```math
f_{\mathrm{fused}} = f_A + \gamma \odot \hat f_B
```

즉,

- A와 B를 동등하게 평균내는 구조가 아니고
- B가 A를 덮어쓰는 구조도 아니며
- **A를 본체 semantic body로 두고**
- **B를 correction stream처럼 필요한 만큼만 주입하는 구조**

다.

## 7.4 A-only baseline일 때

Branch B를 끄면 `NullBranchB + IdentityFusion` 경로를 탄다.

그 경우:

```math
f_{\mathrm{fused}} = f_A
```

즉 A-only baseline은

- Branch A feature만 사용
- Branch B 기여 0
- Fusion도 identity

구조다.

---

# 8. Final Decoder

최종 decoder는 fusion feature를 받아 1채널 최종 mask logits를 낸다.

`RefinementDecoder` 실제 구성:

1. `ConvBNReLU(256 → 256)`
2. `ConvBNReLU(256 → 256)`
3. `Conv2d(256 → 1, kernel=1)`

즉:

```math
z_{\mathrm{final}} = \mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(\mathrm{ConvBNReLU}_{256\to256}(f_{\mathrm{fused}})))
```

최종 segmentation probability는

```math
\hat y = \sigma(z_{\mathrm{final}})
```

이다.

평가/저장은 입력 해상도로 다시 업샘플된 결과를 사용한다.

---

# 9. 전체 Loss 구조

전체 loss는 개념적으로 다음처럼 쓸 수 있다.

```math
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{final}}
+
\mathcal{L}_{A}
+
\mathcal{L}_{\mathrm{bdry}}
+
\mathcal{L}_{\mathrm{cand}}
+
\mathcal{L}_{\mathrm{topo}}
+
\mathcal{L}_{\mathrm{aff}}
```

하지만 실제 활성 loss 항은 config와 코드 버전에 따라 달라진다.

## 9.1 Final Segmentation Loss

```math
\mathcal{L}_{\mathrm{final}} = \mathcal{L}_{\mathrm{seg}}(z_{\mathrm{final}}, y)
```

## 9.2 Branch A Auxiliary Loss

```math
\mathcal{L}_A
=
\lambda_c \mathcal{L}_{\mathrm{seg}}(z_{\mathrm{coarse}}, y)
+
\lambda_f \mathcal{L}_{\mathrm{seg}}(z_{\mathrm{fine}}, y)
+
\lambda_o \mathcal{L}_{\mathrm{seg}}(z_{\mathrm{obj}}, y)
+
\lambda_b \mathcal{L}_{\mathrm{seg}}(z_{\mathrm{bp}}, y_{\mathrm{bdry}})
```

## 9.3 Boundary Loss

```math
\mathcal{L}_{\mathrm{bdry}} = \lambda_{\mathrm{bdry}} \cdot \mathrm{BCEWithLogits}(z_{\mathrm{bdry}}, y_{\mathrm{bdry}})
```

필요하면 candidate에도 loss를 건다.

```math
\mathcal{L}_{\mathrm{cand}} = \lambda_{\mathrm{cand}} \cdot \mathrm{BCEWithLogits}(z_{\mathrm{cand}}, y_{\mathrm{bdry}})
```

## 9.4 Topology Loss

이름은 topology지만, v1 코드에서는 강한 topological optimization이라기보다 smoothness/continuity surrogate에 가깝게 구현되는 경우가 많다.

대표적인 형태는 개념적으로 다음과 유사하다.

```math
\mathcal{L}_{\mathrm{topo}}
\approx
\frac{1}{2}\left(
\|z_{\mathrm{bdry}}[:,:,1:,:]-z_{\mathrm{bdry}}[:,:,:-1,:]\|_1
+
\|z_{\mathrm{bdry}}[:,:,:,1:]-z_{\mathrm{bdry}}[:,:,:,:-1]\|_1
\right)
```

## 9.5 Affinity Loss

Affinity 구조는 edge-level logits를 낼 수 있지만, loss 연결 여부는 버전별로 다르다.

개념적으로는 다음과 같은 BCE-with-logits 형태다.

```math
\mathcal{L}_{\mathrm{aff}} = \lambda_{\mathrm{aff}} \cdot \mathrm{BCEWithLogits}(z_{\mathrm{aff}}, y_{\mathrm{aff}})
```

단, `y_aff`를 어떻게 만드는지는 구현 버전에 따라 달라질 수 있다.

---

# 10. Tensor 흐름으로 다시 정리

아래는 forward를 거의 코드 순서대로 다시 쓴 것이다.

## 10.1 입력

```math
x \in \mathbb{R}^{B \times 3 \times H \times W}
```

## 10.2 Backbone / FPN

```math
(c_2,c_3,c_4,c_5)=\mathrm{ResNet50Backbone}(x)
```

```math
(p_2,p_3,p_4,p_5)=\mathrm{SimpleFPNNeck}(c_2,c_3,c_4,c_5)
```

## 10.3 Branch A

```math
f_A = \mathrm{ConvBNReLU}_{256\to256}(\mathrm{ConvBNReLU}_{1024\to256}([p_2,U(p_3),U(p_4),U(p_5)]))
```

```math
f_{\mathrm{coarse}} = \mathrm{ConvBNReLU}_{256\to256}(p_4)
```

```math
z_{\mathrm{coarse}} = U(\mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(f_{\mathrm{coarse}})))
```

```math
z_{\mathrm{fine}} = \mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(\mathrm{ConvBNReLU}_{256\to256}(f_A)))
```

```math
z_{\mathrm{obj}} = \mathrm{Conv2d}_{256\to1}(f_A),\qquad o=\sigma(z_{\mathrm{obj}})
```

```math
z_{\mathrm{bp}} = \mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(f_A)),\qquad b=\sigma(z_{\mathrm{bp}})
```

```math
u = \mathrm{sigmoid\_entropy\_from\_logits}(z_{\mathrm{fine}})
```

## 10.4 Ablation Switch

```math
o^\* = \alpha_o o,\qquad u^\* = \alpha_u u,\qquad b^\* = \alpha_b b
```

## 10.5 Branch B Base Feature

```math
f_0 = \mathrm{ConvBNReLU}_{256\to256}(\mathrm{ConvBNReLU}_{512\to256}([p_2,U(p_4)]))
```

## 10.6 ROI Gating

```math
r = \sigma(\mathrm{Conv2d}_{64\to1}(\mathrm{ConvBNReLU}_{64\to64}(\mathrm{ConvBNReLU}_{3\to64}([o^\*,u^\*,b^\*]))))
```

## 10.7 ROI-based Feature Emphasis

```math
f_B = f_0 \odot (1+r)
```

## 10.8 Boundary Candidate

```math
z_{\mathrm{cand}} = \mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(f_B)),\qquad c=\sigma(z_{\mathrm{cand}})
```

## 10.9 Tokenization

```math
\bar f=A_{16}(f_B),\qquad \bar c=A_{16}(c),\qquad \bar r=A_{16}(r)
```

```math
s=\bar c\odot\bar r
```

```math
T=\mathrm{Flatten}(\bar f)
```

```math
m_i = \mathbf{1}(s_i > 0.05)
```

## 10.10 Graph Reasoning

```math
E=\mathrm{KNN}(q,8)
```

```math
m_{i\to j}=\mathrm{MLP}_{514\to256\to256}([T_i,T_j,q_j-q_i])
```

```math
a_j=\sum_{i\in\mathcal{N}(j)} m_{i\to j}
```

```math
T'_j=\mathrm{LayerNorm}(T_j+\mathrm{MLP}_{512\to256\to256}([T_j,a_j]))
```

## 10.11 Closure

```math
z_{\mathrm{cls}}^{\mathrm{tok}}=\mathrm{Linear}_{256\to1}(\mathrm{ReLU}(\mathrm{Linear}_{256\to256}(T^{(L)})))
```

```math
z_{\mathrm{cls}}^{\mathrm{map}}=\mathrm{Reshape}(z_{\mathrm{cls}}^{\mathrm{tok}})
```

```math
\tilde z_{\mathrm{cls}}=U(z_{\mathrm{cls}}^{\mathrm{map}}\to f_B)
```

## 10.12 Refined Boundary

```math
z_{\mathrm{bdry}}=\mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{257\to256}([f_B,\tilde z_{\mathrm{cls}}]))
```

## 10.13 Fusion

```math
\hat f_B=\mathrm{ConvBNReLU}_{256\to256,1\times1}(f_B)
```

```math
\gamma = \sigma(\mathrm{Conv2d}_{256\to256}(\mathrm{ConvBNReLU}_{515\to256}([f_A,\hat f_B,U(z_{\mathrm{bdry}}),U(\tilde z_{\mathrm{cls}}),U(u^\*)])))
```

```math
f_{\mathrm{fused}}=f_A+\gamma\odot\hat f_B
```

## 10.14 Final Prediction

```math
z_{\mathrm{final}}=\mathrm{Conv2d}_{256\to1}(\mathrm{ConvBNReLU}_{256\to256}(\mathrm{ConvBNReLU}_{256\to256}(f_{\mathrm{fused}})))
```

```math
\hat y=\sigma(z_{\mathrm{final}})
```

---

# 11. Branch A와 Branch B의 역할 차이

## 11.1 Branch A의 역할

Branch A는 다음 질문에 답하려는 브랜치다.

- 어디가 물체일 가능성이 큰가?
- 어디가 foreground-like한가?
- 어디가 경계처럼 보이는가?
- 어디서 segmentation이 애매한가?

즉 Branch A는 **전역 semantic prior 생성기**다.

```math
(p_2,p_3,p_4,p_5) \mapsto (z_{\mathrm{coarse}},z_{\mathrm{fine}},o,u,b,f_A)
```

## 11.2 Branch B의 역할

Branch B는 다음 질문에 답하려는 브랜치다.

- A가 중요하다고 한 위치 안에서 어떤 경계 fragment가 실제 contour를 이룰까?
- 끊긴 boundary 조각들을 어떻게 연결해야 할까?
- 닫힌 윤곽으로 정리할 수 있을까?

즉 Branch B는 **ROI-guided contour refinement / closure reasoning branch**다.

```math
(p_2,p_4,o^\*,u^\*,b^\*) \mapsto (r,z_{\mathrm{cand}},z_{\mathrm{cls}},z_{\mathrm{bdry}},z_{\mathrm{aff}},f_B)
```

## 11.3 한 문장 요약

- **Branch A**는 “어디를 봐야 하는지”를 알려주고
- **Branch B**는 “그 경계를 어떻게 닫고 정리해야 하는지”를 해결한다.

---

# 12. 각 Ablation이 정확히 의미하는 것

## 12.1 A-only Baseline

Branch B 자체를 끈 경우:

```math
f_{\mathrm{fused}} = f_A
```

즉,

- ROI 없음
- boundary candidate 없음
- token graph 없음
- closure 없음
- refined boundary 없음
- affinity 없음

오직 Branch A + decoder만 남는다.

## 12.2 Objectness Off

```math
o^\* = 0,\qquad u^\* = u,\qquad b^\* = b
```

의미:

- objectness head는 계속 예측된다.
- 하지만 Branch B/Fusion은 objectness를 실제 입력으로 안 쓴다.

## 12.3 Uncertainty Off

```math
o^\* = o,\qquad u^\* = 0,\qquad b^\* = b
```

의미:

- fine logits에서 uncertainty는 계속 계산된다.
- 하지만 downstream은 그 uncertainty를 사용하지 않는다.

## 12.4 Boundary Prior Off

```math
o^\* = o,\qquad u^\* = u,\qquad b^\* = 0
```

의미:

- boundary prior head는 살아 있다.
- 하지만 Branch B/Fusion은 그 prior를 사용하지 않는다.

---

# 13. 최종 핵심 요약

이 모델의 핵심은 다음 한 문장으로 정리할 수 있다.

> **Branch A가 전역 semantic prior(objectness, uncertainty, boundary prior, coarse/fine segmentation)를 만들고, Branch B가 그 prior를 바탕으로 ROI를 정하고 boundary fragment를 token graph로 reasoning하여 closure-aware boundary를 만든 뒤, Fusion이 A를 본체로 두고 B를 선택적으로 주입하여 최종 segmentation을 완성한다.**

수식적으로 다시 압축하면:

```math
x
\rightarrow
(c_2,c_3,c_4,c_5)
\rightarrow
(p_2,p_3,p_4,p_5)
\rightarrow
(z_{\mathrm{coarse}},z_{\mathrm{fine}},o,u,b,f_A)
\rightarrow
(o^\*,u^\*,b^\*)
\rightarrow
(r,z_{\mathrm{cand}},T,E,z_{\mathrm{cls}},z_{\mathrm{bdry}},z_{\mathrm{aff}},f_B)
\rightarrow
f_{\mathrm{fused}}
\rightarrow
z_{\mathrm{final}}
```

그리고 역할 분담은 다음처럼 정리된다.

- **Branch A**: semantic / objectness / uncertainty / boundary prior 생성
- **Branch B**: ROI-guided boundary fragment reasoning + closure + refined boundary
- **Fusion**: A를 본체로, B를 보정항으로 사용
- **Decoder**: 최종 mask logits 생성

---

# 14. 아주 짧은 초압축 버전

## Branch A

```text
coarse / fine segmentation, objectness, uncertainty, boundary prior를 만든다.
```

## Branch B

```text
A가 준 signal을 바탕으로 ROI를 정하고, boundary fragment를 token graph로 reasoning하여 closure-aware refined boundary를 만든다.
```

## Fusion

```text
A feature를 본체로 두고, B feature를 256채널 spatial gate로 선택적으로 주입한다.
```

## Final

```text
fused feature를 decoder에 넣어 최종 camouflage mask logits를 만든다.
```
