# Branch A / Branch B 동작 원리 정리

## 코드 기준 수식화 + 역할 + Tensor 흐름 + Ablation 의미

이 문서는 현재 업로드된 `COCO_V1` 코드 기준으로, `Branch A`와 `Branch B`가  
무엇을 입력받고, 어떤 중간 표현을 만들고, 어떻게 최종 출력으로 이어지는지를  
**GitHub Markdown 수식 문법** 기준으로 다시 정리한 것이다.

---

# 0. 전체 큰 흐름

모델 전체를 가장 압축해서 쓰면 다음과 같다.

```math
x
\;\xrightarrow{\text{Backbone+FPN}}\;
(p_2,p_3,p_4,p_5)
\;\xrightarrow{\text{Branch A}}\;
(z_{\mathrm{coarse}}, z_{\mathrm{fine}}, o, u, b, f_A)
\;\xrightarrow{\text{Branch B}}\;
(r, z_{\mathrm{cand}}, z_{\mathrm{cls}}, z_{\mathrm{bdry}}, f_B)
\;\xrightarrow{\text{Fusion}}\;
f_{\mathrm{fused}}
\;\xrightarrow{\text{Decoder}}\;
z_{\mathrm{final}}
```

각 기호는 다음 의미를 가진다.

- $x$: 입력 이미지
- $p_2, p_3, p_4, p_5$: FPN multi-scale feature
- $z_{\mathrm{coarse}}$: Branch A coarse segmentation logits
- $z_{\mathrm{fine}}$: Branch A fine segmentation logits
- $o$: objectness map
- $u$: uncertainty map
- $b$: boundary prior
- $f_A$: Branch A feature
- $r$: ROI mask
- $z_{\mathrm{cand}}$: boundary candidate logits
- $z_{\mathrm{cls}}$: closure logits / closure map
- $z_{\mathrm{bdry}}$: Branch B refined boundary logits
- $f_B$: Branch B feature
- $f_{\mathrm{fused}}$: fusion 결과 feature
- $z_{\mathrm{final}}$: 최종 camouflage segmentation logits

---

# 1. 표기와 기본 연산

이 문서에서는 아래 표기를 쓴다.

- $\sigma(\cdot)$: sigmoid
- $U(\cdot)$: bilinear upsampling
- $[\cdot]$: channel concat
- $\odot$: element-wise multiplication
- $A_G(\cdot)$: adaptive average pooling to $G \times G$
- 기본 token grid는 $G=16$
- 따라서 token 수는 $N = G^2 = 256$

실제 코드에는 `ConvBNReLU`, 작은 CNN block, MLP 등이 쓰이지만,  
설명에서는 이를 $\phi, \psi, h, g$ 같은 함수 기호로 묶어서 적는다.

---

# 2. Backbone + FPN

입력 이미지를 다음처럼 둔다.

```math
x \in \mathbb{R}^{B \times 3 \times H \times W}
```

Backbone은 ResNet50 계열이고, encoder feature를 다음처럼 만든다.

```math
(c_2, c_3, c_4, c_5) = \mathrm{Backbone}(x)
```

대략적인 shape는 다음처럼 이해하면 된다.

- $c_2 \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}$
- $c_3 \in \mathbb{R}^{B \times 512 \times H/8 \times W/8}$
- $c_4 \in \mathbb{R}^{B \times 1024 \times H/16 \times W/16}$
- $c_5 \in \mathbb{R}^{B \times 2048 \times H/32 \times W/32}$

그 다음 simple FPN neck이 이들을 모두 256채널로 맞춰서 `p2~p5`를 만든다.

```math
p_5 = S_5(W_5 c_5)
```

```math
p_4 = S_4(W_4 c_4 + U(p_5 \to c_4))
```

```math
p_3 = S_3(W_3 c_3 + U(p_4 \to c_3))
```

```math
p_2 = S_2(W_2 c_2 + U(p_3 \to c_2))
```

여기서

- $W_i$: lateral $1 \times 1$ projection
- $S_i$: smoothing conv block

이다.

결국 FPN 출력은 대략 다음처럼 본다.

- $p_2 \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}$
- $p_3 \in \mathbb{R}^{B \times 256 \times H/8 \times W/8}$
- $p_4 \in \mathbb{R}^{B \times 256 \times H/16 \times W/16}$
- $p_5 \in \mathbb{R}^{B \times 256 \times H/32 \times W/32}$

---

# 3. Branch A: Global / Objectness Branch

Branch A는 **전역 semantic cue 생성기**다.

이 브랜치의 목적은 먼저 다음을 만드는 것이다.

1. 대략 어디에 물체가 있는지
2. 보다 세밀한 segmentation 힌트가 무엇인지
3. 어디가 애매한지
4. 어디가 경계일 법한지

즉 Branch A는 최종 contour를 직접 끝까지 닫는 브랜치가 아니라,  
먼저 **semantic prior / objectness prior / uncertainty / boundary prior**를 만들고,  
이 신호들을 Branch B와 Fusion에 공급한다.

---

## 3.1 Branch A의 입력

Branch A는 FPN 출력 전체를 받는다.

```math
(p_2, p_3, p_4, p_5)
```

---

## 3.2 Branch A의 핵심 feature $f_A$

먼저 $p_3, p_4, p_5$를 모두 $p_2$ 해상도로 맞춘다.

```math
\tilde p_3 = U(p_3 \to p_2), \qquad
\tilde p_4 = U(p_4 \to p_2), \qquad
\tilde p_5 = U(p_5 \to p_2)
```

그 다음 concat 후 conv block을 태워 fused feature를 만든다.

```math
f_A = \phi_A([p_2, \tilde p_3, \tilde p_4, \tilde p_5])
```

shape는 대략

```math
f_A \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
```

이다.

이 $f_A$가 Branch A의 중심 feature이며, 이후 fine/objectness/boundary prior가 여기서 나온다.

---

## 3.3 Coarse Head

Coarse prediction은 주로 $p_4$ 기반의 고수준 feature를 활용한다.

먼저 coarse용 feature를 만든다.

```math
f_{\mathrm{coarse}} = \psi(p_4)
```

그 다음 coarse logits를 예측한다.

```math
z_{\mathrm{coarse}} = U(h_{\mathrm{coarse}}(f_{\mathrm{coarse}}) \to p_2)
```

즉 coarse prediction은

- 더 고수준 semantic 정보
- 더 거친 공간 해상도
- 물체의 대략적 위치와 큰 윤곽

을 담는 보조 segmentation 출력이다.

---

## 3.4 Fine Head

Fine prediction은 $f_A$에서 직접 나온다.

```math
z_{\mathrm{fine}} = h_{\mathrm{fine}}(f_A)
```

shape는 대략 다음처럼 본다.

```math
z_{\mathrm{fine}} \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
```

이것은 Branch A가 내는 보다 세밀한 segmentation logits이다.

---

## 3.5 Objectness Head

Objectness도 $f_A$에서 나온다.

```math
z_{\mathrm{obj}} = h_{\mathrm{obj}}(f_A)
```

sigmoid를 취하면 objectness map이다.

```math
o = \sigma(z_{\mathrm{obj}})
```

shape는

```math
o \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
```

이다.

이 objectness는 detection-style objectness라기보다,  
실질적으로는 **foreground-like confidence map**에 가깝다.

즉,

- 이 위치가 camouflage object에 속할 것 같은가
- foreground 쪽일 가능성이 높은가

를 나타내는 map이다.

---

## 3.6 Boundary Prior Head

Boundary prior도 $f_A$에서 나온다.

```math
z_{\mathrm{bp}} = h_{\mathrm{bp}}(f_A)
```

```math
b = \sigma(z_{\mathrm{bp}})
```

shape는

```math
b \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
```

이다.

이 map은 “여기가 경계처럼 보일 가능성”에 대한 prior다.

---

## 3.7 Uncertainty는 별도 head가 아니다

이 부분이 매우 중요하다.

이 코드에서 uncertainty는 별도의 conv head가 아니다.  
uncertainty는 **fine logits로부터 계산되는 entropy map**이다.

먼저 fine logits를 확률로 바꾼다.

```math
p = \sigma(z_{\mathrm{fine}})
```

수치 안정성을 위해 clip을 두면

```math
\bar p = \mathrm{clip}(p, \epsilon, 1-\epsilon)
```

그 다음 entropy를 계산한다.

```math
u = -\Big( \bar p \log \bar p + (1-\bar p)\log(1-\bar p) \Big)
```

즉 uncertainty는

- $p \approx 0$ 또는 $1$이면 낮고
- $p \approx 0.5$이면 높다

이 말은 곧, uncertainty map $u$가 **fine prediction이 애매한 위치**를 의미한다는 뜻이다.

핵심적으로는 다음처럼 이해하면 정확하다.

```math
u = \mathcal{H}(\sigma(z_{\mathrm{fine}}))
```

즉 uncertainty는 **fine segmentation 결과의 불확실성**이지,  
별도의 supervised uncertainty head가 아니다.

---

## 3.8 Branch A 전체 출력

정리하면 Branch A는 다음 함수다.

```math
\mathrm{BranchA}(p_2,p_3,p_4,p_5)
\mapsto
(z_{\mathrm{coarse}}, z_{\mathrm{fine}}, o, u, b, f_A)
```

각 출력의 의미는 다음과 같다.

- $z_{\mathrm{coarse}}$: 거친 localization
- $z_{\mathrm{fine}}$: 보다 세밀한 segmentation logits
- $o$: foreground/objectness cue
- $u$: fine prediction의 entropy 기반 uncertainty
- $b$: boundary prior
- $f_A$: 이후 fusion의 기본 본체가 되는 feature

---

# 4. Branch A의 Supervision과 Loss

Branch A 관련 supervision은 크게 다음 두 종류다.

- **mask supervision**
- **boundary supervision**

---

## 4.1 기본 segmentation loss 형태

코드의 `BCEDiceLoss`는 이름과 달리 실제 구현은 **BCE + IoU surrogate**에 가깝다.

대략 다음 형태로 이해하면 된다.

```math
\mathcal{L}_{\mathrm{seg}}(z,y)
=
\mathrm{BCEWithLogits}(z,y)
+
\lambda_{\mathrm{iou}}
\left(
1-
\frac{\sum \sigma(z)y + 1}
{\sum \sigma(z)+\sum y-\sum \sigma(z)y + 1}
\right)
```

여기서

- $z$: logits
- $y$: target mask 또는 target boundary

이다.

---

## 4.2 Branch A auxiliary loss

Branch A의 aux loss는 대략 다음처럼 묶인다.

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

- coarse: mask GT로 supervision
- fine: mask GT로 supervision
- objectness: mask GT로 supervision
- boundary prior: boundary GT로 supervision

중요한 점은 uncertainty map $u$에는 **직접적인 supervision이 없다**는 것이다.

uncertainty는 fine prediction에서 파생되기 때문에,  
직접 “uncertainty target”을 두고 학습하는 구조가 아니다.

---

## 4.3 Boundary target 생성

이 프로젝트는 boundary GT를 외부 edge ground truth에 전적으로 의존하지 않고,  
기본적으로 mask에서 내부 생성한다.

개념적으로는 다음과 유사하다.

```math
y_{\mathrm{bdry}}
=
\mathbf{1}\big(\mathrm{dilate}(y) - \mathrm{erode}(y) > 0\big)
```

즉 mask 경계의 morphology-based band를 boundary target으로 사용한다.

---

# 5. Raw Signal vs Used Signal

이 구조의 중요한 실험 축은 Branch A signal ablation이다.

Branch A가 만든 raw signal을 다음처럼 두자.

```math
(o, u, b)
```

하지만 Branch B와 Fusion에 실제로 들어가는 것은 raw가 아니라 **used signal**이다.

```math
o^\* = \alpha_o\, o,\qquad
u^\* = \alpha_u\, u,\qquad
b^\* = \alpha_b\, b
```

여기서

```math
\alpha_o, \alpha_u, \alpha_b \in \{0,1\}
```

이다.

즉,

- objectness ablation: $\alpha_o = 0$
- uncertainty ablation: $\alpha_u = 0$
- boundary prior ablation: $\alpha_b = 0$

중요한 점은 다음이다.

- raw signal $o,u,b$는 계속 계산된다.
- Branch A head도 계속 학습된다.
- 단지 Branch B/Fusion이 실제로 사용하는 버전만 0으로 바뀐다.

이 구조 덕분에 다음 두 질문을 분리해서 볼 수 있다.

1. 그 head가 실제로 의미 있는 map을 만들어내는가?
2. 그 map을 downstream이 실제로 사용했을 때 성능 기여가 있는가?

---

# 6. Branch B: Contour Closure Branch

Branch B는 semantic segmentation을 다시 한 번 하는 브랜치가 아니다.

Branch B의 핵심 목적은

- 중요한 위치를 고르고
- 경계 후보를 만들고
- 경계 fragment들을 token graph로 reasoning하고
- 더 닫힌 contour에 가까운 boundary를 만드는 것

이다.

즉 Branch B는 **구조적 contour refinement branch**다.

---

## 6.1 Branch B의 입력

Branch B는 다음 입력을 받는다.

### Feature 입력
- $p_2$: low-level feature
- $p_4$: high-level feature

### Signal 입력
- $o^\*$: used objectness
- $u^\*$: used uncertainty
- $b^\*$: used boundary prior

즉 Branch B는 원본 이미지가 아니라,  
**FPN feature + Branch A가 만든 prior signal**을 사용한다.

---

## 6.2 Base Feature 생성

먼저 high-level feature $p_4$를 $p_2$ 해상도로 올린다.

```math
\tilde p_4 = U(p_4 \to p_2)
```

그 다음 concat 후 conv로 base feature를 만든다.

```math
f_0 = \phi_B([p_2, \tilde p_4])
```

shape는 대략

```math
f_0 \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
```

이다.

이 $f_0$는 Branch B가 경계 reasoning을 시작할 기본 feature다.

---

## 6.3 ROI Gating

Branch B의 첫 핵심은 ROI gating이다.

used signal 세 개를 channel concat해서 작은 CNN에 넣고 1채널 ROI mask를 만든다.

```math
r = \sigma(g([o^\*, u^\*, b^\*]))
```

shape는

```math
r \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
```

이다.

이 $r$는 의미적으로 다음과 비슷하다.

```math
r(x,y) \approx P\big(\text{이 위치는 contour reasoning에 중요하다}\mid o^\*,u^\*,b^\*\big)
```

즉

- objectness가 높고
- uncertainty가 높고
- boundary prior도 높은 곳

같은 위치를 더 집중해서 보게 만든다.

---

## 6.4 ROI로 Feature 강조

ROI mask는 feature reweighting에 쓰인다.

```math
f_B = f_0 \odot (1+r)
```

이 식의 의미는 매우 직관적이다.

- $r=0$이면 원래 feature 그대로
- $r>0$이면 그 위치 feature 강화

즉 Branch B는 전체 장면을 균일하게 보지 않고,  
**A가 알려준 중요 위치를 더 강조해서 본다.**

여기서 $f_B$는 Branch B의 핵심 feature가 된다.

```math
f_B \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
```

---

## 6.5 Boundary Candidate Head

강조된 feature $f_B$에서 경계 후보를 예측한다.

```math
z_{\mathrm{cand}} = h_{\mathrm{cand}}(f_B)
```

```math
c = \sigma(z_{\mathrm{cand}})
```

shape는 다음처럼 본다.

```math
z_{\mathrm{cand}}, c \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
```

이 단계는 “어느 픽셀이 경계일 가능성이 높은가”를  
픽셀 수준에서 먼저 뽑는 단계다.

---

## 6.6 Fragment Tokenization

이제 Branch B는 픽셀 단위 reasoning을 그대로 하지 않고,  
feature를 $16 \times 16$ token grid로 줄인다.

먼저 adaptive average pooling:

```math
\bar f = A_G(f_B), \qquad
\bar c = A_G(c), \qquad
\bar r = A_G(r)
```

여기서 $G=16$ 이므로

- $\bar f \in \mathbb{R}^{B \times 256 \times 16 \times 16}$
- $\bar c \in \mathbb{R}^{B \times 1 \times 16 \times 16}$
- $\bar r \in \mathbb{R}^{B \times 1 \times 16 \times 16}$

가 된다.

이제 token score를 만든다.

```math
s = \bar c \odot \bar r
```

즉 score는

- boundary candidate가 높고
- ROI도 높은

셀에서 커진다.

그 다음 feature token을 flatten한다.

```math
T = \mathrm{Flatten}(\bar f)
\in \mathbb{R}^{B \times N \times C}
```

여기서

- $N=256$
- $C=256$

이므로

```math
T \in \mathbb{R}^{B \times 256 \times 256}
```

이다.

---

## 6.7 Valid Token Mask

모든 token을 reasoning에 쓰지 않는다.  
score가 threshold를 넘는 token만 valid로 본다.

```math
m_i = \mathbf{1}(s_i > \tau)
```

기본 threshold는 대략

```math
\tau = 0.05
```

이다.

즉 Branch B는  
**ROI 안에 있고, 경계 candidate도 충분히 높은 token만 본다.**

---

## 6.8 Token 좌표

각 token에는 정규화된 grid 좌표가 붙는다.

```math
q_i \in [0,1]^2
```

이 좌표는 learned coordinate가 아니라 regular grid에서 만들어진다.

즉 이 구조는

- content-adaptive superpixel graph가 아니라
- fixed grid token graph

이다.

다만 valid mask $m_i$ 덕분에 실제 reasoning 대상은 동적으로 달라진다.

---

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

중요한 점은 이 graph가 **feature similarity 기반 graph가 아니라는 것**이다.

즉 “특징이 비슷한 token끼리” 연결하는 것이 아니라,

- 공간적으로 가까운 token끼리
- contour가 이어질 법한 지역적 관계를 따라

연결하는 구조다.

---

## 6.10 Graph Message Passing

각 edge $(i,j)$에 대해 message를 만든다.

```math
m_{i\to j}
=
\mathrm{MLP}_{\mathrm{msg}}
\big(
[T_i, T_j, q_j - q_i]
\big)
```

그 다음 destination 쪽으로 들어오는 message를 합친다.

```math
a_j = \sum_{i \in \mathcal{N}(j)} m_{i \to j}
```

그리고 update:

```math
T'_j
=
\mathrm{LN}
\Big(
T_j + \mathrm{MLP}_{\mathrm{upd}}([T_j, a_j])
\Big)
```

이걸 여러 층 반복한다.

```math
T^{(\ell+1)} = \mathrm{GraphLayer}(T^{(\ell)}, q, E)
```

보통 layer 수는 2다.

---

## 6.11 Valid Edge Mask

모든 edge를 다 쓰지는 않는다.  
코드상 valid token끼리 이어진 edge만 실제 reasoning에 사용된다.

```math
e_{ij}^{\mathrm{valid}} = m_i \land m_j
```

즉 의미상 Branch B graph reasoning은

- ROI 밖
- boundary candidate 없음

같은 token을 거의 무시하고,  
**실제로 contour reasoning이 필요한 조각들 사이에서만 관계를 추론**한다.

---

## 6.12 Closure Prediction

Graph reasoning이 끝난 token에서 closure logits를 만든다.

```math
z_{\mathrm{cls}}^{\mathrm{tok}}
=
\mathrm{MLP}_{\mathrm{cls}}(T^{(L)})
\in \mathbb{R}^{B \times N \times 1}
```

이를 다시 $16 \times 16$ map으로 reshape한다.

```math
z_{\mathrm{cls}}^{\mathrm{map}}
=
\mathrm{Reshape}(z_{\mathrm{cls}}^{\mathrm{tok}})
\in \mathbb{R}^{B \times 1 \times G \times G}
```

그 다음 feature resolution로 업샘플:

```math
\tilde z_{\mathrm{cls}} = U(z_{\mathrm{cls}}^{\mathrm{map}} \to f_B)
```

이 closure map은 의미적으로

> 끊긴 경계 fragment들을 닫힌 contour처럼 정리하도록 돕는 learned structural cue

라고 보면 된다.

---

## 6.13 Boundary Refinement

최종 boundary logits는 $f_B$와 closure map을 합쳐서 만든다.

```math
z_{\mathrm{bdry}}
=
h_{\mathrm{ref}}([f_B, \tilde z_{\mathrm{cls}}])
```

즉 Branch B의 최종 boundary는 단순 boundary candidate가 아니라,

- ROI 정보
- feature 강조
- boundary candidate
- token graph reasoning
- closure map

까지 반영한 **refined boundary**다.

shape는 대략

```math
z_{\mathrm{bdry}} \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
```

이다.

---

## 6.14 Affinity Head

Branch B에는 affinity head도 존재한다.

개념적으로는 edge pair마다 다음을 예측할 수 있다.

```math
z_{\mathrm{aff}}^{(i,j)}
=
\mathrm{MLP}_{\mathrm{aff}}([T_i, T_j, q_j - q_i])
```

하지만 현재 공개 v1 코드 기준으로는

- affinity 구조는 존재
- 선택적으로 logits 계산 가능
- 그러나 핵심 학습축은 아님
- loss도 placeholder에 가까움

이라고 이해하는 게 맞다.

---

## 6.15 Branch B 전체를 하나의 함수로 쓰면

```math
\mathrm{BranchB}(p_2,p_4,o^\*,u^\*,b^\*)
\mapsto
(r, z_{\mathrm{cand}}, z_{\mathrm{cls}}, z_{\mathrm{bdry}}, f_B)
```

즉 Branch B를 한 문장으로 요약하면 다음과 같다.

> **Branch A가 알려준 중요 위치 안에서 경계 fragment를 뽑고, token graph로 그 관계를 reasoning하여, 더 닫힌 contour에 가까운 refined boundary를 만드는 브랜치**

---

# 7. Fusion: Branch A와 Branch B를 어떻게 합치는가

이 모델은 Branch A와 Branch B를 단순 평균하거나 concat만 하는 구조가 아니다.  
A를 본체로 두고 B를 gate로 얹는 구조다.

---

## 7.1 B Feature Projection

먼저 Branch B feature를 projection한다.

```math
\hat f_B = W_B(f_B)
```

shape는 A feature와 맞춰진다.

```math
\hat f_B \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
```

---

## 7.2 Fusion Gate 계산

Fusion gate는 다음 입력을 본다.

- $f_A$
- $\hat f_B$
- $z_{\mathrm{bdry}}$ upsample
- $z_{\mathrm{cls}}$ upsample
- $u^\*$ upsample

수식으로 쓰면:

```math
\gamma
=
\sigma\Big(
\phi_F(
[
f_A,\hat f_B,U(z_{\mathrm{bdry}}),U(z_{\mathrm{cls}}),U(u^\*)
]
)
\Big)
```

여기서 $\gamma$는 1채널 마스크가 아니라 **채널별 gate**다.

```math
\gamma \in \mathbb{R}^{B \times 256 \times H/4 \times W/4}
```

---

## 7.3 최종 Fusion

Fusion 결과는 다음과 같다.

```math
f_{\mathrm{fused}} = f_A + \gamma \odot \hat f_B
```

이 식의 의미는 중요하다.

이 구조는

- A와 B를 동등하게 평균내는 구조가 아니고
- B가 A를 덮어쓰는 구조도 아니며
- **A를 기본 semantic body로 두고**
- **B를 correction stream처럼 필요한 만큼만 주입하는 구조**

다.

즉 B는 A를 대체하는 것이 아니라 **보정**한다.

---

## 7.4 A-only baseline일 때

Branch B를 끄면 `NullBranchB + IdentityFusion` 경로를 탄다.

그 경우:

```math
f_{\mathrm{fused}} = f_A
```

즉 A-only baseline은 정말 말 그대로

- Branch A feature만 사용
- Branch B 기여 0
- Fusion도 identity

구조다.

---

# 8. Final Decoder

최종 decoder는 fusion feature를 받아 1채널 최종 mask logits를 낸다.

```math
z_{\mathrm{final}} = D(f_{\mathrm{fused}})
```

최종 segmentation probability는

```math
\hat y = \sigma(z_{\mathrm{final}})
```

이다.

shape는 대략 다음처럼 보면 된다.

```math
z_{\mathrm{final}} \in \mathbb{R}^{B \times 1 \times H/4 \times W/4}
```

마지막에는 입력 해상도로 업샘플되어 평가와 저장에 쓰인다.

---

# 9. 전체 Loss 구조

전체 loss는 대략 다음처럼 생각할 수 있다.

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

하지만 현재 공개 코드 기준으로 핵심 active loss는 주로 다음이다.

---

## 9.1 Final Segmentation Loss

```math
\mathcal{L}_{\mathrm{final}}
=
\mathcal{L}_{\mathrm{seg}}(z_{\mathrm{final}}, y)
```

---

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

---

## 9.3 Boundary Loss

```math
\mathcal{L}_{\mathrm{bdry}}
=
\lambda_{\mathrm{bdry}}
\cdot
\mathrm{BCEWithLogits}(z_{\mathrm{bdry}}, y_{\mathrm{bdry}})
```

필요하면 boundary candidate에도 loss를 건다.

```math
\mathcal{L}_{\mathrm{cand}}
=
\lambda_{\mathrm{cand}}
\cdot
\mathrm{BCEWithLogits}(z_{\mathrm{cand}}, y_{\mathrm{bdry}})
```

---

## 9.4 Topology Loss

이름은 topology지만 현재 구현은 본격적인 topological loss라기보다,  
boundary logits의 이웃 차이를 줄이는 smoothness regularizer에 가깝다.

개념적으로는 다음과 비슷하다.

```math
\mathcal{L}_{\mathrm{topo}}
\approx
\frac{1}{2}
\left(
\|z_{\mathrm{bdry}}[:, :, 1:, :] - z_{\mathrm{bdry}}[:, :, :-1, :]\|_1
+
\|z_{\mathrm{bdry}}[:, :, :, 1:] - z_{\mathrm{bdry}}[:, :, :, :-1]\|_1
\right)
```

---

## 9.5 Affinity Loss

현재 코드 기준 affinity loss는 실질적으로 placeholder에 가깝다.

즉

- affinity 구조는 존재
- 그러나 pair-level label 학습 축은 현재 완성형이 아님
- 기본 실험의 핵심 기여는 boundary/closure 쪽에 있음

으로 이해하면 된다.

---

# 10. Tensor 흐름으로 다시 정리

아래는 forward를 거의 코드 순서대로 다시 쓴 것이다.

---

## 10.1 입력

```math
x \in \mathbb{R}^{B \times 3 \times H \times W}
```

---

## 10.2 Backbone / FPN

```math
(c_2,c_3,c_4,c_5)=\mathrm{Backbone}(x)
```

```math
(p_2,p_3,p_4,p_5)=\mathrm{FPN}(c_2,c_3,c_4,c_5)
```

---

## 10.3 Branch A

```math
f_A = \phi_A([p_2,U(p_3),U(p_4),U(p_5)])
```

```math
z_{\mathrm{coarse}} = U(h_{\mathrm{coarse}}(\psi(p_4)))
```

```math
z_{\mathrm{fine}} = h_{\mathrm{fine}}(f_A)
```

```math
o = \sigma(h_{\mathrm{obj}}(f_A))
```

```math
b = \sigma(h_{\mathrm{bp}}(f_A))
```

```math
u = -(\bar p\log \bar p + (1-\bar p)\log(1-\bar p)),
\qquad
\bar p = \mathrm{clip}(\sigma(z_{\mathrm{fine}}))
```

---

## 10.4 Ablation Switch

```math
o^\* = \alpha_o o,\qquad
u^\* = \alpha_u u,\qquad
b^\* = \alpha_b b
```

---

## 10.5 Branch B Base Feature

```math
f_0 = \phi_B([p_2,U(p_4)])
```

---

## 10.6 ROI Gating

```math
r = \sigma(g([o^\*,u^\*,b^\*]))
```

---

## 10.7 ROI-based Feature Emphasis

```math
f_B = f_0 \odot (1+r)
```

---

## 10.8 Boundary Candidate

```math
z_{\mathrm{cand}} = h_{\mathrm{cand}}(f_B),\qquad
c = \sigma(z_{\mathrm{cand}})
```

---

## 10.9 Tokenization

```math
\bar f = A_G(f_B),\qquad
\bar c = A_G(c),\qquad
\bar r = A_G(r)
```

```math
s = \bar c \odot \bar r
```

```math
T = \mathrm{Flatten}(\bar f)
```

```math
m_i = \mathbf{1}(s_i > \tau)
```

---

## 10.10 Graph Reasoning

```math
E = \mathrm{KNN}(q,k)
```

```math
m_{i\to j}
=
\mathrm{MLP}_{\mathrm{msg}}([T_i,T_j,q_j-q_i])
```

```math
a_j = \sum_{i \in \mathcal{N}(j)} m_{i\to j}
```

```math
T'_j =
\mathrm{LN}\big(T_j + \mathrm{MLP}_{\mathrm{upd}}([T_j,a_j])\big)
```

---

## 10.11 Closure

```math
z_{\mathrm{cls}}^{\mathrm{tok}} = \mathrm{MLP}_{\mathrm{cls}}(T^{(L)})
```

```math
z_{\mathrm{cls}}^{\mathrm{map}} = \mathrm{Reshape}(z_{\mathrm{cls}}^{\mathrm{tok}})
```

```math
\tilde z_{\mathrm{cls}} = U(z_{\mathrm{cls}}^{\mathrm{map}} \to f_B)
```

---

## 10.12 Refined Boundary

```math
z_{\mathrm{bdry}} = h_{\mathrm{ref}}([f_B,\tilde z_{\mathrm{cls}}])
```

---

## 10.13 Fusion

```math
\hat f_B = W_B(f_B)
```

```math
\gamma
=
\sigma\Big(
\phi_F(
[
f_A,\hat f_B,U(z_{\mathrm{bdry}}),U(\tilde z_{\mathrm{cls}}),U(u^\*)
]
)
\Big)
```

```math
f_{\mathrm{fused}} = f_A + \gamma \odot \hat f_B
```

---

## 10.14 Final Prediction

```math
z_{\mathrm{final}} = D(f_{\mathrm{fused}})
```

```math
\hat y = \sigma(z_{\mathrm{final}})
```

---

# 11. Branch A와 Branch B의 역할 차이

이 부분은 직관적으로도 명확히 정리할 필요가 있다.

---

## 11.1 Branch A의 역할

Branch A는 다음 질문에 답하려는 브랜치다.

- 어디가 물체일 가능성이 큰가?
- 어디가 foreground-like한가?
- 어디가 경계처럼 보이는가?
- 어디서 segmentation이 애매한가?

즉 Branch A는 **전역 semantic prior 생성기**다.

수식적으로는 다음처럼 쓸 수 있다.

```math
(p_2,p_3,p_4,p_5)
\mapsto
(z_{\mathrm{coarse}},z_{\mathrm{fine}},o,u,b,f_A)
```

---

## 11.2 Branch B의 역할

Branch B는 다음 질문에 답하려는 브랜치다.

- A가 중요하다고 한 위치 안에서
- 어떤 경계 fragment가 실제 contour를 이룰까?
- 끊긴 boundary 조각들을 어떻게 연결해야 할까?
- 닫힌 윤곽으로 정리할 수 있을까?

즉 Branch B는 **ROI-guided contour refinement / closure reasoning branch**다.

수식적으로는 다음과 같다.

```math
(p_2,p_4,o^\*,u^\*,b^\*)
\mapsto
(r,z_{\mathrm{cand}},z_{\mathrm{cls}},z_{\mathrm{bdry}},f_B)
```

---

## 11.3 한 문장 요약

- **Branch A**는 “어디를 봐야 하는지”를 알려주고
- **Branch B**는 “그 경계를 어떻게 닫고 정리해야 하는지”를 해결한다

---

# 12. 각 Ablation이 정확히 의미하는 것

---

## 12.1 A-only Baseline

Branch B 자체를 끈 경우:

```math
f_{\mathrm{fused}} = f_A
```

즉

- ROI 없음
- candidate 없음
- token graph 없음
- closure 없음
- refined boundary 없음

오직 Branch A + decoder만 남는다.

---

## 12.2 Objectness Off

```math
o^\* = 0,\qquad u^\* = u,\qquad b^\* = b
```

의미:

- objectness head는 계속 예측된다
- 하지만 Branch B/Fusion은 objectness를 실제 입력으로 안 쓴다

---

## 12.3 Uncertainty Off

```math
o^\* = o,\qquad u^\* = 0,\qquad b^\* = b
```

의미:

- fine logits에서 uncertainty는 계속 계산된다
- 하지만 downstream은 그 uncertainty를 사용하지 않는다

---

## 12.4 Boundary Prior Off

```math
o^\* = o,\qquad u^\* = u,\qquad b^\* = 0
```

의미:

- boundary prior head는 살아 있다
- 하지만 Branch B/Fusion은 그 prior를 사용하지 않는다

---

# 13. 최종 핵심 요약

이 모델의 핵심은 다음 한 문장으로 정리할 수 있다.

> **Branch A가 전역 semantic prior(objectness, uncertainty, boundary prior, coarse/fine segmentation)를 만들고, Branch B가 그 prior를 바탕으로 ROI를 정하고 boundary fragment를 token graph로 reasoning하여 closure-aware boundary를 만든 뒤, Fusion이 A를 본체로 두고 B를 선택적으로 주입하여 최종 segmentation을 완성한다.**

수식적으로 다시 압축하면:

```math
x
\rightarrow
(p_2,p_3,p_4,p_5)
\rightarrow
(z_{\mathrm{coarse}},z_{\mathrm{fine}},o,u,b,f_A)
\rightarrow
(o^\*,u^\*,b^\*)
\rightarrow
(r,z_{\mathrm{cand}},T,E,z_{\mathrm{cls}},z_{\mathrm{bdry}},f_B)
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

```math
\text{semantic prior 생성}
```

## Branch B

```math
\text{경계 fragment를 구조적으로 정리}
```

## Fusion

```math
\text{A 본체 위에 B 보정}
```

## Final

```math
\text{최종 camouflage mask 출력}
```