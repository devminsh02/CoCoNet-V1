# CCR-COD v1 (COCO_V1)

PyTorch 기반의 **Camouflaged Object Detection (COD)** 실험 코드입니다.  
이 저장소는 **ResNet50 backbone + simple FPN + dual-branch reasoning (Branch A / Branch B) + gated fusion + refinement decoder** 구조를 중심으로, 학습·평가·시각화·ablation 실험까지 한 번에 재현할 수 있도록 구성되어 있습니다.

## 프로젝트 개요

이 프로젝트는 COD를 다음과 같은 역할 분리로 다룹니다.

### Branch A: global / objectness branch
- coarse semantic localization
- fine semantic prediction
- objectness estimation
- uncertainty estimation
- boundary prior estimation

### Branch B: contour closure reasoning branch
- ROI gating
- boundary candidate extraction
- fragment tokenization
- graph reasoning
- closure prediction
- boundary refinement

### Fusion and final decoding
- Branch A와 Branch B feature를 gating 기반으로 결합
- refinement decoder를 통해 최종 camouflage mask 예측

## 이 저장소에서 바로 할 수 있는 것

- 학습용 split 생성
- 모델 구조 sanity check
- dev 학습 및 평가
- full-train 실험
- **A-only baseline** 실행
- **Branch A signal ablation** 실행
  - `objectness_map`
  - `uncertainty_map`
  - `boundary_prior`
- 샘플 단위 prediction / branch visualization 저장

## 현재 릴리스 범위

이 README는 **현재 포함된 코드 기준**으로 작성되었습니다.

- 메인 실험 흐름은 **A+B base model** 및 **Branch A signal ablation**에 맞춰 정리되어 있습니다.
- config 안에 `GT_Edge`, `GT_Instance` 관련 경로가 존재할 수 있지만, 현재 릴리스의 기본 학습 흐름은 **image + mask 중심**으로 동작합니다.
- boundary target은 기본적으로 mask에서 내부적으로 생성되는 형태를 사용합니다.
- affinity / topology 관련 훅은 코드 구조에 포함되어 있으나, 기본 공개 설정에서는 핵심 활성 실험 축이 아닙니다.

즉, 이 저장소는 **v1 실험 코드**로 이해하는 것이 가장 정확합니다.

## 저장소 구조

```text
.
├── configs/                 # 실험 설정 YAML
├── data/
│   └── splits/              # prepare-splits 결과
├── datasets/                # dataset / transforms / target 생성
├── engine/                  # trainer / evaluator
├── losses/                  # segmentation / boundary / auxiliary losses
├── metrics/                 # COD metric 계산
├── models/
│   ├── backbones/           # ResNet50 backbone
│   ├── necks/               # simple FPN neck
│   ├── branches/            # Branch A / Branch B
│   ├── fusion/              # gated / identity fusion
│   ├── decoders/            # refinement decoder
│   └── cod_model.py         # 전체 조립 모델
├── scripts/                 # 반복 실험용 PowerShell 스크립트
├── tools/                   # 환경 점검 / 결과 수집
├── utils/                   # config / visualization / 공통 유틸
├── Results/                 # 학습 결과 저장 위치
├── main.py                  # 진입점
└── README.md
```

## 실행 엔트리포인트

`main.py`는 아래 네 가지 커맨드를 제공합니다.

```bash
python main.py prepare-splits
python main.py sanity-model
python main.py train
python main.py eval
```

## 데이터셋 설정

기본 설정 파일은 Windows 환경 예시 경로를 사용합니다.  
실행 전에 **반드시 각 YAML의 dataset / results 경로를 자신의 환경에 맞게 수정**해야 합니다.

기본적으로 다음 데이터셋을 가정합니다.

- COD10K
- CAMO
- CHAMELEON
- NC4K

예시 설정 위치:

```yaml
paths:
  datasets:
    cod10k:
      train_image_root: ...
      train_mask_root: ...
      test_image_root: ...
      test_mask_root: ...
    camo:
      train_image_root: ...
      test_image_root: ...
      mask_root: ...
    chameleon:
      image_root: ...
      mask_root: ...
    nc4k:
      image_root: ...
      mask_root: ...
  results:
    root: ...
```

split 파일은 `prepare-splits` 실행 후 `data/splits/` 아래에 생성됩니다.

## 설치

먼저 가상환경을 만들고 활성화합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

PyTorch는 사용 중인 CUDA / OS 환경에 맞게 별도로 설치한 뒤, 나머지 의존성을 설치합니다.

```powershell
pip install -r requirements.txt
```

환경 점검:

```powershell
python .\tools\check_env.py
```

## 빠른 시작

### 1) split 생성

```powershell
python .\main.py prepare-splits --config .\configs\model_v1_resnet50.yaml
```

### 2) 모델 sanity check

```powershell
python .\main.py sanity-model --config .\configs\model_v1_resnet50_debug_ab_noamp.yaml
```

### 3) dev 학습

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_debug_ab_noamp.yaml --run-name debug_ab_noamp --skip-prepare
```

### 4) dev 평가

`best_salpha.pth`가 생성된 경우 해당 checkpoint를 우선 사용할 수 있고, 그렇지 않으면 `last.pth`를 평가하면 됩니다.

```powershell
python .\main.py eval --config .\configs\model_v1_resnet50_debug_ab_noamp.yaml --run-name debug_ab_noamp_eval --checkpoint .\Results\checkpoints\debug_ab_noamp\last.pth --skip-prepare
```

## 주요 실험

### Base full-train

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_base.yaml --run-name fulltrain_base_v1 --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_base.yaml --run-name fulltrain_base_v1_eval --checkpoint .\Results\checkpoints\fulltrain_base_v1\last.pth --skip-prepare
```

### A-only baseline

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_a_only.yaml --run-name fulltrain_a_only --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_a_only.yaml --run-name fulltrain_a_only_eval --checkpoint .\Results\checkpoints\fulltrain_a_only\last.pth --skip-prepare
```

### Branch A signal ablation

#### Objectness off

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_ablate_objectness_map.yaml --run-name ablate_objectness --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_ablate_objectness_map.yaml --run-name ablate_objectness_eval --checkpoint .\Results\checkpoints\ablate_objectness\last.pth --skip-prepare
```

#### Uncertainty off

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_ablate_uncertainty_map.yaml --run-name ablate_uncertainty --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_ablate_uncertainty_map.yaml --run-name ablate_uncertainty_eval --checkpoint .\Results\checkpoints\ablate_uncertainty\last.pth --skip-prepare
```

#### Boundary prior off

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_ablate_boundary_prior.yaml --run-name ablate_boundaryprior --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_ablate_boundary_prior.yaml --run-name ablate_boundaryprior_eval --checkpoint .\Results\checkpoints\ablate_boundaryprior\last.pth --skip-prepare
```

### 반복 실험 스크립트

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_fulltrain_ablation_sequence.ps1
```

## 결과 저장 위치

```text
Results/
├── checkpoints/<run_name>/
├── logs/<run_name>/
├── metrics/<run_name>/
├── predictions/<run_name>/
├── vis/<run_name>/
└── debug/<run_name>/
```

## 시각화 산출물

`eval`을 실행하면 sample별 branch visualization이 저장됩니다.

<details>
<summary>생성되는 주요 시각화 파일 보기</summary>

```text
Results/vis/<run_name>/<dataset>/<sample_id>/
  00_input.jpg
  01_gt.png
  02_gt_overlay.jpg
  03_final_pred.png
  04_final_overlay.jpg
  05_error_fp_fn.png
  06_a_coarse_heatmap.png
  07_a_fine_heatmap.png
  08_a_objectness_raw_heatmap.png
  09_a_objectness_used_heatmap.png
  10_a_uncertainty_raw_heatmap.png
  11_a_uncertainty_used_heatmap.png
  12_a_boundary_prior_raw_heatmap.png
  13_a_boundary_prior_used_heatmap.png
  14_a_feature_energy_heatmap.png
  15_b_roi_mask.png
  16_b_boundary_candidate_heatmap.png
  17_b_boundary_refined_heatmap.png
  18_b_closure_heatmap.png
  19_b_feature_energy_heatmap.png
  20_fusion_gate_heatmap.png
  21_fused_feature_energy_heatmap.png
  22_affinity_graph.png
  23_branch_summary_board.jpg
```

</details>

이 구조를 통해 다음을 함께 확인할 수 있습니다.

- Branch A signal head가 실제로 반응하는지
- 해당 signal이 downstream에서 실제로 사용되는지
- Branch B와 fusion이 어디에서 기여하는지

## 로그 해석

### dev split 사용 시
- `train_log.csv`에 train loss와 validation metric이 함께 기록됩니다.

### full-train 사용 시
- `val_split: null`이면 epoch별 validation metric은 기록되지 않고 train loss 중심으로 저장됩니다.
- 최종 metric 비교는 학습 후 `eval` 단계에서 수행하는 것이 일반적입니다.

## 권장 비교 순서

아래 순서로 비교하면 해석이 깔끔합니다.

1. `fulltrain_a_only`
2. `fulltrain_base_v1`
3. `ablate_objectness`
4. `ablate_uncertainty`
5. `ablate_boundaryprior`

이렇게 하면 다음을 분리해서 볼 수 있습니다.

- Branch B 전체 기여
- Branch A signal별 기여

## 결과 집계

평가가 끝난 뒤 metric CSV를 하나로 모을 수 있습니다.

```powershell
python .\tools\collect_ablation_results.py --metrics-root .\Results\metrics --output-csv .\Results\ablation_summary.csv
```

## 주의사항

- `loss_affinity`, `loss_topology`가 0이라고 해서 Branch B 전체가 비활성화된 것은 아닙니다. 해당 loss가 현재 설정에서 비활성화된 것일 수 있습니다.
- `eval` 시 저장되는 heatmap은 시각화 목적이며, 샘플별 정규화 결과로 해석하는 것이 안전합니다.
- Windows 경로와 PowerShell / `.bat` 스크립트를 기준으로 작성되어 있으므로, Linux 환경에서는 경로와 실행 스크립트를 적절히 바꿔야 합니다.
- 공개용 정리 단계에서는 config naming, dependency hygiene, 실험 naming consistency를 한 번 더 정리하는 것을 권장합니다.

## 체크리스트

GitHub에 올리기 전에 최소한 아래 항목은 확인하는 편이 안전합니다.

- [ ] 각 YAML의 dataset / results 경로가 현재 환경과 일치하는가
- [ ] `prepare-splits`가 정상적으로 split txt를 생성했는가
- [ ] `tools/check_env.py`가 통과하는가
- [ ] `sanity-model`이 shape mismatch 없이 끝나는가
- [ ] `Results/` 하위에 checkpoint / logs / metrics / vis가 정상 생성되는가

---

이 저장소는 **COD 구조 실험용 v1 코드베이스**이며, 특히 **A+B base pipeline과 Branch A signal ablation 분석**에 초점을 둡니다.
