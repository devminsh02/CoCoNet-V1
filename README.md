# COCO_V1 / CCR-COD Complete Project

이 압축본은 **다른 컴퓨터에서도 그대로 압축 해제 후 실행 가능한 전체 코드 원본**이다.

## 프로젝트 목표

이 프로젝트는 Camouflaged Object Detection(COD)을 위해 다음 철학을 따른다.

- **Branch A**: global/objectness branch
  - coarse semantic localization
  - fine semantic prediction
  - objectness / uncertainty / boundary prior 추정
- **Branch B**: contour closure reasoning branch
  - ROI gating
  - boundary candidate extraction
  - fragment tokenization
  - graph reasoning
  - closure prediction
- **Fusion**: Branch A와 Branch B를 gating 기반으로 결합
- **Final decoder**: 최종 camouflage mask 예측

또한 이 프로젝트는 **Branch A signal ablation**을 실험 가능하게 설계했다.
즉 Branch A가 리턴하는 아래 signal을 downstream에서 **raw는 유지한 채 used signal만 끌 수 있다.**

- `objectness_map`
- `uncertainty_map`
- `boundary_prior`

그래서 아래 두 가지를 동시에 볼 수 있다.

1. head 자체는 해당 signal을 예측했는가?
2. 그 signal을 Branch B / Fusion에서 실제로 사용했을 때 기여가 있는가?

## 데이터셋 구조

프로젝트는 아래 Windows 경로를 기본으로 가정한다.

```text
E:\COD\Datasets\COD10K-v3
E:\COD\Datasets\CAMO-V.1.0-CVIU2019
E:\COD\Datasets\CHAMELEON
E:\COD\Datasets\NC4K
```

`prepare-splits`를 실행하면 `data/splits/` 아래에 split txt가 생성된다.

## 설치

먼저 가상환경을 만들고 PyTorch를 설치한다.

```powershell
cd /d E:\COD\COCO_V1
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

그 다음 공식 PyTorch selector에서 네 환경에 맞는 `torch` / `torchvision`을 설치하고, 나머지를 설치한다.

```powershell
pip install -r requirements.txt
```

환경 확인:

```powershell
python .\tools\check_env.py
```

## 가장 기본 실행 순서

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

### 4) dev checkpoint 평가

```powershell
python .\main.py eval --config .\configs\model_v1_resnet50_debug_ab_noamp.yaml --run-name debug_ab_noamp_bestS_eval --checkpoint E:\COD\COCO_V1\Results\checkpoints\debug_ab_noamp\best_salpha.pth --skip-prepare
```

## Full-train 메인 실험

### Base full-train

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_base.yaml --run-name fulltrain_base_v1 --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_base.yaml --run-name fulltrain_base_v1_eval --checkpoint E:\COD\COCO_V1\Results\checkpoints\fulltrain_base_v1\last.pth --skip-prepare
```

### A-only baseline

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_a_only.yaml --run-name fulltrain_a_only --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_a_only.yaml --run-name fulltrain_a_only_eval --checkpoint E:\COD\COCO_V1\Results\checkpoints\fulltrain_a_only\last.pth --skip-prepare
```

### Branch A signal ablation

#### objectness off

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_ablate_objectness_map.yaml --run-name ablate_objectness --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_ablate_objectness_map.yaml --run-name ablate_objectness_eval --checkpoint E:\COD\COCO_V1\Results\checkpoints\ablate_objectness\last.pth --skip-prepare
```

#### uncertainty off

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_ablate_uncertainty_map.yaml --run-name ablate_uncertainty --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_ablate_uncertainty_map.yaml --run-name ablate_uncertainty_eval --checkpoint E:\COD\COCO_V1\Results\checkpoints\ablate_uncertainty\last.pth --skip-prepare
```

#### boundary prior off

```powershell
python .\main.py train --config .\configs\model_v1_resnet50_fulltrain_ablate_boundary_prior.yaml --run-name ablate_boundaryprior --skip-prepare
python .\main.py eval --config .\configs\model_v1_resnet50_fulltrain_ablate_boundary_prior.yaml --run-name ablate_boundaryprior_eval --checkpoint E:\COD\COCO_V1\Results\checkpoints\ablate_boundaryprior\last.pth --skip-prepare
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

## 브랜치 시각화

`eval`을 실행하면 sample별로 아래와 같은 시각화가 저장된다.

```text
Results\vis\<run_name>\<dataset>\<sample_id>\
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

이 구조는 raw/used signal을 모두 보여주므로,
- signal head는 살아 있는지
- downstream에서 그 signal을 실제로 쓰는지
를 함께 확인할 수 있다.

## 학습 로그 해석

### dev split 사용 시
`train_log.csv`에 train loss와 validation metric이 함께 찍힌다.

### full-train 사용 시
`val_split: null`이라 epoch별 metric이 로그에 안 찍히고, train loss만 기록된다.
이 경우 metric은 학습 후 `eval`에서 따로 계산한다.

## 실험 비교 정리

권장 비교 순서:

1. `fulltrain_a_only`
2. `fulltrain_base_v1`
3. `ablate_objectness`
4. `ablate_uncertainty`
5. `ablate_boundaryprior`

이렇게 해야
- Branch B 전체 기여
- Branch A signal별 기여
를 분리해서 해석할 수 있다.

## 결과 수집

평가가 끝난 뒤 CSV 집계:

```powershell
python .\tools\collect_ablation_results.py --metrics-root .\Results\metrics --output-csv .\Results\ablation_summary.csv
```

## 주의사항

- `loss_affinity`, `loss_topology`가 0이면 Branch B 전체가 죽었다는 뜻이 아니라, 해당 loss가 비활성화된 것이다.
- full-train에서는 `last.pth`를 우선 평가하고, 필요하면 이후 별도 best selection을 도입한다.
- 고해상도 시각화는 evaluator에서 메모리 안전하게 처리되지만, 필요하면 `eval.save_vis_per_dataset`를 줄일 수 있다.
