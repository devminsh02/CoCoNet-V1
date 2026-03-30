$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot\..
$cfgs = @(
  @{ cfg = '.\\configs\\model_v1_resnet50_fulltrain_a_only.yaml'; run = 'fulltrain_a_only' },
  @{ cfg = '.\\configs\\model_v1_resnet50_fulltrain_base.yaml'; run = 'fulltrain_base' },
  @{ cfg = '.\\configs\\model_v1_resnet50_fulltrain_ablate_objectness_map.yaml'; run = 'ablate_objectness' },
  @{ cfg = '.\\configs\\model_v1_resnet50_fulltrain_ablate_uncertainty_map.yaml'; run = 'ablate_uncertainty' },
  @{ cfg = '.\\configs\\model_v1_resnet50_fulltrain_ablate_boundary_prior.yaml'; run = 'ablate_boundary_prior' }
)
foreach ($item in $cfgs) {
  python .\main.py train --config $item.cfg --run-name $item.run --skip-prepare
  python .\main.py eval --config $item.cfg --run-name ($item.run + '_eval') --checkpoint (".\\Results\\checkpoints\\" + $item.run + "\\last.pth") --skip-prepare
}
