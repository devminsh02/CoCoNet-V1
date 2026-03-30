from __future__ import annotations

import importlib
import platform


def main() -> None:
    print('Python:', platform.python_version())
    print('Platform:', platform.platform())
    for pkg in ['torch', 'torchvision', 'yaml', 'cv2', 'numpy', 'py_sod_metrics']:
        try:
            mod = importlib.import_module(pkg)
            print(f'{pkg}:', getattr(mod, '__version__', 'unknown'))
        except Exception as exc:
            print(f'{pkg}: NOT AVAILABLE ({exc})')
    try:
        import torch
        print('CUDA available:', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('CUDA device count:', torch.cuda.device_count())
            print('CUDA device name:', torch.cuda.get_device_name(0))
    except Exception as exc:
        print('Torch CUDA check failed:', exc)


if __name__ == '__main__':
    main()
