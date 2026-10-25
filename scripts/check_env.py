"""Sanity-check the runtime environment: python scripts/check_env.py"""
import pandas
import pymatgen.core
import scipy
import sklearn
import torch
import torch_geometric
import torch_scatter

print(f'torch {torch.__version__} (CUDA build {torch.version.cuda})')
print(f'torch_geometric {torch_geometric.__version__}, torch_scatter {torch_scatter.__version__}')
print(f'pymatgen {pymatgen.core.__version__}, scikit-learn {sklearn.__version__}, '
      f'scipy {scipy.__version__}, pandas {pandas.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 0: {torch.cuda.get_device_name(0)}')
else:
    print('Training and evaluation require a CUDA GPU.')
