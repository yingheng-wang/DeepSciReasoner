# DeepSciReasoner

Official implementation of **Deep Scientific Reasoning under Physical Constraints: Structure-Aware Spectrum Prediction** (ICML 2026).

Yingheng Wang, Tao Yu, Shufeng Kong, Francesco Ricci, John M. Gregoire, Carla P. Gomes

[[Paper]](https://arxiv.org/abs/2610.34021) [[Project page]](https://yingheng-wang.com/DeepSciReasoner/) [[Dataset]](https://huggingface.co/datasets/yingheng/DeepSciReasoner)

DeepSciReasoner predicts structured scientific spectra from crystal structures while respecting hard physical constraints. It works in two stages:

- **Stage I: structure-aware spectrum decoding.** An edge-aware crystal graph transformer encodes the crystal. A two-stage transformer decoder then attends from spectral-bin queries to the atom embeddings to produce an initial spectrum.
- **Stage II: mass-conserving iterative refinement (MIR).** A velocity network, conditioned on the frozen crystal embedding, is trained against optimal-transport displacement interpolation between the Stage I prediction and the target. At inference, the spectrum's CDF is transported with an upwind scheme, which preserves nonnegativity and total spectral mass.

We evaluate on electronic density of states (eDOS, CalTech and TRI), phonon DOS (phDOS), XANES (full multi-absorber and oxygen K-edge), and Raman spectra.

## Repository layout

```text
train_stage1.py        Stage I training (structure-aware decoder)
train_stage2.py        Stage II training (MIR on a frozen Stage I model)
test.py                Test-set evaluation of a two-seed Stage I/II ensemble
parser.py              Shared command-line arguments
Sci_Reasoner/
  data.py              Benchmark registry and crystal-graph dataset
  stage1.py            Graph encoder, spectrum decoder, Stage I loss
  stage2.py            Velocity network and MIR training/inference
  transformer.py       Transformer building blocks and the Stage I LR schedule
  utils.py             Metrics (R2, MAE, MSE, Wasserstein distance)
scripts/
  download_data.sh     Fetch the prepared benchmarks from Hugging Face
  train_stage1.sh      Stage I launcher (bash or sbatch)
  train_stage2.sh      Stage II launcher (bash or sbatch)
  test.sh              Evaluation launcher (bash or sbatch)
  check_env.py         Environment sanity check
environment.yml        Conda environment
```

## Installation

The code targets Linux x86_64 with an NVIDIA GPU: Python 3.12, PyTorch 2.5.1 + CUDA 12.4, and PyTorch Geometric 2.6.1. Use a GPU with compute capability sm_50–sm_90 (e.g. A100, H100) and a driver that supports CUDA 12.

```bash
git clone https://github.com/yingheng-wang/DeepSciReasoner.git
cd DeepSciReasoner
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yml
conda activate dsr
python -m pip check && python scripts/check_env.py
```

## Data

The six benchmarks are hosted at [huggingface.co/datasets/yingheng/DeepSciReasoner](https://huggingface.co/datasets/yingheng/DeepSciReasoner) as prepared crystal graphs with spectrum targets. Download all of them (about 500 MB compressed), or only the ones you need:

```bash
bash scripts/download_data.sh
```

```bash
bash scripts/download_data.sh edos_caltech phdos
```

Each benchmark is unpacked to `DATA/<benchmark>/`:

```text
DATA/<benchmark>/
  manifest.csv     sample IDs (first column); split indices refer to these rows
  split.pkl        fixed (train, val, test) row indices
  graphs/<id>.chkpt
```

| Benchmark | Spectrum | `--data_src` | `--label_dim` | Spectral grid | Samples | Train / Val / Test |
| --- | --- | --- | ---: | --- | ---: | --- |
| `edos_caltech` | eDOS, CalTech | `binned_dos_128` | 128 | [−4, 4] eV | 37,565 | 30,052 / 3,757 / 3,756 |
| `edos_tri` | eDOS, TRI | `binned_dos_256` | 256 | [−5, 5] eV | 14,140 | 11,312 / 1,414 / 1,414 |
| `phdos` | Phonon DOS | `ph_dos_51` | 51 | [0, 1000] cm⁻¹ | 1,524 | 1,220 / 152 / 152 |
| `xanes_full` | K-edge XANES, all absorbers | `xanes_128_sigma0.5` | 128 | [−5, 40] eV rel. to E₀ | 49,895 | 39,916 / 4,990 / 4,989 |
| `xanes_o` | K-edge XANES, oxygen | `xanes_O_128_sigma0.5` | 128 | [−5, 40] eV rel. to E₀ | 5,094 | 4,075 / 510 / 509 |
| `raman` | Raman | `raman_128_sigma10` | 128 | [0, 1500] cm⁻¹ | 2,283 | 1,826 / 229 / 228 |

Curation details are in Appendix D.1 of the paper and on the [dataset card](https://huggingface.co/datasets/yingheng/DeepSciReasoner). For the XANES benchmarks, the model is also conditioned on the absorbing element. To keep the data elsewhere, symlink that location to `DATA/`.

## Training and evaluation

Run every command from the repository root with the `dsr` environment active. The launchers take a benchmark name and a seed. They apply the configuration used in the paper: 200 epochs per stage and batch size 128. The model uses `d_model=512` and 6 decoder layers per decoder module. Stage I trains with `lr=1` (Noam-style warmup/decay); Stage II trains with AdamW at `lr=1e-3` and 100 MIR steps. Extra arguments are forwarded to the Python entry point and override these defaults.

```bash
bash scripts/train_stage1.sh edos_caltech 1
```

```bash
bash scripts/train_stage1.sh edos_caltech 4
```

```bash
bash scripts/train_stage2.sh edos_caltech 1
```

```bash
bash scripts/train_stage2.sh edos_caltech 4
```

```bash
bash scripts/test.sh edos_caltech 1 4
```

Stage II for seed `N` loads the Stage I run from the same seed. `test.py` averages the refined spectra of the two Stage I/II pairs with equal weights. It then reports R², MAE, MSE and Wasserstein distance on the test split. The code runs on the first visible GPU; select one with `CUDA_VISIBLE_DEVICES`.

**Slurm.** The same scripts can be submitted with `sbatch` from the repository root; logs go to `<job name>-<job id>.out` and `.err` there (e.g. `dsr_stage1-123456.out`). The `#SBATCH` headers request 1 GPU, 16 CPUs, 32 GB of memory and 24 h. Add site-specific options on the command line, e.g. `sbatch -p <partition> --qos=<qos> ...`. Dependencies chain the stages:

```bash
s1=$(sbatch --parsable scripts/train_stage1.sh edos_caltech 1)
```

```bash
sbatch --dependency=afterok:$s1 scripts/train_stage2.sh edos_caltech 1
```

**Outputs.** The best-validation-MSE checkpoint and a log are written to `TRAINED/stage1/<benchmark>_stage1_seed<N>/` and `TRAINED/stage2/<benchmark>_stage2_seed<N>/`. TensorBoard summaries go to `summary/`. Re-running with the same benchmark and seed overwrites the previous run. Pass `--save_dir <name>` for a separate run; for Stage II, also pass `--stage1_ckpt <name>`.

## Results

Test-set performance of DeepSciReasoner reported in the paper (mean ± std over seeds):

| Benchmark | R² ↑ | MAE ↓ | MSE ↓ | WD ↓ |
| --- | --- | --- | --- | --- |
| eDOS, CalTech | 0.815 ± 0.007 | 1.983 ± 0.021 | 19.612 ± 0.225 | 0.191 ± 0.017 |
| eDOS, TRI | 0.761 ± 0.009 | 2.357 ± 0.046 | 27.162 ± 0.563 | 0.305 ± 0.011 |
| phDOS | 0.636 ± 0.013 | 0.062 ± 0.004 | 0.021 ± 0.002 | 0.061 ± 0.002 |
| XANES, full | 0.764 ± 0.009 | – | – | 0.098 ± 0.003 |
| XANES, O | 0.474 ± 0.010 | – | – | 0.093 ± 0.003 |
| Raman | 0.264 ± 0.002 | – | – | 0.423 ± 0.017 |

## Citation

```bibtex
@inproceedings{wang2026deepscireasoner,
  title     = {Deep Scientific Reasoning under Physical Constraints: Structure-Aware Spectrum Prediction},
  author    = {Wang, Yingheng and Yu, Tao and Kong, Shufeng and Ricci, Francesco and Gregoire, John M. and Gomes, Carla P.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {306},
  year      = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The benchmark data are derived from the Materials Project and other public sources; see the [dataset card](https://huggingface.co/datasets/yingheng/DeepSciReasoner) for their licenses and the works to cite.
