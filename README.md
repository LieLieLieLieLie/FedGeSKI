# FedGeSKI

Official implementation of **FedGeSKI: Geometric-Statistical Knowledge
Integration for Task-Free Federated Continual Intrusion Detection under
Model-Age Staleness**.

FedGeSKI addresses federated continual intrusion detection when attack classes
arrive in client-dependent orders, local feature distributions drift, only a
subset of clients participates, and updates may have different model ages. The
method combines regular-simplex anchors, classwise moment statistics,
stable-plastic projection, reliability-staleness-aware aggregation, and
record-free transient server consolidation. Default prediction uses the global
encoder and classifier directly; posterior correction is retained only as an
optional diagnostic control.

This repository contains the complete Python pipeline for deterministic data
preprocessing, federated simulation, baseline evaluation, statistical
summarization, prediction extraction, and paper-style PDF visualization. Dataset
archives, processed data, trained checkpoints, and generated results are not
included.

## Requirements

- Python 3.11 is recommended.
- PyTorch 2.0 or newer.
- A CUDA-capable device is optional. The implementation selects CUDA when it is
  available and otherwise falls back to CPU.

Create an isolated environment and install the dependencies:

```bash
git clone https://github.com/LieLieLieLieLie/FedGeSKI.git
cd FedGeSKI
python -m venv .venv
```

Activate the environment, then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset acquisition

The experiments use labeled, tabular intrusion-detection data derived from two
public benchmarks:

- [Edge-IIoTset](https://doi.org/10.21227/mbc1-1h68), originally released for
  centralized and federated IoT/IIoT intrusion detection.
- [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html), collected
  from a 105-device IoT topology under benign traffic and 33 attacks.

For exact reproduction, download the following six-class federated archives
from the public [Zenodo record](https://doi.org/10.5281/zenodo.11315294):

- `df_FL_Edge-IIoTset_6_classes.rar`
- `df_FL_CICIoT2023_6_classes.rar`

Extract the CSV files and create this directory structure at the repository
root:

```text
FedGeSKI/
|-- data/
|   |-- EdgeIIoTset/
|   |   `-- df_FL_Edge-IIoTset_6_classes.csv
|   `-- CICIoT2023/
|       `-- df_FL_CICIoT2023_6_classes.csv
|-- config.py
|-- prepare_data.py
`-- ...
```

The spelling and capitalization of both directory and file names must match the
tree above. The code does not redistribute either dataset; users remain
responsible for complying with the providers' access and usage conditions.

## Preprocessing

Prepare both datasets with the protocol used in the experiments:

```bash
python prepare_data.py --all
```

The command hashes each complete numerical feature vector before sampling,
allocates each hash group to exactly one train/validation/test subset, performs
deterministic classwise sampling, applies training-only robust scaling over the
5th--95th percentile range, and clips to `[-12, 12]`. Processed arrays and
machine-readable overlap checks are written to `data/processed/`. Because the
six-class derivatives omit timestamps, device IDs, sessions, and flow IDs, this
protocol prevents exact-feature duplicates across subsets but does not claim
temporal-, device-, or session-disjoint generalization.

To change the per-class sample count or preprocessing seed:

```bash
python prepare_data.py --all --per-class 6000 --seed 2026
```

## Quick verification

Run a short FedGeSKI experiment before launching the complete benchmark:

```bash
python run_experiment.py \
  --dataset edgeiiot \
  --method fedgeski \
  --seed 99 \
  --rounds 2 \
  --local-steps 1 \
  --overwrite
```

Available dataset identifiers are `edgeiiot` and `ciciot`. Available method
identifiers are `fedgcc`, `afcl_csc`, `fedavgm`, `fedadam`, `fedyogi`,
`fedasync`, `fedbuff`, `glfc`, `evofedids`, `fedta`, `fedagc`, and
`fedgeski`. `fedgcc` and `afcl_csc` are protocol-aligned adaptations for the
common fixed-label tabular setting; they are not bit-for-bit reproductions of
the source visual/pretrained systems.

## Reproducing the evaluation

Run the experiment suites separately so that completed configurations can be
cached and resumed:

```bash
python run_all.py --suite main
python run_all.py --suite asynchronous
python run_all.py --suite stress
python run_all.py --suite ablation
python run_all.py --suite mechanism
python run_all.py --suite sensitivity
python run_all.py --suite extreme
```

Use `python run_all.py --suite all` to execute every suite. By default, an
existing completed checkpoint is reused; pass `--overwrite` only when a run
must be recomputed.

After training, generate the numerical summaries, cached predictions, and PDF
figures:

```bash
python summarize_results.py
python extract_predictions.py
python visualize_results.py
```

## Output structure

The scripts create all output directories automatically:

```text
results/
|-- figures/   # vector PDF figures
|-- models/    # checkpoints, histories, metrics, and prediction caches
|-- tables/    # CSV and LaTeX-ready numerical summaries
`-- logs/      # execution logs
```

Visualization reads cached artifacts and does not retrain the models. The
eleven baselines and FedGeSKI share the same tabular encoder, evolving stream,
client sampling process, optimizer budget, evaluation implementation, and FP16
model-increment codec. FedGeSKI's input-moment capsule is counted
separately, as is the FP16 server-to-client input-moment snapshot used for
transient local statistical augmentation. Per-run JSON files expose uplink,
downlink, and total communication independently. Replay, server-momentum,
adaptive-moment, and other method-specific
auxiliary states are included in the reported memory and communication
accounting.

## Reproducibility notes

- Main comparisons and component controls use five matched seeds (0--4).
- Stress tests vary client participation, Dirichlet label skew, and maximum
  model age. The three prespecified extreme conditions are repeated over five
  seeds for Fed-GCC, AFCL-CSC, and FedGeSKI.
- The bounded-model-age suite compares AFCL-CSC, FedAsync, FedBuff, and
  FedGeSKI under matched delay schedules and bidirectional model encoding.
- The mechanism suite repeats the full and statistics-only variants over five
  seeds and records historical-gradient coverage and cross-version alignment
  diagnostics.
- The code records the complete experiment configuration in every JSON history.
- Random generators and deterministic backend options are initialized from the
  experiment seed.
- A CPU run is supported but will generally take longer than CUDA execution.

## Citation

The accompanying manuscript is under peer review. If this implementation is
useful in your research, please cite the paper by its title; complete
bibliographic information will be added after publication.
