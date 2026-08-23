# NASA C-MAPSS Dataset

This project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** turbofan engine degradation dataset for Remaining Useful Life (RUL) prediction.

## Dataset

The experiments use the following four benchmark subsets:

* FD001
* FD002
* FD003
* FD004

The dataset contains multivariate time-series measurements collected from simulated turbofan engines operating until failure.

## Dataset Source

The dataset can be obtained from:

https://www.kaggle.com/datasets/bishals098/nasa-turbofan-engine-degradation-simulation

## Expected Files

After downloading the dataset, place the required files inside:

```text
data/raw/
```

The expected CMAPSS files are:

```text
train_FD001.txt
train_FD002.txt
train_FD003.txt
train_FD004.txt

test_FD001.txt
test_FD002.txt
test_FD003.txt
test_FD004.txt

RUL_FD001.txt
RUL_FD002.txt
RUL_FD003.txt
RUL_FD004.txt
```

The original NASA dataset README can also be placed inside `data/raw/`.

## Important

The raw dataset files are **not included in this repository**. The `.gitignore` configuration prevents the dataset files from being accidentally committed.

The repository should therefore contain:

```text
data/
├── README.md
└── raw/
    └── .gitkeep
```

After downloading the dataset, `data/raw/` should contain the CMAPSS files listed above.

## Dataset Usage

The notebook preprocesses the raw data by:

1. Loading engine and sensor measurements.
2. Calculating Remaining Useful Life (RUL).
3. Applying the RUL cap.
4. Generating degradation-state labels.
5. Clustering operating conditions.
6. Applying condition-aware normalization.
7. Applying EMA-based sensor denoising.
8. Generating sliding-window sequences for model training.
