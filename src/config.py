# ============================================================
# Configuration
# Aircraft Engine RUL Prediction
# ============================================================

import torch
# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42
BENCHMARK_SEEDS = [42, 43, 44]

# ------------------------------------------------------------
# Dataset Configuration
# ------------------------------------------------------------

DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']

SEQ_LENGTH = 40
RUL_CAP = 125
NUM_CLUSTERS = 6

# ------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------

BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 150
PATIENCE = 45


# ------------------------------------------------------------
# Device Configuration
# ------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')

else:
    DEVICE = torch.device('cpu')
