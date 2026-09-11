"""
Global configuration for the KINDRA plasmonic breath-analysis study.

Every number that controls the optical model, the adsorption model, the
synthetic cohort, or the training protocol is declared here so that a single
file documents the full experimental configuration.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import os

SEED_LIST: Tuple[int, ...] = (11, 23, 37, 53, 71)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURE_DIR = os.path.join(ROOT, "figures")
CACHE_DIR = os.path.join(ROOT, "cache")
for _d in (RESULTS_DIR, FIGURE_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Optical stack (Kretschmann configuration, wavelength interrogation)
# --------------------------------------------------------------------------
PRISM = "SF10"                 # Schott SF10, Sellmeier dispersion in optics.py
INCIDENCE_DEG = 40.25          # fixed internal angle giving a deep resonance for every spot
CR_THICKNESS_NM = 2.0          # chromium adhesion layer
AU_THICKNESS_NM = 47.0         # nominal gold film thickness
LAMBDA_SCAN_NM = (600.0, 1000.0)
LAMBDA_STEPS = 801             # 0.5 nm grid for the reflectance scan

N_CHANNELS = 12                # functionalised sensing spots in the array
# Nominal sorbent overlayer thickness (nm) and refractive index per channel.
# The twelve coatings span the sorbent classes used in plasmonic VOC sensing:
# ZIF-8 / HKUST-1 type MOFs, polymer sorbents and peptide-functionalised films.
COATING_THICKNESS_NM: Tuple[float, ...] = (
    24.0, 31.0, 38.0, 45.0, 52.0, 60.0, 28.0, 35.0, 42.0, 49.0, 56.0, 64.0)
COATING_INDEX: Tuple[float, ...] = (
    1.26, 1.29, 1.32, 1.35, 1.38, 1.41, 1.27, 1.31, 1.34, 1.37, 1.40, 1.43)

N_ADLAYER = 1.47               # refractive index of the condensed VOC adlayer
MONOLAYER_THICKNESS_NM = 0.62  # effective thickness of a saturated VOC monolayer

# --------------------------------------------------------------------------
# Breath sampling protocol
# --------------------------------------------------------------------------
FS_HZ = 0.5                    # spectrometer read-out rate (one full spectral scan per 2 s)
RECORD_S = 180                 # total record length in seconds
T_STEPS = 90                   # samples per record
BASELINE_S = 30                # ambient purge before the first exhalation
N_CYCLES = 3
EXPOSURE_S = 20                # exhalation window
PURGE_S = 20                   # inter-breath purge window
RECOVERY_S = 30

# --------------------------------------------------------------------------
# Cohort
# --------------------------------------------------------------------------
CLASSES: Tuple[str, ...] = ("Healthy", "LungCancer", "COPD", "Asthma")
N_SUBJECTS: Dict[str, int] = {"Healthy": 760, "LungCancer": 600, "COPD": 600, "Asthma": 440}
SESSIONS_PER_SUBJECT = 2
N_DEVICES = 8                  # fabrication batches / physical sensor chips
STUDY_MONTHS = 18              # longitudinal span, used for film ageing
CAP_LOSS_PER_MONTH = 0.013     # fractional loss of binding-site density per month

# --------------------------------------------------------------------------
# Instrument noise and nuisance processes
# --------------------------------------------------------------------------
WAVELENGTH_NOISE_NM = 2.5e-3   # 1-sigma resonance-tracking noise
QUANT_NM = 5.0e-4              # read-out quantisation
BREATH_TEMP_RISE_K = 11.0       # exhaled air is warmer than ambient
BREATH_RH_RISE = 0.55
TEMP_RAMP_K = 4.5              # 1-sigma within-record drift of the stack temperature
TEMP_OU_K = 0.45               # 1-sigma short-term thermal fluctuation
TEMP_OU_TAU_S = 38.0          # absolute change in relative humidity during exhalation
DRIFT_LINEAR_RIU_PER_S = 1.4e-5
DRIFT_OU_SIGMA_RIU = 1.8e-4
DRIFT_OU_TAU_S = 42.0
SPEC_DRIFT_NM = (7.0, 4.2)   # 1-sigma read-out drift, offset and dispersive term
SPEC_DRIFT_TAU_S = 55.0

# --------------------------------------------------------------------------
# Model / training
# --------------------------------------------------------------------------
@dataclass
class TrainConfig:
    epochs: int = 45
    batch_size: int = 64
    lr: float = 1.8e-3
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    patience: int = 12
    latent_analytes: int = 6   # latent volatile components resolved by the cell
    hidden_fast: int = 64      # GRU width
    proj_rank: int = 3         # rank of the estimated nuisance subspace
    lambda_kin: float = 0.60   # weight of the kinetic-consistency residual
    lambda_adv: float = 0.15   # weight of the drift-adversarial term
    lambda_evi: float = 0.02   # weight of the evidential regulariser
    head_width: int = 96
    head_dropout: float = 0.15
    warmup_epochs: int = 8
    label_smooth: float = 0.0
    num_workers: int = 0

TRAIN = TrainConfig()

METRICS: List[str] = ["accuracy", "balanced_accuracy", "macro_f1", "mcc",
                      "macro_auroc", "macro_auprc", "ece", "brier", "nll"]
