# PCSE

This repository is a revision-stage partial code release. It provides the core model modules (see `model.py`) and the split lists used in our experiments. It is not a full end-to-end reproduction package.

## Files

- `model.py`: core model modules (uncertainty-aware refinement / progression mask related components).
- `assembly101_exoV1_egoE4_split.txt`: split list for Assembly101 (ExoV1 → EgoE4).
- `egoexo4d_exoCam01_egoCam214_split.txt`: split list for Ego-Exo4D (ExoCam01 → EgoCam214).
- `requirements.txt`, `assets/`.

## Note

The full training pipeline (dataloaders, complete loss definitions, preprocessing, checkpoints, and evaluation scripts) is not included in this public snapshot. `train.py` depends on additional internal modules and is provided for reference only.

