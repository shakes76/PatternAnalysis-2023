# TODO:
- Make code right (Time, Plots, Different Model, Model Save), Meta Learning?
- Graph Everything (Batch Norm, Train Test)
- Validation Set (https://www.geeksforgeeks.org/training-neural-networks-with-validation-using-pytorch/)

# NOTES:
- Data Augmentation
- Path: /home/Student/s4737925/Project/PatternAnalysis-2023/recognition/ADNI_TRANSFORMER_47379251
- tensorboard --logdir=runs
- In ReadMe add briefly something about Alzheimers (The motivation)
- https://augmentor.readthedocs.io/en/stable/
- Augmix
- Regularization (Dropouts), Normalization
- Weights and Biases (Google)
- No model files in git but save it and datafiles
- 20 marks algo - solve problem, design, comment
- Use images in ReadMe (good documentation) 
- Test if pull request works by creating another repo
- Results and Plots
- Header block in code

# Hyper-Param:
- Learning Rate - 3e-4 (58.6), 1e-4 (58.12), 1e-3 (54), 5e-4 (57), 2e-4 (52.98)
- Optimiser
- Batch Size - 64 (55.88), 32 (56.25), 128 (56.255)
- Patch Size - 64 (56.255), 128 (56.7), 32 (54.388), 256 (44.8), 16 (51.7)
- Dimensionality Head - 1024 (56.7)(56.7), 256 (58.22), 128 (57.84), 512 (58.6), 768 (55.82)
- Normalisation
- Resize (128x128, patch - 128) (65.6667)
- Crop
- Flip
- LR Scheduler
- Optimizer
- Criterion
- Model
- Epoch - 35 (64.2), 50 (65), 60 (65.3), 120 (66), 200 (66.7), 300 (65)
- Dropout
# Jobs:
750 - DViT - 60.6%
754 - CCT - 71.73%
783 - CCT 2 layers - 72.3%
975 - CCT 2 layers + RandomRotation (5) - 74%
1112 - RandomAugment - 76.377%
1152 - RA + 150 E - 76%
1199 - RA + 150 E + n_ops = 3 - 76.5%
1220 - TrivialAugmentWide, E - 80 - 75.244%
1231 - TrivialAugmentWide, E - 200 - 76%
253 - RA + 100 E + n_ops = 4 - 77.155% (STANDARD)
1307 - RA + 100 E + n_ops = 3 + LR: 1e-4 - 75.64%
1364 - Data Leakage (only test) - 75.5%
1438 - RA + 100 E + n_ops = 5 - 76%
1439 - RA + 100 E + n_ops = 4 + magnitude = 11 - 76.53%
1478 - RA + 100 E + n_ops = 4 + Heads = 8  - 76.244%
1590 - RA + 55 E + n_ops = 4 + Heads = 8 - 75%
1604, 1607 - 10E + LR Scheduler - 69.5%
1642 - OnceCycle (outside) - 72.6%
1654 - OnceCycle (inside) - 
1728 - ReduceLROnPlateau - 

GIT PUSH