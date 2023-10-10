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
- Crop
- Flip
- LR Scheduler
- Optimizer
- Criterion
- Model
- Epoch - 35 (64.2), 50 (65), 60 (65.3), 120 (66), 200 (66.7), 300 (65)
- Dropout
# Jobs:
190 - val
188 - 200e 
76 - test_m
77 - test_val_m
78 - 0.1 dp test
79 - 0.1 dp t+v