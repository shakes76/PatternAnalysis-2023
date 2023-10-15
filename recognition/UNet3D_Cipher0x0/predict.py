from train import *

test_data = NiiImageLoader("test/Case_004_Week0_LFOV.nii.gz",
                           "test/Case_004_Week0_SEMANTIC_LFOV.nii.gz")

# path in rangpur
# test_data = NiiImageLoader("/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/Case_004_Week0_LFOV.nii.gz",
#                            "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/Case_004_Week0_SEMANTIC_LFOV.nii.gz")

for X, y in test_data:
    X = X.unsqueeze(0).float().to(device)
    y = y.long().to(device)
    pred = unet(X)



