# Configurable constants for the project

# Filepath to the training data
# Expected directory structure:
# /AD
#     /[PatientID]_[imageID].jpeg
#     ...
# /NC
#     /[PatientID]_[imageID].jpeg
#     ...
TRAIN_PATH = '/home/groups/comp3710/ADNI/AD_NC/train/'

# Filepath to the test data
# Expected directory structure:
# /AD
#     /[PatientID]_[imageID].jpeg
#     ...
# /NC
#     /[PatientID]_[imageID].jpeg
#     ...
TEST_PATH = '/home/groups/comp3710/ADNI/AD_NC/test/'

# Filepath to load pre-trained models from
# Expected directory structure:
# /one_checkpoint.tar
# /another_checkpoint.tar
MODEL_PATH = "/home/Student/s4641725/COMP3710/project_results/"

# Filepath where results, including plots and model checkpoints, will be saved
RESULTS_PATH = "/home/Student/s4641725/COMP3710/project_results/"