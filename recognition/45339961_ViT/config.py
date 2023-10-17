""" Customisable configuration for ViT model. """

# General user preferences
will_save = False
will_load = True
show_model_summary = False
will_train = False
will_test = False

results_path = "recognition/45339961_ViT/results"

# Dataloader specific parameters
data_path = "C:/Users/Jacqu/Downloads/AD_NC"
batch_size = 64
n_channels = 1
image_size = 224
n_classes = 2
train_mean = 0.1155
train_std = 0.2224
test_mean = 0.1167
test_std = 0.2228
data_split = 0.8

# Training specific parameters
n_epochs = 10
learning_rate = 0.0005

# Transformer model specific parameters
patch_size = 8
n_heads = 6
n_layers = 2
mlp_size = 768
embedding_dim = 24
mlp_dropout = 0.1
attn_dropout = 0.0
embedding_dropout = 0.1

# Model storage specific parameters
load_path = "C:/Users/Jacqu/Downloads/model.pth"
save_path = "C:/Users/Jacqu/Downloads/model.pth"