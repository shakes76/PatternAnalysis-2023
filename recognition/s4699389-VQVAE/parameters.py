VQVAE_PATH = "./vqvae_model.txt"
TRAIN_OUTPUT_PATH = "./train_output.out"

num_epochs = 5  # Change to desired epochs

# Settings for train.py
save_model = True
save_model_output = True

# Settings for predict.py
save_figure = True      # True if you wish to save VQVAE reconstruction results to file, False otherwise
show_one_figure = True  # True if you only want to show one test example, False shows all

save_graphs = True
train_loss = True
mean_ssim = True

# Hyperparameters
batch_size = 32
learning_rate = 0.0002
commitment_cost = 0.25
num_hiddens = 128
num_residual_hiddens = 32
num_channels = 1
embedding_dim = 64
num_embeddings = 512