from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib import cycler

from dataset import load_test_data, load_train_data
from modules import *

# Stylize plots
def plot_configs():
    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                    '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
        axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('xtick', direction='out', color='gray')
    plt.rc('ytick', direction='out', color='gray')
    plt.rc('patch', edgecolor='#E6E6E6')
    plt.rc('lines', linewidth=2)

    font = {'size': 12}
    plt.rc('font', **font)
    plt.rc('legend', fontsize = 12)

#path = r"C:\Users\dcp\Documents\OFFLINE-Projects\DATASETS\ADNI"  # Laptop Path
path = r"C:\Users\deepp\Documents\Offline Projects\ML Datasets\ADNI" # PC path

# Model Parameters
image_shape = (240, 240)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
d_latent = 256
embed_dim = 128
transformer_depth = 1
num_heads = 1
n_perceiver_blocks = 6
num_epochs = 50
batch_size = 5
n_classes = 2
lr = 0.005

if __name__ == "__main__":
    
    # Dataset loader and Transforms
    transforms = transforms.Compose([
        transforms.CenterCrop(240), 
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Lambda(lambda x: torch.flatten(x, start_dim = 1))
    ])
    train_dataset = load_train_data(path, batch_size, transforms = transforms)

    # Model
    model = Perceiver(
        d_latent,
        embed_dim,
        num_heads,
        transformer_depth,
        n_perceiver_blocks,
        n_classes,
        batch_size
    ).to(device = device)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    # Training the model
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0
        correct = 0
        total = 0

        for i, (samples, labels) in enumerate(train_dataset):

            # Get samples and labels of batch
            samples = samples.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            outputs = model(samples)

            # Get current loss and update weights
            loss = loss_fn(outputs, labels)
            loss.backward()
            optim.step()

            # Get accuracy and loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

            # if i % 200 == 0:
            #     print(f"Batch {i} / {len(train_dataset)}")

        losses.append(running_loss / len(train_dataset))
        accuracies.append(100 * (correct / total))

        print(f"Epoch {epoch + 1} / {num_epochs}, loss: {losses[-1]}, accuracy: {accuracies[-1]}")

    # Testing the model
    test_dataset = load_test_data(path, batch_size, transforms = transforms)

    model.eval()
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(test_dataset): 
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Model Test Accuracy: {test_accuracy:.4f}%")

plot_configs()
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (30, 10))

for ax in axs:
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_xlim([0, num_epochs])
    ax.legend()

# Plot training loss vs epoch
axs[0].plot(range(1, num_epochs + 1), losses, label = 'Training Loss')
axs[0].set_title('Training Loss vs. Epochs')

# Plot training accuracy vs epoch
axs[1].plot(range(1, num_epochs + 1), accuracies, label = 'Training Accuracy')
axs[1].set_title('Training Accuracy vs. Epochs')
axs[1].set_ylim([0, 100])

plt.show()
torch.save(model.state_dict(), "./model.pth")