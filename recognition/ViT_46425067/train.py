import torch
import torch.nn as nn
import torch.optim as optim
from modules import ViT
from dataset import load_data
from types import SimpleNamespace
from tqdm.auto import tqdm

#setup random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def train_epoch(model: nn.Module, 
                data_loader: torch.utils.data.DataLoader,
                loss_fn: nn.Module,
                optimiser: optim.Optimizer,
                device: str):
    train_loss, train_acc = 0, 0
    model.train()
    
    for batch, (X, y) in enumerate(tqdm(data_loader)):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        train_acc += accuracy(y_pred, y)
        #backpropagation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    train_acc = train_acc / len(data_loader)
    train_loss = train_loss / len(data_loader)
    return train_loss, train_acc

def test_epoch(model: nn.Module, 
                data_loader: torch.utils.data.DataLoader,
                loss_fn: nn.Module,
                device: str):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_acc += accuracy(y_pred, y)
    test_loss = test_loss / len(data_loader)
    test_acc = test_acc / len(data_loader)
    return test_loss, test_acc

def accuracy(y_pred, y):
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc = (y_pred_class == y).sum().item() / len(y_pred)
    return train_acc

def train_model(model: nn.modules,
                train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                optimiser: optim.Optimizer,
                loss_fn: nn.modules,
                device: str,
                config):
    # track results of training
    results  = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc":  [],}
    # main training loop
    for epoch in tqdm(range(config.epochs)):
        #training loss and accuracu
        train_loss, train_acc = train_epoch(model=model,
                                            data_loader=train_loader,
                                            loss_fn=loss_fn,
                                            optimiser=optimiser,
                                            device=device)
        # testing loss and accuracy
        test_loss, test_acc = test_epoch(model=model,
                                            data_loader=test_loader,
                                            loss_fn=loss_fn,
                                            device=device)
        # save results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results

if __name__ == "__main__":
    #device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #hyperparmeters
    config = SimpleNamespace(
        epochs=1,
        batch_size=32,
        img_size=(224, 224),
        patch_size=16,
        img_channel=1,
        num_classes=2,
        embed_dim=768,
        depth=1,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        drop_prob=0.1,
        lr=1e-3
    )
    
    #load dataloaders
    train_loader, test_loader, _, _ = load_data(config.batch_size, config.img_size)
    #create model
    model = ViT(img_size=config.img_size[0],
                patch_size=config.patch_size,
                img_channels=config.img_channel,
                num_classes=config.num_classes,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                drop_prob=config.drop_prob).to(device)
    #loss function + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = optim.AdamW(model.parameters(), lr=config.lr)
    
    #Train the model and store the results
    results = train_model(model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            optimiser=optimiser,
                            loss_fn=loss_fn,
                            device=device,
                            config=config)
    print(results)