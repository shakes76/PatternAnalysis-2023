import torch
import torch.nn as nn
import torch.optim as optim
from modules import ViT, ViT_torch
from dataset import load_data
from types import SimpleNamespace
from tqdm.auto import tqdm
import wandb
import torchinfo
from torchinfo import summary

#setup random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def accuracy(y_pred, y):
    y_pred_label = torch.round(torch.sigmoid(y_pred))
    correct = torch.eq(y, y_pred_label).sum().item()
    acc = (correct / len(y))
    return acc

def train_epoch(model: nn.Module, 
                data_loader: torch.utils.data.DataLoader,
                loss_fn: nn.Module,
                optimiser: optim.Optimizer,
                scheduler: optim.Optimizer,
                grad_scaler,
                device: str,
                mix_precision: bool,
                lr_scheduler: bool):
    #setup for training
    train_loss, train_acc = 0, 0
    model.train()
    
    # training loop
    for batch, (X, y) in enumerate(tqdm(data_loader)):
        # mixed precision
        X, y = X.to(device), y.float().to(device)
        with torch.cuda.amp.autocast(enabled=mix_precision, dtype=torch.float16):
            # model prediction
            y_pred_logits = model(X).squeeze()
            loss = loss_fn(y_pred_logits, y)
        # save loss
        train_loss += loss.item()
        
        # model accuracy
        acc = accuracy(y_pred_logits, y)
        train_acc += acc
        
        #backpropagation
        optimiser.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimiser)
        if lr_scheduler:
            scheduler.step()
        grad_scaler.update()
    # loss, accuracy average over 1 epoch
    train_acc = train_acc / len(data_loader)
    train_loss = train_loss / len(data_loader)
    return train_loss, train_acc

def test_epoch(model: nn.Module, 
                data_loader: torch.utils.data.DataLoader,
                loss_fn: nn.Module,
                mix_precision: bool,
                device: str):
    # test setup
    test_loss, test_acc = 0, 0
    model.eval()
    
    # testing loop
    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.float().to(device)
            # mixed precision
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=mix_precision,):
                y_pred_logits = model(X).squeeze()
                loss = loss_fn(y_pred_logits, y)
            # save loss
            test_loss += loss.item()
            
            #model accuracy
            acc =  accuracy(y_pred_logits, y)
            test_acc += acc
            
    # average loss, accuracy over epoch
    test_loss = test_loss / len(data_loader)
    test_acc = test_acc / len(data_loader)
    return test_loss, test_acc

def train_model(model: nn.modules,
                train_loader: torch.utils.data.DataLoader,
                test_loader: torch.utils.data.DataLoader,
                optimiser: optim.Optimizer,
                loss_fn: nn.modules,
                scheduler: optim.Optimizer,
                grad_scaler,
                device: str,
                config):
    # track results of training
    results  = {
                "train_loss": [],
                "train_acc": [],
                "test_loss": [], 
                "test_acc":  [],
                }
    # main training loop
    for epoch in tqdm(range(config.epochs)):
        #training loss and accuracu
        train_loss, train_acc = train_epoch(model=model,
                                            data_loader=train_loader,
                                            loss_fn=loss_fn,
                                            optimiser=optimiser,
                                            scheduler=scheduler,
                                            grad_scaler=grad_scaler,
                                            device=device,
                                            mix_precision=config.mix_precision,
                                            lr_scheduler=config.lr_scheduler)
        # testing loss and accuracy
        test_loss, test_acc = test_epoch(model=model,
                                            data_loader=test_loader,
                                            loss_fn=loss_fn,
                                            device=device, 
                                            mix_precision=config.mix_precision)
        wandb.log({"train/epoch/loss": train_loss,
                    "train/epoch/acc": train_acc,
                    "train/epoch/epoch_num": epoch + 1,
                    "test/epoch/loss": test_loss,
                    "test/epoch/acc": test_acc,
                    "test/epoch/epoch_num": epoch + 1})
        # save results
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results

def save_model(path, model):
    pass

if __name__ == "__main__":
    WANDB = True
    #device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hyperparmeters
    config = SimpleNamespace(
        epochs=100,
        img_channel=1,
        num_classes=1,  
        batch_size=800,
        img_size=(224, 224),
        patch_size=16,  
        embed_dim=32,  #patch embedding dimension
        depth=6,        #number of transform encoders
        num_heads=8,    #attention heads
        mlp_ratio=4,    #the amount of hidden units in feed forward layer in proportion to the input dimension  
        # qkv_bias=True,  #bias for q, v, k calculations
        drop_prob=0.1,  #dropout prob used in the ViT network
        lr=1e-3,
        optimiser="ADAM",
        linear_embed=True,
        data_augments=[],
        weight_decay=1e-6,
        mix_precision=True,  
        lr_scheduler=True,      
        max_lr=0.01,
    )
    # load dataset
    train_loader, test_loader, _, _ = load_data(config.batch_size, config.img_size)
    # create model
    model = ViT_torch(img_size=config.img_size[0],
                patch_size=config.patch_size,
                img_channels=config.img_channel,
                num_classes=config.num_classes,
                embed_dim=config.embed_dim,
                depth=config.depth,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                drop_prob=config.drop_prob,
                linear_embed=config.linear_embed).to(device)
    
    # summarise model architecture
    summary(model, input_size=(1, 1, 224, 224), device=device)
    
    # loss function 
    loss_fn = nn.BCEWithLogitsLoss()
    
    # optimiser
    optimiser = optim.SGD(model.parameters(),
                            lr=config.lr,
                            momentum=0.9,
                            weight_decay=config.weight_decay)
    if config.optimiser == "ADAM":
        optimiser = optim.Adam(model.parameters(),
                                lr=config.lr,
                                weight_decay=config.weight_decay)
    elif config.optimiser == "ADAMW":
        optimiser = optim.AdamW(model.parameters(),
                                lr=config.lr,
                                weight_decay=config.weight_decay)
        
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimiser,
                                                max_lr=config.max_lr,
                                                steps_per_epoch=len(train_loader),
                                                epochs=config.epochs)
    # grad scaler for mixed precision
    grad_scaler = torch.cuda.amp.GradScaler(enabled=config.mix_precision)
    
    # logging
    if WANDB:
        wandb.init(project="ViT", job_type="Train", config=config)        #Login into wandb
        wandb.login(anonymous="allow")
    
    # Train the model and store the results
    results = train_model(model=model,
                            train_loader=train_loader,
                            test_loader=test_loader,
                            optimiser=optimiser,
                            loss_fn=loss_fn,
                            scheduler=scheduler,
                            grad_scaler=grad_scaler,
                            device=device,
                            config=config)
    
    #end logging
    if WANDB:
        wandb.finish()
