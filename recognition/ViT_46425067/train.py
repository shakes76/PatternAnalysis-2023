"""
Created on Wednesday September 20 2023

Script to train, test and evaluate a visual transformer model on the ADNI 
dataset

@author: Rodger Xiang s4642506
"""
import utils
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from types import SimpleNamespace
from tqdm.auto import tqdm
from dataset import load_data
from modules import ViT
# Determines if a wandb sweep is used or not
WANDB_SWEEP = False
# hyperparmeters
CONFIG = SimpleNamespace(
        epochs=1,
        img_channel=1,      
        num_classes=1,          
        batch_size=512,         
        img_size=32,           #image sizes (img_size, img_size)
        patch_size=16,          #sizes of patchs (patch_size, patch_size)
        embed_dim=16,          #patch embedding dimension
        depth=1,                #number of transform encoders
        num_heads=1,            #attention heads
        mlp_dim=16,            #the amount of hidden units in feed forward layer in proportion to the input dimension  
        drop_prob=0.2,          #dropout prob used in the ViT network
        lr=0.001,                #learning rate for optimiser
        optimiser="SGD",        #optimiser: SGD, ADAM, ADAMW
        data_augments="H_V_R",       #specifies what data augments have been used
        weight_decay=1e-7,       #optimiser weight decay
        mix_precision=True,     #enable float16 mix precision during loss, model calcs
        lr_scheduler=True,      #enable one cycle learning rate scheduler 
        max_lr=0.01,            #max learning rate for learning scheduler
    )

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
        acc = utils.accuracy(y_pred_logits, y)
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
    """ validates the current model's accuracy and loss on a validation dataset

    Args:
        model (nn.Module): _description_
        data_loader (torch.utils.data.DataLoader): _description_
        loss_fn (nn.Module): _description_
        mix_precision (bool): _description_
        device (str): _description_

    Returns:
        _type_: _description_
    """
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
            acc =  utils.accuracy(y_pred_logits, y)
            test_acc += acc
        # average loss, accuracy over epoch
        test_loss = test_loss / len(data_loader)
        test_acc = test_acc / len(data_loader)
    return test_loss, test_acc

def train_model(config):
    with wandb.init(project="ViT", config=config):
        config = wandb.config
        # load dataset
        train_loader, val_loader, _ = load_data(config.batch_size, config.img_size)
        # create model
        model = ViT(img_size=config.img_size,
                    patch_size=config.patch_size,
                    img_channels=config.img_channel,
                    num_classes=config.num_classes,
                    embed_dim=config.embed_dim,
                    depth=config.depth,
                    num_heads=config.num_heads,
                    mlp_dim=config.mlp_dim,
                    drop_prob=config.drop_prob).to(device)
        # summarise model architecture 
        summary(model, input_size=(1, config.img_channel, config.img_size, config.img_size), device=device)
        # loss function 
        loss_fn = nn.BCEWithLogitsLoss()
        # optimiser
        optimiser = optim.SGD(model.parameters(),
                                lr=config.lr,
                                momentum=0.9,
                                weight_decay=config.weight_decay)
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimiser,
                                                    max_lr=config.max_lr,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=config.epochs)
        # grad scaler for mixed precision
        grad_scaler = torch.cuda.amp.GradScaler(enabled=config.mix_precision)
        
        # track results of training
        results  = {
                    "train_loss": [],
                    "train_acc": [],
                    "val_loss": [], 
                    "val_acc":  [],
                    }
        # main training loop
        for epoch in tqdm(range(config.epochs)):
            #train loss and accuracy over 1 epoch
            train_loss, train_acc = train_epoch(model=model,
                                                data_loader=train_loader,
                                                loss_fn=loss_fn,
                                                optimiser=optimiser,
                                                scheduler=scheduler,
                                                grad_scaler=grad_scaler,
                                                device=device,
                                                mix_precision=config.mix_precision,
                                                lr_scheduler=config.lr_scheduler)
            # test loss and accuracy over 1 epoch
            val_loss, val_acc = test_epoch(model=model,
                                                data_loader=val_loader,
                                                loss_fn=loss_fn,
                                                device=device, 
                                                mix_precision=config.mix_precision)
            #Wandb Logging
            wandb.log({"train/epoch/loss": train_loss,
                        "train/epoch/acc": train_acc,
                        "val/epoch/loss": val_loss,
                        "val/epoch/acc": val_acc,
                        "epoch_num": epoch + 1,
                        })
            # save results
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)
            
            # Stop training if accuracy drops to zero
            if val_acc <= 0.0 or train_acc <= 0.0:
                break
        
        #save model
        wandb.run.name
        utils.save_model(model=model, model_name=wandb.run.name)
            
    return results

if __name__ == "__main__":    
    #setup random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    #device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # hyperparameter sweeps or a normal run using the specified config above
    if WANDB_SWEEP:
        # Login into wandb
        wandb.login(anonymous="allow")
        # Launch agent to connect to sweep
        wandb.agent(sweep_id="rodxiang2/ViT_Sweep/k6g6ewwu", function=train_model)
    else:
        # Normal Training without wandb sweep
        results = train_model(config=CONFIG)
