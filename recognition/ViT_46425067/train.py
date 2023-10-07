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
from vit_pytorch import ViT, SimpleViT
from vit_pytorch.deepvit import DeepViT
import utils

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
        train_loader, test_loader = load_data(config.batch_size, config.img_size)
        # create model
        model = ViT_torch(img_size=config.img_size,
                    patch_size=config.patch_size,
                    img_channels=config.img_channel,
                    num_classes=config.num_classes,
                    embed_dim=config.embed_dim,
                    depth=config.depth,
                    num_heads=config.num_heads,
                    mlp_dim=config.mlp_dim,
                    drop_prob=config.drop_prob,
                    linear_embed=config.linear_embed).to(device)
        # model = ViT(
        #     image_size=config.img_size,
        #     patch_size=config.patch_size,
        #     num_classes=1,
        #     dim=config.embed_dim,
        #     depth=config.depth,
        #     heads=config.num_heads,
        #     mlp_dim=config.mlp_dim,
        #     dropout=config.drop_prob,
        #     emb_dropout=config.drop_prob,
        #     channels=1).to(device)
        # summarise model architecture
        summary(model, input_size=(1, 1, 256, 256), device=device)
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
                                    weight_decay=config.weight_decay,)
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
            if test_acc <= 0.0 or train_acc <= 0.0:
                break
    return results

if __name__ == "__main__":
    WANDB_SWEEP = False
    
    #setup random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    #device agnostic code
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hyperparmeters
    config = SimpleNamespace(
        epochs=100,
        img_channel=1,      
        num_classes=1,          
        batch_size=512,         
        img_size=128,           #image sizes (img_size, img_size)
        patch_size=64,          #sizes of patchs (patch_size, patch_size)
        embed_dim=128,          #patch embedding dimension
        depth=6,                #number of transform encoders
        num_heads=8,            #attention heads
        mlp_dim=1024,            #the amount of hidden units in feed forward layer in proportion to the input dimension  
        drop_prob=0,          #dropout prob used in the ViT network
        lr=0.001,                #learning rate for optimiser
        optimiser="SGD",        #optimiser: SGD, ADAM, ADAMW
        linear_embed=True,      #to use linear embed instead of conv method
        data_augments=[],       #specifies what data augments have been used
        weight_decay=0.0,       #optimiser weight decay
        mix_precision=True,     #enable float16 mix precision during loss, model calcs
        lr_scheduler=True,      #enable one cycle learning rate scheduler 
        max_lr=0.01,            #max learning rate for learning scheduler
    )
    # logging with hyperparameter sweeps or normal runs
    if WANDB_SWEEP:
        # Login into wandb
        wandb.login(anonymous="allow")
        # Launch agent to connect to sweep
        wandb.agent(sweep_id="rodxiang2/ViT_Sweep/k6g6ewwu", function=train_model)
    else:
        # Normal Training without wandb sweep
        results = train_model(config=config)
