import torch
import time
import torch.nn.functional as F
# import Siamese from modules
# import Config from utils

class Train() :
    
    def __init__(self, model, dataloader, config) :
        self.model = model
        self.dataloader = dataloader

        # Optimisation parameters
        self.best = config.best
        self.best
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
    
    def train(self) :
        self.model.train()

        for epoch in range(self.epochs) :
            start = time.time()
            for batch_idx, (data, target) in enumerate(self.dataloader) :
                self.optimiser.zero_grad()
                output = self.model(data[0], data[1])
                loss = F.binary_cross_entropy_with_logits(output, target)
                loss.backward()
                self.optimiser.step()
                if batch_idx % 10 == 0 :
                    print(f"Epoch: {epoch+1}/{self.epochs} Batch: {batch_idx+1}/{len(self.dataloader)} Loss: {loss.item():.6f}")
            end = time.time()
            print(f"Time: {end-start:.2f}s")
            self.test()
