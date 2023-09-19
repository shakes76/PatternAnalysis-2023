import torch
import time
import torch.nn.functional as F
# import Siamese from modules
# import Config from utils

class Train() :
    
    def __init__(self, model, dataset: Dataset, config) :
        self.model = model
        self.dataset = dataset

        # Optimisation parameters
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self) :
        self.dataset.load_train()
        self.model.train()

        for epoch in range(self.epochs) :
            start = time.time()
            for batch_idx, (data, target) in enumerate(self.dataset.get_train()) :
                self.optimiser.zero_grad()
                output = self.model(data[0], data[1])
                loss = self.criterion(output.squeeze(), target)
                loss.backward()
                self.optimiser.step()
                if batch_idx % 10 == 0 :
                    print(f"Epoch: {epoch+1}/{self.epochs} Batch: {batch_idx+1}/{len(self.dataset.get_train())} Loss: {loss.item():.6f}")
            end = time.time()
            print(f"Time: {end-start:.2f}s")
            self.test()

    def test(self) :
        self.dataset.load_test()
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad() :
            for batch_idx, (data, target) in enumerate(self.dataset.get_test()) :
                output = self.model(data[0], data[1])
                test_loss += self.criterion(output.squeeze(), target).item()
                pred = torch.round(torch.sigmoid(output))
                correct += (pred == target).float().sum()
        test_loss /= len(self.dataset.dataset)
        print(f"Test loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.dataset.dataset.get_test())} ({100.*correct/len(self.dataset.get_test()):.0f}%)")
