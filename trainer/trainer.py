import torch
import torch.nn as nn
from utils.metrics import accuracy

class TrainerClassification():
    def __init__(self, 
                 model: nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 optimizer,
                 criterion,
                 num_epochs: int,
                 device=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = num_epochs
        self.model.to(self.device)

    def training_step(self, batch):
        images, labels = batch 
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model.forward(images)
        loss = self.criterion(out, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        images, labels = images.to(self.device), labels.to(self.device)
        out = self.model.forward(images)
        acc = accuracy(out, labels)
        return {'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_acc']))
        
    def train_mode_on(self):
        self.model.train()
    
    def eval_mode_on(self):
        self.model.eval()
        
    @torch.no_grad()
    def evaluate(self):
        self.eval_mode_on()
        outputs = [self.validation_step(batch) for batch in self.val_dataloader]
        return self.validation_epoch_end(outputs)
    
    def fit(self):
        history = []
        for epoch in range(self.epochs):      
            self.train_mode_on()
            train_losses = []
            for batch in self.train_dataloader:
                loss = self.training_step(batch)
                train_losses.append(loss)
                
            result = self.evaluate()
            result['train_loss'] = torch.stack(train_losses).mean().item()
            self.epoch_end(epoch, result)
            history.append(result)
        return history