import torch
import os
from tqdm import tqdm
from utils import save_model, load_model, stable_sigmoid
from sklearn.metrics import hamming_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import wandb


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, device, args):
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.args = args

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for idx, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False):
            inputs, targets = batch['images'], batch['labels']
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if epoch < self.args.warmup-1:
                self.scheduler.step()
            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{self.args.epochs} - loss: {total_loss / len(self.train_loader):.4f}")
        return_dict = {
            'loss_train': total_loss / len(self.train_loader),
        }

        return return_dict

    def validate_all(self, epoch):
        self.model.eval()
        total_loss = 0
        total_num = 0
        correct_num = 0
        total_num_class = np.zeros(self.args.num_classes)
        correct_num_class = np.zeros(self.args.num_classes)

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch['images'], batch['labels']
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                total_num += targets.size(0)*targets.size(1)
                total_num_class += targets.size(0)
                outputs_logistic = stable_sigmoid(outputs.detach())
                correct_num += np.sum((outputs_logistic >
                                      0.5) == targets.cpu().numpy())
                correct_num_class += np.sum((outputs_logistic > 0.5)
                                            == targets.cpu().numpy(), axis=0)

        class_acc = correct_num_class / total_num_class
        acc_dict = {str(index): acc for index, acc in enumerate(class_acc)}

        print(
            f"Epoch {epoch+1}/{self.args.epochs} - loss: {total_loss / len(self.val_loader):.4f} - acc: {correct_num / total_num:.4f}")
        return_dict = {
            'loss_val': total_loss / len(self.val_loader),
            'acc_val': correct_num / total_num,
            **acc_dict
        }

        return return_dict

    def validate_hair(self, epoch):
        self.model.eval()
        total_loss = 0
        total_num = 0
        correct_num = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch['images'], batch['labels']
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                total_num += targets.size(0)
                outputs_softmax = torch.softmax(outputs.detach(), dim=1)
                _, predicted = torch.max(outputs_softmax, 1)
                correct_num += (predicted == targets).sum().item()

        print(
            f"Epoch {epoch+1}/{self.args.epochs} - loss: {total_loss / len(self.val_loader):.4f} - acc: {correct_num / total_num:.4f}")
        return_dict = {
            'loss_val': total_loss / len(self.val_loader),
            'acc_val': correct_num / total_num,
        }

        return return_dict

    def validate_smile(self, epoch):
        self.model.eval()
        total_loss = 0
        total_num = 0
        correct_num = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch['images'], batch['labels']
                inputs, targets = inputs.to(
                    self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                total_num += targets.size(0)
                outputs_logistic = torch.sigmoid(outputs.detach())
                predicted = torch.argmax(outputs_logistic, dim=1)
                correct_num += (predicted == targets).sum().item()

        print(
            f"Epoch {epoch+1}/{self.args.epochs} - loss: {total_loss / len(self.val_loader):.4f} - acc: {correct_num / total_num:.4f}")
        return_dict = {
            'loss_val': total_loss / len(self.val_loader),
            'acc_val': correct_num / total_num,
        }

        return return_dict

    def train(self):
        start_epoch = 0

        pth_dir = os.path.join(self.args.output_dir, 'pth')
        os.makedirs(pth_dir, exist_ok=True)

        if self.args.resume:
            start_epoch = load_model(
                self.model, self.optimizer, self.scheduler, self.args.resume)
        for epoch in range(start_epoch, self.args.epochs):
            trian_dict = self.train_epoch(epoch)
            if self.args.task == "all":
                val_dict = self.validate_all(epoch)
            elif self.args.task == "hair":
                val_dict = self.validate_hair(epoch)
            elif self.args.task == "smile":
                val_dict = self.validate_smile(epoch)

            lr = self.scheduler.get_last_lr()[0]
            if self.args.wandb:
                wandb.log({**trian_dict, **val_dict, 'lr': lr})

            if epoch > self.args.warmup and (epoch-self.args.warmup) % 5 == 0:
                self.scheduler.step()

        pth_path = os.path.join(pth_dir, f"model_epoch_{epoch}.pth")
        save_model(self.model, self.optimizer,
                   self.scheduler, epoch, pth_path)
