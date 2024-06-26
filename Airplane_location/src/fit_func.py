from tqdm.notebook import tqdm
import numpy as np
import torch
from time import time

class FitTrainEval:
  def __init__(self, model, S:int, B:int, loss, optimizer, metrics, path_save=None, save_best=True, device='cuda'):
    self.model = model.to(device)
    self.S = S
    self.B = B
    self.criterion = loss
    self.optimizer = optimizer
    self.path_save = path_save
    self.save_best = save_best
    self.device = device
    self.names = [metric.__name__ for metric in metrics]
    self.metrics = {name: metric for name, metric in zip(self.names, metrics)}

  def define_metrics_losses(self):
    # history
    self.loss_history, self.val_loss_history = [-np.inf], [-np.inf]
    self.metric_history, self.val_metric_history = {}, {}
    for name in self.names:
      self.metric_history[name] = [-np.inf]
      self.val_metric_history[name] = [-np.inf]
    # by epoch
    train_losses, val_losses = [], []
    train_metrics, val_metrics = {}, {}
    for name in self.names:
      train_metrics[name] = []
      val_metrics[name] = []
    self.train_losses = train_losses
    self.val_losses = val_losses
    self.train_metrics = train_metrics
    self.val_metrics = val_metrics

  def clear_metric_loss(self):
    self.train_losses, self.val_losses = [], []
    for name in self.names:
      self.train_metrics[name], self.val_metrics[name] = [], []

  def define_history(self):
    self.history = {
        'loss': self.loss_history,
        'metric': self.metric_history,
        'val_loss': self.val_loss_history,
        'val_metric': self.val_metric_history
    }

  def save_model(self):
    objects = {
          'model': self.model.state_dict(),
          'optimizer': self.optimizer.state_dict(),
          'loss_history': self.loss_history,
          'metric_history': self.metric_history,
          'val_loss_history': self.val_loss_history,
          'val_metric_history': self.val_metric_history
      }
    print("Saving model")
    torch.save(objects, self.path_save)


  def fit_train_eval(self, data_train, data_eval, epochs):
      tot_train, tot_eval = len(data_train), len(data_eval)

      self.define_metrics_losses()

      for epoch in range(epochs):
        start = time()
        self.clear_metric_loss()
        print(f"Epoch {epoch+1}/{epochs}")
        print("Training")
        for i, batch in tqdm(enumerate(data_train), total=tot_train):
          batch_x = batch["image"].to(self.device)
          batch_y = batch["target"].to(self.device)

          self.model.train()

          y_pred = self.model(batch_x)
          loss = self.criterion(y_pred, batch_y)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          self.train_losses.append(loss.item())
          for name in self.names:
            y_pred = y_pred.view(-1, self.S, self.S, 5*self.B)
            metric = self.metrics[name](y_pred, batch_y).item()
            self.train_metrics[name].append(metric)


        self.model.eval() # <- different behaviour for some training layers
        with torch.no_grad(): # <- do not compute the comp. graph
          print("Evaluating")
          for i, batch in tqdm(enumerate(data_eval), total=tot_eval):
            batch_x = batch["image"].to(self.device)
            batch_y = batch["target"].to(self.device)

            y_pred = self.model(batch_x)
            loss = self.criterion(y_pred, batch_y)

            self.val_losses.append(loss.item())
            for name in self.names:
              y_pred = y_pred.view(-1, self.S, self.S, 5*self.B)
              metric = self.metrics[name](y_pred, batch_y).item()
              self.val_metrics[name].append(metric)

        self.loss_history.append(np.mean(self.train_losses))
        for name in self.names:
          self.metric_history[name].append(np.mean(self.train_metrics[name]))

        self.val_loss_history.append(np.mean(self.val_losses))
        for name in self.names:
          self.val_metric_history[name].append(np.mean(self.val_metrics[name]))

        print(f"Train loss: {self.loss_history[-1]} \t Val loss: {self.val_loss_history[-1]}")
        for name in self.names:
          print(f"Train {name}: {self.metric_history[name][-1]} \t Val {name}: {self.val_metric_history[name][-1]}")
        print()

        # save with respect to value in first defined metric
        metric_save = self.val_metric_history[self.names[0]]
        current_metric = self.val_metric_history[self.names[0]][-1]
        best_metric = max(metric_save[:-1])
        print(f"Best metric: {best_metric}")
        print(f"Current metric: {current_metric}")
        finish = time()
        print(f"Epoch time: {finish - start} s")

        print(f"Checker: {current_metric} > {best_metric}")
        if (current_metric > best_metric) and (self.path_save is not None) and self.save_best:
          self.save_model()
        elif (self.path_save is not None) and (not self.save_best):
          self.save_model()

      self.define_history()

      return self