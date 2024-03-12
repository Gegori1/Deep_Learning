from tqdm.notebook import tqdm
import numpy as np
import torch
from time import time

class FitTrainEvalCovidNet:
  def __init__(self, model, loss_seg, loss_class, weights: list, optimizer, metrics: dict, path_save=None, device='cuda'):
    self.model = model.to(device)
    self.loss_segmentation = loss_seg
    self.loss_classification = loss_class
    self.optimizer = optimizer
    self.path_save = path_save
    self.device = device
    self.metrics_ = metrics
    self.weights = weights
    self.obtain_metrics_order()
    self.obtain_metrics()


  def obtain_metrics_order(self):
    if list(self.metrics_.keys()) != ["classification", "segmentation"]:
      raise ValueError("Metrics must be classification and segmentation")
    self.names_by_problem = {}
    for problem in self.metrics_:
      self.names_by_problem[problem] = [metric.__name__ for metric in self.metrics_[problem]]

  def obtain_metrics(self):
    self.metrics, self.names = {}, []
    for problem, names in self.names_by_problem.items():
      for i, name in enumerate(names):
        self.metrics[name] = self.metrics_[problem][i]
        self.names.append(name)

  def define_metrics_losses(self):
    # history
    self.loss_history, self.val_loss_history = [-np.inf], [-np.inf]
    self.metric_history, self.val_metric_history = {}, {}
    for metric in self.names:
      self.metric_history[metric] = [-np.inf]
      self.val_metric_history[metric] = [-np.inf]
    # by epoch
    self.train_losses, self.val_losses = [], []
    self.train_metrics, self.val_metrics = {}, {}
    for metric in self.names:
      self.train_metrics[metric] = []
      self.val_metrics[metric] = []


  def clear_metric_loss(self):
    self.train_losses, self.val_losses = [], []
    for metric in self.names:
      self.train_metrics[metric] = []
      self.val_metrics[metric] = []

  def define_history(self):
    self.history = {
        'loss': self.loss_history,
        'metric': self.metric_history,
        'val_loss': self.val_loss_history,
        'val_metric': self.val_metric_history
    }
    
  def compound_loss(self, y_pred_seg, y_true_seg, y_pred_class, y_true_class):
    loss_seg = self.loss_segmentation(y_pred_seg[:, 0, :, :], y_true_seg)
    loss_class = self.loss_classification(y_pred_class, y_true_class)
    return (loss_seg * self.weights[0]) + (loss_class * self.weights[1])
  
  def calculate_metric(self, metric, y_pred_seg, y_true_seg, y_pred_class, y_true_class):
    if metric in self.names_by_problem["segmentation"]:
      return self.metrics[metric](y_pred_seg[:, 0, :, :], y_true_seg).item()
    elif metric in self.names_by_problem["classification"]:
      return self.metrics[metric](y_pred_class, y_true_class).item()

    


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
          batch_mask = batch["mask"].to(self.device)
          batch_y = batch["class"].to(self.device)

          self.model.train()

          y_pred_seg, y_pred_class = self.model(batch_x)
          loss = self.compound_loss(y_pred_seg, batch_mask, y_pred_class, batch_y)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          self.train_losses.append(loss.item())

          for name in self.names:
            metric = self.calculate_metric(name, y_pred_seg, batch_mask, y_pred_class, batch_y)
            self.train_metrics[name].append(metric)


        self.model.eval() # <- different behaviour for some training layers
        with torch.no_grad(): # <- do not compute the comp. graph
          print("Evaluating")
          for i, batch in tqdm(enumerate(data_eval), total=tot_eval):
            batch_x = batch["image"].to(self.device)
            batch_mask = batch["mask"].to(self.device)
            batch_y = batch["class"].to(self.device)

            y_pred_seg, y_pred_class = self.model(batch_x)
            loss = self.compound_loss(y_pred_seg, batch_mask, y_pred_class, batch_y)
            self.val_losses.append(loss.item())

            for name in self.names:

              metric = self.calculate_metric(name, y_pred_seg, batch_mask, y_pred_class, batch_y)
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

        # save with respect to value in first defined metric in each problem
        metric_save = [
            (i*self.weights[0])+(j*self.weights[1]) for i, j in
            zip(self.val_metric_history[self.names_by_problem["segmentation"][0]],
            self.val_metric_history[self.names_by_problem["classification"][0]])
        ]
        current_metric = metric_save[-1]
        best_metric = max(metric_save[:-1])
        print(f"Best metric: {best_metric}")
        print(f"Current metric: {current_metric}")
        finish = time()
        print(f"Epoch time: {finish - start} s")

        if (current_metric > best_metric) and (self.path_save is not None):
          self.save_model()

      self.define_history()

      return self