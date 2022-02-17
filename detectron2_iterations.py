# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 23:39:47 2022

@author: hofer
"""

from math import ceil

class iteration_handler():
  """
  Detectron2 uses iterations (a single batch) rather than the standard epoch
  (all the batches in the training set). This class converts between the two.
  """

  def get_learning_rate_schedule(self, lr_step_iterations, training_iterations):
      max_steps = int(ceil(training_iterations / lr_step_iterations))
      schedule = [lr_step_iterations * (i + 1) - 1 for i in range(0, max_steps)]
      return schedule

  
  def epochs_to_iterations(self, epochs, num_imgs, batch_size):
      return int(epochs * (num_imgs / batch_size))


  def get_iters(self, num_imgs, train_epochs, eval_epochs, lrs_step, batch_size):
    train_iters = int(self.epochs_to_iterations(train_epochs, num_imgs, batch_size))
    eval_iters = int(self.epochs_to_iterations(eval_epochs, num_imgs, batch_size))
    lr_step_iters = int(self.epochs_to_iterations(lrs_step, num_imgs, batch_size))
    lr_sched = self.get_learning_rate_schedule(lr_step_iters, train_iters)
    return train_iters, eval_iters, lr_sched