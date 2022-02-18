# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 01:05:12 2022

@author: hofer
"""
import pandas as pd

class get_training_results():
  """Detectron 2 spits out a training/validation metrics json file. This is kinda
  a jumble and here we parse it to get out a training df, an overall eval df and
  then we split the eval df into eval dfs for each validation dataset
  """  

  def get_evaluation_metrics(self, results_dir):
     eval_results = pd.read_json(results_dir + '/metrics.json', lines=True)
     return eval_results


  def get_results(self, results_dir, eval_dir=None):
    eval_results = self.get_evaluation_metrics(results_dir)
    return eval_results


  def sort_organize_df(self, dataframe):
    dataframe = dataframe.dropna(axis=1, how='any') #drop columns with na values
    dataframe = dataframe.sort_values(by=['iteration']) #sort by training iteration
    dataframe = dataframe.reset_index(drop=True)
    return dataframe
  

  def get_evaluation_df(self, metricsDF):
    """Here we split the evaluation portion of the metrics df off
    """
    evaluation_df = metricsDF.dropna(subset=["bbox/APs"]) #split validation metric
    evaluation_df = self.sort_organize_df(evaluation_df)
    return evaluation_df


  def get_training_df(self, metricsDF):
    """Here we split the training portion of the metrics df off
    """
    training_df = metricsDF.dropna(subset=["total_loss"]) # split training metrics
    training_df = self.sort_organize_df(training_df)
    return training_df


  def get_metrics(self, metrics_path):
    '''Does all the splitting of the metrics df into its respective components.'''
    metricsDF = self.get_results(metrics_path)
    training_df = self.get_training_df(metricsDF)
    evaluation_df = self.get_evaluation_df(metricsDF)
    return training_df, evaluation_df