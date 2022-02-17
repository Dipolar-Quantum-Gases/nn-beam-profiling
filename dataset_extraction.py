# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 23:49:44 2022

@author: hofer
"""

import os
import subprocess

class dataset_extractor():

  def __init__(self, dataset, url_string, zip_dir, unzip_dir):
    zip_path = zip_dir / (dataset + '.zip')

    self.make_local_data_dir(zip_dir)
    self.download_zip(str(zip_path), url_string)
    self.make_local_data_dir(unzip_dir)
    self.unpack_zip(str(zip_path), str(unzip_dir))


  def make_local_data_dir(self, local_data_path):
      if not os.path.isdir(local_data_path):
        os.makedirs(local_data_path)


  def download_zip(self, zip_path, url_string):
    result = subprocess.run(['wget', '-O', zip_path, url_string])
    return result

  def unpack_zip(self, zip_path, unzip_path):
    result = subprocess.run(['unzip', str(zip_path), '-d', str(unzip_path)])
    return result