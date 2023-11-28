"""Load data from the specified paths and format them for training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np
import tensorflow as tf

LABEL_NAME = "SEMS"
DATA_NAME = "rows"


class DataLoader(object):
  """Loads data and prepares for training."""

  def __init__(self, train_data_path, valid_data_path, test_data_path,
               seq_length):
    self.dim = 10
    self.seq_length = seq_length
    self.label2id = {"0": 0, "1": 1, "2": 2, "3": 3,  "4": 4,  "5": 5,  "6": 6,  "7": 7,  "8": 8}
    self.train_data, self.train_label, self.train_len = self.get_data_file(
        train_data_path, "train")
    self.valid_data, self.valid_label, self.valid_len = self.get_data_file(
        valid_data_path, "valid")
    self.test_data, self.test_label, self.test_len = self.get_data_file(
        test_data_path, "test")

  def get_data_file(self, data_path, data_type):
    """Get train, valid and test data from files."""
    data = []
    label = []
    with open(data_path, "r") as f:
      lines = f.readlines()
      for idx, line in enumerate(lines):
        # for each line from the JSON, which represents a Dataset in Dict form, convert it back into python objects 
        dic = json.loads(line)
        data.append(dic[DATA_NAME])
        label.append(dic[LABEL_NAME])
    length = len(label)
    print(data_type + "_data_length:" + str(length))
    return data, label, length

  def sliceData(self, data, seq_length):
    """Get slice of Data"""
    slicedData = []
    tmp_data = []
    tmp_data[(seq_length -
              min(len(data), seq_length)):] = data[:min(len(data), seq_length)]
    tmp_data[:min(len(data), seq_length)] = data[:min(len(data), seq_length)]
    slicedData.append(tmp_data)
    return slicedData

  def format_support_func(self, slice_num, length, data, label):
    """Support function for format.(Helps format train, valid and test.)"""
    # Add 1 padding, initialize data and label
    length *= slice_num
    features = np.zeros((length, self.seq_length, self.dim))
    labels = np.zeros(length)
    # Get slices for train, valid and test
    for idx, (data, label) in enumerate(zip(data, label)):
      sliced_data = self.sliceData(data, self.seq_length)
      for num in range(slice_num):
        features[slice_num * idx + num] = sliced_data[num]
        labels[slice_num * idx + num] = self.label2id[label]
    # Turn into tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (features, labels.astype("int32")))
    print(dataset)
    return length, dataset
  
  def format(self):
    """Format data(including padding, etc.) and get the dataset for the model."""
    slice_num = 1
    self.train_len, self.train_data = self.format_support_func(
        slice_num, self.train_len, self.train_data, self.train_label)
    self.valid_len, self.valid_data = self.format_support_func(
        slice_num, self.valid_len, self.valid_data, self.valid_label)
    self.test_len, self.test_data = self.format_support_func(
        slice_num, self.test_len, self.test_data, self.test_label)
