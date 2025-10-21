import tensorflow as tf
from keras import models

import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from prefab.processor import *

class Predictor():
    def __init__(self, type, fab, process, version, model_nums):
        self.type = type
        self.fab = fab
        self.process = process
        self.version = version
        self.model_nums = model_nums

        self.models = []
        for j in self.model_nums:
            self.models.append(models.load_model("../models/" + type + "_" + fab + "_"
                + process + "_" + str(version) + "_" + str(j) + ".pb"))

        self.slice_length = int(np.sqrt(self.models[0].weights[-1].shape)[0])
        self.slice_size = (self.slice_length, self.slice_length)

    def predict(self, device, step_length, binary=False):
        # slice image up
        x_slices_4D = np.lib.stride_tricks.sliding_window_view(device, self.slice_size)[::step_length, ::step_length]
        x_slices = x_slices_4D.reshape(-1, *self.slice_size)
        x_slices = tf.reshape(x_slices, [len(x_slices), self.slice_length, self.slice_length, 1])

        # make predictions
        y_sum = 0
        for model in self.models:
            y = model(x_slices)
            y = tf.cast(y, tf.float64)
            y_sum += y
        y_slices = y_sum/len(self.models)
        y_slices = np.squeeze(y_slices).reshape(x_slices_4D.shape)

        # stitch slices back together (needs a better method)
        y = np.zeros(device.shape)
        avg_mtx = np.zeros(device.shape)
        for k in range(0, device.shape[0]-self.slice_length+1, step_length):
            for j in range(0, device.shape[1]-self.slice_length+1, step_length):
                y[k:k+self.slice_length, j:j+self.slice_length] += y_slices[k//step_length,j//step_length]
                avg_mtx[k:k+self.slice_length, j:j+self.slice_length] += np.ones(self.slice_size)
        prediction = y/avg_mtx

        # binarize or leave as raw (showing uncertainty)
        if binary == True:
            prediction = binarize(prediction)

        return prediction
        