import os
import re
import string
import nltk
import math
from random import shuffle
from sklearn import svm
import numpy
from datetime import timedelta

# time in seconds of how big an accepted interval must be
interval_base = 120
SECONDS_PER_MINUTE = 60

class MyRecord:

  def __init__(self, my_class, my_file):
    self.intervals = list()
    self.features = dict()
    self.preset_features = dict()
    self.my_class = my_class
    self.id = my_file.split("_")[1].split(".")[0]
    self.onset = timedelta(seconds=interval_base)

  def process(self, file_p):
    lines = file_p.readlines()[1:]
    for line in lines:
      words = line.split()
      curr_time = self.get_curr_time(words[0].split(":"))
      self.intervals.append((curr_time, float(words[2])))
    if (len(self.intervals) > 0):
      self.generate_features()

  def get_curr_time(self, curr_time):
    # spontaneous vt/vf stored in different format
    if (self.my_class == "VF_RR" or self.my_class == "VT_RR"):
      curr_time = timedelta(minutes=float(curr_time[0]),
                            seconds=float(curr_time[1].split(".")[0]),
                            milliseconds=(float(curr_time[1].split(".")[1])
                                          * 1000)
                            )
      return curr_time
    if (len(curr_time) == 2):
      curr_time = timedelta(seconds=float(curr_time[0]),
                            milliseconds=float(curr_time[1]))
    else:
      curr_time = timedelta(minutes=float(curr_time[0]),
                            seconds=float(curr_time[1]),
                            milliseconds=float(curr_time[2]))
    return curr_time

  def generate_median(self):
    sorted_list = sorted(self.intervals)
    length = len(sorted_list)
    median = 0
    if (length % 2 == 0):
      index = int(length / 2)
      median = (sorted_list[index][1] + sorted_list[index + 1][1]) / 2
    else:
      index = int((length + 1) / 2)
      median = sorted_list[index][1]
    self.features['median'] = median

  def generate_rmsdd(self):
    last_time = 0
    rmsdd_sum = 0
    sdsd_sum = 0
    nn_50 = 0

    for (time, item) in self.intervals:
      diff = last_time - item
      rmsdd_sum += pow(diff, 2)
      sdsd_sum += math.sqrt(pow(diff, 2))
      if (math.sqrt(pow(diff, 2)) > 0.05):
        nn_50 += 1

    rmsdd_sum /= len(self.intervals)
    sdsd_sum /= len(self.intervals)
    self.features['rmsdd'] = math.sqrt(rmsdd_sum)
    self.features['sdsd'] = sdsd_sum
    self.features['nn_50'] = nn_50
    self.features['p_nn_50'] = nn_50 / len(self.intervals)

  def generate_outlier(self):
    num_outliers = 0
    last_interval = self.intervals[0][1]
    for (time, item) in self.intervals:
      diff = abs(item - last_interval) / last_interval
      if diff > 1.2 or diff < 0.8:
        num_outliers += 1

    self.features['outlier'] = num_outliers

  def generate_sdhr(self):
    mean_hr = 0
    sdhr = 0
    hrs = list()

    for (time, item) in self.intervals:
      hr = SECONDS_PER_MINUTE / item
      hrs.append(hr)
      mean_hr += hr
    mean_hr /= len(self.intervals)

    for hr in hrs:
      sdhr += pow(mean_hr - hr, 2)
    sdhr /= math.sqrt(len(hrs))

    self.features['sdhr'] = sdhr

  def generate_lf(self):
    pvlf = 0
    plf = 0
    a_total = 0

    self.features['pvlf'] = pvlf
    self.features['plf'] = plf
    self.features['a_total'] = a_total

  def generate_presets(self):
    self.preset_features['outlier'] = self.features['outlier']
    self.preset_features['sdhr'] = self.features['sdhr']
    self.preset_features['atotal'] = self.features['atotal']
    self.preset_features['pvlf'] = self.features['pvlf']
    self.preset_features['plf'] = self.features['plf']
    self.preset_features['sd1'] = self.features['sdsd']
    # self.preset_features['alpha1']
    # features of murukesan et al
    # aTotal = total abs power of vlf, lf, and hf bands (very low, low, high frequency)
    # pvlf = percentage of absolute power of very low frequency band over total power (atotal)
    # plf = peak frequency in low frequency band
    # sd1 = instantaneous std dev of instantaneous beat-to-beat variability (short term variability)
    # alpha1 = the strength of the short term correlation properties of RR interval data

  def generate_features(self):
    my_sum = 0
    my_max = 0
    my_min = 1000
    std_dev = 0

    for (time, item) in self.intervals:
      my_sum += item
      if (item > my_max):
        my_max = item
      if (item < my_min):
        my_min = item

    mean = float(my_sum / len(self.intervals))

    for (time, item) in self.intervals:
      std_dev += pow(item - mean, 2)
    std_dev = math.sqrt(std_dev / len(self.intervals))

    self.features['mean'] = mean
    self.features['min'] = my_min
    self.features['max'] = my_max
    self.features['std_dev'] = std_dev
    self.generate_median()
    self.generate_rmsdd()
    self.generate_outlier()
    self.generate_sdhr()
    # self.generate_lf()
    # self.generate_presets()
