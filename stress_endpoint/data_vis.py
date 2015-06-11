#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import copy
import operator
import numpy
from random import shuffle
from my_record import MyRecord, interval_base
import pickle
import matplotlib
from sklearn.preprocessing import normalize
from logitboost import dataset_activities, stress_mappings

records = list()
axes_array = list()

def parallel_coordinates(data_sets, style=None):
  dims = len(data_sets[0])
  x = range(dims)
  fig, axes = plt.subplots(1, dims - 1, sharey=False)

  if style is None:
    style = ['r-'] * len(data_sets)

  # Calculate the limits on the data
  min_max_range = list()
  for m in zip(*data_sets):
    mn = min(m)
    mx = max(m)
    if mn == mx:
        mn -= 0.5
        mx = mn + 1.
    r = float(mx - mn)
    min_max_range.append((mn, mx, r))

  # Normalize the data sets
  norm_data_sets = list()
  for ds in data_sets:
    nds = [(value - min_max_range[dimension][0]) /
            min_max_range[dimension][2]
            for dimension, value in enumerate(ds)]
    norm_data_sets.append(nds)
  data_sets = norm_data_sets

  # Plot the datasets on all the subplots
  for i, ax in enumerate(axes):
    for dsi, d in enumerate(data_sets):
      ax.plot(x, d, style[dsi], linewidth=0.5)
    ax.set_xlim([x[i], x[i + 1]])

  # Set the x axis ticks
  for dimension, (axx, xx) in enumerate(zip(axes, x[:-1])):
    axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
    ticks = len(axx.get_yticklabels())
    labels = list()
    step = min_max_range[dimension][2] / (ticks - 1)
    mn = min_max_range[dimension][0]
    for i in range(ticks):
      v = mn + i * step
      labels.append('%4.2f' % v)
    axx.set_yticklabels(labels)

  # Move the final axis' ticks to the right-hand side
  axx = plt.twinx(axes[-1])
  dimension += 1
  axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
  ticks = len(axx.get_yticklabels())
  step = min_max_range[dimension][2] / (ticks - 1)
  mn = min_max_range[dimension][0]
  labels = ['%4.2f' % (mn + i * step) for i in range(ticks)]
  axx.set_yticklabels(labels)

  # Stack the subplots
  plt.subplots_adjust(wspace=0)
  # plt.xlabel(str(axes_array))
  return plt

def normalize_features(norm_features, stress_features):
  feature_array = [list(norm_features.values()), list(stress_features.values())]
  for record in records:
    feature_array.append(list(record.features.values()))
  return normalize(feature_array, norm='l2', axis=0, copy=False)

def sort_features():
  stress_features = dict()
  norm_features = dict()
  stress_count = 0
  norm_count = 0

  for key in records[0].features.keys():
    stress_features[key] = 0
    norm_features[key] = 0

  for record in records:
    if record.my_class == 1:
      stress_count += 1
      for key in record.features.keys():
        stress_features[key] += record.features[key]
    else:
      norm_count += 1
      for key in record.features.keys():
        norm_features[key] += record.features[key]

  for key in records[0].features.keys():
    norm_features[key] /= norm_count
    stress_features[key] /= stress_count

  return (norm_features, stress_features)

def load():
  my_records_band = pickle.load(open("band_records.pickle", "rb"))

  for index, record in enumerate(my_records_band):
    record.my_class = stress_mappings[dataset_activities[index]]
    records.append(record)

def sort_axes(sort_type, axes_array, norm_features, stress_features):
  deltas_dict = dict()
  for index, key in enumerate(axes_array):
    if sort_type == "absolute":
      deltas_dict[key] = -abs(norm_features[index] - stress_features[index])
    elif sort_type == "norm":
      deltas_dict[key] = stress_features[index]
    elif sort_type == "stress":
      deltas_dict[key] = norm_features[index]
    else:
      deltas_dict[key] = norm_features[index] - stress_features[index]

  sorted_deltas = sorted(deltas_dict.items(), key=operator.itemgetter(1))

  new_axes = list()
  for (key, value) in sorted_deltas:
    new_axes.append(key)
  return new_axes

def run(sort_type, normalized=True):
  (norm_features, stress_features) = sort_features()
  axes_array = list(norm_features.keys())
  dataset = normalize_features(norm_features, stress_features)
  (norm_features, stress_features) = dataset[:2]
  dataset = dataset[2:]

  axes_array = sort_axes(sort_type, axes_array, norm_features, stress_features)
  axes_array.remove("p_nn_50")
  print(axes_array)

  feature_array = list()
  for record in records:
    feature_list = list()
    for key in axes_array:
      feature_list.append(record.features[key])
    feature_array.append(feature_list)

  if normalized:
    feature_array = normalize(feature_array, norm='l2', axis=0, copy=False)

  colors = list()
  for record in records:
    if record.my_class == 1:
      colors.append("r")
    else:
      colors.append("b")

  parallel_coordinates(feature_array, style=colors).show()
