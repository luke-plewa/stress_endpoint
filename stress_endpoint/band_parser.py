import os
import re
import string
import nltk
import math
import pickle
from my_record import MyRecord, interval_base

rootdir = '/Users/luke/Documents/thesis_sca/zagreus/samples/BAND_HR'
outputdir = '/Users/luke/Documents/thesis_sca/zagreus/samples/BAND_NORM'
records = list()

# converts band hr data files to usable hr data files

def process(file_p):
  heart_rates = list()
  temperatures = list()

  lines = file_p.readlines()
  for line in lines:
    words = line.split()
    if words[0][0] != '+':
      heart_rates.append("%.3f\n" % (60.0 / int(words[0])))

  return heart_rates

def write_file(heart_rates, index):
  file_p = open(os.path.join(outputdir, "BAND_" + str(index) + ".txt"), 'w')
  file_p.writelines(heart_rates)
  file_p.close()

def create_record(heart_rates, my_file):
  record = MyRecord("NORM_RR", my_file)
  record.filename = my_file
  curr_time = 0
  for rate in heart_rates:
    curr_time += float(rate)
    record.intervals.append((curr_time, float(rate)))
  record.generate_features()
  records.append(record)

def parse():
  index = 0
  for subdir, dirs, files in os.walk(rootdir):
    for my_file in files:
      if my_file != 'dataset_description.txt':
        filename = os.path.join(subdir, my_file)
        file_p = open(filename, 'r')
        heart_rates = process(file_p)
        print(heart_rates)
        file_p.close()

        write_file(heart_rates, index)
        index += 1

        create_record(heart_rates, "BAND_" + str(index) + ".txt")

  pickle.dump(records, open("band_records.pickle", "wb"))

parse()
