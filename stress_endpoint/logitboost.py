import copy
import operator
import numpy
from random import shuffle
from my_record import MyRecord, interval_base
import pickle

# from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import Perceptron
from logitboost_sample import LogitBoostClassifier
# svm, knn, MLP, ANN
from sklearn.metrics import f1_score, precision_score, recall_score


dataset_activities = [
  1,
  1,
  0,
  9,
  9,
  10,
  11,
  11,
  11,
  11,
  11,
  11,
  11,
  11,
  11,
  11,
  4,
  2,
  9,
  0,
  7,
  7,
  9,
  7,
  7,
  7,
  7,
  7,
  1,
  1,
  1,
  1,
  1,
  1,
  9,
  9,
  10,
  10,
  10,
  1,
  1,
  1,
  1,
  1,
  8,
  8,
  8,
  8,
  11,
  11,
  11,
  11,
  11,
  11,
  11,
  8,
  8,
  4,
  4,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  7,
  5,
  5,
  5,
  2,
  2,
  2,
  2,
  10,
  10,
  10,
  10,
  10,
  9,
  9,
  5,
  5,
  5,
  2,
  2,
  2,
  10,
  10,
  10,
  5,
  5,
  5,
  2,
  2,
  0,
  0,
  11
]

stress_mappings = [
  0,
  1,
  0,
  0,
  0,
  1,
  0,
  1,
  0,
  0,
  1,
  1,
]

rootdir = '/Users/luke/Documents/ekg_analysis/zagreus/samples/'
repositories = ['SDDB_RR', 'NORM_RR', 'VF_RR', 'VT_RR']
records = list()


estimator_list = numpy.array([DecisionTreeClassifier(max_depth=1),
                              MultinomialNB(),
                              Perceptron(),
                              SVC(kernel="linear", C=0.1)])
boosting_classifier_size = 20

def load():
  my_records_band = pickle.load(open("band_records.pickle", "rb"))

  for index, record in enumerate(my_records_band):
    record.my_class = stress_mappings[dataset_activities[index]]
    records.append(record)

class MyBoostClassifier(LogitBoostClassifier):
  def __init__(self,
               base_estimator=DecisionTreeClassifier(max_depth=1),
               n_estimators=50,
               estimator_params=tuple(),
               learning_rate=1.,
               algorithm='SAMME.R'):
    self.base_estimator = base_estimator
    self.n_estimators = n_estimators
    self.estimator_params = estimator_params
    self.learning_rate = learning_rate
    self.algorithm = algorithm

  def set_estimators(self, estimators):
    self.n_estimators = len(estimators)
    self.estimators_ = numpy.array(estimators)

  def fit(self, X, y, sample_weight=None):
    # Check parameters.
    if self.learning_rate <= 0:
      raise ValueError("learning_rate must be greater than zero.")

    if sample_weight is None:
      # Initialize weights to 1 / n_samples.
      sample_weight = numpy.empty(X.shape[0], dtype=numpy.float)
      sample_weight[:] = 1. / X.shape[0]
    else:
      # Normalize existing weights.
      sample_weight = sample_weight / sample_weight.sum(dtype=numpy.float64)

    # Check that the sample weights sum is positive.
    if sample_weight.sum() <= 0:
      raise ValueError(
        "Attempting to fit with a non-positive "
        "weighted number of samples.")

    # Clear any previous fit results.
    # self.estimators_ = []
    self.estimator_weights_ = numpy.zeros(self.n_estimators, dtype=numpy.float)
    self.estimator_errors_ = numpy.ones(self.n_estimators, dtype=numpy.float)

    for (iboost, estimator) in enumerate(self.estimators_):
      # Fit the estimator.
      estimator.fit(X, y, sample_weight=sample_weight)

      if iboost == 0:
        self.classes_ = getattr(estimator, 'classes_', None)
        self.n_classes_ = len(self.classes_)

      # Generate estimator predictions.
      y_pred = estimator.predict(X)

      # Instances incorrectly classified.
      incorrect = y_pred != y

      # Error fraction.
      estimator_error = numpy.mean(
        numpy.average(incorrect, weights=sample_weight, axis=0))

      # Boost weight using multi-class AdaBoost SAMME alg.
      estimator_weight = self.learning_rate * (
        numpy.log((1. - estimator_error) / estimator_error) +
        numpy.log(self.n_classes_ - 1.))

      # Only boost the weights if there is another iteration of fitting.
      if not iboost == self.n_estimators - 1:
        # Only boost positive weights (logistic loss).
        sample_weight *= numpy.log(1 + numpy.exp(estimator_weight * incorrect *
                                   ((sample_weight > 0) |
                                    (estimator_weight < 0))))

      self.estimator_weights_[iboost] = estimator_weight
      self.estimator_errors_[iboost] = estimator_error

def random_split(train_split_size):
  stress_count = 0
  for record in records:
    if record.my_class == 1:
      stress_count += 1
  print(stress_count, len(records))
  shuffle(records)
  split = int(len(records) * train_split_size)
  train_set = records[:split]
  test_set = records[split:]
  print("split:", train_split_size, "train:", len(train_set),
                                    "test:", len(test_set))
  return (train_set, test_set)

def report():
  (train_set, test_set) = random_split(0.66)
  train_features = list()
  train_classes = list()
  test_features = list()
  test_classes = list()

  for record in train_set:
    train_features.append(list(record.features.values()))
    train_classes.append(record.my_class)
  for record in test_set:
    test_features.append(list(record.features.values()))
    test_classes.append(record.my_class)

  svm_classifier = SVC(kernel="linear", C=0.1)
  svm_classifier.fit(train_features, train_classes)
  print("linear kernel svm accuracy: " +
        str(svm_classifier.score(test_features, test_classes)))
  print("svm f1 score: " +
        str(f1_score(numpy.array(test_classes),
                     svm_classifier.predict(numpy.array(test_features)),
                     pos_label=1)))

  nb_classifier = MultinomialNB()
  nb_classifier.fit(train_features, train_classes)
  print("naive bayes accuracy: " +
        str(nb_classifier.score(test_features, test_classes)))
  print("naive bayes score: " +
        str(f1_score(numpy.array(test_classes),
                     nb_classifier.predict(numpy.array(test_features)),
                     pos_label=1)))

  # svm_classifier = SVC(kernel="linear", C=0.1)
  # svm_classifier.fit(train_features, train_classes)
  # print("linear kernel svm accuracy: " +
  #       str(svm_classifier.score(test_features, test_classes)))
  # print("svm accuracy: " +
  #       str(svm_classifier.score(numpy.array(test_features),
  #                                numpy.array(test_classes))))
  # print("svm precision: " +
  #       str(precision_score(numpy.array(test_classes),
  #                           svm_classifier.predict(numpy.array(test_features)),
  #                           pos_label=1)))
  # print("svm recall: " +
  #       str(recall_score(numpy.array(test_classes),
  #                        svm_classifier.predict(numpy.array(test_features)),
  #                        pos_label=1)))
  # print("svm f1 score: " +
  #       str(f1_score(numpy.array(test_classes),
  #                    svm_classifier.predict(numpy.array(test_features)),
  #                    pos_label=1)))

  # nb_classifier = MultinomialNB()
  # nb_classifier.fit(train_features, train_classes)
  # print("naive bayes accuracy: " +
  #       str(f1_score(numpy.array(test_classes),
  #                    nb_classifier.predict(numpy.array(test_features)),
  #                    pos_label=1)))

  # perceptron_classifier = Perceptron()
  # perceptron_classifier.fit(train_features, train_classes)
  # print("SGD accuracy: " +
  #       str(f1_score(numpy.array(test_classes),
  #                    perceptron_classifier.predict(numpy.array(test_features)),
  #                    pos_label=1)))

  # dt_classifier = DecisionTreeClassifier(max_depth=1)
  # dt_classifier.fit(train_features, train_classes)
  # print("tree accuracy: " +
  #       str(f1_score(numpy.array(test_classes),
  #                    dt_classifier.predict(numpy.array(test_features)),
  #                    pos_label=1)))

  classifier = MyBoostClassifier(algorithm="SAMME")
  new_estimator_list = list()
  for estimator in estimator_list:
    for index in range(boosting_classifier_size):
      new_estimator_list.append(copy.deepcopy(estimator))
  classifier.set_estimators(new_estimator_list)
  classifier.fit(numpy.array(train_features), numpy.array(train_classes))
  print("logitboost accuracy: " +
        str(classifier.score(numpy.array(test_features),
                             numpy.array(test_classes))))
  # print("logitboost precision: " +
  #       str(precision_score(numpy.array(test_classes),
  #                           classifier.predict(numpy.array(test_features)),
  #                           pos_label=1)))
  # print("logitboost recall: " +
  #       str(recall_score(numpy.array(test_classes),
  #                        classifier.predict(numpy.array(test_features)),
  #                        pos_label=1)))
  print("logitboost f1 score: " +
        str(f1_score(numpy.array(test_classes),
                     classifier.predict(numpy.array(test_features)),
                     pos_label=1)))

load()
report()
report()
report()
report()
report()
