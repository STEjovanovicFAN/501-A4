from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow_core.python.keras import regularizers

train_file_path = './heart.csv'
test_file_path = './heart_test.csv'

#train_file_path = pd.read_csv("heart_train.csv")
#test_file_path = pd.read_csv("heart_test.csv")

np.set_printoptions(precision=3, suppress=True)

LABEL_COLUMN = 'chd'
LABELS = [0, 1]

def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,
        label_name=LABEL_COLUMN,
        ignore_errors=True, 
        **kwargs)
    return dataset

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))    


#show_batch(raw_train_data)
#CSV_COLUMNS = ['row.names', 'sbp', 'tobacco', 'ldl', 'adiposity', 'famhist', 'typea', 'obesity', 'alcohol', 'age', 'chd']
#temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)
#show_batch(temp_dataset)
SELECT_COLUMNS = ['sbp', 'tobacco', 'ldl', 'adiposity', 'typea','obesity','alcohol','age']
#temp_dataset = get_dataset(train_file_path)
#temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)
temp_train_data = get_dataset(train_file_path)
temp_test_data = get_dataset(train_file_path)
#show_batch(temp_train_data)
#show_batch(temp_test_data)

class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_freatures = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_freatures]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features

        return features, labels

packed_train_data = temp_train_data.map(PackNumericFeatures(SELECT_COLUMNS))
packed_test_data = temp_test_data.map(PackNumericFeatures(SELECT_COLUMNS))

show_batch(packed_train_data)

desc = pd.read_csv(train_file_path)[SELECT_COLUMNS].describe()
MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])

def normalize_numeric_data(data, mean, std):
  # Center the data
  return (data-mean)/std

normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)
numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(SELECT_COLUMNS)])
numeric_columns = [numeric_column]

CATEGORIES = {
    'famhist': ['Present', 'Absent']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)
#print(preprocessing_layer(example_batch).numpy()[0]

print('\n~~~~~~~~BuildingModel~~~~~~~~~')
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=(int(1e4)/5)*10000,
  decay_rate=1,
  staircase=False)

model = tf.keras.Sequential([
  preprocessing_layer,
  tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001),
                activation='elu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001),
                activation='elu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(128, kernel_regularizer=regularizers.l2(0.0001),
                activation='elu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='sigmoid')

  #tf.keras.layers.Dense(64, activation='elu', input_shape=(28,)),
  #tf.keras.layers.Dense(64, activation='elu'),
  #tf.keras.layers.Dense(64, activation='elu'),
  #tf.keras.layers.Dense(1, activation='sigmoid')
  
  #tf.keras.layers.Dense(128, activation='relu'),
  #tf.keras.layers.Dense(128, activation='relu'),
  #tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                #optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'binary_crossentropy'])
#model.compile(
#    loss='binary_crossentropy',
#    optimizer='adam',
#    metrics=['accuracy'])

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data.shuffle(500)

model.fit(train_data, epochs=20,steps_per_epoch=128)


print('\n~~~~~~~~EvaluatingModel~~~~~~~~~')
print("Evaluating Train Data:")
model_loss, model_acc, yes = model.evaluate(train_data, verbose=2, steps=128)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

print("Evaluating Test Data:")
model_loss, model_acc, yes = model.evaluate(test_data, verbose=2, steps=128)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")