import tensorflow as tf
from tensorflow import keras
import functools

def lr(in_dim, out_dim, is_standard):
  return tf.keras.models.Sequential([
                  tf.keras.layers.InputLayer(input_shape=(in_dim,)),
                  tf.keras.layers.Dense(out_dim, activation='sigmoid'),
                  tf.keras.layers.Softmax(),
                ])

def mlp(in_dim, out_dim, hidden):
  ret_model = tf.keras.models.Sequential([
                  tf.keras.layers.InputLayer(input_shape=(in_dim,)),
                ])
  
  for i in hidden:
    ret_model.add(tf.keras.layers.Dense(i, activation='relu'))
  
  ret_model.add(tf.keras.layers.Dense(out_dim))
  ret_model.add(tf.keras.layers.Softmax())
  
  return ret_model

def linear_regression(in_dim):
  return tf.keras.models.Sequential([
                  tf.keras.layers.InputLayer(input_shape=(in_dim,)),
                  tf.keras.layers.Dense(1),
                ])

def create_original_fedavg_cnn_model(only_digits=False):
  data_format = 'channels_last'
  input_shape = [28, 28, 1]
  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)
  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10 if only_digits else 62),
  ])
  return model

def lenet(out_dim):
  data_format = 'channels_last'
  input_shape = [32, 32, 1]
  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='valid',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='valid',
      data_format=data_format,
      activation=tf.nn.relu)
  model = tf.keras.models.Sequential([
      conv2d(filters=6, input_shape=input_shape),
      max_pool(),
      conv2d(filters=16),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(120, activation=tf.nn.relu),
      tf.keras.layers.Dense(84, activation=tf.nn.relu),
      tf.keras.layers.Dense(out_dim),
      tf.keras.layers.Softmax()
  ])
  return model

def lstm(out_dim, embedding_size=128, latent_size=128, num_layers=1):
  inputs = tf.keras.Input(shape=(None,), dtype="int32")
  a = tf.keras.layers.Embedding(input_dim=out_dim, output_dim=embedding_size)(inputs)
  b = tf.keras.layers.LSTM(latent_size, return_sequences=True)(a)
  c = tf.keras.layers.Dense(out_dim - 1)(b)
  d = tf.keras.layers.Softmax()(c)
  paddings = tf.constant([[0, 0,], [1, 0]])
  f = tf.pad(tf.reshape(d, [-1, out_dim - 1]), paddings, mode='CONSTANT', constant_values=0)
  f = tf.reshape(f, [-1, 10, out_dim])

  return tf.keras.Model(inputs=inputs, outputs=f)


def alexnet():
  model = keras.models.Sequential([
      keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
      keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
      keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
      keras.layers.BatchNormalization(),
      keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
      keras.layers.BatchNormalization(),
      keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
      keras.layers.BatchNormalization(),
      keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
      keras.layers.Flatten(),
      keras.layers.Dense(4096, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(4096, activation='relu'),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(2, activation='softmax')
  ])
  return model

def resnet101():
  return tf.keras.applications.resnet.ResNet101(
              include_top=True,
              weights=None,
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=2,
              **kwargs
          )