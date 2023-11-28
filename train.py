#"""Build and train for the AI Models."""
#imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
from data_load import DataLoader
import numpy as np
import tensorflow as tf

model_name = ""


def reshape_function(data, label):
  reshaped_data = tf.reshape(data, [-1, 10, 1])
  return reshaped_data, label


def calculate_model_size(model):
  print(model.summary())
  var_sizes = [
      np.product(list(map(int, v.shape))) * v.dtype.size
      for v in model.trainable_variables
  ]
  print("Model size:", sum(var_sizes) / 1024, "KB")


def build_cnn(seq_length):
  """Builds a convolutional neural network in Keras."""
  #ERROR ValueError: Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 128, 3, 1), found shape=(None, 640, 3, 1)
  #CHANGE added fixed seq_length
  #seq_length = 640
  global model_name

  if args.modelnumber == "0":
    model_name = "-CNN_model-0"
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(
            10, (20, 10),
            padding="same",
            activation="relu",
            input_shape=(seq_length, 10, 1)))
    model.add(tf.keras.layers.MaxPooling2D((3, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(9, activation='linear'))
    model.summary()
  elif args.modelnumber == "1":
    model_name = "-CNN_model-1"
    model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          10, (20, 10),
          padding="same",
          activation="relu",
          input_shape=(seq_length, 10, 1)),  # output_shape=(batch, 128, 3, 8)
      tf.keras.layers.MaxPool2D((3, 3)),  # (batch, 42, 1, 8)
      tf.keras.layers.Dropout(0.1),  # (batch, 42, 1, 8)
      tf.keras.layers.Conv2D(16, (10, 1), padding="same",
                             activation="relu"),  # (batch, 42, 1, 16)
      tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 14, 1, 16)
      tf.keras.layers.Flatten(),  # (batch, 224)
      tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 16)
      tf.keras.layers.Dense(9, activation="relu")  # (batch, 4)
    ])


  model_path = os.path.join("./netmodels", "CNN")
  print("Built CNN.")
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  #wheights disabled !!!!! TODO
  #model.load_weights("./netmodels/CNN/weights.h5")
  return model, model_path


def build_lstm(seq_length):
  """Builds an LSTM in Keras."""
  #CHANGE input_shape=(seq_length, 15)
  #tf.keras.layers.LSTM(22),
  #tf.keras.layers.Dense(10


  #LSTM Sequential model with 2 layers, 100 neurons in first layer after it a flatten and then a dense-layer with 9 neurons
  #Best performing model till now 28.11.2023 14:26
  #RMSE 1.4 -> but no accurate predictions epochs 30 -> seq 20 -> batch 64
  #Loss: 0.939727783203125, RMSE: 0.9693955779075623 -> epochs 30 -> batch 64 -> seq 20

  global model_name

  #TODO add modelnumber to foldername
  if args.modelnumber == "0":
    model_name = "-LSTM_model-0"
    model = tf.keras.Sequential([
            tf.keras.Input(shape=(seq_length, 10)),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dense(units=9, activation="relu"),
        ])
    model.summary()
  if args.modelnumber == "1":
    model_name = "-LSTM_model-1"
    model = tf.keras.Sequential([
            tf.keras.Input(shape=(seq_length, 10)),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=9, activation="linear"),
        ])
    model.summary()
  elif args.modelnumber == "2":
    model_name = "-LSTM_model-2"
    #LSTM Sequential model with 2 layers, 100 neurons in first layer after it a Dropoutlayer with 20% and then a dense-layer with 9 neurons
    model = tf.keras.Sequential([
          tf.keras.Input(shape=(seq_length, 10)),
          tf.keras.layers.LSTM(100),
          #tf.keras.layers.Flatten(),
          tf.keras.layers.Dropout(0.2),
          #tf.keras.layers.Dense(units=84, activation="relu"),
          tf.keras.layers.Dense(units=9, activation="linear"),
      ])
    model.summary()
  elif args.modelnumber == "3":
    model_name = "-LSTM_model-3"
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_length, 10)),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=9, activation="softmax"),
    ])
    model.summary()
  elif args.modelnumber == "4":
    model_name = "-LSTM_model-4"
    #LSTM Sequential model with 3 layers, 100 neurons in first layer, 100 neurons in second layer and then a dense-layer with 9 neurons
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(seq_length, 10)),
        tf.keras.layers.LSTM(100, return_sequences = True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(units=9, activation="linear"),
    ])
    model.summary()
  elif args.modelnumber == "5":
    model_name = "-LSTM_model-5"
    model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(100, return_sequences = True),
        input_shape=(seq_length, 10)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(100),
      tf.keras.layers.Dense(units=9, activation="linear")
    ])

  model_path = os.path.join("./netmodels", "LSTM")
  print("Built LSTM.")
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  return model, model_path


def load_data(train_data_path, valid_data_path, test_data_path, seq_length):
  data_loader = DataLoader(
      train_data_path, valid_data_path, test_data_path, seq_length=seq_length)
  data_loader.format()
  return data_loader.train_len, data_loader.train_data, data_loader.valid_len, \
      data_loader.valid_data, data_loader.test_len, data_loader.test_data


def build_net(args, seq_length):
  if args.model == "CNN":
    model, model_path = build_cnn(seq_length)
  elif args.model == "LSTM":
    model, model_path = build_lstm(seq_length)
  else:
    print("Please input correct model name.(CNN  LSTM)")
  return model, model_path


def train_net(
    model,
    model_path,  # pylint: disable=unused-argument
    train_len,  # pylint: disable=unused-argument
    train_data,
    valid_len,
    valid_data,  # pylint: disable=unused-argument
    test_len,
    test_data,
    kind):
  



  """
  print("\n\n\n TEST PREDICTION \n\n\n")
  model.predict(train_data)
  # Load the TFLite model in TFLite Interpreter
  interpreter = tf.lite.Interpreter(model.tflite)
  # There is only 1 signature defined in the model,
  # so it will return it by default.
  # If there are multiple signatures then we can pass the name.
  my_signature = interpreter.get_signature_runner()

  # my_signature is callable with input as arguments.
  output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
  # 'output' is dictionary with all outputs from the inference.
  # In this case we have single output 'result'.
  print(output['result'])
  """







  """Trains the model."""
  calculate_model_size(model)
  #RMSE 1,7 -> 10 epochs -> batch 64 -> sequenc 20
  epochs = 30 #15 # maybe 26 better
  #The batch_size argument specifies how many pieces of training data to feed into the network before measuring its accuracy and updating its weights and biases.
  #CHANGE batch_size = 64
  #batch_size = 16
  
  #batch_size = 16
  #batch_size = 10
  batch_size = 64
  
  """
  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"])
  
  #TODO try different optimizer
  #model with meanquare error out
  """
  rmse = tf.keras.metrics.RootMeanSquaredError()
  #model.compile(optimizer="adam", loss='mean_squared_error',
  #            metrics=[rmse,'mae'])
  model.compile(
    optimizer='adam',
    loss='mse',
    metrics=[tf.keras.metrics.RootMeanSquaredError(), "accuracy"])

  
  if kind == "CNN":
    train_data = train_data.map(reshape_function)
    test_data = test_data.map(reshape_function)
    valid_data = valid_data.map(reshape_function)
  test_labels = np.zeros(test_len)
  idx = 0
  for data, label in test_data:  # pylint: disable=unused-variable
    test_labels[idx] = label.numpy()
    print(str(label))
    idx += 1
  

  #load train_data_entry for test
  print("--> trainTest_labels: ")
  trainTest_labels = np.zeros(train_len)
  idx = 0
  for data, label in train_data:  # pylint: disable=unused-variable
    trainTest_labels[idx] = label.numpy()
    print(str(label))
    idx += 1
  trainTest_data = train_data.batch(batch_size)


  train_data = train_data.batch(batch_size).repeat()
  valid_data = valid_data.batch(batch_size)
  test_data = test_data.batch(batch_size)
  #print("TEST-DATA: ")
  #print(test_data)
  #CHANGED -> steps_per_epoch=1000

  #EaelyStop
  #EarlyStopping() saves us a lot of time, it stops the model training once it realizes that there will be no more decrease in loss in further epochs and training can now be stopped earlier than described epochs.
  early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 2)

  



  model.fit(
      train_data,
      epochs=epochs,
      validation_data=valid_data,
      steps_per_epoch=1000,
      #validation_steps=int((valid_len - 1) / batch_size + 1),
      validation_steps=1,
      #callbacks=[tensorboard_callback, early_stop])
      callbacks=[tensorboard_callback])
  #loss, acc, val_mae = model.evaluate(test_data)
  loss, rmse, acc= model.evaluate(test_data)
  pred = np.argmax(model.predict(test_data), axis=1)
  print("\n\n\n TEST PREDICTION \n\n\n")
  print("\n Prediction should be:")
  print(test_labels)
  print("\n Prediction")
  print(pred)
  print("\n\n\n TEST PREDICTION END \n\n\n")
  #num_classes: The possible number of labels the classification task can
  #TODO research what is confusion matrix
  confusion = tf.math.confusion_matrix(
      labels=tf.constant(test_labels),
      predictions=tf.constant(pred),
      num_classes=9)
  print(confusion)
  #TODO what is val_mae
  #print("Loss {}, Accuracy {}".format(loss, acc))
  #print("Loss {}, RMSE {}, val_mae {}".format(loss, acc, val_mae))
  print("Loss: {}, RMSE: {}, Accuracy: {}".format(loss, rmse, acc))
  # Convert the model to the TensorFlow Lite format without quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)

  """
  print("\n\n\n BEFORE INFERENCE")
  # Load the TFLite model and allocate tensors.
  interpreter = tf.lite.Interpreter(model_path="model.tflite")
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Test the model on random input data.
  input_shape = input_details[0]['shape']
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])





  print("\n\n\n AFTER INFERENCE")
  print(output_data)
  """


  #predict_fn = tf.contrib.predictor.from_saved_model("model.tflite")
  #predictions = predict_fn(test_data)
  #print(predictions['scores'])

  


  #MIHI

  #converter.optimizations = [tf.lite.Optimize.DEFAULT]
  #converter.experimental_new_converter=True
  #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
  converter._experimental_lower_tensor_list_ops = False

  #MIHI END
  
  
  tflite_model = converter.convert()

  # Save the model to disk
  open("model.tflite", "wb").write(tflite_model)

  # Convert the model to the TensorFlow Lite format with quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  
  #MIHI

  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
  converter._experimental_lower_tensor_list_ops = False

  #MIHI END

  tflite_model = converter.convert()

  # Save the model to disk
  open("model_quantized.tflite", "wb").write(tflite_model)

  basic_model_size = os.path.getsize("model.tflite")
  print("Basic model is %d bytes" % basic_model_size)
  quantized_model_size = os.path.getsize("model_quantized.tflite")
  print("Quantized model is %d bytes" % quantized_model_size)
  difference = basic_model_size - quantized_model_size
  print("Difference is %d bytes" % difference)


if __name__ == "__main__":

  #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m")
  parser.add_argument("--modelnumber", "-mn")
  parser.add_argument("--person", "-p")
  args = parser.parse_args()
  args.model = "LSTM"
  args.modelnumber = "0"

  #seq_length data window size
  #seq_length = 2988
  #seq_length = 128
  #seq_length = 640
  #seq_length = 64
  #seq_length = 128
  #20 window
  #seq_length = 10
  #je kleiner die seq_length umso ungenauer bzw größerer RMSE ??? why -> weil das fenster zu klein und das model somit keinen gescheiten zusammenhang erkennen kann ??
  #128 -> RMSE 1.378 -> early stop 17 epochs
  #seq 400 batch 16 -> RMSE
  #seq_length = 20 # RMSE LSTM -> 2.3 -> 10 Epochs
  #without 0 rows RMSE 2.3 and 1.8 -> with 0 rows RMSE 2.5
  #seq_length = 128 # RMSE LSTM -> 1.7 -> 10 Epochs
  seq_length = 20


  print("Start to load data...")
  #  if args.person == "true":
  #    train_len, train_data, valid_len, valid_data, test_len, test_data = \
  #        load_data("./person_split/train", "./person_split/valid",
  #                  "./person_split/test", seq_length)
  #  else:
  train_len, train_data, valid_len, valid_data, test_len, test_data = \
          load_data("./Data/train/train.json", "./Data/valid/valid.json", "./Data/test/test.json", seq_length)

  print("Start to build net...")
  model, model_path = build_net(args, seq_length)

  logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + model_name
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

  print("Start training...")
  train_net(model, model_path, train_len, train_data, valid_len, valid_data,
              test_len, test_data, args.model)

  print("Training finished!")



#LIST OF TESTED LSTM MODELS
  """
  #Loss: 2.5077505111694336, RMSE: 1.583587884902954 -> 5 epochs
  model = tf.keras.Sequential([
      tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(20),
          input_shape=(seq_length, 10)),  # output_shape=(batch, 44)
          #tf.keras.layers.Dropout(0.2),
          #tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(11, activation="sigmoid")  # (batch, 4)
  ])

  model.summary()
  """


  """
  #good model 2 -> RMSE 1.4 ohne dropout layer 24epochs batch 64 seq 20-> mit dropout layer RMSE
  #22.11.2023 - 14:34
  model = tf.keras.Sequential([
      tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(100, return_sequences = True),
          input_shape=(seq_length, 10)),  # output_shape=(batch, 44)
          tf.keras.layers.LSTM(100),
          tf.keras.layers.Dropout(0.2),
      #tf.keras.layers.Dense(11, activation="sigmoid")  # (batch, 4)
      tf.keras.layers.Dense(11)#, activation="relu")  # (batch, 4)
      #tf.keras.layers.Dense(11, activation="linear")  # (batch, 4)
  ])
  """

  """
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer((seq_length,15)),
      #tf.keras.layers.LSTM(100, return_sequences = True),     
      tf.keras.layers.LSTM(100),
      #tf.keras.layers.LSTM(50),
      #tf.keras.layers.Dense(8, activation = 'relu'),
      #tf.keras.layers.Dense(30, activation = 'relu'),
      tf.keras.layers.Dense(11, activation = 'linear')
      #tf.keras.layers.Dense(11, activation = 'softmax')
  ])
  """
  """
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer((seq_length,15)),
      #tf.keras.layers.LSTM(100, return_sequences = True),     
      tf.keras.layers.LSTM(15, return_sequences = True),
      tf.keras.layers.LSTM(30),
      tf.keras.layers.Dense(15),
      #tf.keras.layers.LSTM(50),
      #tf.keras.layers.Dense(8, activation = 'relu'),
      #tf.keras.layers.Dense(30, activation = 'relu'),
      ##tf.keras.layers.Dropout(0.1),
      ##tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(11, activation = 'softmax')
      #tf.keras.layers.Dense(11, activation = 'softmax')
  ])
  """
  """
  n_features = 15                        

  model = tf.keras.Sequential()

  model.add(tf.keras.layers.InputLayer((seq_length,n_features)))
  model.add(tf.keras.layers.LSTM(15, return_sequences = True))     
  model.add(tf.keras.layers.LSTM(100, return_sequences = True))
  model.add(tf.keras.layers.LSTM(50))
  #model.add(tf.keras.layers.Dense(8, activation = 'relu'))
  model.add(tf.keras.layers.Dense(11, activation = 'linear'))

  model.summary()
  """
  """
  #seq 2000 batch 16 -> RMSE 1.41 after 6 epochs
  n_features = 15                        

  model = tf.keras.Sequential()

  model.add(tf.keras.layers.InputLayer((seq_length,n_features)))
  model.add(tf.keras.layers.LSTM(100))
  #model.add(tf.keras.layers.LSTM(100, return_sequences = True))
  #model.add(tf.keras.layers.LSTM(50))
  #model.add(tf.keras.layers.Dense(8, activation = 'relu'))
  model.add(tf.keras.layers.Dense(11, activation = 'linear'))

  model.summary()
  """
  """
  n_features = 15                        

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(100),
          input_shape=(seq_length, 15)))
  ##model.add(tf.keras.layers.InputLayer((seq_length,n_features)))
  ##model.add(tf.keras.layers.LSTM(100))
  ###model.add(tf.keras.layers.LSTM(100))
  ###model.add(tf.keras.layers.LSTM(100))
  #model.add(tf.keras.layers.LSTM(100, return_sequences = True))
  #model.add(tf.keras.layers.LSTM(50))
  #model.add(tf.keras.layers.Dense(8, activation = 'relu'))
  model.add(tf.keras.layers.Dropout(0.1))
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(11, activation="linear"))

  model.summary()
  """
  """
  #WORKING 0.9 RMSE
  model = tf.keras.Sequential([
      tf.keras.layers.InputLayer((seq_length,15)),
      tf.keras.layers.LSTM(100, return_sequences = True),     
      tf.keras.layers.LSTM(100, return_sequences = True),
      tf.keras.layers.LSTM(50),
      #tf.keras.layers.Dense(8, activation = 'relu'),
      tf.keras.layers.Dense(30, activation = 'relu'),
      tf.keras.layers.Dense(11, activation = 'linear')
      #tf.keras.layers.Dense(11, activation = 'softmax')
  ])
  """

  """
  model = tf.keras.Sequential([
          tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(100),
          input_shape=(seq_length, 15)),
    #tf.keras.layers.LSTM(100, return_sequences = True),     
    #tf.keras.layers.LSTM(100, return_sequences = True),
    #tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(8, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'linear')
  ])
  """

  """
  model = tf.keras.Sequential
  model.add(tf.keras.layers.InputLayer((seq_length,15)))
  model.add(tf.keras.layers.LSTM(100, return_sequences = True))     
  model.add(tf.keras.layers.LSTM(100, return_sequences = True))
  model.add(tf.keras.layers.LSTM(50))
  model.add(tf.keras.layers.Dense(8, activation = 'relu'))
  model.add(tf.keras.layers.Dense(1, activation = 'linear'))
  """

  #LIST OF TESTED CNN MODELS
  """
  model_0 = tf.keras.Sequential(
    [
        #tf.keras.layers.Input(shape=input_shape),        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        #tf.keras.layers.Dense(num_classes_0, activation='softmax')
    ]
  )
  """

  """
  #good model 
  n_features = 10                        

  model = tf.keras.Sequential()

  model.add(tf.keras.layers.InputLayer((seq_length,n_features)))
  #model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.LSTM(70, return_sequences = True))
  #model.add(tf.keras.layers.BatchNormalization())     
  #model.add(tf.keras.layers.LSTM(100, return_sequences = True))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.LSTM(50))
  #model.add(tf.keras.layers.Dense(8, activation = 'relu'))
  ##model.add(tf.keras.layers.Dense(11, activation = 'linear'))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(11, activation = 'linear'))

  model.summary()
  """