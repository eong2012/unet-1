# Code (c) Sam Russell 2017
import pandas
import argparse
import numpy as np
import keras
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt

class BaseTrainer:
  batch_size = 128
  epochs = 100
  validation_percentage = 1.00

  def run(self):
    #self.load_args()
    self.load_data()
    shaped_values, shaped_labels = self.load_training_data()
    testing_values, testing_labels = self.load_testing_data()
    training_values, validation_values = self.split_data(shaped_values)
    training_labels, validation_labels = self.split_data(shaped_labels)
    #training_values = training_values[:1000]
    #training_labels = training_labels[:1000]

    print('values shape:', shaped_values.shape)
    print(training_values.shape[0], 'training samples')
    print(validation_values.shape[0], 'validation samples')

    self.build_models(input_shape=training_values.shape[1:])
    self.model.summary()

    self.model.load_weights("model.h5")

    self.model.fit(training_values, training_labels, batch_size=64, epochs=self.epochs, verbose=1, shuffle=True)

    self.model.save_weights("model.h5")

    self.save_results("output.png", testing_values)

  def save_results(self, filename, testing_values):
    # save some samples
    input_indices = np.random.choice(testing_values.shape[0], 8)
    input_images = [testing_values[x] for x in input_indices]

    masks = self.model.predict(np.array(input_images))
    #output_images = [masks[x] for x in masks.shape[0]]
    plt.figure(figsize=(10,10))

    output = [val for pair in zip(input_images, masks) for val in pair]

    for i in range(16):
      plt.subplot(4, 4, i+1)
      image = output[i]
      if image.shape[2] == 1:
        image = np.reshape(image, [self.img_rows, self.img_cols])
      elif K.image_data_format() == 'channels_first':
        image = image.transpose(1,2,0)
      # implicit no need to transpose if channels are last
      plt.imshow(image, cmap='gray')
      plt.axis('off')
    plt.tight_layout()

    plt.savefig(filename)
    plt.close('all')

  #def test_results(self, testing_values, testing_labels):
    #predictions = self.model.predict(testing_values)
    #df = pandas.DataFrame(data=np.argmax(predictions, axis=1), columns=['Label'])
    #df.insert(0, 'ImageId', range(1, 1 + len(df)))

    # save results
    #df.to_csv(self.commandline_args.output, index=False)

  #def load_args(self):
  #  self.commandline_args = self.parser.parse_args()

  def scale_values(self, values):
    return values.astype('float32') / 255

  def reshape_values(self, values):
    # TODO make it work when data comes pre-shaped
    if K.image_data_format() == 'channels_first':
        reshaped_values = values.reshape(values.shape[0], self.img_channels, self.img_rows, self.img_cols)
    else:
        reshaped_values = values.reshape(values.shape[0], self.img_rows, self.img_cols, self.img_channels)

    return reshaped_values

  def split_data(self, data):
    landmark = int(data.shape[0] * self.validation_percentage)
    return data[:landmark], data[landmark:]

  def build_models(self, input_shape):
    raise NotImplementedError("Must be implemented by subclass")
