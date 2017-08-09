from unet import UnetTrainer
import keras
from PIL import Image
import numpy as np

unet = UnetTrainer()
# assumes channels_last
unet.build_models((None, None, 3))
unet.model.load_weights("model.h5")

image = Image.open("beetil1.png").convert("RGB")

def save_scaled_heatmap(image, scale):
  width = scale*16*4
  height = scale*16*3
  image = image.resize((width, height))
  scaled_image_data = np.asarray(image, dtype='float32') / 255.

  output = unet.model.predict(np.array([scaled_image_data]))

  output_data = (output * 255).astype('uint8')

  output_image = Image.fromarray(output_data[0,:,:,0]).convert("RGB")

  output_image.save("beetil1_heatmap%dx%d.png" % (width, height))

save_scaled_heatmap(image, 6)
save_scaled_heatmap(image, 5)
save_scaled_heatmap(image, 4)
save_scaled_heatmap(image, 3)
save_scaled_heatmap(image, 2)
save_scaled_heatmap(image, 1)
