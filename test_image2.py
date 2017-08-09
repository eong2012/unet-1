from unet import UnetTrainer
import keras
from PIL import Image
import numpy as np

unet = UnetTrainer()
# assumes channels_last
unet.build_models((None, None, 3))
unet.model.load_weights("model.h5")

image = Image.open("beetil1.png").convert("RGB")

def scale_image(image, scale):
  width = scale*16*4
  height = scale*16*3
  image = image.resize((width, height))

  return image

scaled_image = scale_image(image, 6)

width, height = scaled_image.size

scaled_image_data = np.asarray(image, dtype='float32') / 255.

rows = []
for y in xrange(0, height-16, 16):
  cells = []
  for x in xrange(0, width-16, 16):
    tile = scaled_image_data[y:y+32, x:x+32, :]
    output = unet.model.predict(np.array([tile]))
    output_data = (output * (255. / 4.)).astype('uint8')
    cells.append(output_data[0])

  # recombine
  even_cells = [cells[i] for i in xrange(0, len(cells), 2)]
  odd_cells = [cells[i] for i in xrange(1, len(cells), 2)]
  even_row = np.concatenate(even_cells, axis=1)
  odd_row = np.concatenate(odd_cells, axis=1)
  odd_row = np.pad(odd_row, ((0,0), (16,16), (0,0)), 'constant')
  row = even_row + odd_row
  rows.append(row)

# recombine
even_rows = [rows[i] for i in xrange(0, len(rows), 2)]
odd_rows = [rows[i] for i in xrange(1, len(rows), 2)]
even_image_data = np.concatenate(even_rows, axis=0)
odd_image_data = np.concatenate(odd_rows, axis=0)
odd_image_data = np.pad(odd_image_data, ((16,16), (0,0), (0,0)), 'constant')
full_image_data = even_image_data + odd_image_data
output_image = Image.fromarray(full_image_data[:,:,0]).convert("RGB")
output_image.save("beetil1_heatmap%dx%d.png" % (width, height))
