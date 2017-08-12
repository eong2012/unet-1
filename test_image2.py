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

tile_pixels = 32
step_pixels = 8
overlap_count = (tile_pixels / step_pixels)
overlap_area = overlap_count * overlap_count * 1.0

rows = []
for y in xrange(0, height-(tile_pixels-step_pixels), step_pixels):
  cells = []
  for x in xrange(0, width-(tile_pixels-step_pixels), step_pixels):
    tile = scaled_image_data[y:y+tile_pixels, x:x+tile_pixels, :]
    output = unet.model.predict(np.array([tile]))
    output_data = (output * (255. / overlap_area)).astype('uint8')
    cells.append(output_data[0])

  # recombine
  even_cells = [cells[i] for i in xrange(0, len(cells), overlap_count)]
  sub1_cells = [cells[i] for i in xrange(1, len(cells), overlap_count)]
  sub2_cells = [cells[i] for i in xrange(2, len(cells), overlap_count)]
  sub3_cells = [cells[i] for i in xrange(3, len(cells), overlap_count)]
  even_row = np.concatenate(even_cells, axis=1)
  sub1_row = np.concatenate(sub1_cells, axis=1)
  sub2_row = np.concatenate(sub2_cells, axis=1)
  sub3_row = np.concatenate(sub3_cells, axis=1)
  sub1_row = np.pad(sub1_row, ((0,0), (step_pixels*1,step_pixels*3), (0,0)), 'constant')
  sub2_row = np.pad(sub2_row, ((0,0), (step_pixels*2,step_pixels*2), (0,0)), 'constant')
  sub3_row = np.pad(sub3_row, ((0,0), (step_pixels*3,step_pixels*1), (0,0)), 'constant')
  row = even_row + sub1_row + sub2_row + sub3_row
  rows.append(row)

# recombine
even_rows = [rows[i] for i in xrange(0, len(rows), overlap_count)]
sub1_rows = [rows[i] for i in xrange(1, len(rows), overlap_count)]
sub2_rows = [rows[i] for i in xrange(2, len(rows), overlap_count)]
sub3_rows = [rows[i] for i in xrange(3, len(rows), overlap_count)]
even_image_data = np.concatenate(even_rows, axis=0)
sub1_image_data = np.concatenate(sub1_rows, axis=0)
sub2_image_data = np.concatenate(sub2_rows, axis=0)
sub3_image_data = np.concatenate(sub3_rows, axis=0)
sub1_image_data = np.pad(sub1_image_data, ((step_pixels*1,step_pixels*3), (0,0), (0,0)), 'constant')
sub2_image_data = np.pad(sub2_image_data, ((step_pixels*2,step_pixels*2), (0,0), (0,0)), 'constant')
sub3_image_data = np.pad(sub3_image_data, ((step_pixels*3,step_pixels*1), (0,0), (0,0)), 'constant')
# square root to pull things up towards 1.0
full_image_data = even_image_data + sub1_image_data + sub2_image_data + sub3_image_data
full_image_data = np.sqrt(full_image_data / 255.) * 255.
output_image = Image.fromarray(full_image_data[:,:,0]).convert("RGB")
output_image.save("beetil1_heatmap%dx%d.png" % (width, height))
