import PIL.Image as Image
import numpy as np
def load_image(filename, max_size=None):
	image = Image.open(filename)
	print(image)
	if max_size is not None:
		factor = max_size / np.max(image.size)
		#scale image height and width
		print(image.size)
		size = np.array(image.size)*factor
		size = size.astype(int)
		image = image.resize(size, Image.LANCZOS)
	return np.float32(image)

load_image("stickman.jpg",max_size=1024)