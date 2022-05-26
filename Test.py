# Imports PIL module
from PIL import Image
import os

# open method used to open different extension image file
filepath = os.path.join(os.path.dirname(__file__),
                        '../small_flower_dataset/daisy/5794839_200acd910c_n.jpg')
im = Image.open(filepath,)

# This method will show image in any image viewer
im.show()
