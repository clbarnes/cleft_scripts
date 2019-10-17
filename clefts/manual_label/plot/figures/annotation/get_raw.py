import numpy as np
from catpy import CatmaidClient
from catpy.image import ImageFetcher
from matplotlib import pyplot as plt
from PIL import Image

from clefts.constants import CREDENTIALS_PATH, STACK_ID

client = CatmaidClient.from_json(CREDENTIALS_PATH)
fetcher = ImageFetcher.from_catmaid(client, STACK_ID)

arr = fetcher.get(np.array([[2660, 19170, 15900], [2661, 19280, 16030]])).squeeze()

im = Image.fromarray(arr)
im.save("raw.png")

plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
plt.show()
