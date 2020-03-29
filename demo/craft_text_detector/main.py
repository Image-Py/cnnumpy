import sys
sys.path.append('../../')
from npcnn import read_onnx, resize
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, transform
from time import time

def normalize(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):  
    img = img.astype(np.float32)
    img.reshape((-1,3))[:] -= np.array(mean) * 255
    img.reshape((-1,3))[:] /= np.array(variance) * 255
    return img

# size to be 32x
def makesize32(img):
    h, w = img.shape[-2:]
    w = w // 32 * 32
    h = h // 32 * 32
    return resize(img, (h, w))

img =  io.imread('test.jpg')[:, :, ::-1]
x = normalize(img).transpose(2, 0, 1)
x = makesize32(x[None, :, :, :])

# 3 files needed, craft.txt, craft.npy, craft_bn.npy
net = read_onnx('craft')
y = net(x)
start = time()
for i in range(1):
    y = net(x)
print('craft detect time:', time()-start)

plt.subplot(131)
plt.imshow(img[:, :, ::-1].astype('uint8'))
# text map
plt.subplot(132)
plt.imshow(y[0, 0, :, :])
plt.title('text map')
# link map
plt.subplot(133)
plt.imshow(y[0, 1, :, :])
plt.title('link map')

plt.show()
