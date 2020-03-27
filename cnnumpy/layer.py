from .util import conv, maxpool, upsample
import numpy as np

class Layer:
    name = 'layer'
    def __init__(self, name):
        self.name = name

    def forward(self, x): pass

    def backward(self, grad_y): pass

    def para(self): return None

    def load(self, buf): return 0

    def __call__(self, x):
        return self.forward(x)

class Dense(Layer):
    name = 'dense'
    def __init__(self, c, n):
        self.K = np.zeros((n, c), dtype=np.float32)
        self.bias = np.zeros(n, dtype=np.float32)

    def para(self): return self.K.shape

    def forward(self, x):
        y = x.dot(self.K.T)
        y += self.bias.reshape((1, -1))
        return y

    def load(self, buf):
        sk, sb = self.K.size, self.bias.size
        self.K.ravel()[:] = buf[:sk]
        self.bias.ravel()[:] = buf[sk:sk+sb]
        return sk + sb

class Conv2d(Layer):
    name = 'conv'
    def __init__(self, c, n, w, s=1, d=1):
        self.n, self.c, self.w = n, c, w
        self.s, self.d = s, d
        self.K = np.zeros((n, c, w, w), dtype=np.float32)
        self.bias = np.zeros(n, dtype=np.float32)

    def para(self): 
        return self.n, self.c, self.w, self.s, self.d

    def forward(self, x):
        out = conv(x, self.K, (self.s, self.s), (self.d, self.d))
        out += self.bias.reshape((1, -1, 1, 1))
        return out

    def load(self, buf):
        sk, sb = self.K.size, self.bias.size
        self.K.ravel()[:] = buf[:sk]
        self.bias.ravel()[:] = buf[sk:sk+sb]
        return sk + sb

class ReLU(Layer):
    name = 'relu'
    def __init__(self):pass

    def forward(self, x):
        return (x > 0) * x

class Flatten(Layer):
    name = 'flatten'
    def __init__(self):pass

    def forward(self, x):
        return x.reshape((1, -1))

class Sigmoid(Layer):
    name = 'sigmoid'
    def __init__(self):pass

    def forward(self, x):
        return 1/(1 + np.exp(-x))

class Softmax(Layer):
    name = 'softmax'
    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        eX = np.exp((x.T - np.max(x, axis=self.axis)).T)
        return (eX.T / eX.sum(axis=self.axis)).T

class Maxpool(Layer):
    name = 'maxpool'
    def __init__(self, w=2, stride=2):
        self.w = w
        self.stride = stride

    def para(self): return (self.stride,)

    def forward(self, x):
        return maxpool(x, (self.w, self.w), (self.stride, self.stride))

class UpSample(Layer):
    name = 'upsample'
    def __init__(self, k):
        self.k = k

    def para(self): return (self.k,)

    def forward(self, x):
        return upsample(x, self.k)

class Concatenate(Layer):
    name = 'concat'
    def __init__(self): pass

    def forward(self, x):
        return np.concatenate(x, axis=1)
        
class BatchNorm(Layer):
    name = 'batchnorm'
    def __init__(self, c):
        self.c = c
        self.k = np.zeros(c, dtype=np.float32)
        self.b = np.zeros(c, dtype=np.float32)
        self.m = np.zeros(c, dtype=np.float32)
        self.v = np.zeros(c, dtype=np.float32)
    
    def forward(self, x):
        x = (x - self.m.reshape(1, -1, 1, 1))
        x /= np.sqrt(self.v.reshape(1, -1, 1, 1))
        x *= self.k.reshape(1, -1, 1, 1) 
        x += self.b.reshape(1, -1, 1, 1)
        return x

    def load(self, buf):
        c = self.c
        self.k[:] = buf[0*c:1*c]
        self.b[:] = buf[1*c:2*c]
        self.m[:] = buf[2*c:3*c]
        self.v[:] = buf[3*c:4*c]
        return self.c * 4

layerkey = {'dense':Dense, 'conv':Conv2d, 'relu':ReLU, 'batchnorm':BatchNorm,
    'flatten':Flatten, 'sigmoid':Sigmoid, 'softmax': Softmax,
    'maxpool':Maxpool, 'upsample':UpSample, 'concat':Concatenate}

if __name__ == "__main__":
    pass
