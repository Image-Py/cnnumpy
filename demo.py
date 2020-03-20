from cnnumpy import *

# ========== write a net manually ========== 
class CustomNet(Net):
    def __init__(self):
        self.conv = Conv2d(3, 64, 3, 1)
        self.relu = ReLU()
        self.pool = Maxpool(2)
        self.upsample = UpSample(2)
        self.concatenate = Concatenate()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        y = self.pool(x)
        y = self.upsample(y)
        z = self.concatenate([x, y])
        return self.sigmoid(z)

net = CustomNet()
rst = net(np.zeros((2, 3, 64, 64), dtype=np.float32))

print('this is a manually net ')

# ========== load net from json ========== 
layer = [('conv', 'conv', (3, 64, 3, 1)),
        ('relu', 'relu', None),
        ('pool', 'maxpool', (2,)),
        ('up', 'upsample', (2,)),
        ('concat', 'concat', None),
        ('sigmoid', 'sigmoid', None)]

flow = [('x', ['conv', 'relu'], 'x'),
        ('x', ['pool', 'up'], 'y'),
        (['x','y'], ['concat', 'sigmoid'], 'z')]

net = Net()
net.load_json(layer, flow)
net(np.zeros((2, 3, 64, 64), dtype=np.float32))

# generate some code similiar to torch
print('\n', '='*10, 'generate layer code', '='*10)
print(net.layer2code('list'))

print('\n', '='*10, 'generate layer code', '='*10)
print(net.layer2code('self'))

print('\n', '='*10, 'generate flow code', '='*10)
print(net.flw2code('list'))

print('\n', '='*10, 'generate flow code', '='*10)
print(net.flw2code('self'))

# ========== load net with data ========== 
shp = {'conv':[[64, 3, 3, 3], [1, 64]]}
data = np.zeros(64*3*3*3+1*64, dtype=np.float32)

net.load_data(data, shp)

load_a_net = '''
write the layer, flow, shp as json format, 
and rename as xx.lay, xxx.flw, xxx.shp
and the data as xx.npy, 1d array
then we can use read_net('xx') to load a net
'''
print(load_a_net)
