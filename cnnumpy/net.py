from .layer import layerkey as key
import numpy as np
import json

class Net:
	def __init__(self):
		self.body = []
		self.cmds = []

	def load_json(self, body, cmds):
		for i in body:
			para = i[2] or []
			self.body.append((i[0], key[i[1]](*para)))
		self.cmds = cmds

	def forward(self, x):
		dic = dict(self.body)
		rst = {self.cmds[0][0]: x}
		for x, ls, y in self.cmds:
			for l in ls:
				out = x if l == ls[0] else y
				if isinstance(out, list):
					p = [rst[i] for i in out]
				else: p = rst[out]
				rst[y] = dic[l](p)
		return rst[y]

	def layer2code(self, style='list'):
		if style == 'list':
			body = ['self.body = [']
			for i in self.body:
				body.append('\t("%s", %s, %s),'%(i[0], 
					i[1].__class__.__name__, i[1].para()))
			body.append(']')
		if style == 'self':
			body = []
			for i in self.body:
				body.append('self.%s = %s%s'%(i[0], 
					i[1].__class__.__name__, i[1].para() or ()))
		return '\n'.join(body)

	def layer2json(self):
		body = []
		invk = dict(zip(key.values(),key.keys()))
		for i in self.body:
			body.append((i[0], invk[i[1].__class__], i[1].para()))
		return body

	def flw2code(self, style='list'):
		body = []
		if style=='list':
			body.append('dic = dict(self.body)')
			for x, ls, y in self.cmds:
				for l in ls:
					out = x if l == ls[0] else y
					if isinstance(out, list): 
						out = str(out).replace("'",'')
					body.append("%s = dic['%s'](%s)"%(y,l,out))
				body.append('')
		if style=='self':
			for x, ls, y in self.cmds:
				for l in ls:
					out = x if l == ls[0] else y
					if isinstance(out, list): 
						out = str(out).replace("'",'')
					body.append('%s = self.%s(%s)'%(y,l,out))
				body.append('')
		return '\n'.join(body)

	def load_data(self, data, shpkey):
		s = 0
		for i in shpkey:
			for j in range(len(shpkey[i])):
				sp = shpkey[i][j]
				l = np.cumprod(sp)[-1]
				shpkey[i][j] = data[s:s+l].reshape(sp)
				s += l
		dic = dict(self.body)
		for i in shpkey:
			dic[i].K = shpkey[i][0]
			dic[i].bias = shpkey[i][1]

	def __call__(self, x):
		return self.forward(x)

def read_net(path):
	net = Net()
	with open(path+'.lay') as f: lay = json.load(f)
	with open(path+'.flw') as f: flw = json.load(f)
	net.load_json(lay, flw)
	data = np.load(path+'.npy')
	with open(path+'.shp') as f: shp = json.load(f)
	net.load_data(data, shp)
	return net

if __name__ == '__main__':
	pass