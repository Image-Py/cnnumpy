from .layer import layer_map as key

class Net:
	def __init__(self):
		self.body = []
		self.cmds = []

	def load_json(self, body, cmds):
		for i in body:
			para = i[2] or []
			self.body.append((i[0], key[i[1]](*para)))
		self.cmds = cmds

		# for better naming
		self.layer = body
		self.flow = cmds

	def forward(self, x):
		dic = dict(self.body)
		rst = {self.cmds[0][0]: x}
		for x, ls, y in self.cmds:
			for l in ls:
				out = x if l == ls[0] else y
				if not isinstance(out, str):
					p = [rst[i] for i in out]
				else: p = rst[out]
				rst[y] = dic[l](p)
				#print(l, 'in:', out, 'out', y, rst[y].sum())
		return rst[y]

	def layer2code(self, style='list'):
		body = []
		if style == 'list':
			body = ['self.layer = [']
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
			# body.append('dic = dict(self.body)')
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

	def load_weights(self, data):
		s = 0
		for i in self.body: 
			s += i[1].load(data[s:])
		# print(data.shape, s)
		# print(data_bn.shape, b)

	def show(self, info = True):
		from .plot import plot_net
		plot_net(self.layer, self.flow, info).show()

	def __call__(self, x):
		return self.forward(x)

if __name__ == '__main__':
	pass