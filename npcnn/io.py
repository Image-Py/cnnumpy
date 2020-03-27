import json, re, numpy as np
from .net import Net
import os

def read_net(path):
	net = Net()
	with open(path+'.lay') as f: lay = json.load(f)
	with open(path+'.flw') as f: flw = json.load(f)
	net.load_json(lay, flw)
	net.load_weights(np.load(path+'.npy'))
	return net

def parse(matched):
	gps = list(matched.groups())
	if len(matched.groups())==0: return ''

	for i in range(len(gps)):
		if '%' in gps[i]: 
			gps[i] = gps[i].replace('%',"'")
			gps[i] = gps[i].replace(')',"')")
			gps[i] = gps[i].replace(',',"',")
		elif gps[i][-1] == ')' and len(gps[i])>2: 
			gps[i] = gps[i][:-1]+',)'

	return str(gps)+'\n'

conv = re.compile(r'.*%(.+?) .+?(Conv).+?dilations=(\[\d+?, \d+?\]).+?strides=(\[\d+?, \d+?\]).+?(\(%.+?, %.+?, %.+?\)).+?\n')
relu = re.compile(r'.*%(.+?) .+?(Relu)\(%(.+?)\).+?\n')
sigmoid = re.compile(r'.*%(.+?) .+?(Sigmoid)\(%(.+?)\).+?\n')
maxpool = re.compile(r'.*%(.+?) .+?(MaxPool).+?kernel_shape=(\[\d+?, \d+?\]).+?strides=(\[\d+?, \d+?\]).+?\(%(.+?)\).+?\n')
upsample = re.compile(r'.*%.+? .+?Constant\[value=.+?(\d+\.?\d*) \[.+?\n.+?%(.+?) .+?(Upsample).+?\(%(.+?),.+?\n')
flatten = re.compile(r'.*%.+?Constant.+?\n.+?Shape.+?\n.+?Gather.+?\n.+?Constant.+?\n.+?Unsqueeze.+?\n.+?Unsqueeze.+?\n.+?Concat.+?\n.+?%(.+?) .+?(Reshape)\(%(.+?),.+?\n')
dense = re.compile(r'.*%(.+?) .+?(Gemm).+(\(%.+?, %.+?, %.+?\)).+?\n')
concat = re.compile(r'.*%(.+?) .+?(Concat).+(\(%.+?\)).+?\n')
batchnorm = re.compile(r'.*%(.+?) .+?(BatchNormalization).+?(\(.+?\)).+?\n')
add = re.compile(r'.*%(.+?) .+?(Add)(\(%.+?\)).+?\n')
weight = re.compile(r'.*%(.+?) .+?(\(.*?\)).*\n')

res = (flatten, upsample, conv, relu, sigmoid, maxpool, dense, concat, add, batchnorm, weight)

def read_onnx(path):
	with open(path+'.txt') as f:
		cont = f.read()
	for i in res: cont = i.sub(parse, cont)
	# for i in cont.split('\n'): print(i)
	cont = [eval(i) for i in cont.split('\n') if len(i)>0 and i[0]=='[']
	cont = [[eval(j) if (',' in j) else j for j in i] for i in cont]

	body = []
	flow = []
	key = {}
	for i in cont:
		if len(i)==2: key[i[0]] = i[1]
		elif i[1]=='Conv':
			num = len(body)
			shp = [key[i[4][1]][j] for j in (1,0,2)] + [i[3][0], i[2][0]]
			body.append(('conv_%s'%num, 'conv', shp))
			flow.append((i[4][0], ['conv_%s'%num], i[0]))
		elif i[1]=='Gemm':
			num = len(body)
			body.append(('dense_%s'%num, 'dense', key[i[2][1]]))
			flow.append((i[2][0], ['dense_%s'%num], i[0]))
		elif i[1]=='Sigmoid':
			num = len(body)
			body.append(('sigmoid_%s'%num, 'sigmoid', None))
			flow.append((i[2], ['sigmoid_%s'%num], i[0]))
		elif i[1]=='Relu':
			num = len(body)
			body.append(('relu_%s'%num, 'relu', None))
			flow.append((i[2], ['relu_%s'%num], i[0]))
		elif i[1]=='Add':
			num = len(body)
			body.append(('add_%s'%num, 'add', None))
			flow.append((i[2], ['add_%s'%num], i[0]))
		elif i[1]=='Concat':
			num = len(body)
			body.append(('concat_%s'%num, 'concat', None))
			flow.append((i[2], ['concat_%s'%num], i[0]))
		elif i[1]=='MaxPool':
			num = len(body)
			body.append(('maxpool_%s'%num, 'maxpool', [i[2][0], i[3][0]]))
			flow.append((i[4], ['maxpool_%s'%num], i[0]))
		elif i[2]=='Upsample':
			num = len(body)
			body.append(('upsample_%s'%num, 'upsample', [int(float(i[0]))]))
			flow.append((i[3], ['upsample_%s'%num], i[1]))
		elif i[1]=='BatchNormalization':
			num = len(body)
			body.append(('batchnorm_%s'%num, 'batchnorm', [key[i[2][1]][0]]))
			flow.append((i[2][0], ['batchnorm_%s'%num,], i[0]))
		elif i[1]=='Reshape':
			num = len(body)
			body.append(('flatten_%s'%num, 'flatten', None))
			flow.append((i[2], ['flatten_%s'%num], i[0]))

	# for i in body: print(i)
	# for i in flow: print(i)
	
	net = Net()
	net.load_json(body, flow)
	net.load_weights(np.load(path+'.npy'))
	return net