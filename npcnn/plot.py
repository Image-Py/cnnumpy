import networkx as nx
from matplotlib import pyplot as plt 
#blue, orange',green,red, purple, brown, pink, gray, olive, cyan
colors={'conv':'blue','relu':'orange','maxp':'purple','upsa':'blue','dense':'blue','pool':'olive', 'up':'orange',
		'conc':'pink','sigm':'olive','batc':'cyan','add':'purple','gap':'blue','flatten':'pink','batchnorm':'cyan'}
size={'conv':300,'relu':300,'maxp':300,'upsa':300,'dense':300,'pool':300, 'up':300,
		'conc':300,'sigm':300,'batc':300,'add':300,'gap':300,'flatten':300,'batchnorm':300}
# ‘so^>v<dph8’.black
shape={'conv':'^','relu':'p','maxp':'h','upsa':'o','dense':'o','pool':'o', 'up':'o',
		'conc':'8','sigm':'o','batc':'>','add':'o','gap':'o','flatten':'o','batchnorm':0}
def plot_net(net,nots_colors=colors,nodes_size=size,nodes_shape=shape):
	flow,body=net.flow,net.layer
	#read the input/output layer's name(input or input1)
	str_input,str_output=flow[0][0],flow[-1][-1]
	G = nx.DiGraph()
	dic={str_input:str_input,'x':'x','y':'y'}
	G.add_node(str_input)

	for i in flow:
		print(i)
		dic[i[2]]=i[1][-1]
		G.add_node(dic[i[2]])
		# multi-input
		if isinstance(i[0],tuple):
			for j in i[0] : G.add_edge(dic[j],i[1][0])
		else : G.add_edge(dic[i[0]],i[1][0])
		# multi-node
		if len(i[1])>1:
			for j in range(len(i[1])-1) : G.add_edge(i[1][j], i[1][j+1])
		if i[2]==str_output:
			G.add_node(str_output)
			G.add_edge(i[1][-1], str_output)

	# positions for all nodes
	pos = nx.kamada_kawai_layout(G)  

	# the node's default color is black
	# nx.draw_networkx_nodes(G, pos,  node_color='black')
	#draw the nods using diffenren colors
	for i in nots_colors.keys():
		#node_size = 500,
		nx.draw_networkx_nodes(G, pos, nodelist=[j for j in G if i in j], 
			node_size=nodes_size[i],node_color=nots_colors[i],node_shape=nodes_shape[i])
	#draw input and outpot layer in red
	nx.draw_networkx_nodes(G, pos, nodelist=[str_input,str_output], node_color='red')

	edges = nx.draw_networkx_edges(G, pos, edges_color='olive',arrowstyle='->',arrowsize=10)
	#add the para of layer under the nodes
	for i in body:
		xy=pos[i[0]]
		if str(i[2])!='None':plt.text(xy[0],xy[1]-0.08,s=str(i[2]), fontsize=6,horizontalalignment='center')

	nx.draw_networkx_labels(G, pos,font_size=8 ,labels={i:i.split('_')[0]  for i in G},font_family='sans-serif')



	plt.axis('off')
	return plt