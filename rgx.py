import numpy as np
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
colors = np.array(['#ff3333', '#ff6633','#ff9933',
							'#ffcc33', '#ffff33','#ccff33',
							'#99ff33', '#33ff33', '#33ff99',
							'#33ffff', '#3399ff', '#3333ff',
							'#9933ff', '#ff33ff', '#ff3366'])
							
def visualize(d, t=None, title="plot", mode="supervised"):
	global colors
	mark = 'o'
	if mode=="supervised":
		tn = len(np.unique(t))
		for tc in range(0, tn):
			ind = np.where(t==tc+1)
			plot.scatter(d[ind,0], d[ind,1], marker=mark, color=colors[tc])
	elif mode=="unsupervised":
		plot.scatter(d[:,0], d[:,1], marker=mark, color=colors[2])
	plot.title(title)
	return plot
	
def visualize_using_centro(d, c, title="plot"):
	global colors
	mark = 'o'
	
	k = len(c)
	dist_mat = np.zeros((len(d), k))
	for j in range(0,k):
		dist_mat[:,j] = euclid(d[:,:], c[j,:])
	
	a = np.argmin(dist_mat,axis=1)
	
	for j in range(0,k):
		ind = (np.where(a == j))[0]
		plot.scatter(d[ind,0], d[ind,1], marker=mark, color=colors[j])
	plot.title(title)
	return plot
	
def visualize_centro(c):
	mark = 'x'
	plot.scatter(c[:,0], c[:,1], marker=mark, color='#000000')
	return plot
	
def train_kmeans(x, t=None, k=2, maxiter=1000):
	print('K-Means training begins')
	c_ind = np.random.choice(len(x),k)
	c = x[c_ind,:]
	
	if t is not None:
		tn = len(np.unique(t))
		for tc in range(0, tn):
			ind = np.where(t==tc+1)
			s_ind = np.random.permutation(ind[0])
			
			c[tc,:] = x[s_ind[0],:]
			if tc == k: break
	
	dist_mat = np.zeros((len(x), k))
	
	saturation = 0
	converged = False
	iter = 0
	prev_sse = 1e10
	
	for iter in range(maxiter):
		# get distance between data to all centroid
		for j in range(0,k):
			dist_mat[:,j] = euclid(x[:,:], c[j,:])

		# get nearest centroid 
		a = np.argmin(dist_mat,axis=1)
		current_sse = sse(x,c,k,a=a)
		
		# update centroid
		for j in range(0,k):
			ind = (np.where(a == j))[0]
			if len(ind) == 0: continue
			c[j,:] = np.mean(x[ind,:], axis=0)
		
		print('sse:\t', current_sse)
		if np.abs(current_sse-prev_sse) <= 1:
			saturation += 1
			if saturation == 10:
				break	
		else:
			saturation = 0
			
		prev_sse = current_sse

	return c

def euclid(p, q):
	return np.sqrt(np.sum(np.power(p - q, 2.0), axis=1))
	
def sse(x, c, k, a=None):
	r = 0
	if a is None:
		dist_mat = np.zeros((len(x), k))
		for j in range(0,k):
			dist_mat[:,j] = euclid(x[:,:], c[j,:])
		
		a = np.argmin(dist_mat,axis=1)
		
	for j in range(0, k):
		ind = np.where(a == j)
		r += np.sum(np.sum(np.power(x[ind,:] - c[j,:], 2.0), axis=1)) 

	return r