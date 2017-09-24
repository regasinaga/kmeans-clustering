from rgx import *
import matplotlib.pyplot as plot
import numpy as np

dfile = 'dataset/R15.csv'
csv = np.genfromtxt(dfile, delimiter=',')
data = csv[:,0:2]
target = csv[:,2:3].astype(int)

# 1 A
plot = visualize(data, t=target,title="visualisasi data", mode="unsupervised")
plot.figure()

# 1 B i
c = train_kmeans(data, k=15, maxiter=100)

# 1 B ii
sserror = sse(data, c, 15)
print('SSE:\t',sserror)

# 1 C
plot = visualize_using_centro(data, c, title="visualisasi pengelompokkan dgn centroid")
plot = visualize_centro(c)
plot.figure()

# 1 D
plot = visualize(data, t=target,title="visualisasi data dan centroid", mode="supervised")
plot = visualize_centro(c)
plot.figure()

# 1 F
c2 = train_kmeans(data, t=target, k=15, maxiter=100)
plot = visualize_using_centro(data, c2,title="visualisasi pengelompokkan dgn centroid 2")
plot = visualize_centro(c2)

plot.show()