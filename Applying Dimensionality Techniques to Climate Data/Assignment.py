from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import scale
import pandas
import matplotlib
import os
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# This I/O operation is to remove 0.00T and convert into 0.00.
o = open("output.txt","a") #open for append
for line in open("53727641925dat.txt"):
   line = line.replace("0.00T","0.00 ")
   o.write(line)
o.close()
os.remove("53727641925dat.txt")
os.replace("output.txt","53727641925dat.txt")

# Reading data as csv.
data = pandas.read_csv("53727641925dat.txt",sep="\s+",header=0,low_memory=False)

# That is the first scaling manually to remove string and encode string codes into integer values.
data = data.replace({'*':0})
data = data.replace({'**':0})
data = data.replace({'***':0})
data = data.replace({'****':0})
data = data.replace({'*****':0})
data = data.replace({'******':0})
data = data.replace({'CLR':0})
data = data.replace({'BKN':1})
data = data.replace({'SCT':2})
data = data.replace({'OBS':3})
data = data.replace({'SKC':4})
data = data.replace({'OVC':5})

# That is to remove column if all elements of it are equal.
uniques = data.apply(lambda x: x.nunique()) #elimination of same columns
data = data.drop(uniques[uniques==1].index, axis=1)

# We eliminate two extra columns from our data.
data.drop('USAF',1)
data.drop('YR--MODAHRMN',1)

# That is the second scaling manually to take mean of some columns and replace 0 values
# that we set at first scaling with its mean value.
usaf_mean = data['DIR'].astype(float).mean()
data['DIR'] = data['DIR'].replace({0:usaf_mean})
mw_mean = data['MW'].astype(float).mean()
data['MW'] = data['MW'].replace({0:mw_mean})
w_mean = data['W'].astype(float).mean()
data['W'] = data['W'].replace({0:w_mean})
max_mean = data['MAX'].astype(float).mean()
data['MAX'] = data['MAX'].replace({0:max_mean})
min_mean = data['MIN'].astype(float).mean()
data['MIN'] = data['MIN'].replace({0:min_mean})
pcp06_mean = data['PCP06'].astype(float).mean()
data['PCP06'] = data['PCP06'].replace({0:pcp06_mean})
pcp24_mean = data['PCP24'].astype(float).mean()
data['PCP24'] = data['PCP24'].replace({0:pcp24_mean})
pcpxx_mean = data['PCPXX'].astype(float).mean()
data['PCPXX'] = data['PCPXX'].replace({0:pcpxx_mean})
sd_mean = data['SD'].astype(float).mean()
data['SD'] = data['SD'].replace({0:sd_mean})

# That is the third and last scaling using sklearn library.
data = scale(data)

#------------------------- PCA ----------------------------#
pca = PCA(n_components=2)
x_pca = pca.fit_transform(data)
plt.scatter(x_pca[:,0],x_pca[:,1],edgecolors='none',alpha=0.5)
plt.xlabel('label1')
plt.ylabel('label2')
plt.show()

# When using PCA, we scale our data between particular bounds. It gives simplier operations
# for finding covariance matrix. After finding covariance matrix, we apply eigen decomposition
# to covariance matrix and extract eigenvalues and eigenvectors. To use these, we fit our data
# and transform into two component data.
#------------------------- PCA ----------------------------#


#------------------------- MDS ----------------------------#
mds = MDS(n_components=2)
x_mds = mds.fit_transform(data[0:1500])
plt.scatter(x_mds[:,0],x_mds[:,1],edgecolors='none',alpha=0.5)
plt.xlabel('label1')
plt.ylabel('label2')
plt.show()

# When using MDS, we calculate euclidean distance between the coordinates. This calculation
# helps us to find loss function for our analysis. After that, that gives a matrix composed by
# square of euclidean distance between points. We apply double centering operation to this matrix
# and it helps us to find eigenvalues and eigenvectors for our data. Since MDS is derived from
# optimization problem, It gives better solution for fitting our data.
#------------------------- MDS ----------------------------#


#------------------------- ISOMAP ----------------------------#
isomap = Isomap(n_components=2)
x_isomap = isomap.fit_transform(data[0:1500])
plt.scatter(x_isomap[:,0],x_isomap[:,1],edgecolors='none',alpha=0.5)
plt.xlabel('label1')
plt.ylabel('label2')
plt.show()

# Isomap uses the same basic idea as PCA, the difference being that linearity is only preserved locally
# We define neighbors for each data point. After that, we find euclidean distance as a matrix M. M can
# actually be thought of as the covariance matrix for the space whose dimensions are defined by the data point
# Since the stated goal of Isomap is to preserve geodesic distances rather than Euclidean distances, it gives
# different dimensinality matrix. Last operation is to find eigenvalues and eigenvectors and data is fitted by the
# result of them.
#------------------------- ISOMAP ----------------------------#


#------------------------- LLE ----------------------------#
lle = LocallyLinearEmbedding(n_components=2)
x_lle = lle.fit_transform(data[0:1500])
plt.scatter(x_lle[:,0],x_lle[:,1],edgecolors='none',alpha=0.5)
plt.xlabel('label1')
plt.ylabel('label2')
plt.show()

# it finds a nonlinear manifold by stitching together small linear neighborhoods. The difference between the
# two algorithms is in how they do the stitching. LLE does it by finding a set of weights that perform local linear
# interpolations that closely approximate the data. First operation is to find neighbours for each data point and
# we should find weights that allow neighbors to interpolate original data accurately. After that, given those weights,
# find new data points that minimize interpolation error in lower dimensional space. We can find matrix M using
# (I −W)^T*(I −W) formula. Lats operation is also to find eigenvalues and eigenvectors and data is fitted by the
# result of them.
#------------------------- LLE ----------------------------#