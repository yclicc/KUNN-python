import numpy as np
from numpy.linalg import multi_dot

def top_axis_elems(array, k, axis):
	# Keeps the top k largest elements along an axis and sets the rest to 0
	arr = array.copy()
	# Because of the way broadcasting works it is easiest to transpose the whole thing
	# if axis = 1 and then do everything for the 0th axis.
	if axis == 1:
		arr = arr.T
	index = arr.argpartition(-k, axis=0)
	other_axis_length = index.shape[1]
	arr[index[:-k,:], np.arange(other_axis_length)] = 0
	# Trnaspose back again.
	if axis == 1:
		arr =  arr.T
	return arr

class KUNN(object):
	def __init__(self, R, kU, kI):
		self.R = R
		self.cI = R.sum(axis=0)
		self.cU = R.sum(axis=1)
		self.kU = kU
		self.kI = kI

		self.cIrooted = np.diag(1.0 / np.sqrt(self.cI))
		self.cUrooted = np.diag(1.0 / np.sqrt(self.cU))

		self.Rscaled = multi_dot([self.cUrooted, self.R, self.cIrooted])

		self.iSim = multi_dot([self.cIrooted, self.R.T, self.Rscaled])
		self.uSim = multi_dot([self.Rscaled, self.R.T, self.cUrooted])

		np.fill_diagonal(self.iSim, 0)
		np.fill_diagonal(self.uSim, 0)

		self.iKNN = top_axis_elems(self.iSim, self.kI, axis=0)
		self.uKNN = top_axis_elems(self.uSim, self.kU, axis=1)

		self.sI = multi_dot([self.cUrooted, self.R, self.iKNN])
		self.sU = multi_dot([self.uKNN, self.R, self.cIrooted])

		self.s = self.sI + self.sU

	def pred(self, u):
		# Returns two arrays, one with all suggested items and another with just those the
		# user has not already interacted with. Each item is listed as a tuple with it's:
		# (index, name, notional score, boolean if it has been interacted with already by user)
		preds = self.s[u,:]
		best_items = np.argsort(preds)[::-1]
		recommendations = list(zip(best_items, preds[best_items], self.R[u, best_items]))
		new_recommendations = [tup for tup in recommendations if not tup[2]]
		return recommendations, new_recommendations

if __name__=="__main__":
	num_users =  10000
	num_items = 5000
	p = 0.01
	R = np.random.choice(a=[False, True], size=(num_users, num_items), p=[1-p, p])
	KUNN30 = KUNN(R, 30, 30)
