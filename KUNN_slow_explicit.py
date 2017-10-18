import numpy as np
from math import sqrt

class KUNN(object):
	def __init__(self, R, kU, kI):
		self.R = R
		self.cI = R.sum(axis=0)
		self.cU = R.sum(axis=1)
		self.kU = kU
		self.kI = kI

	def u_sim(self, u, v):
		common_items = np.nonzero(self.R[u, :] & self.R[v, :])[0]
		if len(common_items) == 0:
			return 0
		else:
			user_product = sqrt(self.cU[u]) * sqrt(self.cU[v])
			sim = 0
			for i in common_items:
				sim += 1 / (sqrt(self.cI[i]) * user_product)
			return sim

	def i_sim(self, i, j):
		common_users = np.nonzero(self.R[:, i] & self.R[:, j])[0]
		if len(common_users) == 0:
			return 0
		else:
			item_product = sqrt(self.cI[i]) * sqrt(self.cI[j])
			sim = 0
			for u in common_users:
				sim += 1 / (sqrt(self.cU[u]) * item_product)
			return sim

	def u_knn(self, u):
		f = np.vectorize(lambda v: self.u_sim(u, v))
		# Find all similarities to u
		all_sims = f(np.arange(num_users))
		# Find the largest kU + 1 elements
		knn = np.argpartition(all_sims, -self.kU-1)[-self.kU-1:]
		# Remove the index of U if present then just in case it wasn't present keep only the largest kU.
		knn = knn[knn != u][-self.kU:]
		return knn, all_sims[knn]

	def i_knn(self, i):
		f = np.vectorize(lambda j: self.i_sim(i, j))
		# Find all similarities to i
		all_sims = f(np.arange(num_items))
		# Find the largest kI + 1 elements
		knn = np.argpartition(all_sims, -self.kI-1)[-self.kI-1:]
		# Remove the index of i if present then just in case it wasn't present keep only the largest kU.
		knn = knn[knn != i][-self.kU:]
		return knn, all_sims[knn]

	def s_u(self, u, i):
		knn, knn_sims = self.u_knn(u)
		denom = sqrt(self.cI[i])
		s = 0
		for v, sim in zip(knn, knn_sims):
			if self.R[v, i]:
				s += sim
		return s/denom

	def s_i(self, u, i):
		knn, knn_sims = self.i_knn(i)
		denom = sqrt(self.cU[u])
		s = 0
		for j, sim in zip(knn, knn_sims):
			if self.R[u, j]:
				s += sim
		return s/denom

	def s(self, u, i):
		return self.s_u(u, i) + self.s_i(u, i)

	def pred(self, u):
		f = np.vectorize(lambda i: self.s(u, i))
		preds = f(np.arange(num_items))
		best_items = np.argsort(preds)[::-1]
		recommendations = zip(best_items, preds[best_items], self.R[u, best_items])
		new_recommendations = [tup for tup in recommendations if not tup[2]]
		return recommendations, new_recommendations

if __name__=="__main__":
	num_users = 10000
	num_items = 5000
	p = 0.01
	R = np.random.choice(a=[False, True], size=(num_users, num_items), p=[1 - p, p])
	R = KUNN(R, 5, 5)