import numpy as np

def zeroth_order():
	raise NotImplementedError("Under development")

def first_order(values):
	return values[:-1] - values[1:]

def second_order(values, positions):
	delta = positions[:-1] - positions[1:]
	# res = A*pjp1 + B*pj + C*pjn1
	raise NotImplementedError("Under development")

def zeroth_order_derivative():
	raise NotImplementedError("Under development")

def first_order_derivative(npars):
	indices = np.arange(1, npars)
	matrix = np.eye(npars, k=0)
	matrix[indices, indices-1] = -1

	return matrix

def second_order_derivative():
	raise NotImplementedError("Under development")

class DepthRegularisation(object):
	def __init__(self):
		pass