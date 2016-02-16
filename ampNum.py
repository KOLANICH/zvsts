'''AMP General Numerical Computation Module
	Author: Alex M. Pronschinske
	
	List of classes: -none-
	List of functions:
		central_diff
		nn_smooth
		sgSm
	Module dependencies: 
		numpy
'''

import scipy as np

#===============================================================================
def central_diff(X, Y):
	'''Central Difference Derivative Method
	
	Args:
		X (list|ndarray): X data
		Y (list|ndarray): Y data
	Returns:
		(ndarray)  dY/dX
	'''
	if type(X).__name__ is not 'ndarray':
		Y = np.array(X)
	if type(Y).__name__ is not 'ndarray':
		Y = np.array(Y)
	
	dX = np.zeros(len(Y))
	dX[0] = X[1] - X[0]
	dX[1:-1] = X[2:] - X[:-2]
	dX[-1] = X[-1] - X[-2]
	dY = np.zeros(len(Y))
	dY[0] = Y[1] - Y[0]
	dY[1:-1] = Y[2:] - Y[:-2]
	dY[-1] = Y[-1] - Y[-2]
	
	return dY/dX
# END central_diff

#===============================================================================
def nn_smooth(Y, window_size):
	'''Near-Neighbor Smoothing Function
	
	Args:
		Y (list): Curve to be smoothed.
		window_size (int): Should be an odd positive integer.
	Returns:
		(numpy.ndarray)  Resulting smoothed curve data.
	'''
	L = len(Y)
	sY = np.zeros(L)
	try:
		window_size = np.abs(np.int(window_size))
	except ValueError as msg:
		raise ValueError('window_size must be of type int')
	if window_size % 2 != 1 or window_size < 1:
		raise ValueError("window_size size must be an odd number")
	# END try
	half_window = (window_size -1) / 2
	left_pad = Y[0] - np.abs( Y[1:half_window+1][::-1] - Y[0] )
	right_pad = Y[-1] + np.abs(Y[-half_window-1:-1][::-1] - Y[-1])
	Y = left_pad.extend(Y)
	Y = Y.extend(right_pad)
		
	for i in range(0,L):
		sY[i] = np.mean(Y[i-half_window:i+half_window])
	
	return sY
# END nn_smooth

#===============================================================================
def sgSm(y, window_size, order, deriv=0):
	'''Savitzky-Golay Smoothing & Differentiating Function
	
	Args:
		Y (list): Objective data array.
		window_size (int): Number of points to use in the local regressions.
						Should be an odd integer.
		order (int): Order of the polynomial used in the local regressions.
					Must be less than window_size - 1.
		deriv = 0 (int): The order of the derivative to take.
	Returns:
		(ndarray)  The resulting smoothed curve data. (or it's n-th derivative)
	Test:
		t = np.linnp.ce(-4, 4, 500)
		y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
		ysg = sg_smooth(y, window_size=31, order=4)
		import matplotlib.pyplot as plt
		plt.plot(t, y, label='Noisy signal')
		plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
		plt.plot(t, ysg, 'r', label='Filtered signal')
		plt.legend()
		plt.show()
	'''
	
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError as msg:
		raise ValueError("window_size and order have to be of type int")
	# END try
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	
	order_range = list(range(order+1))
	half_window = (window_size -1) / 2
	
	# precompute coefficients
	b = np.mat(
		[
			[k**i for i in order_range] for k in range(
				-half_window, half_window+1
			)
		]
	)
	m = np.linalg.pinv(b).A[deriv]
	
	# pad the function at the ends with reflections
	left_pad = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
	right_pad = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
	y = np.concatenate((left_pad, y, right_pad))
	
	return ((-1)**deriv) * np.convolve( m, y, mode='valid')
# END sg_smooth