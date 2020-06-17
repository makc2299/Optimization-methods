This is a module containing numerical methods of unconditional multidimensional minimization such as:

optimize_module.CoordinateDescent()

optimize_module.NelderMead()

optimize_module.GradientDescent()

optimize_module.ConjugateGradients()

optimize_module.Newton()

optimize_module.Marquardt()

Zero order methods

	1. Coordinate descent method
		call example: optimize_module.CoordinateDescent(f,x,eps),
		f - is function object that you declared above such as def f(x): return 4.*(x[0]-5)**2.+(x[1]-6)**2
		x - point coordinate exmple: x =  np.array([1., -1.])
		eps - up to this number, the algorithm will work
	2. Nelder-Mead method
		call example: optimize_module.NelderMead(f,x1, x2, x3, eps),
		f - is function object that you declared above such as def f(x): return 4.*(x[0]-5)**2.+(x[1]-6)**2
		x1,x2,x3 - point coordinates exmple: x1=np.array([9. , 2.]), x2=np.array([3., 5.]), x3=np.array([4. , 10.])
First order methods

	1. Method of the fastest gradient descent
		call example: optimize_module.GradientDescent(f,grad,x,eps),
		f - is function object that you declared above such as def f(x): return 4.*(x[0]-5)**2.+(x[1]-6)**2
		grad - this is a function object that returns a gradient from f function
			example: def grad(x): return np.array([8.*(x[0]-5), 2.*(x[1]-6)])
		x - point coordinate exmple: x =  np.array([1., -1.])
		eps - up to this number, the algorithm will work
	2. Method of conjugate gradients
		call example: optimize_module.ConjugateGradients(f,grad,x,eps),
		f - is function object that you declared above such as def f(x): return 4.*(x[0]-5)**2.+(x[1]-6)**2
		grad - this is a function object that returns a gradient from f function
			example: def grad(x): return np.array([8.*(x[0]-5), 2.*(x[1]-6)])
		x - point coordinate exmple: x =  np.array([1., -1.])
		eps - up to this number, the algorithm will work

Second order methods

	1. Newton's method
		call example: optimize_module.Newton(f,grad,hesse,x,eps),
		f - is function object that you declared above such as def f(x): return 4*(x[0]-5)**2 + (x[1]-6)**2
		grad - this is a function object that returns a gradient from f function
			example: def grad(x): return np.array([8*(x[0]-5), 2*(x[1]-6)])
		hesse - is function object that returns the matrix of partial derivatives of grad
			example: def hesse(x): return np.array([[8., 0.], [0., 2.]])
		x - point coordinate exmple: x =  np.array([1., -1.])
		eps - up to this number, the algorithm will work
	2. The McWard method
		call example: optimize_module.Marquardt(f,grad,hesse,x,eps),
		f - is function object that you declared above such as def f(x): return 4*(x[0]-5)**2 + (x[1]-6)**2
		grad - this is a function object that returns a gradient from f function
			example: def grad(x): return np.array([8*(x[0]-5), 2*(x[1]-6)])
		hesse - is function object that returns the matrix of partial derivatives of grad
			example: def hesse(x): return np.array([[8., 0.], [0., 2.]])
		x - point coordinate exmple: x =  np.array([1., -1.])
		eps - up to this number, the algorithm will work
