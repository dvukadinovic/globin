def invert():
	"""
	As input we expect all data to be present :)

	Our procedure steps are:
		1. we do initial synthesis using reference atmosphere
		2. we go to compute RFs (and build from nodes and solve SE equation)
		3. arange RFs in global matrix and find the inverse (set the lambda)
		4. estimate the next steps and calculate chi2
		5. decide to take new step or revert back to 3. and re do
		6. iterate until max number of steps or tolerance is approached
	"""
	pass