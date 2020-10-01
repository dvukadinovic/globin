from globin import atmos
import os

#--- curent working directory 
cwd = os.getcwd()

class Input(object):
	"""
	Class for storing input data parameters.
	"""

	def __init__(self):
		atm = None
		n_thread = 1

def ReadInputFile(fname="params.input"):
	"""
	Function which opens 'params.input' file for reading
	input data.

	Parameters:
	---------------
	fname : str
		File name in which are stored input parameters.

	Returns:
	---------------
	Input : struct
		Input class in which we store all the informations
	"""

	lines = open(fname, "r").readlines()

	input_data = Input()

	for line in lines:
		if line[0]!="#":
			line = line.rstrip("\n").split("=")
			keyword = line[0].replace(" ", "")
			if keyword=="atmosphere":
				atm_path = line[1].replace(" ", "")
				atm = atmos.Atmosphere(atm_path)
			if keyword=="n_threads":
				n_thread = int(line[1].replace(" ", ""))

	input_data.atm = atm
	input_data.n_thread = n_thread

	# return atm, n_thread
	return input_data