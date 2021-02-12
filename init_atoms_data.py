import numpy as np

class Line(object):

	def __init__(self, lineNo=None, lam0=None, loggf=None):
		self.lineNo = lineNo
		self.lam0 = lam0
		self.loggf = loggf
		self.loggf_min = None
		self.loggf_max = None
		self.dlam_min = None
		self.dlam_max = None

	def __str__(self):
		return "<LineNo: {}, lam0: {}, loggf: {}\n  loggf_min: {}, loggf_max: {}>".format(self.lineNo, self.lam0, self.loggf, self.loggf_min, self.loggf_max)

def read_RLK_lines(fpath):
	"""
	Read RLK line list for given spectral region.
	"""
	lines = open(fpath, "r").readlines()
	
	RLK_lines = []

	for i_, line in enumerate(lines):
		lam0 = float(line[0:10])
		loggf = float(line[10:17])

		RLK_lines.append(Line(i_+1, lam0, loggf))

	return lines, RLK_lines

def write_init_line_pars(lines):
	out = open("test_line_pars", "w")

	out.write("# parID   LineNo   initial   min     max\n")

	for line in lines:
		out.write("loggf    ")
		out.write("{: 3d}    ".format(line.lineNo))
		out.write("{: 4.3f}    ".format(line.loggf))
		out.write("{: 4.3f}    ".format(line.loggf_min))
		out.write("{: 4.3f}\n".format(line.loggf_max))

	out.close()

# lineNo = list(range(18))
lineNo = np.arange(0,18)

_, RLK_lines = read_RLK_lines("../../Atoms/Kurucz/spinor_window_original")

lines = []
for lineID in lineNo:
	for i_ in range(len(RLK_lines)):
		if RLK_lines[i_].lineNo-1==lineID:
			lines.append(RLK_lines[i_])
			lines[-1].loggf_min = lines[-1].loggf-3
			if lines[-1].loggf_min<-10:
				lines[-1].loggf_min = -10
			lines[-1].loggf_max = lines[-1].loggf+3
			if lines[-1].loggf_max>1:
				lines[-1].loggf_max = 1
			lines[-1].loggf += np.random.normal(0, 1)

write_init_line_pars(lines)