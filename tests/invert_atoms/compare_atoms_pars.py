from astropy.io import fits
import numpy as np
import globin

_, RLK_lines = globin.read_RLK_lines("lines_4016")
loggf0 = [line.loggf for line in RLK_lines]

def get_final_chi2(chi2):
	nx, ny, nit = chi2.shape

	best_chi2 = np.zeros((nx,ny))
	for idx in range(nx):
		for idy in range(ny):
			inds_non_zero = np.nonzero(chi2[idx,idy])[0]
			best_chi2[idx,idy] = chi2[idx,idy,inds_non_zero[-1]]

	return best_chi2

def get_atomic_data(run_name, frac=0.5):
	fpath = f"{run_name}/inverted_atoms.fits"
	hdu_atoms = fits.open(fpath)

	chi2 = fits.open(f"{run_name}/chi2.fits")[0].data
	# chi2 = get_final_chi2(chi2)

	atoms_data = {}
	lineID = {}

	for parameter in ["loggf", "dlam"]:
		ind = hdu_atoms.index_of(parameter)
		data = hdu_atoms[ind].data
		nx, ny, _, nl = data.shape
		atoms_data[parameter] = data[:,:,1,:]
		lineID[parameter] = np.array(data[0,0,0,:], dtype=np.int)
		
		# mode = int(hdu_atoms[ind].header["MODE"])

	# for idx in range(nx):
	# 	for idy in range(ny):
	# 		print(chi2[idx,idy])
	# 		print("-----------------------------------")

	return atoms_data, lineID

run_name = "runs/dummy_m2"
pars, IDs = get_atomic_data(run_name)

idx, idy = 0,0
par = pars["loggf"][idx,idy]

print("---------------------------")
print(" ID  Expect  Invert  Diff")
print("---------------------------")
for idl in range(len(IDs["loggf"])):
	lineNo = IDs["loggf"][idl]
	diff = np.abs(loggf0[lineNo] - par[idl])
	print("{:2d}  {:+4.3f}  {:+4.3f}   {:4.3f}".format(\
			lineNo+1, loggf0[lineNo], par[idl], diff))