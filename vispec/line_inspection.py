import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import CheckButtons
import globin

import ipywidgets as widgets
from IPython.display import display

PSE = globin.atoms.PSE()

path = "/home/dusan/Documents/globin_inversion/gris_aug18_trelles/obs/gris_20180829_003_TrellesArjona.p2.fits"
spectrum = globin.Observation(path, spec_type="hinode")

idx, idy = 0, 0

_, RLK_lines = globin.atoms.read_RLK_lines("/home/dusan/Documents/globin_inversion/gris_aug18_trelles/lines_trelles21_LS_lande_lam_ABO")

atoms = {}
for line in RLK_lines:
	element = PSE.get_element_symbol(line.ion)
	element += "I" if line.state==0 else "II"
	if element in atoms:
		atoms[element].append(line)
	else:
		atoms[element] = [line]

plt.plot(spectrum.wavelength, spectrum.I[idx,idy])

lines = {}
for element in atoms:
	lines[element] = []
	for line in atoms[element]:
		l = plt.axvline(x=line.lam0, c="grey", lw=0.75)
		lines[element].append(l)

checkbox = widgets.Checkbox(
	value=False, 
	description='Check me', 
	disabled=False,
	indent=False)

display(checkbox)

plt.xlim([spectrum.wavelength[0], spectrum.wavelength[-1]])

def callback(checkbox):
	print(checkbox)
	element = "FeI"
	if checkbox:
		for line in lines[element]:
			line.set_visible(True)
			line.figure.canvas.draw_idle()
	else:
		for line in lines[element]:
			line.set_visible(True)
			line.figure.canvas.draw_idle()	

	plt.show()

widgets.interactive_output(callback, {"checkbox" : True})

# x0, y0, w, h = plt.gca().get_position().bounds
# rax = plt.gca().inset_axes([x0+w+0.10, 0, 0.12, 0.2])
# check = CheckButtons(
# 	ax=rax,
# 	labels=list(atoms.keys()),
# 	actives=[lines[element][0].get_visible() for element in lines]
# 	)

# def callback(element):
# 	print(element)
# 	for line in lines[element]:
# 		line.set_visible(not line.get_visible())
# 		line.figure.canvas.draw_idle()

# check.on_clicked(callback)