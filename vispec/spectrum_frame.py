import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo, askyesno

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np

import sys

from topwindow import TopWindow

try:
	import globin
except ImportError:
	sys.exit("Could not find 'globin' module. Please install it before using the 'vispec'.")

from constants import PADX, PADY, FONT

filetypes = (("fits files", "*.fits"),
			 ("All files", "*.*"))

TITLE = {"I" : "Stokes I", "Q" : "Stokes Q", 
		 "U" : "Stokes U", "V" : "Stokes V"}
IDS = {"I" : 0, "Q" : 1, "U" : 2, "V" : 3}

class SpectrumFrame(ttk.Frame):
	def __init__(self, container, master):
		tk.Frame.__init__(self, container)
		# self.config(width=300, height=150, relief=tk.RAISED)

		self.inverted_spectrum = None
		self.observed_spectrum = None
		self.map_window = None

		button = tk.Button(self, text="Inverted spectrum...", command=self.load_inverted_spectrum,
					width=20)
		button.grid(column=0, row=0, padx=PADX, pady=PADY, sticky="NW")
		self.inv_flag = tk.IntVar()
		self.inv_lbl = tk.Checkbutton(self, text="Inverted spectrum", fg="red", font=FONT,
						variable=self.inv_flag, onvalue=1, offvalue=0)
		self.inv_lbl.grid(column=0, row=1, padx=PADX, pady=PADY, sticky="W")

		button = tk.Button(self, text="Observed spectrum...", command=self.load_observed_spectrum,
					width=20)
		button.grid(column=1, row=0, padx=PADX, pady=PADY, sticky="NW")
		self.obs_flag = tk.IntVar()
		self.obs_lbl = tk.Checkbutton(self, text="Observed spectrum", fg="red", font=FONT,
						variable=self.obs_flag, onvalue=1, offvalue=0)
		self.obs_lbl.grid(column=1, row=1, padx=PADX, pady=PADY, sticky="W")

		button = tk.Button(self, text="Get chi2", command=self.get_chi2, width=15)
		button.grid(column=2, row=0, padx=PADX, pady=PADY, sticky="W")

		#--- Stokes components (checkboxes)
		self.stokesI = tk.IntVar()
		self.stokesQ = tk.IntVar()
		self.stokesU = tk.IntVar()
		self.stokesV = tk.IntVar()

		cb = tk.Checkbutton(self, text="Stokes I", onvalue=1, offvalue=0, font=FONT, variable=self.stokesI)
		cb.grid(column=0, row=2, sticky="W")
		cb = tk.Checkbutton(self, text=r"Stokes U", onvalue=1, offvalue=0, font=FONT, variable=self.stokesU)
		cb.grid(column=0, row=3, sticky="W")
		cb = tk.Checkbutton(self, text=r"Stokes Q", onvalue=1, offvalue=0, font=FONT, variable=self.stokesQ)
		cb.grid(column=1, row=2, sticky="W")
		cb = tk.Checkbutton(self, text=r"Stokes V", onvalue=1, offvalue=0, font=FONT, variable=self.stokesV)
		cb.grid(column=1, row=3, sticky="W")

		#--- maps
		button = tk.Button(self, text="Plot map(s)", command=self.plot_spectral_maps)
		button.grid(column=0, row=4, sticky="W")

		label = tk.Label(self, text="logt")
		label.grid(column=0, row=5, sticky="E")
		self.logtau = tk.StringVar("")
		entry = tk.Entry(self, textvariable=self.logtau)
		entry.grid(column=1, row=5, sticky="W")

		label = tk.Label(self, text="lam0")
		label.grid(column=0, row=6, sticky="E")
		self.logtau = tk.StringVar("")
		entry = tk.Entry(self, textvariable=self.logtau)
		entry.grid(column=1, row=6, sticky="W")

		#--- compare
		button = tk.Button(self, text="Compare", command=self.compare_1D_spectra)
		button.grid(column=1, row=4, sticky="W")

		label = tk.Label(self, text="X")
		label.grid(column=0, row=7, sticky="E")
		self.idx = tk.StringVar("")
		entry = tk.Entry(self, textvariable=self.idx)
		entry.grid(column=1, row=7, sticky="W")

		label = tk.Label(self, text="Y")
		label.grid(column=0, row=8, sticky="E")
		self.Y = tk.StringVar("")
		entry = tk.Entry(self, textvariable=self.Y)
		entry.grid(column=1, row=8, sticky="W")

	def load_inverted_spectrum(self):
		if (self.inverted_spectrum is not None):
			answer = askyesno(title="Checkpoint",
							  message="Inverted spectrum already loaded. Do you want to load new one?",
							  icon="info")
			if not answer:
				return

		self.inverted_spectrum = self._load_spectrum()
		if self.inverted_spectrum is not None:
			self.inv_lbl.config(fg="green")

	def load_observed_spectrum(self):
		if (self.observed_spectrum is not None):
			answer = askyesno(title="Checkpoint",
							  message="Observed spectrum already loaded. Do you want to load new one?",
							  icon="info")
			if not answer:
				return

		self.observed_spectrum = self._load_spectrum()
		if self.observed_spectrum is not None:
			self.obs_lbl.config(fg="green")

	def _load_spectrum(self):
		filenames = fd.askopenfilenames(
			title="Open files",
			initialdir="/media/dusan/storage/RH/globin_results/custom_atmos_v2/",
			filetypes=filetypes)
		
		if filenames:
			fpath = filenames[0]
			if "fits" in fpath:
				return globin.Observation(fpath)
			else:
				showinfo(title="Unknown file type",
						 message="Only .fit(s) file type is allowed.",
						 icon="error")
		return None

	def get_chi2(self):
		if (self.inverted_spectrum is None):
			showinfo(title="Missing inverted spectrum",
					 message="Load inverted spectrum before computing chi2.",
					 icon="error")
			return False
		if (self.observed_spectrum is None):
			showinfo(title="Missing observed spectrum",
					 message="Load observed spectrum before computing chi2.",
					 icon="error")
			return False

		if self.inverted_spectrum.shape!=self.observed_spectrum.shape:
			showinfo(title="Incompatible dimensions",
					 message="Inverted and observed spectra have different dimensions. Please, load the correct spectra for comparison.",
					 icon="error")
			return False			
		
		Ndata = len(self.inverted_spectrum.wavelength)
		self.chi2 = np.sum( (self.inverted_spectrum.spec - self.observed_spectrum.spec)**2, axis=(2,3)) / Ndata

	def _get_map_layout(self):
		self.stokes = []
		num = 0
		if self.stokesI.get()==1:
			num += 1
			self.stokes.append("I")
		if self.stokesQ.get()==1:
			num += 1
			self.stokes.append("Q")
		if self.stokesU.get()==1:
			num += 1
			self.stokes.append("U")
		if self.stokesV.get()==1:
			num += 1
			self.stokes.append("V")

		if num==1:
			return 1, 1
		if num==2:
			return 1, 2
		if num==3:
			return 1, 3
		if num==4:
			return 2, 2
		else:
			showinfo(title="Map layour",
					 message="None of Stokes components is selected for the map creating.",
					 icon="error")
			return None, None

	def map_mouse_click(self):
		pass

	def plot_spectral_maps(self):
		if self.inv_flag.get() and (self.inverted_spectrum is not None):
			spectral_map = self.inverted_spectrum
		elif self.obs_flag.get() and (self.observed_spectrum is not None):
			spectral_map = self.observed_spectrum
		else:
			showinfo(title="Spectrum map",
					 message="Neither of spectra has been loaded or selected.",
					 icon="info")
			return None

		nrows, ncols = self._get_map_layout()
		if nrows is None:
			return None

		if self.map_window is not None:
			self.map_window.destroy()
			self.map_window.update()

		idl = 0

		width = ncols*4 + 1
		height = nrows*4 + 1
		figure = Figure(figsize=(width, height), dpi=90)

		# create FigureCanvasTkAgg object
		self.map_window = TopWindow(self)
		canvas = FigureCanvasTkAgg(figure, self.map_window)
		canvas.get_tk_widget().grid(column=0, row=0, padx=PADX, pady=PADY)

		# create map
		gs = figure.add_gridspec(ncols=ncols, nrows=nrows)
		for idx in range(nrows):
			for idy in range(ncols):
				ida = idx * ncols + idy
				ax = figure.add_subplot(gs[idx,idy])

				ids = IDS[self.stokes[ida]]

				ax.set_title(TITLE[self.stokes[ida]])
				im = ax.imshow(spectral_map.spec[:,:,idl,ids], cmap="gray", origin="lower")

		canvas.draw()

		figure.canvas.callbacks.connect('button_press_event', self.map_window.map_mouse_click)

	def compare_1D_spectra(self):
		if (self.inverted_spectrum is None) or \
		   (self.observed_spectrum is None):
			showinfo(title="1D spectrum cmpare",
					 message="Observed/inverted spectra is missing.",
					 icon="error")
			return None