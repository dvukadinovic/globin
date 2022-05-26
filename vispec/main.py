import tkinter as tk
from tkinter import ttk

from spectrum_frame import SpectrumFrame
from atmosphere_frame import AtmosphereFrame

from constants import PADX, PADY

class MainWindow(tk.Tk):
	def __init__(self):
		super().__init__()
		# self.geometry(width=615, height=180)

		container = ttk.Frame(self)
		container.grid(column=0, row=1, padx=PADX, pady=PADY, sticky="EW")

		self.spec_frame = SpectrumFrame(container, self)
		self.spec_frame.grid(column=0, row=0, padx=PADX, pady=PADY, sticky="N")

		# sep = ttk.Separator(container, orient="vertical")
		# sep.pack()

		self.atmos_frame = AtmosphereFrame(container, self)
		self.atmos_frame.grid(column=2, row=0, padx=PADX, pady=PADY, sticky="N")

app = MainWindow()
app.mainloop()