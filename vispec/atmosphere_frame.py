import tkinter as tk
from tkinter import ttk

from constants import PADX, PADY

class AtmosphereFrame(ttk.Frame):
	def __init__(self, container, master):
		tk.Frame.__init__(self, container)
		self.config(width=300, height=150, relief=tk.RAISED)