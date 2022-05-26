import tkinter as tk
from tkinter import ttk

class TopWindow(tk.Toplevel):
	def __init__(self, origin):
		tk.Toplevel.__init__(self)
		self.origin = origin

	def map_mouse_click(self, event):
		pass