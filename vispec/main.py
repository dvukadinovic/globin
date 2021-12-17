import tkinter as tk
from astropy.io import fits

"""
At start we have small window with TextBox's and Button.
Additionally, we initialize the window for plotting to which
we draw everything.

This figure will be cleared before we replot something else.

Field for path to observed spectrum.
Field for path to inverted spectrum (optional).
Field for path to atmosphere.
Field for path to inverted atmosphere (optional).

Plot options:
	-- which Stokes components we plot (single or all)
	-- which atmospheric parameter we plot (one or more)

Field for picking the pixel position to plot.

Button to generate the spectrum in separate window.
"""