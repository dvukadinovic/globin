.. globin documentation master file, created by
   sphinx-quickstart on Wed Nov 18 18:03:13 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _Home:

Welcome to the globin's documentation!
======================================

**globin** is a Python package for spectropolarimetric inversion. It is based on the forward modeling by the `RH code <https://www2.hao.ucar.edu/spectropolarimetry/rh>`_ . 

The **globin** inversion is based on spatial coupling of spectra from different pixels through atomic line parameters. We requiere that the atomic line parameter is unique in the observed field of view. This coupling loweres down the number of free parameters (per pixel) and introduces a large constrain on the atomic line parameters. This permits an accurate retrieval of atomic line parameters and pixel-by-pixel atmospheric parameters. 

Currently, supported atomic line parameters are transition probability (through log(gf) measure) and line rest wavelength (given as a wavelength shift in respect to the input wavelength). In future, we aim to account also for the element abundance and the energy of lower level. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user/introduction
   user/installation
   user/keywords
   user/input_files

   changelog

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
