.. globin documentation master file, created by
   sphinx-quickstart on Wed Nov 18 18:03:13 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _Home:

Welcome to the globin's documentation!
======================================

**globin** is a Python module for perfmoring non-LTE spectropolarimetric inversions of high-spectral and high-spatial resolution observations of the solar atmosphere. Primary motive for developing **globin** is a self-consistent inference of atmospheric and atomic line parameters. This is achieved by spatially coupling atomic line parameters over all pixels in the observed field of view and retriving a single value that simultaneously fit all spectra. The coupled method is described in more detail in `Vukadinović et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.262V/abstract>`_.

.. Currently, supported atomic line parameters are transition probability (through log(gf) measure) and line rest wavelength (given as a wavelength shift in respect to the input wavelength). In future, we aim to account also for the element abundance and the energy of lower level. 

.. It is based on the forward modeling by the `RH code <https://www2.hao.ucar.edu/spectropolarimetry/rh>`_ . 

.. The **globin** inversion is based on spatial coupling of spectra from different pixels through atomic line parameters. We requiere that the atomic line parameter is unique in the observed field of view. This coupling loweres down the number of free parameters (per pixel) and introduces a large constrain on the atomic line parameters. This permits an accurate retrieval of atomic line parameters and pixel-by-pixel atmospheric parameters. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   user/installation
   user/quickstart
   user/introduction
   user/input_files
   user/keywords

   changelog

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
