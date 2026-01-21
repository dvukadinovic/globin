.. _intro:

Introduction
============

**globin** is spectropolarimetric inversion code developed to self-consistently infer atmospheric and atomic line parameters from high-spectral and high-spatial resolution observations of the solar atmosphere from the near ultraviolet to infrared wavelengths. Main purpose of **globin** is to infer the atomic line parameters, such as, transition probability, line centra wavelength and element abundance, spatially coupling spectra from each pixel from the observed field of view. This spatial coupling improves the inference by reducing the coupling between atomic and atmospheric parameters. More information about the method and its implementation is presented in `Vukadinovic et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.262V/abstract>`_.

The forward modelling of **globin** module is based on the well-tested non-LTE code `RH <https://github.com/han-uitenbroek/RH>`_. RH is modified and ported into Python environment using Cython as a separate module named `pyrh <https://github.com/dvukadinovic/pyrh>`_. 

The modifications of RH code are performed to enable parallel executions on superclusters using MPI protocol using **mpi4py** module from **schwimmbad** module. 

.. note::
    **globin** is well tested for LTE forward modelling and inversion. Tests performed in non-LTE mode show machine precision difference in spectra from **globin** and original RH, verifying that modifications of RH did not impact in any way the core of the code. Inversions in non-LTE are possible, but are not verifyed yet. 

Inverter
============

Main class that is used for synthesizing spectra from given atmospheric model, or inverting the observations is the ``Inverter``. Through inverter we specify the input files path that contain all neccessary variables for running the **globin**.

Quick start
---------------

Atmosphere
---------------

Input atmosphere

Synthesis
---------------

Inversion
---------------
