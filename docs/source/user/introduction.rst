.. _intro:

Introduction
============

Main purpose of **globin** package is to invert high-resolution NUV solar observations from SUSI instrument onboard SUNRISE III balloon-borne mission for inference of atomic line parameters. Package is able to invert ocilator strength (log(gf)) and rest wavelength of spectral line along with atmospheric parameters (temperature, vertical velocity, micro-turbulent velocity, magnetic field strength, field inclination and azimuth).

Parametrization of atmospheric structure is done using node approach where in node we have value of given physical parameters. Code minimizes difference between provided observed profiles and synthesised spectra using Levenberg-Marquardt algorithm.

In case of atomic parameters we have implemented a global minimization. In this approach, atomic parameters are called global and are the same for each observed Stokes profile in given field-of-view. This way, we have coupled information from all the pixels to have better inference of atomic parameters.

Atmospheric structure is assumed to be given in optical depth scale which is then finly interpolated on scale -6 to 1 with step size of 0.1. Interpolation between atmospheric nodes is performed with Bezier's 2nd and 3rd order polynomials.

Response function, necessary for spectropolarimetric inversion using LM algorithm, are calculated numerical using ``rf_ray`` executable of RH. It is modified executable of ``sovleray``.

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
