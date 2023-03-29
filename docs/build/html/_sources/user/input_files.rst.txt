.. _inputs:


Input files
============

Since the **globin** is based on the RH code, we rely on the same input parameters as are used for the RH. For detailed description of each file and underlying parameters, refer to the RH documentation.

The input file for the **globin** is, by default, named 'parameters.input'. It contains all neccesssary parameters for the synthesis/inversion.

* 'mode': state in which **globin** will be run. It is used to specify which paremeters are to be read and performs sanity check.
    #. mode=0 -- synthesize spectra from given atmosphere model.
    #. mode=1 -- the pixel-by-pixel inversion of atmospheric parameters only.
    #. mode=2 -- the pixel-by-pixel inversion of atmospheric and atomic parameters.
    #. mode=3 -- the pixel-by-pixel inversion of atmospheric parameters and the global inversion of atomic parameters

    Aditionally, we can invert for grey stray light in pixel-by-pixel model (for modes 1-3). Also, we have an option for inversion of opacity fudge coefficients that are always pixel-by-pixel inverted.

* 'n_thread': number of threads used for multiprocessing computation.
* 'interp_deg': interpolation degree for building atmosphere structure from the nodes.
