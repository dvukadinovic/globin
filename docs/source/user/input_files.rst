.. _inputs:


Input files
============

The **globin** relies on the same input files as those used by the RH code, such as, ``atoms.input``, ``molecules.input``, ``keywords.input``, and ``kurucz.input`` (``ray.input`` is obsolete). The names of these files are hardcoded and should not be altered. Additional input file ``params.input`` is used to control the **globin**. Here, we will only cover **globin** specific keywords used to synthesize spectra or perform inversions of spectropoalrimetric observations. Users unfamiliar with the RH input files are suggested to consult the codes own documentation. This is especially important if non-LTE modelling is to be used since many parameters can be adjusted and will impact the output spectrum. In case of the LTE modelling, RH parameters are set by default inside **globin** and users do not need to know the specifics of the RH code.

* 'mode': state in which **globin** will be run. It is used to specify which paremeters are to be read and performs sanity check.
    #. mode=0 -- synthesize spectra from given atmosphere model.
    #. mode=1 -- the pixel-by-pixel inversion of atmospheric parameters only.
    #. mode=2 -- the pixel-by-pixel inversion of atmospheric and atomic parameters.
    #. mode=3 -- the pixel-by-pixel inversion of atmospheric parameters and the global inversion of atomic parameters

    Aditionally, we can invert for grey stray light in pixel-by-pixel model (for modes 1-3). Also, we have an option for inversion of opacity fudge coefficients that are always pixel-by-pixel inverted.

* 'n_thread': number of threads used for multiprocessing computation.
* 'interp_deg': interpolation degree for building atmosphere structure from the nodes.
