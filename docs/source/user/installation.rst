.. _install:


Installation
============

To have an updated version of **globin** package, download it from `gitlab repo <https://gitlab.gwdg.de/dusan.vukadinovic01/atoms_invert>`_. To install the **globin** package type:

.. code-block:: bash
	
	pip3 install -e /path/to/package

The package is based on the Python implementation of RH through the use of Cython language. Before using the **globin** you must also install and **pyrh** package from `here <https://gitlab.gwdg.de/dusan.vukadinovic01/pyrh>`_.

Test
---------------

To test the package, copy the sample files located in 'globin/test' directory and simply run ``python run.py``. In the terminal, it will write down the current progress of the test. Test results will then be writen inside `runs` directory under the sub-directory that was given in the ``run.py`` script.

Dependencies
---------------

Package is written for Python 3.6+ interpreter and it is depended on following packages:

subprocess>=

multiprocessing>=

astropy>=

os>=

sys>=

time>=

copy>=

numpy>=

matplotlib>=

time>=

rh

  io

  xdrlib
