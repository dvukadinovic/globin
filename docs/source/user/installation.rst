.. _install:


Installation
============

To have an updated version of **globin** package, download it from `gitlab repo <https://gitlab.gwdg.de/dusan.vukadinovic01/atoms_invert>`_. To install the **globin** package type:

.. code-block:: bash
	
	pip3 install -e /path/to/package

Since the package is wrapped around RH it requieres also and functional installation of the code which can be downloaded from ``here``.

Test
---------------

To test the package, copy the sample files located in 'globin/test' directory and simply run ``python run.py``. In the terminal, it will write down the current progress of the test. Test results will then be writen inside `result` directory.

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