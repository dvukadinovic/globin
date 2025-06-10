.. _install:


Installation
============

Prior to installing **globin** module, it is recommended to create a new Python environment and install all requierd packages from ``requierments.txt``. The minimal recommended Python version is 3.6. 

To install **globin** in a newly created environment type:

.. code-block:: bash
	
	pip3 install -e /path/to/package

The forward modelling and inversion mode of **globin** rely on the `RH code <https://github.com/han-uitenbroek/RH>`_. The modified version of RH, **pyrh**, optimized to be used directly from Python environment, can be installed from `here <https://github.com/dvukadinovic/pyrh>`_. The **pyrh** module is not necessary only if user wishes to visualise inversion results or manipulate with atmospheric models.

Test
---------------

To test the package, copy the sample files located in 'globin/test' directory and simply run ``python run.py``. In the terminal, it will write down the current progress of the test. Test results will then be writen inside `runs` directory under the sub-directory that was given in the ``run.py`` script.