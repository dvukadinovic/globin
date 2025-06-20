.. _install:


Installation
============

.. note::
	Prior to installing **globin** module, it is recommended to create a new Python environment and install all requierd packages from ``requierments.txt``. Check :ref:`Create environment`.

To install **globin** in a newly created environment type:

.. code-block:: bash

	pip install -e /absolute/path/to/package

The forward modelling and inversion mode of **globin** rely on the `RH code <https://github.com/han-uitenbroek/RH>`_. The modified version of RH, **pyrh**, optimized to be used directly from Python environment, can be installed from `here <https://github.com/dvukadinovic/pyrh>`_. The **pyrh** module is not necessary only if user wishes to visualise inversion results or manipulate with atmospheric models.

Create environment
---------------

Create a new conda virtual environment using Python 3.10 (the minimal recommended version is 3.6).

.. code-block:: bash

	conda create --name globin_env python=3.10	

Due to some required packages, such as, ``emcee``, we need to add the ``conda-forge`` in the package search channels:

.. code-block:: bash

	conda config --add channels conda-forge

Now, install all the required packages:

.. code-block:: bash

	conda install --file requirements.txt
