# Limit the number of threads used by numpy/scipy when we run using MPI
import os
os.environ["OMP_NUM_THREADS"] = "1"

from .atmos import Atmosphere
from .inversion import Inverter
from .chi2 import Chi2
from .spec import Observation, Spectrum
from .visualize import show, plot_atmosphere, plot_spectra, plot_rf, plot_chi2
from .utils import Stats
from .chi2 import compute_chi2
from .atoms import AtomPars
from .rh import *

import globin.invert as invert

__all__ = ["rh", "atmos", "atoms", "inversion", "spec", "tools", "input", "visualize", "utils", "invert"]
__name__ = "globin"
__path__ = os.path.dirname(__file__)

#--- container for statistical information about the code performance
# collect_stats = False
# statistics = Stats()

from scipy.interpolate import splrep
import numpy as np

#--- FAL C model (ref.): reference model if not given otherwise
FALC = Atmosphere(f"{__path__}/data/falc.dat", atm_type="spinor")
FALC_temp_tck = splrep(FALC.data[0,0,0], FALC.data[0,0,1])
FALC_pg_tck = splrep(FALC.data[0,0,0], FALC.pg[0,0])

#--- HSRA model
HSRA = Atmosphere(f"{__path__}/data/hsrasp_vmic0.dat", atm_type="spinor")
T0_HSRA = HSRA.T[0,0, np.where(HSRA.logtau==0)[0][0]]