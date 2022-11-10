import globin
#from scipy.interpolate import splrep, splev
#from scipy.constants import k as kb
#import numpy as np

#nHtot = np.sum(globin.falc.data[0,0,8:], axis=0) # [1/cm3]
#ne = globin.falc.data[0,0,2] # [1/cm3]
#temp = globin.falc.data[0,0,1] # [K]
#pg = (nHtot + ne)*1e6 * kb * temp # [N/m2]
#tck = splrep(globin.falc.logtau, pg)

atmos = globin.utils.construct_atmosphere_from_nodes("node_atmosphere_uniform", 
	vmac=0, output_atmos_path="atmos_uniform.fits")
print(atmos.pg_top)