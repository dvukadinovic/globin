import globin

globin.interp_degree = 3
atmos = globin.construct_atmosphere_from_nodes("node_atmosphere", vmac=0, output_atmos_path="atmos.fits")