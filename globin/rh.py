class RHAtoms(object):
    pass

class RHMolecules(object):
    pass

class RHKeywords(object):
    def __init__(self, keywor_input=None):
        if keywor_input:
            self._read_keyword_input_file(keywor_input)

        self.set_LTE_parameters()

    def _read_keyword_input_file(self, fpath):
        pass

    def set_parameters(self, **parameters):
        keywords = self.dict()
        for key in parameters:
            if key in keywords:
                keywords[key] = parameters[key]

    def set_LTE_parameters(self):
        self.NRAYS = 1
        self.N_MAX_SCATTER = 0
        self.I_SUM = -1

        self.N_MAX_ITER = 1
        self.ITER_LIMIT = 1e-2

        self.NG_ORDER = 0
        self.NG_DELAY = 10
        self.NG_PERIOD = 3

        self.PRD_N_MAX_ITER = 1
        self.PRD_ITER_LIMIT = 1e-2
        self.PRD_NG_DELAY = None
        self.PRD_NG_ORDER = None
        self.PRD_NG_PERIOD = None

        self.PRD_ANGLE_DEP = None
        self.XRD = None

        self.J_FILE = "J.dat"
        self.STARTING_J = "NEW_J"
        self.BACKGROUND_FILE = "background.dat"
        self.OLD_BACKGROUND = False

        self.METALLICITY = None

        self.KURUCZ_DATA = "kurucz.input"
        self.SOLVE_NE = "NONE"
        self.RLK_SCATTER = False

        self.HYDROGEN_LTE = True
        self.HYDROSTATIC = None

        self.OPACITY_FUDGE = None
        self.SPECTRUM_OUTPUT = None

        self.OPACITY_OUTPUT = None
        self.RADRATE_OUTPUT = None
        self.DAMPING_OUTPUT = None
        self.COLLRATE_OUTPUT = None

        self.VMICRO_CHAR = 5 # [km/s]
        self.VMACRO_TRESH = 0 # [km/s]

        self.S_INTERPOLATION = "S_BEZIER3"
        self.S_INTERPOLATION_STOKES = "DELO_BEZIER3"

        self.LAMBDA_REF = 500.0 # [nm]
        self.VACUUM_TO_AIR = False

        self.STOKES_MODE = "FULL_STOKES"
        self.MAGNETO_OPTICAL = False

        self.BACKGROUND_POLARIZATION = True
        self.LIMIT_MEMORY = False
        self.ALLOW_PASSIBE_BB = False

        self.PRINT_CPU = False
        self.N_THREADS = None

    def create_input_file(self, fpath):
        with open(fpath, "w") as file:
            keywords = self.dict()
            for key in keywords:
                if key==True:
                    value = "TRUE"
                elif key==False:
                    value = "FALSE"
                elif key is None:
                    prefix = "#"
                    value = ""
                else:
                    value = str(keywords[key])
                file.write(f"{prefix}  {key} = {value}\n")
                prefix = ""
