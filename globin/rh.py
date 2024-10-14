def create_kurucz_input(line_list_name, fpath):
    out = open(fpath, "w")

    out.write(f"{line_list_name}\n")
    out.close()

class AtoMol(object):
    def __init__(self, name, state="PASSIVE", initial_population="LTE_POPULATIONS"):
        self.name = name
        self.state = state
        self.initial_population = initial_population
        self.output_file = f"pops.{name}.out"

    def __str__(self):
        return f"AtoMol(name='{self.name}', state={self.state}, initial_population={self.initial_population})"

    def __repr__(self):
        return self.__str__()

class RHAtomsMolecules(object):
    def __init__(self):
        self.atoms = []
        self.molecules = []

    def add_atom(self, atom):
        self.atoms.append(atom)

    def add_molecule(self, molecule):
        self.molecules.append(molecule)

    def create_list(self, fpath, type):
        out = open(fpath, "w")

        if type=="atoms":
            values = self.atoms
        if type=="molecules":
            values = self.molecules
        
        N = len(values)

        out.write(f"  {N}\n\n")

        for atomol in values:
            out.write(f"  {atomol.name:10s}    {atomol.state:7s}    {atomol.initial_population:18s}")
            if type=="atoms":
                out.write(f"    {atomol.output_file:20s}\n")
            if type=="molecules":
                out.write("\n")

        out.write("\n")
        out.close()

    def create_atoms_list(self, fpath="atoms.input"):
        self.create_list(fpath, "atoms")

    def create_molecules_list(self, fpath="molecules.input"):
        self.create_list(fpath, "molecules")

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

# atoms
H_6 = AtoMol("H_6.atom")
He = AtoMol("He.atom")
C = AtoMol("C.atom")
N = AtoMol("N.atom")
O = AtoMol("O.atom")
S = AtoMol("S.atom")
Fe = AtoMol("Fe.atom")
Si = AtoMol("Si.atom")
Al = AtoMol("Al.atom")
Na = AtoMol("Na.atom")
Mg = AtoMol("Mg.atom")

# molecules
H2 = AtoMol("H2.molecule")
H2p = AtoMol("H2+.molecule")
C2 = AtoMol("C2.molecule")
N2 = AtoMol("N2.molecule")
O2 = AtoMol("O2.molecule")
CH = AtoMol("CH.molecule")
CO = AtoMol("CO.molecule")
CN = AtoMol("CN.molecule")
NH = AtoMol("NH.molecule")
NO = AtoMol("NO.molecule")
OH = AtoMol("OH.molecule")
H2O = AtoMol("H2O.molecule")

atmols = RHAtomsMolecules()

RHatoms = [H_6, He, C, N, O, S, Fe, Si, Al, Na, Mg]
for atom in RHatoms:
    atmols.add_atom(atom)

RHmolecules = [H2, H2p, C2, N2, O2, CH, CO, CN, NH, NO, OH, H2O]
for molecule in RHmolecules:
    atmols.add_molecule(molecule)