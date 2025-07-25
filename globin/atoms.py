from astropy.io import fits
import numpy as np

import globin

orbitals = {"S": 0, "P": 1, "D": 2, "F": 3, 
            "G": 4, "H": 5, "I": 6, "J": 7,
            "K": 8, "L": 9, "M": 10, "N": 11,
            "O": 12, "Q": 13, "R": 14, "T": 15}

def gamma_function(a, b, c):
    if a==0:
        return 0

    result = a*(a+1) + b*(b+1) - c*(c+1)
    result /= 2*a*(a+1)

    return result

def get_JK_Lande(J, K, l, S1, L1, J1):
    gJK = 1 + gamma_function(J, 1/2, K)
    gJK += gamma_function(J,K,1/2) * gamma_function(K,J1,l) * gamma_function(J1,S1,L1)
    return gJK

def get_LS_Lande(S, L, J):
    gLS = 1 + gamma_function(J, S, L)
    return gLS

def get_effective_Lande(gLlow, gLup, Jlow, Jup):
    return 0.5*(gLup+gLlow) + 0.25*(gLlow-gLup) * (Jlow*(Jlow+1.0) - Jup*(Jup+1.0))

class Line(object):
    """
    Class for storing spectral line data.
    """
    def __init__(self, lineNo=None, lam0=None,
                    loggf=None, loggf_min=None, loggf_max=None,
                    dlam=None, dlam_min=None, dlam_max=None,
                    ion=None, state=None, elow=None, eup=None,
                    gLlow=None, gLup=None, Jlow=None, Jup=None,
                    Grad=None,
                    config_low=None, config_up=None,
                    swap=False):
        self.lineNo = lineNo
        self.lam0 = lam0 # [1/nm]
        self.loggf = loggf
        self.loggf_min = loggf_min
        self.loggf_max = loggf_max
        self.dlam = dlam # [mA]
        self.dlam_min = dlam_min
        self.dlam_max = dlam_max

        self.Jlow = Jlow
        self.Jup = Jup

        if Grad is not None:
            self.Grad = 10**Grad # [1/s]

        self.config_low = config_low
        self.config_up = config_up

        self.ion = ion
        self.state = state
        self.elow = elow # [eV]
        self.eup = eup # [eV]

        self.gLlow = gLlow
        self.gLup = gLup

        self.swapped = swap
        if swap:
            self._swap()

        if (self.config_low is not None) and (self.config_up is not None):
            self.get_LS_numbers()

    def __str__(self):
        return "<LineNo: {}, lam0: {}, loggf: {}\n  loggf_min: {}, loggf_max: {}\n  dlam: {}, dlam_min: {}, dlam_max: {}>".format(self.lineNo, self.lam0, self.loggf, self.loggf_min, self.loggf_max, self.dlam, self.dlam_min, self.dlam_max)

    def _swap(self):
        """
        Swap levels' data since they were reversed in the Kurucz line list.
        """
        self.Jlow, self.Jup = self.Jup, self.Jlow
        self.elow, self.eup = self.eup, self.elow
        self.gLlow, self.gLup = self.gLup, self.gLlow
        self.config_low, self.config_up = self.config_up, self.config_low

    def get_effective_Lande(self):
        self.gLeff = 0.5*(self.gLup + self.gLlow) + 0.25*(self.gLup - self.gLlow) * (self.Jup*(self.Jup + 1.0) - self.Jlow*(self.Jlow + 1.0))

        return self.gLeff

    def get_LS_numbers(self):
        # lower level
        config_low = self.config_low[-2:]
        self.term_low = config_low
            
        self.has_low_LS_numbers = True
        try:
            self.Slow = int(config_low[0]) - 1
            self.Slow /= 2
        except:
            self.Slow = None
            self.has_low_LS_numbers = False
        
        try:   
            self.Llow = orbitals[config_low[1]]
        except:
            self.Llow = None
            self.has_low_LS_numbers = False

        # upper level
        config_up = self.config_up[-2:]
        self.term_up = config_up
        
        self.has_up_LS_numbers = True
        try:
            self.Sup = int(config_up[0]) - 1
            self.Sup /= 2
        except:
            self.Sup = None
            self.has_up_LS_numbers = False
        
        try:   
            self.Lup = orbitals[config_up[1]]
        except:
            self.Lup = None
            self.has_up_LS_numbers = False

        self.LS_line = False
        if self.has_low_LS_numbers and self.has_up_LS_numbers:
            self.LS_line = True

    def get_LS_Lande(self):
        """
        Compute the Lande factor for each level assuming LS coupling sheme.
        """
        if not self.LS_line:
            return None, None

        gLlow = 0
        if self.Jlow!=0:
            gLlow = self.Jlow*(self.Jlow+1) + self.Slow*(self.Slow+1) - self.Llow*(self.Llow+1)
            gLlow /= 2*self.Jlow*(self.Jlow+1)
            gLlow += 1

        gLup = 0
        if self.Jup!=0:
            gLup = self.Jup*(self.Jup+1) + self.Sup*(self.Sup+1) - self.Lup*(self.Lup+1)
            gLup /= 2*self.Jup*(self.Jup+1)
            gLup += 1

        self.gLlow = gLlow
        self.gLup = gLup

        # print Lande factors in the order of levels as it is in the file
        # if self.swapped:
        #     return gLup, gLlow

        # return gLlow, gLup

    @property
    def Aji(self):
        glow = 2*self.Jlow + 1
        gup = 2*self.Jup + 1
        gf = 10**self.loggf
        nu = globin.LIGHT_SPEED/self.lam0/1e-9
        Aji = (2*np.pi*globin.ELECTRON_CHARGE**2*nu**2)/(globin.EPSILON_0*globin.ELECTRON_MASS*globin.LIGHT_SPEED**3) * gf/gup
        return Aji

    @property
    def Bji(self):
        nu = globin.LIGHT_SPEED/self.lam0/1e-9
        Bji = globin.LIGHT_SPEED**2/(2*globin.PLANCK*nu**3) * self.Aji
        return Bji

    @property
    def Bij(self):
        glow = 2*self.Jlow + 1
        gup = 2*self.Jup + 1
        Bij = gup/glow * self.Bji
        return Bij

def read_RLK_lines(fpath):
    """
    Read RLK line list for given spectral region.

    Line list is given in Kurucz format, except, there is one less blanko
    space at beginning of line.

    Parameters:
    ---------------
    fpath : str
        path to file containing the Kurucz line list.

    Return:
    ---------------
    text_lines : list(str)
        list of lines in input file at 'fpath'.
    RLK_lines : list(Lines)
        list of Lines() objects which storse Kurucz line list data (lam0 and log(gf)).
    """
    text_lines = open(fpath, "r").readlines()
    
    RLK_lines = []

    for idl, line in enumerate(text_lines):
        # ignore blank lines in Kurucz line list file
        # if len(line)<160:
        #     continue
        lam0 = float(line[0:10])
        loggf = float(line[10:17])
        aux = float(line[17:23])
        decimal, integer = np.modf(aux)
        ion = int(integer)
        state = round(decimal*100)
        elow = float(line[23:35])/ 8065.544 # [1/cm --> eV]
        eup = float(line[51:63])/ 8065.544 # [1/cm --> eV]
        gLlow = float(line[143:148]) / 1e3
        gLup = float(line[149:154]) / 1e3
        Jlow = float(line[35:40])
        Jup = float(line[63:68])
        Grad = float(line[81:86])
        config_low = line[42:54].rstrip(" ")
        config_up = line[70:81].rstrip(" ")

        swap = False
        if elow>eup:
            swap = True

        RLK_lines.append(Line(lineNo=idl+1, lam0=lam0, loggf=loggf, 
                              ion=ion, state=state,
                              elow=elow, eup=eup,
                              gLlow=gLlow, gLup=gLup,
                              Jlow=Jlow, Jup=Jup,
                              Grad=Grad,
                              config_low=config_low, config_up=config_up,
                              swap=swap))

    return text_lines, RLK_lines

def read_init_line_parameters(fpath):
    """
    Read input data for lines.

    Each line has following structure:

        parameter   line_number   initial_value   min_value   max_value

    'parameter' --> parameter name (currently supported are 'loggf' and 'dlam' in mA)
    'line_number' --> position of line in RLK line list
    'initial_value' --> initial value for the parameter
    'min_value' --> lower limit value for the parmaeter
    'max_value' --> upper limit value for the parmaeter

    Parameters:
    ---------------
    fpath : str
        path to file in which input atomic data parameters are stored.

    Return:
    ---------------

    """
    lines = open(fpath, "r").readlines()

    lines_to_fit = []

    for line in lines:
        line = list(filter(None,line.rstrip("\n").split(" ")))
        # if line[0]!="#":
        if "#" not in line[0]:
            lineNo = int(line[1])-1
            par = float(line[2])
            par_min = float(line[3])
            par_max = float(line[4])
            if line[0]=="loggf":
                flag = [True if spec_line.lineNo==lineNo else False for spec_line in lines_to_fit]
                # if we have already inputed the parameters for given line, just update rest of parameters
                if any(flag):
                    ind = flag.index(True)
                    lines_to_fit[ind].loggf = par
                    lines_to_fit[ind].loggf_min = par_min
                    lines_to_fit[ind].loggf_max = par_max
                else:
                    spec_line = Line(lineNo=lineNo, loggf=par, loggf_min=par_min, loggf_max=par_max)
                    lines_to_fit.append(spec_line)
            
            elif line[0]=="dlam":
                flag = [True if spec_line.lineNo==lineNo else False for spec_line in lines_to_fit]
                # if we have already inputed the parameters for given line, just update rest of parameters
                if any(flag):
                    ind = flag.index(True)
                    lines_to_fit[ind].dlam = par
                    lines_to_fit[ind].dlam_min = par_min
                    lines_to_fit[ind].dlam_max = par_max
                else:
                    spec_line = Line(lineNo=lineNo, dlam=par, dlam_min=par_min, dlam_max=par_max)
                    lines_to_fit.append(spec_line)

    return lines_to_fit

def init_line_pars(lineNo, RLK_line_list_path, line_pars_path=None, min_max={"loggf" : 1, "dlam" : 25}):
    _, RLK_lines = read_RLK_lines(RLK_line_list_path)

    pars = list(lineNo.keys())

    if line_pars_path is not None:
        out = open(line_pars_path, "w")
        out.write("# parID   LineNo   initial   min     max\n")

    outing_lines = []
    for par in pars:
        lines = []
        dpar = min_max[par]
        for lineID in lineNo[par]:
            for i_ in range(len(RLK_lines)):
                if RLK_lines[i_].lineNo==lineID:
                    # check if this line is not already in the list
                    # if it is not, append and set index to -1
                    # if it is, get the index of line in list                        
                    lines.append(RLK_lines[i_])

                    if par=="loggf":
                        # set line min/max for log(gf)
                        lines[-1].loggf_min = lines[-1].loggf-dpar
                        if lines[-1].loggf_min<-10:
                            lines[-1].loggf_min = -10
                        lines[-1].loggf_max = lines[-1].loggf+dpar
                        if lines[-1].loggf_max>1:
                            lines[-1].loggf_max = 1
                        
                        # check if log(gf) is in min/max range
                        # lines[-1].loggf += np.random.normal(0, dpar/5)
                        if lines[-1].loggf > lines[-1].loggf_max:
                            lines[-1].loggf = lines[-1].loggf_max
                        if lines[-1].loggf < lines[-1].loggf_min:
                            lines[-1].loggf = lines[-1].loggf_min

                    if par=="dlam":
                        # set line min/max for dlam
                        lines[-1].dlam_min = -dpar
                        lines[-1].dlam_max = dpar
                        
                        # check if dlam is in min/max range
                        lines[-1].dlam = np.random.normal(0, dpar/5)
                        if lines[-1].dlam > lines[-1].dlam_max:
                            lines[-1].dlam = lines[-1].dlam_max
                        if lines[-1].dlam < lines[-1].dlam_min:
                            lines[-1].dlam = lines[-1].dlam_min
                    
                    ind = -1
                    for k_, line in enumerate(outing_lines):
                        if lineID == line.lineNo:
                            ind = k_
                            break
                    if ind==-1:
                        outing_lines.append(lines[-1])

        if line_pars_path is not None:
            for line in lines:
                if par=="loggf":
                    out.write("loggf    ")
                    out.write("{: 3d}    ".format(line.lineNo))
                    out.write("{: 4.3f}    ".format(line.loggf))
                    out.write("{: 4.3f}    ".format(line.loggf_min))
                    out.write("{: 4.3f}\n".format(line.loggf_max))
                if par=="dlam":
                    out.write("dlam     ")
                    out.write("{: 3d}   ".format(line.lineNo))
                    out.write("{: 5.3f}   ".format(line.dlam))
                    out.write("{: 5.3f}   ".format(line.dlam_min))
                    out.write("{: 5.3f}\n".format(line.dlam_max))

    if line_pars_path is not None:
        out.close()

    return outing_lines

def check_init_loggf():
    """
    For a given lines log(gf) initial values, test if these lines are seen
    in spectrum. We requier that line is 1% stronger than continuum intensity.
    """
    pass

def write_line_pars(fpath, loggf=None, loggfID=None, dlam=None, dlamID=None, min_max={"loggf" : 1, "dlam" : 25}):
    out = open(fpath, "w")

    out.write("# parID   LineNo   initial   min     max\n")

    if loggf is not None:
        for val,idl in zip(loggf, loggfID):
            out.write("loggf    ")
            out.write("{: >3d}    ".format(idl+1))
            out.write("{: 4.3f}    ".format(val))
            out.write("{: 4.3f}    ".format(val-min_max["loggf"]))
            out.write("{: 4.3f}\n".format(val+min_max["loggf"]))
    
    if dlam is not None:
        for val,idl in zip(dlam, dlamID):
            out.write("dlam     ")
            out.write("{: >3d}    ".format(idl+1))
            out.write("{: 5.3f}    ".format(val))
            out.write("{: 5.3f}    ".format(val-min_max["dlam"]))
            out.write("{: 5.3f}\n".format(val+min_max["dlam"]))

    out.close()

def update_line_list(fpath, parameter, line_number, value):
    """
    Update the Kurucz line list for the given atomic parameter.

    Parameters:
    -----------
    parameter : str 
        name of the parameter to be updated
    line_number : list or ndarray
        the corresponding line number position from line list to be updated. 
        Counting starts from 1.
    value : list or ndarray
        values of atomic parameter to update the Kurucz line list. Needs to be
        the same dimension as 'line_number'. For 'dlam' parameter values are
        assumed to be given in mA.
    """
    lines = open(fpath, "r").readlines()

    if parameter not in ["loggf", "dlam"]:
        raise ValueError(f"Currently unsupported parameter type '{parameter}'")

    if len(line_number)!=len(value):
        raise ValueError("Unequal size of the 'line_number' and 'value' lists.")

    for idl, line_num in enumerate(line_number):
        # line_num -= 1 # to comply with Python counting
        kurucz_line = list(lines[line_num])
        if parameter=="loggf":
            kurucz_line[10:17] = "{:7.3f}".format(value[idl])
        if parameter=="dlam":
            lam0 = float("".join(kurucz_line[0:10]))
            kurucz_line[0:10] = "{:10.4f}".format(value[idl]/1e4 + lam0)
        lines[line_num] = "".join(kurucz_line)

    out = open(fpath, "w")
    out.writelines(lines)
    out.close()

class AtomPars(object):
    """
    Class storing atomic parameters from inversion for easier access and
    analysis of results.
    """
    units = {"loggf" : "dex",
            "dlam"  : "mA"}

    def __init__(self, fpath=None):
        self.data = {"loggf"    : None,
                     "loggfIDs" : None,
                     "dlam"     : None,
                     "dlamIDs"  : None}

        self.header = {"loggf"  : None,
                       "dlam"   : None}

        self.limit_values = {"loggf" : None,
                             "dlam"  : None}

        self.mode = 0
        self.nx, self.ny = None, None
        self.nl = {"loggf" : None,
                   "dlam"  : None}

        if fpath:
            self.read_atom_pars(fpath)

    @property
    def loggf(self):
        return self.data["loggf"]

    @property
    def dlam(self):
        return self.data["dlam"]
    
    def read_atom_pars(self, fpath):
        hdu = fits.open(fpath)

        try:
            self.mode = hdu[0].header["MODE"]
        except:
            pass

        pars = ["loggf", "dlam"]
        hdu_ind = []
        for parameter in pars:
            try:
                ind = hdu.index_of(parameter)
                self.header[parameter] = hdu[ind].header
                self.data[parameter] = np.array(hdu[ind].data[:,:,1,:], dtype=np.float64)
                self.data[parameter+"IDs"] = np.array(hdu[ind].data[0,0,0], dtype=np.int32)
                self.nx, self.ny = self.data[parameter].shape[:-1]
                self.nl[parameter] = len(self.data[parameter+"IDs"])
            except Exception as e:
                print(parameter)
                print(e)

            try:
                ind = hdu.index_of(f"MINMAX_{parameter}")
                self.limit_values[parameter] = hdu[ind].data
            except:
                print(f"[Info] Did not found boundary values for {parameter}.")

class PSE(object):
    def __init__(self):
        self.symbols = ["H",  "He", 
                        "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
                        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
                        "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                        "Ga", "Ge", "As", "Se", "Br", "Kr",
                        "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                        "In", "Sn", "Sb", "Te", "I",  "Xe",
                        "Cs", "Ba", 
                        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
                        "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
                        "Fr", "Ra",
                        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm"]

    def get_element_symbol(self, element_number):
        return self.symbols[int(element_number-1)]

    def get_element_number(self, element_symbol):
        return self.symbols.index(symbol)+1

class EnergyLevel(object):
    def __init__(self, energy, g, configuration, stage, levelID, abo_level=None):
        self.E = energy
        self.g = g
        self.configuration = configuration
        self.stage = stage
        self.levelID = levelID
        self.abo_level = abo_level

        self.get_term()

    def get_term(self):
        config = list(filter(None, self.configuration.split(" ")))
        try:
            J = int(config[-1])
            term = config[-2]
        except:
            J = None
            term = config[-1]

        S, L, parity = term[-3:]
        S = (int(S) - 1)/2
        L = orbitals[L]

        self.J = J
        self.S = S
        self.L = L
        self.term = term[-4:-1]

class Transition(Line):
    def __init__(self, lower_level, upper_level, loggf):
        super().__init__(loggf=loggf)

        self.lower_level = lower_level
        self.upper_level = upper_level

    def __str__(self):
        msg = f"<Transition: {self.lower_level} --> {self.upper_level}; log(gf) = {self.loggf}>"
        return msg

class Atom(object):
    def __init__(self):
        self.ID = None

    def read_model(self, name):
        text = open(name, "r").readlines()

        text = [line.rstrip("\n") for line in text if (len(line.rstrip("\n"))>1 and line[0]!="#")]

        self.ID = text[0]
        self.Nlevel, self.Nline, self.Ncont, self.Nfixed = map(int, list(filter(None, text[1].split(" "))))

        #--- read in the information of each level

        self.levels = [None]*self.Nlevel

        for idl in range(self.Nlevel):
            text_line = list(filter(None, text[2+idl].split("'")))
            conf = text_line[1]
            E, g = map(float, filter(None, text_line[0].split(" ")))
            _text_line = list(filter(None, text_line[2].split(" ")))
            # stage, levelNo = map(int, filter(None, text_line[2].split(" ")))
            if len(_text_line)==3:
                stage, levelNo, abo_level = map(int, _text_line)
            else:
                abo_level = None
                stage, levelNo = map(int, _text_line)

            self.levels[idl] = EnergyLevel(E, g, conf, stage, levelNo, abo_level)

        #--- read in the transitions

        self.transitions = [None]*self.Nline

        for idt in range(self.Nline):
            text_line = list(filter(None, text[2+self.Nlevel+idt].split(" ")))
            lower_level, upper_level = map(int, text_line[:2])
            oscillator_strength = float(text_line[2])
            profile_type = text_line[3]
            NQ, symmetry, Qcore, Qwing = text_line[4:8]
            vdwapproximation = text_line[8]
            vdw_H = text_line[9:11]
            vdw_He = text_line[11:13]
            radiative = text_line[13]
            Stark = text_line[14]

            loggf = np.log10(oscillator_strength*self.levels[upper_level].g)

            self.transitions[idt] = Transition(lower_level=lower_level, 
                                          upper_level=upper_level,
                                          loggf=loggf)

        #--- read in the continuum transitions

if __name__=="__main__":
    lineNo = {"loggf" : [1,2,3,4], "dlam" : [11,12]}
    init_line_pars(lineNo, "../../rh/Atoms/Kurucz/spinor_window_original", "/home/dusan/line_pars")