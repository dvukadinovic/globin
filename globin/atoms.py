import numpy as np

class Line(object):
    """
    Class for storing spectral line data.
    """
    def __init__(self, lineNo=None, lam0=None,
                    loggf=None, loggf_min=None, loggf_max=None,
                    dlam=None, dlam_min=None, dlam_max=None):
        self.lineNo = lineNo
        self.lam0 = lam0
        self.loggf = loggf
        self.loggf_min = loggf_min
        self.loggf_max = loggf_max
        self.dlam = dlam
        self.dlam_min = dlam_min
        self.dlam_max = dlam_max

    def __str__(self):
        return "<LineNo: {}, lam0: {}, loggf: {}\n  loggf_min: {}, loggf_max: {}\n  dlam: {}, dlam_min: {}, dlam_max: {}>".format(self.lineNo, self.lam0, self.loggf, self.loggf_min, self.loggf_max, self.dlam, self.dlam_min, self.dlam_max)

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

    for i_, line in enumerate(text_lines):
        lam0 = float(line[0:10])
        loggf = float(line[10:17])

        RLK_lines.append(Line(lineNo=i_+1, lam0=lam0, loggf=loggf))

    return text_lines, RLK_lines

def read_init_line_parameters(fpath):
    """
    Read input data for lines.

    Each line has following structure:

        parameter   line_number   initial_value   min_value   max_value

    'parameter' --> parameter name (currently supported are 'loggf' and 'dlam')
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
        if line[0]!="#":
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

def init_line_pars(lineNo, RLK_line_list_path, line_pars_path="test_line_pars"):
    _, RLK_lines = read_RLK_lines(RLK_line_list_path)

    lines = []
    for lineID in lineNo:
        for i_ in range(len(RLK_lines)):
            if RLK_lines[i_].lineNo==lineID:
                lines.append(RLK_lines[i_])

                lines[-1].loggf_min = lines[-1].loggf-2
                if lines[-1].loggf_min<-10:
                    lines[-1].loggf_min = -10
                lines[-1].loggf_max = lines[-1].loggf+2
                if lines[-1].loggf_max>1:
                    lines[-1].loggf_max = 1
                
                lines[-1].loggf *= (1 + 0.5*np.random.normal(0, 1))
                if lines[-1].loggf > lines[-1].loggf_max:
                    lines[-1].loggf = lines[-1].loggf_max
                if lines[-1].loggf < lines[-1].loggf_min:
                    lines[-1].loggf = lines[-1].loggf_min

    out = open(line_pars_path, "w")

    out.write("# parID   LineNo   initial   min     max\n")

    for line in lines:
        out.write("loggf    ")
        out.write("{: 3d}    ".format(line.lineNo))
        out.write("{: 4.3f}    ".format(line.loggf))
        out.write("{: 4.3f}    ".format(line.loggf_min))
        out.write("{: 4.3f}\n".format(line.loggf_max))

    out.close()

def check_init_loggf():
    """
    For a given lines log(gf) initial values, test if these lines are seen
    in spectrum. We requier that line is 1% stronger than continuum intensity.
    """
    pass