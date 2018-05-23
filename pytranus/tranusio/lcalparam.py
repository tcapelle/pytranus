#! /usr/bin/env python
from pytranus.support import TranusConfig
from pytranus.support import BinaryInterface
import numpy as np
import logging
import sys
import os.path


def line_remove_strings(L): # takes the lines of section 2.1 L1E, and returns a line without strings
    return [x for x in L if is_float(x)]

def is_float(s):
    ''' little functions to verify if a string can be converted into a number '''
    try:
        float(s)
        return True
    except ValueError:
        return False
        
class LcalParam:
    '''LcalParam class:
    This class is meant to read all the Tranus LCAL input files, and store
    the lines in local variables.
    '''

    def __init__(self, t, normalize = True):
        '''LcalParam(tranusConfig)
        Constructor of the class, this object has a local variable for each of the
        Tranus lines relevants to LCAL.

        Parameters
        ----------
        tranusConfig : TranusConfig object
            The corresponding TranusConfig object of your project.

        Class Attributes
        ----------------
        list_zones: list
            List of zones.
        list_zones_ext: list
            List of external zones.
        nZones: integer  (len(list_zones))
            Number of zones in list_zones.
        list_sectors: list
            List of Economical Sectors.
        nSectors: integer (len(list_sectors))
            Number of Economical Sectors.
        housingSectors: 1-dimensional ndarray
            Subset of list_sectors including the land-use housing sectors.
        substitutionSectors: 1-dimensional ndarray
            Subset of list_sectors that have substitution.

        Variables from L0E section 1.1, and their corresponding initialized value:
        ExogProd = np.zeros((nSectors,nZones))
            Exogenous base-year Production X* for each sector and zone.
        InduProd = np.zeros((nSectors,nZones))
            Induced base-year Production X_0 for each sector and zone. 
        ExogDemand = np.zeros((nSectors,nZones))  
            Exogenous Demand for each sector and zone.
        Price = np.zeros((nSectors,nZones))
            Base-year prices for each sector and zone.
        ValueAdded = np.zeros((nSectors,nZones))
            Value Added for each sector and zone.
        Attractor = np.zeros((nSectors,nZones))
            Attractors W^n_i for each sector and zone.
        Rmin = np.zeros((nSectors,nZones))
            Lower limit to production, is not used in LCAL. 
        Rmax = np.zeros((nSectors,nZones))
            Upeer limit to production, is not used in LCAL.
        
        Variables from L1E section 2.1:
        alfa = np.zeros(nSectors) 
            Exponent in the Attractor formula. 
        beta = np.zeros(nSectors)
            Dispersion parameter in Pr^n_{ij}
        lamda  = np.zeros(nSectors)
            Marginal Localization Utility of price. 
        thetaLoc = np.zeros(nSectors)
            Exponent in normalization in Localization Utility, not used.

        Variables from L1E section 2.2:
        demax = np.zeros((nSectors,nSectors))
            Elastic demand function max value, for housing consuming->land-use 
            sectors.
        demin = np.zeros((nSectors,nSectors))
            Elastic demand function min value, for housing consuming->land-use 
            sectors.
        delta = np.zeros((nSectors,nSectors))
            Disperion parameter in the demand function a^{mn}_i

        Variables from L1E section 2.3:
        sigma = np.zeros(nSectors)
            Dispersion parameter in Substitution logit, no longer used.
        thetaSub = np.zeros(nSectors)
            Exponent in normalization in substitution logit, not used. 
        omega = np.zeros((nSectors,nSectors))
            Relative weight in substitution logit.
        Kn = np.zeros((nSectors,nSectors),dtype=int)
            Set of possible substitution sectors.

        Variables from L1E section 3.2:
        bkn = np.zeros((nSectors,nSectors))
            Coefficients of the attractor weight.

        Disutil transport, monetary cost transport from C1S:
        t_nij = np.zeros((nSectors,nZones,nZones))
            Disutility of transportation between to zones for a given sector.
        tm_nij = np.zeros((nSectors,nZones,nZones))
            Monetary cost of transportation between to zones for a given sector.
        '''

        self.list_zones,self.list_zones_ext = t.numberingZones()  # list of zones [1,2,4,5,8,....
        self.nZones = len(self.list_zones) #number of zones: 225   
        self.sectors_names  = []  #list of names of sectors (from file L1E)
        self.list_sectors   = t.numberingSectors() # list of sectors
        self.nSectors       = len(self.list_sectors)   #number of sectors: 20
        self.housingSectors = np.array([-1])    #array not initialized
        self.substitutionSectors = np.array([-1])    #array not initialized
        #Variables from L0E section 1.1
        self.ExogProd       = np.zeros((self.nSectors,self.nZones))
        self.InduProd       = np.zeros((self.nSectors,self.nZones))
        self.ExogDemand     = np.zeros((self.nSectors,self.nZones))  
        self.Price          = np.zeros((self.nSectors,self.nZones))
        self.ValueAdded     = np.zeros((self.nSectors,self.nZones))
        self.Attractor      = np.zeros((self.nSectors,self.nZones))
        self.Rmin           = np.zeros((self.nSectors,self.nZones))
        self.Rmax           = np.zeros((self.nSectors,self.nZones))
        #Variables from L1E s ection 2.1
        self.alfa           = np.zeros(self.nSectors)  
        self.beta           = np.zeros(self.nSectors)
        self.lamda          = np.zeros(self.nSectors)
        self.thetaLoc       = np.zeros(self.nSectors)
        #Variables from L1E section 2.2
        self.demax          = np.zeros((self.nSectors,self.nSectors))
        self.demin          = np.zeros((self.nSectors,self.nSectors))
        self.delta          = np.zeros((self.nSectors,self.nSectors))
        #Variables from L1E section 2.3
        self.sigma          = np.zeros(self.nSectors)
        self.thetaSub       = np.zeros(self.nSectors)
        self.omega          = np.zeros((self.nSectors,self.nSectors))
        self.Kn             = np.zeros((self.nSectors,self.nSectors), dtype=int)
        #Variables from L1E section 3.2
        self.bkn            = np.zeros((self.nSectors,self.nSectors))
        #Disutil transport, monetary cost transport from C1S
        self.t_nij          = np.zeros((self.nSectors,self.nZones,self.nZones))
        self.tm_nij         = np.zeros((self.nSectors,self.nZones,self.nZones))
        
        self.read_files(t)
        if normalize:
            self.normalize()
        return

    def read_files(self, t):
        '''read_files(t)
        Reads the files L0E, L1E and C1S to load the LCAL lines into the
        Lcalparam object.
        Parameters
        ----------
        t : TranusConfig object
            The TranusConfig file of your project.

        Example
        -------
        You could use this method for realoading the lines from the files
        after doing modifications.
        >>>filename = '/ExampleC_n/'
        >>>t = TranusConfig(nworkingDirectory = filename)
        >>>param = Lcalparam(t)
        >>>param.beta
        array([ 0.,  1.,  1.,  3.,  0.])
        #modify some parameter, for example:
        >>>param.beta[2]=5
        >>>param.beta
        array([ 0.,  1.,  5.,  3.,  0.])
        >>>param.read_files(t)
        >>>param.beta
        array([ 0.,  1.,  1.,  3.,  0.])
        '''
        print "  Loading Lcal object from: %s"%t.workingDirectory
        self.read_C1S(t)
        self.read_L0E(t)
        self.read_L1E(t)
        return

    def read_L0E(self,t):
        '''read_LOE(t)
        Reads the corresponding LOE file located in the workingDirectory.
        Stores what is readed in local variables of the Lcalparam object.
        This method is hardcoded, meaning that if the file structure of the
        LOE file changes, probably this method needs to be updated as well.

        It's not meant to be used externally, it's used when you call the 
        constructor of the class.
        '''

        filename=os.path.join(t.workingDirectory,t.obs_file)
        logging.debug("Reading Localization Data File (L0E), [%s]", filename)
        
        filer = open(filename, 'r')
        lines = filer.readlines()
        filer.close()
        
        length_lines = len(lines)
        for i in range(length_lines):
            lines[i]=str.split(lines[i])
        
        """ Section that we are interested in. """
        string = "1.1"
        """ Getting the section line number within the file. """
        for line in range(len(lines)):
            if (lines[line][0] == string):
                break
        """ End of section. This is horribly harcoded as we depend
            on the format of the Tranus lines files and we don't have any control
            over it. Also, this format will probably change in the future, hopefully
            to a standarized one.
        """
        end_of_section = "*-"

        
        line+=2 #puts you in the first line for reading
        
        while lines[line][0][0:2] != end_of_section:
            n,i=self.list_sectors.index(float(lines[line][0])),self.list_zones.index(float(lines[line][1])) #n,i=sector,zone
            self.ExogProd[n,i]      =lines[line][2]
            self.InduProd[n,i]      =lines[line][3]
            self.ExogDemand[n,i]    =lines[line][4]
            self.Price[n,i]         =lines[line][5]
            self.ValueAdded[n,i]    =lines[line][6]
            self.Attractor[n,i]     =lines[line][7]
            
            line+=1
        
        """ Filter sectors """
        
        string = "2.1"
        """ Getting the section line number within the file. """
        for line in range(len(lines)):
            if (lines[line][0] == string):
                break
            
        line+=2 #puts you in the first line for reading            
        while lines[line][0][0:2] != end_of_section:
            n,i=self.list_sectors.index(float(lines[line][0])),self.list_zones.index(float(lines[line][1]))
            self.ExogDemand[n,i]    =lines[line][2]
            line+=1
        
        string = "2.2"              
        """ Getting the section line number within the file. """
        for line in range(len(lines)):
            if (lines[line][0] == string):
                break
            
        line+=2 #puts you in the first line for reading              
        while lines[line][0][0:2] != end_of_section:
            n,i=self.list_sectors.index(float(lines[line][0])),self.list_zones.index(float(lines[line][1]))
            self.Rmin[n,i]      =lines[line][2]
            self.Rmax[n,i]      =lines[line][3]
            line+=1
            
        string = "3."               
        """ Getting the section line number within the file. """
        for line in range(len(lines)):
            if (lines[line][0] == string):
                break
            
        line+=2 #puts you in the first line for reading 
        list_housing=[]             
        while lines[line][0][0:2] != end_of_section:
            n,i=self.list_sectors.index(float(lines[line][0])),self.list_zones.index(float(lines[line][1]))
            if n not in list_housing:
                list_housing.append(n)
            self.Rmin[n,i]      =lines[line][2]
            self.Rmax[n,i]      =lines[line][3]
            line+=1
        self.housingSectors=np.array(list_housing)
        return
    
    def read_C1S(self,t):
        '''read_C1S(t)
        Reads COST_T.MTX and DISU_T.MTX files generated using ./mats from the 
        C1S file. Normally, this files are created when you first create your
        TranusConfig file.

        Stores what is readed in local variables of the LCALparam object.
        This method is hardcoded, meaning that if the file structure of the
        COST_T.MTX and DISU_T.MTX file changes, probably this method needs 
        to be updated as well.

        It's not meant to be used externally, it's used when you call the 
        constructor of the class.
        '''
        interface = BinaryInterface(t)
        if not os.path.isfile(os.path.join(t.workingDirectory,"COST_T.MTX")):
            logging.debug(os.path.join(t.workingDirectory,"COST_T.MTX")+': not found!')
            logging.debug("Creating Cost Files with ./mats")
            if interface.runMats() != 1:
                logging.error('Generating Disutily files has failed')
                return
        #Reads the C1S file using mats
        #this is hardcoded because we are using intermediate files DISU_T.MTX and COST_T.MTX
        path = t.workingDirectory
        logging.debug("Reading Activity Location Parameters File (C1S) with Mats")
        
        filer = open(os.path.join(path,"COST_T.MTX"), 'r')
        lines = filer.readlines()
        filer.close()
        
        sector_line=4   #line where the 1st Sector is written
        line=9          #line where the matrix begins
        # print 'Zones: %s'%self.nZones
        while line<len(lines):
            n=int(lines[sector_line][0:4])
            z = 0
            while True:
                param_line = (lines[line][0:4]+lines[line][25:]).split()
                if len(param_line)==0:
                    break
                try:
                    i=int(param_line[0])
                    aux_z = self.list_zones.index(i)
                    # print aux_z
                except ValueError:
                    aux_z = z 
                    # print '>> %s'%aux_z
                if z < self.nZones:
                    self.tm_nij[self.list_sectors.index(n),aux_z,:] = param_line[1:self.nZones+1]
                z+=1
                line+=1
                if line==len(lines):
                    break
            sector_line=line+4
            line+=9     #space in lines between matrices
            
        filer = open(os.path.join(path,"DISU_T.MTX"), 'r')
        lines = filer.readlines()
        filer.close()

        sector_line=4
        line=9
        
        while line<len(lines):
            n=int(lines[sector_line][0:4])
            z = 0
            while True:
                param_line = (lines[line][0:4]+lines[line][25:]).split()
                if len(param_line)==0:
                    break
                try:
                    i=int(param_line[0])
                    aux_z = self.list_zones.index(i)
                    # print aux_z
                except ValueError:
                    aux_z = z 
                    # print '>> %s'%aux_z
                if z < self.nZones:
                    self.t_nij[self.list_sectors.index(n),aux_z,:] = param_line[1:self.nZones+1]
                z+=1
                line+=1
                if line==len(lines):
                    break
            sector_line=line+4
            line+=9     #space in lines between matrices

        return
        
    def read_L1E(self,t):
        '''read_L1E(t)
        Reads the corresponding L1E file located in the workingDirectory.
        Stores what is readed in local variables of the LCALparam object.
        This method is hardcoded, meaning that if the file structure of the
        L1E file changes, probably this method needs to be updated as well.

        It's not meant to be used externally, it's used when you call the 
        constructor of the class.
        '''

        filename=os.path.join(t.workingDirectory,t.scenarioId,t.param_file)
        logging.debug("Reading Activity Location Parameters File (L1E), [%s]", filename)
        
        filer = open(filename, 'r')
        lines = filer.readlines()
        filer.close()
        
        length_lines = len(lines)

        
        """ Section that we are interested in. """

        string = "2.1"
        """ Getting the section line number within the file. """
        for line in range(len(lines)):
            # print lines[line][3:6]
            if (lines[line][3:6] == string):
                break
        """ End of section. This is horribly harcoded as we depend
            on the format of the Tranus lines files and we don't have any control
            over it. Also, this format will probably change in the future, hopefully
            to a standarized one.
        """
        end_of_section = "*-"
        line+=3
        # print 'line: %s'%line
        while lines[line][0:2] != end_of_section:
            # print lines[line]
            param_line = line_remove_strings(lines[line].split())  #we remove the strings from each line!
            n = self.list_sectors.index(float(param_line[0]))
            self.sectors_names.append(lines[line][14:33].rstrip()[0:-1])
            self.alfa[n] = param_line[6]
            #print param_line
            self.beta[n] = param_line[1]
            self.lamda[n] = param_line[3]
            self.thetaLoc[n] = param_line[5]
            line += 1
        
        for i in range(length_lines):
            lines[i]=str.split(lines[i])
        
        string = "2.2"
        """ Getting the section line number within the file. """
        for line in range(len(lines)):
            if (lines[line][0] == string):
                break
        line+=2
        while lines[line][0][0:2] != end_of_section:
            m,n=self.list_sectors.index(float(lines[line][0])),self.list_sectors.index(float(lines[line][1]))
            self.demin[m,n]=lines[line][2]
            self.demax[m,n]=lines[line][3]
            if self.demax[m,n]==0:
                self.demax[m,n]=self.demin[m,n]
            self.delta[m,n]=lines[line][4]
            line+=1    


        
        string = "2.3"
        """ Getting the section line number within the file. """
        for line in range(len(lines)):
            if (lines[line][0] == string):
                break
        line+=2
        n=0
        list_sub_sectors=[]
        while lines[line][0][0:2] != end_of_section:
            if len(lines[line])==5:
                n=self.list_sectors.index(float(lines[line][0]))
                self.sigma[n]   =lines[line][1]
                self.thetaSub[n]    =lines[line][2]
                self.Kn[n,self.list_sectors.index(float(lines[line][3]))]=1
                self.omega[n,self.list_sectors.index(float(lines[line][3]))]=lines[line][4]
                list_sub_sectors.append(n)
            if len(lines[line])==2:
                self.Kn[n,self.list_sectors.index(int(lines[line][0]))]=1
                self.omega[n,self.list_sectors.index(float(lines[line][0]))]=lines[line][1]
            
            line+=1
            
        self.substitutionSectors = np.array(list_sub_sectors)
        

        
        string = "3.2"
        """ Getting the section line number within the file. """
        for line in range(len(lines)):
            if (lines[line][0] == string):
                break
        line+=2
        while lines[line][0][0:2] != end_of_section:
            m,n=self.list_sectors.index(float(lines[line][0])),self.list_sectors.index(float(lines[line][1]))
            self.bkn[m,n]=lines[line][2]
            line+=1    
        
        return
    def normalize(self, t = -1, tm = -1, P = -1):
        '''normalize input variables of the utility'''
        if t == -1:
            t = 10**np.floor(np.log10(self.t_nij.max()))
        if tm == -1:
            tm = 10**np.floor(np.log10(self.tm_nij.max()))
        if P == -1:
            P = 10**np.floor(np.log10(self.Price[self.housingSectors,:].max()))

        self.t_nij  /= t
        self.tm_nij /= tm
        self.Price  /= P
        return 

    def __str__(self):
        ex =    """See class docstring"""
        
        return ex
        


if __name__=="__main__":
    pass
