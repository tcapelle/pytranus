  #!/usr/bin/env python

#===============================================================================
# This is the class that interfaces the lcal.o file with python. Here we run 
# lcal proving it with the required inputs from python.
#===============================================================================
import subprocess
import os
import glob
import time
import logging
import platform
from sys import stdout

class BinaryInterface:
    '''BinaryInterface models an interface to communicate with Tranus binaries'''
    # Initialize lcal interface
    def __init__(self, t, numSample=None):
        "LcalInterface constructor. It initializes the needed values."
        self.nSamp = numSample  
        self.outFile = "outlcal.txt"       #output of lcal stored here                 
        self.tranusConf = t  #Configure tranus using TranusConfiguration class
        self.TranusHasBeenRun = 0
        operating_system = platform.system()
        if operating_system.startswith('Windows'):
            self.extension = '.exe'
        else:
            self.extension = ''

    def runLcal(self, freeze=False, additional = False):
        '''runLcal(self, freeze=False):
        Runs LCAL module and write its output in the file indicated by 'self.outFile
        freeze = True : only land use sectors are computed.
        '''
        program = os.path.join(self.tranusConf.tranusBinPath,'lcal'+self.extension)
        if not os.path.isfile(program):
            logging.error('The <lcal> program was not found in %s'%self.tranusConf.tranusBinPath )
            return 0
        outlcal = os.path.join(self.tranusConf.workingDirectory, "outlcal.txt")
        outlcalerr = os.path.join(self.tranusConf.workingDirectory, "outlcalerr.txt")
        args = [program, self.tranusConf.scenarioId]
        if freeze:
            args = [program, self.tranusConf.scenarioId, '-f',]
        if additional:
            args.append('A')
        result = subprocess.Popen(args, stdout=open(outlcal, "w"), stderr=open(outlcalerr, 'w'), cwd=self.tranusConf.workingDirectory).communicate() # Success!
        return 1

    
    def createImploc(self):
        '''Creates the Imploc report from lcal'''
        program = os.path.join(self.tranusConf.tranusBinPath,'imploc'+self.extension)
        if not os.path.isfile(program):
            logging.error('The <imploc> program was not found in %s'%self.tranusBinPath )
            return 0
        outimploc = os.path.join(self.tranusConf.workingDirectory, "outimploc.txt")
        outimplocerr = os.path.join(self.tranusConf.workingDirectory, "outimplocerr.txt")
        args = [program, self.tranusConf.scenarioId,'-J', '-o','imploc_out.txt',' ']
        proc = subprocess.Popen(args, stdout=open(outimploc, 'w'), stderr=open(outimplocerr,'w'), cwd=self.tranusConf.workingDirectory)
        return 1
    

    #Runs Fluj module which transforms the lcal matrices into flows for the transport sector
    def runFluj(self):
        '''Runs Fluj module and write its output in a file'''
        program = os.path.join(self.tranusConf.tranusBinPath,'fluj'+self.extension)
        if not os.path.isfile(program):
            logging.error('The <fluj> program was not found in %s'%self.tranusBinPath )
            return 0
        outfluj = os.path.join(self.tranusConf.workingDirectory, "outfluj.txt")
        outflujerr = os.path.join(self.tranusConf.workingDirectory, "outflujserr.txt")
        args = [program, self.tranusConf.scenarioId]
        result = subprocess.Popen(args, stdout=open(outfluj, "w"), stderr=open(outflujerr, 'w'), cwd=self.tranusConf.workingDirectory).communicate() # Success!
        return 1
    
    #Running the transport Module
    def runTrans(self,loopn):
        '''Runs Trans module and write its output in a file'''
        
        program = os.path.join(self.tranusConf.tranusBinPath,'trans'+self.extension)
        if not os.path.isfile(program):
            logging.error('The <trans> program was not found in %s'%self.tranusBinPath )
            return 0
        outtrans = os.path.join(self.tranusConf.workingDirectory, "outtrans.txt")
        outtranserr = os.path.join(self.tranusConf.workingDirectory, "outtranserr.txt")
        if loopn == 0:
            args = [program, self.tranusConf.scenarioId,'-I',' ']
        else: 
            args = [program, self.tranusConf.scenarioId]
        result = subprocess.Popen(args, stdout = open(outtrans, "w"), stderr = open(outtranserr,'w'), cwd = self.tranusConf.workingDirectory).communicate() # Success!
        return 1
    
    #Running the cost module that will convert the matrices/flows from the transport to costs
    def runCost(self):
        '''Runs Cost module and write its output in a file'''
        program = os.path.join(self.tranusConf.tranusBinPath,'cost' + self.extension)
        if not os.path.isfile(program):
            logging.error('The <cost> program was not found in %s'%self.tranusBinPath )
            return 0
        outcost = os.path.join(self.tranusConf.workingDirectory, "outcost.txt")
        outcosterr = os.path.join(self.tranusConf.workingDirectory, "outcosterr.txt")
        args = [program,self.tranusConf.scenarioId]
        result = subprocess.Popen(args, stdout=open(outcost, "w"), stderr = open(outcosterr, 'w'), close_fds = True, cwd = self.tranusConf.workingDirectory).communicate() # Success!
        return 1

    def runPasos(self):
        '''Runs PASOS module and write its output in a file'''
        program = os.path.join(self.tranusConf.tranusBinPath,'pasos' + self.extension)
        if not os.path.isfile(program):
            logging.error('The <pasos> program was not found in %s'%self.tranusBinPath )
            return 0
        outpasos = os.path.join(self.tranusConf.workingDirectory, "outpasos.txt")
        args = [program, self.tranusConf.scenarioId]
        result = subprocess.Popen(args, stdout=open(outpasos, "w"), stderr = subprocess.PIPE, close_fds = True, cwd = self.tranusConf.workingDirectory) # Success!
        return 1
            
    def runMats(self):
        '''Runs MATS to generate the transport cost and disutilities matrix COST_T.MTX and DISU_T.MTX
        Needs L1S file!'''

        program = os.path.join(self.tranusConf.tranusBinPath, 'mats' + self.extension)
        if not os.path.isfile(program):
            logging.error('The <mats> program was not found in %s'%program )
            return 0
            
        logging.debug('Creating Disutility matrix')        
        outmats = os.path.join(self.tranusConf.workingDirectory, "outmats.txt")
        outmatserr = os.path.join(self.tranusConf.workingDirectory, "outmatserr.txt")

        args = [program, self.tranusConf.scenarioId, "-S", "[k]", "-o", "DISU_T.MTX", " "]
        proc = subprocess.Popen(args, stdout=open(outmats, 'w'), stderr=open(outmatserr, 'w'), cwd=self.tranusConf.workingDirectory)

        args = [program, self.tranusConf.scenarioId, "-K", "[k]", "-o", "COST_T.MTX", " "]
        proc = subprocess.Popen(args, stdout=open(outmats, 'w'), stderr=open(outmatserr, 'w'), cwd=self.tranusConf.workingDirectory)
        return 1   


    '''YOU HAVE TO RUN TRANUS IN A LOOP TO ACHIEVE PROPER CONVERGENCE IN THE SAME SEQUENCE GIVEN HERE IN THE PROGRAM.'''
    def runTranus(self,loopn):
        start = time.time()
        print 'runTrans0'
        self.runTrans(loopn) # Initial Assignment
        print 'runTrans'
        self.runCost()
        print 'runCost'
        self.runLcal()
        print 'runLcal'
        self.runFluj()  
        print 'runFluj'
        self.createImploc()
        print 'CreateImploc'
        print (time.time()-start)
        self.TranusHasBeenRun = 1
        return self.TranusHasBeenRun

        
    ''' RUN THE LCAL MODULE ONLY.'''
    def runOnlyLcal(self):
        self.runLcal()  
        self.createImploc()
        self.TranusHasBeenRun = 1      
        print "TRANUS Convergece %s"%self.converges()     
        return self.TranusHasBeenRun                     
      
class TranusConfig:
    '''Class which defines tranus configuration parameters'''

    def __init__(self,  tranusBinPath, workingDirectory, projectId, scenarioId):

        self.tranusBinPath  = tranusBinPath   
        #check if Tranus binaries are in there:
        if not os.path.isfile(os.path.join(tranusBinPath,'lcal')):
            logging.error('Tranus binaries not found in : %s'%tranusBinPath)

        self.workingDirectory   = workingDirectory#DIRECTORY OF TRANUS MODEL
        self.projectId          = projectId#ID OF THJE PROJECT  EX GRL
        self.scenarioId         = scenarioId#ID OF THE SCENARIO EX 00A
        self.param_file         = 'W_'+projectId+scenarioId+'.L1E'
        self.obs_file           = "W_"+projectId+'.L0E' #Added TC
        self.zone_file          = "W_"+projectId+".Z1E" #Added TC
        self.convFactor         = "0.0001"
        self.nbIterations       = "250"
        self.tranusConf_file    = 'W_TRANUS.CTL'

        self.CTL_filepath = os.path.join(workingDirectory,'W_TRANUS.CTL')
        self.L0E_filepath = glob.glob(os.path.join(workingDirectory,'W_*.L0E'))[0]
        self.Z1E_filepath = glob.glob(os.path.join(workingDirectory,'W_*.Z1E'))[0]

        self.ScenarioPath = os.path.join(workingDirectory,scenarioId)
        self.L1E_filepath = glob.glob(os.path.join(self.ScenarioPath,'W_*.L1E'))[0]

        self.L1S_filepath = os.path.join(self.ScenarioPath,"pyLcal"+"_"+scenarioId+".L1S")


        
    def numberingZones(self):
        '''functions that returns the list number of a zone. Takes into account only first level internal zones'''
        with open(os.path.join(self.workingDirectory,self.zone_file), 'r') as fs:
            logging.debug("Reading zones from file : %s"%os.path.join(self.workingDirectory,self.zone_file) )
            ll = fs.readlines()
            for i in range(0,len(ll)):
                ll[i] = ll[i]#.decode("iso-8859-1")
            numbersZ1E = []
            numbersZ1Eext = []
            n=5  

            '''PRETTY MUCH BRUTE FORCE IMPLEMENTATION OF HOW TO READ THE ZONE NUMBERING FROM THE .Z1E FILE.'''
            while n<len(ll)-1:
                while ll[n][1:3]!="--":
                    numbersZ1E.append(int(str.split(ll[n])[0]))                   
                    n+=1             
                n+=6        #skiping second level zones
                while ll[n][1:3]!="--":
                    numbersZ1Eext.append(int(str.split(ll[n])[0]))                  
                    n+=1              
            return numbersZ1E, numbersZ1Eext

    def numberingSectors(self):
        '''functions that returns the list number of sectors ''' 

        with open(os.path.join(self.workingDirectory,self.scenarioId,self.param_file), 'r') as fs:
            logging.debug("Reading sectors from file : %s"%os.path.join(self.workingDirectory,self.scenarioId,self.param_file) )
            ll=fs.readlines()
            for i in range(0,len(ll)):
                ll[i]=ll[i]#.decode("iso-8859-1")
            numbersL1E=[]
            n=11  
            while n<len(ll):
                while ll[n][1:3]!="--":
                    numbersL1E.append(int(str.split(ll[n])[0]))                   
                    n+=1
                #n+=3               
                n=len(ll)
            return numbersL1E

    
    
# Testing module
if __name__ == '__main__':
    pass
    # log_level = logging.DEBUG
    # logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(message)s',
    #                     level  = log_level,
    #                     stream = stdout)
    # t = TranusConfig()
    # print "TRANUS binaries directory                    : ", t.tranusBinPath
    # print "Directory where is located the .tuz file     : ", t.workingDirectory
    # print "ID of the project that we want to simulate   : ", t.projectId
    # print "ID of the scenario that we want to simulate  : ", t.scenarioId
    # print "Parameters file                              : ", t.param_file
    # print "Observations file                            : ", t.obs_file
    # print "Zone file                                    : ", t.zone_file
    # print "Convergence factor                           : ", t.convFactor
    # print "Number of iterations                         : ", t.nbIterations

    # from LcalInterface import *
    # interface = LcalInterface(t)
    # interface.runMats()
    
