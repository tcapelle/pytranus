#Identify L1E variables after LCAL
from context import pytranus
from pytranus import Lcal, BinaryInterface, L1s, L1sParam, TranusConfig
import numpy as np
import pandas as pd
import logging
import os
from sys import stdout

log_level = logging.DEBUG
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(message)s',
                    level  = log_level,
                    stream = stdout)
np.set_printoptions(precision=5,linewidth=210,suppress=True)

# scn = "00A"
bin = "/Users/thomascapelle/Dropbox/TRAN_fortranfiles/OSX/"

def replace_L1S(t):
	path_scn = os.path.join(t.workingDirectory ,t.scenarioId)
	os.remove(path_scn+ '/' + t.projectId +t.scenarioId+ '.L1S')
	os.rename(path_scn+ '/NEW_LCAL_' + scn + '.L1S', path_scn+ '/' + t.projectId +t.scenarioId+ '.L1S')
	return


# path = "/Users/thomascapelle/Dropbox/TRANUS_LCAL/Grenoble_2017_04_04/"
# t = TranusConfig(tranusBinPath = bin, workingDirectory = path, 
#     projectId='NEW', scenarioId=scn) 
scn = "03A"
path =  "ExampleC/"
t = TranusConfig(tranusBinPath = bin, workingDirectory = path, 
    			 projectId='EXC', scenarioId=scn) 


interface = BinaryInterface(t)

Lcal = Lcal(t, normalize = False)

nSectors = Lcal.param.nSectors
nZones = Lcal.param.nZones

vec = np.random.random(2*nSectors*nZones)
# vec0=vec.copy()

h0, p0 = Lcal.reshape_vec(vec)

Lcal.calc_sp_housing()
# Lcal.param.beta /= 1000
p, h, conv, lamda_opti = Lcal.compute_shadow_prices(h0)

L1sparam = L1sParam(t)
path_L1S = os.path.join(path,scn,t.projectId+t.scenarioId+'.L1S')

def read_L1S(path_L1S, L1sparam):
#241, 25
	out_L1S = L1s(path_L1S,L1sparam.nbSectors,L1sparam.nbTotZones).read() 
	aux = []
	for var in out_L1S:
	    aux.append(var[:,:227])
	return aux
##test 31/8
Lcal.p[0,:] = Lcal.cospro()[0,:]
###
pro, cospro, precio, coscon, utcon, atrac, ajuste, dem = read_L1S(path_L1S, L1sparam)

L1sparam.runParametersExtraction()
L1sparam.GRAL1S(Lcal)   
