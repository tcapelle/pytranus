from pytranus.support import TranusConfig
from pytranus.support import BinaryInterface
from pytranus.tranusio.lcalparam import LcalParam
from pytranus.pylcal.utils.DX import cython_DX_n

import numpy as np
from numpy.linalg import norm

import pandas as pd
from pandas import Series, DataFrame

from scipy import linalg
from scipy.optimize import leastsq, minimize
import multiprocessing

import logging
from time import time

def fun(f,q_in,q_out):
    while True:
        i,x = q_in.get()
        if i is None:
            break
        q_out.put((i,f(x)))

def parmap(f, X, nprocs = multiprocessing.cpu_count()):
    q_in   = multiprocessing.Queue(1)
    q_out  = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun,args=(f,q_in,q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
    [q_in.put((None,None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i,x in sorted(res)]


class Lcal:
    '''Class LCAL
    This class is the bread and butter of Tranus Land Use module. It computes
    all the variables related to land use, and fin the prices and shadowprices
    for calibration. It has many method for computing each variable, also 
    the optimisation methods for computing many of the important land-use
    parameters.

    For reading the Tranus model, it uses a TranusConfig object as input, and
    reads the input files using the LcalParam class, thus storing all the 
    parameters on local variables (Lcal.param.variable).


    '''
    def __init__(self, tranusConfig, normalize = True):
        ''' Lcal(tranusConfig)
        Constructor of LCALn class, this object once constructed, permits to 
        evaluate and calibrate the land-use model LCAL.

        Parameters
        ----------
        tranusConfig : TranusConfig object
            The corresponding TranusConfig object to your project.

        Class Attributes
        ---------------
        jac_calls = 0
            Number of calls to the gradient
        tranusConfig = tranusConfig
            Tranus Configuration class file
        param = LcalParam(tranusConfig)
            Instance of the LCALparam class, all LCAL variables imported 
            from the files L1E and L0E are here
        X = np.zeros((param.nSectors,param.nZones)) (initial value)
            Induced Production
        a_mni = 0 (initial value)
            Demand Function
        U_nij = 0 (initial value)
            Utilities
        multicpu = True
            Computer has multiple cores (use pmap instead of map)
        X_0 = param.ExogProd+param.InduProd
            Base year Production
        A_ni = np.dot(param.bkn,X_0)*param.Attractor
            Attractors for the location probability 
        housingSectors = param.housingSectors  #housing sectors
            np.array of housing sectors
        h_housing = np.zeros((len(housingSectors),param.nZones))   
            Shadow prices of land-use sectors
        X_housing = np.zeros((len(housingSectors),param.nZones))
            Induced production of housing sectors
        genflux_sectors = 
            np.array of sectors that generate flux
        nonTransportableDone = False
            bool that checks if the land-use optimization has been done


        Returns
        -------
        Lcal object.

        Examples
        --------
        >>>filename = '/ExampleC_n/'
        >>>t = TranusConfig(nworkingDirectory = filename) 
        >>>Lcal = LCALn(t)
        '''


        self.tranusConfig = tranusConfig
        self.param = LcalParam(tranusConfig, normalize)
        self.X = np.zeros((self.param.nSectors,self.param.nZones))
        self.h = np.zeros((self.param.nSectors,self.param.nZones))
        self.p = np.zeros((self.param.nSectors,self.param.nZones))
        self.a = 0
        self.logit = True
        #####test parameters
        self.D = np.zeros((self.param.nSectors, self.param.nZones))
        ####################
        self.U_nij = 0
        self.a = 0
        self.S = 0
        self.Pr = 0
        self.multicpu = True
        self.X_0 = self.param.ExogProd+self.param.InduProd
        self.A_ni = (np.dot(self.param.bkn,self.X_0)**self.param.alfa[:,np.newaxis])*self.param.Attractor
        self.housingSectors = self.param.housingSectors                    #housing sectors
        self.h_housing = np.zeros((len(self.housingSectors),self.param.nZones))    #shadow prices of housing sectors
        self.X_housing = np.zeros((len(self.housingSectors),self.param.nZones))    #production of housing sectors
        self.genflux_sectors = np.array([x for x in range(self.param.nSectors) if self.gen_flux(x)])
        self.nonTransportableDone = False                       #bool to check if the sp of housing sectors have been computed

        logging.debug("__LCAL__: Creating instance of LCAL")

        return


    def reset(self, X=0):
        '''reset(self, X):
        Resets the paramteres in LCAL, specially the variable housingHasBeen
        Runned 
        '''
        self.param = LcalParam(self.tranusConfig, False)
        self.a = 0
        self.S = 0
        if not X==0:
            self.X_0 = X
        self.nonTransportableDone = False   

    def gen_flux(self, n):
        '''gen_flux(n)

        Computes if sector n generates flux, as explained in the Mariano 
        Report.

        Parameters
        ----------
        n : integer in range(nSectors)

        Returns
        -------
        True : If sector n generates flux.

        Examples
        --------
        >>>filename = '/ExampleC_n/'
        >>>t=TranusConfig(nworkingDirectory = filename) 
        >>>Lcal=LCALn(t)
        >>>Lcal.gen_flux(0)
        False
        >>>Lcal.gen_flux(1)
        True
        '''
        #Mariano Report, page 12
        beta            =self.param.beta
        demax           =self.param.demax
        demin           =self.param.demin
        aux=demin[:,n]+demax[:,n]
        if beta[n]!=0 and aux.any():
            return True
        else: 
            return False        
    
    def calc_a(self, U_ni):
        '''calc_a(U_ni)

        Computes the demand functions a^{mn}_i for a given value of utility.
        In the practice, normally there is elastic demand for land-use sectors
        only.

        Parameters
        ----------
        U_ni : 2-dimensional ndarray[nSectors, nZones]
            The second output of calc_prob_loc

        Returns
        -------
        a_mni : 3-dimensional ndarray[nSectors, nSectors, nZones]
            The demand function evaluated in U_ni

        Examples
        --------
        >>>filename = '/ExampleC_n/'
        >>>t=TranusConfig(nworkingDirectory = filename) 
        >>>Lcal=LCALn(t)
        >>>Pr, U = Lcal.calc_prob_loc(h0,p0)
        >>>Lcal.calc_a(U)
        array([[[ 0.     ,  0.     ,  0.     ],
                [ 0.     ,  0.     ,  0.     ],
                [ 1.99897,  1.99897,  1.99897],
                [ 1.24813,  1.24813,  1.24813],
                [ 0.00449,  0.00581,  0.00499]],

               [[ 0.     ,  0.     ,  0.     ],
                [ 0.     ,  0.     ,  0.     ],
                [ 1.60924,  1.60924,  1.60924],
                [ 1.44861,  1.44861,  1.44861],
                [ 0.00349,  0.00481,  0.00399]],

               [[ 0.     ,  0.     ,  0.     ],
                [ 0.12035,  0.12035,  0.12035],
                [ 0.     ,  0.     ,  0.     ],
                [ 0.     ,  0.     ,  0.     ],
                [ 0.00341,  0.00451,  0.00383]],

               [[ 0.     ,  0.     ,  0.     ],
                [ 0.15327,  0.15327,  0.15327],
                [ 0.     ,  0.     ,  0.     ],
                [ 0.     ,  0.     ,  0.     ],
                [ 0.00557,  0.00711,  0.00616]],

               [[ 0.     ,  0.     ,  0.     ],
                [ 0.     ,  0.     ,  0.     ],
                [ 0.     ,  0.     ,  0.     ],
                [ 0.     ,  0.     ,  0.     ],
                [ 0.     ,  0.     ,  0.     ]]])

        '''
        if self.nonTransportableDone:
            return self.a
        #computing of a^{mn}_i (eq 9)
        else:
            demin       = self.param.demin
            demax       = self.param.demax
            delta       = self.param.delta
            nSectors    = self.param.nSectors
            nZones      = self.param.nZones

            gap_delta = demax-demin
            arg = -delta[:,:,np.newaxis]*U_ni[np.newaxis,:,:]
            a = demin[:,:,np.newaxis]+gap_delta[:,:,np.newaxis]*np.exp(arg)
            #self.a = np.where(arg==0, 0, a) #beware with transportable sectors!
            self.a = a
            return self.a


    def calc_prob_loc(self, h, p):               
        '''calc_prob_loc(self,h,p)

        Computes the Localization logit Pr_nij, 
        and the localization aggregated disutility U_ni.

        This is done using a logit discrete choice model
        within the sectors that have substitution

        Parameters
        ----------
        h : 2-dimensional ndarray
        p : 2-dimensional ndarray

        Returns
        -------
        Pr_nij : 3-dimensional ndarray[nSectors, nZones, nZones]
            The localisation probabilities.
        U_ni : 2-dimensional ndarray[nSectors, nZones]
            The aggregated localisation disutilities.

        Examples
        --------
        >>>filename = '/ExampleC_n/'
        >>>t=TranusConfig(nworkingDirectory = filename) 
        >>>Lcal=LCALn(t)
        >>>nSectors = Lcal.param.nSectors
        >>>nZones = Lcal.param.nZones
        >>>Lcal.housingSectors =array([4])  #Set the housing sectors       
        >>>h0=array([[   0.5,     0.5,   0.5],
                    [   0.62937,    0.07076,    0.83355],
                    [   0.63249,    0.42116,   0.00018],
                    [   0.0903,     0.65461,    0.06308],
                    [   8.0137 ,    0.99969,  110.96611]])

        >>>p0=random.rand(nSectors,nZones)
        >>>Pr,U = Lcal.calc_prob_loc(h0,p0)
        >>>Pr
        array([[[ 1.     ,  0.     ,  0.     ],
                [ 0.     ,  1.     ,  0.     ],
                [ 0.     ,  0.     ,  1.     ]],

               [[ 0.75958,  0.15648,  0.08394],
                [ 0.35536,  0.53186,  0.11278],
                [ 0.38526,  0.22177,  0.39297]],

               [[ 0.25927,  0.38291,  0.35782],
                [ 0.07311,  0.69867,  0.22822],
                [ 0.0711 ,  0.26285,  0.66604]],

               [[ 0.99602,  0.00007,  0.00391],
                [ 0.33641,  0.54261,  0.12098],
                [ 0.00543,  0.00011,  0.99446]],

               [[ 1.     ,  0.     ,  0.     ],
                [ 0.     ,  1.     ,  0.     ],
                [ 0.     ,  0.     ,  1.     ]]])
        >>>U
        array([[   1.27836,    1.15907,    1.34575],
               [  -6.98221,   -6.44446,   -6.32879],
               [  -7.59404,   -8.23842,   -8.14018],
               [  -1.68315,   -1.13993,   -2.09798],
               [   8.88706,    1.62096,  111.08013]])
        '''
    
        nSectors    =self.param.nSectors
        nZones      =self.param.nZones
        beta        =self.param.beta
        thetaLoc    =self.param.thetaLoc
        lamda       =self.param.lamda
        t_nij       =self.param.t_nij


        # logging.debug("Calculating U_nij")
        aux0 = lamda[:,np.newaxis]*(p+h) #nj

        U_nij = aux0[:,np.newaxis,:]+t_nij
        self.U_nij=U_nij
        logging.debug("A_ni")

        A_ni = self.A_ni

        Pr_nij      =np.zeros((nSectors,nZones,nZones))
        U_ni        =np.zeros((nSectors,nZones))

        
        for n in range(nSectors):
            if not self.gen_flux(n):  # => beta>0
                U_ni[n,:] = self.param.lamda[n] * (p[n,:] + h[n,:])
                Pr_nij[n,:,:] = np.eye(nZones)  #X_ni=D_ni
            else:
                # print "Max U_nij, Min U_nij: (%s,%s)"%(U_nij.max(),U_nij.min())
                # print "Max h, Min h: (%s,%s)"%(h.max(),h.min())
                # print "Max p, Min p: (%s,%s)"%(p.max(),p.min())
                aux = A_ni[n,np.newaxis,:] * np.exp( -beta[n] * U_nij[n,:,:])  #ij
                s = aux.sum(1)
                # print '----------------------------'
                # print 'S:'
                # print s
                # print 'U_nij'
                # print U_nij[n,:,:]                                            #i
                Pr_nij[n,:,:] = aux / s[:,np.newaxis]                                   #ij=ij/i
                # U_ni[n,:]=np.log(s)/(-beta[n]*lamda[n])

        return Pr_nij, U_ni  
   
    def calc_subst(self, U_ni):
        '''calc_subst(U_ni)

        Computes the substitution logit S_mni

        This is done using a logit discrete choice model
        within the sectors that have substitution

        Parameters
        ----------
        U_ni : 2-dimensional ndarray[nSectors, nZones]

        Returns
        -------
        S_mni : 3-dimensional ndarray[nSectors, nSectors, nZones]
            The substituion probabilities.

        Examples
        --------
        '''
        if self.nonTransportableDone:
            return self.S

        nSectors        = self.param.nSectors
        nZones          = self.param.nZones
        Kn              = self.param.Kn
        sigma           = self.param.sigma
        thetaSub        = self.param.thetaSub
        Attractor       = self.param.Attractor
        omega           = self.param.omega

        S_mni           = np.ones((nSectors,nSectors,nZones))
        U_tilde         = omega[:,:,np.newaxis]* self.calc_a(U_ni)* U_ni[np.newaxis,:,:]                  #mni

        for m in Kn.sum(1).nonzero()[0]:    #where there is substitution
            subt_choices = Kn[m].nonzero()[0]
            aux = Attractor[subt_choices, :]* np.exp(-sigma[m]* U_tilde[m, subt_choices, :])       #ni
            s = aux.sum(0)                                                                      #i                                
            S_mni[m, subt_choices, :] = np.where(aux==0, 0, aux /s[np.newaxis, :])                                    #ni=ni/n

        return S_mni

    def calc_induced_prod_housing(self, h_i, i, jac_omega=False, jac_delta=False, jac_sigma=False): #h vector of length 6  #Cheap Way--> Zone i

        '''calc_induced_prod_housing(h, i, jac_omega = False)

        This function returns the induced production of a zone i
        for the values of shadow prices for this zone.

        If jac_omega is True it also returns the derivaves of the production for
        the omega parameter

        Parameters
        ----------
        h_i : 1-dimensional ndarray
            Shadow prices for land-use sectors for zone i.
        i : interger
            Corresponding zone.
        jac_omega: bool, optional
            If True, returns the jac_omega dX_i / domega_mn along 
            the induce production.

        Returns
        -------
        X_i : 1-dimensional ndarray
            The induce production of land-use sectors for zone i
        Jac_omega : 3-dimensional ndarray
            The jacobien dX_i / domega_mn

        Examples
        --------
        '''

        housingSectors = self.housingSectors  #length =6
        nHSectors       = len(housingSectors)
        p_i             = self.param.Price[housingSectors,i]
        nSectors        = self.param.nSectors    #number of sectors: 21
        ExogDemand      = self.param.ExogDemand[housingSectors,i]
        X0              = self.X_0[:,i]
        
        S_mn            = np.ones((nSectors,nHSectors))
        
        demin = self.param.demin[:,housingSectors]
        demax = self.param.demax[:,housingSectors]
        delta = self.param.delta[:,housingSectors]

        U_n = p_i + h_i
        a_mn = demin + (demax-demin)*np.exp(-delta*U_n[np.newaxis,:])
        
        Kn              = self.param.Kn[:,housingSectors]
        sigma           = self.param.sigma
        thetaSub        = self.param.thetaSub
        Attractor       = self.param.Attractor[housingSectors,i]
        omega           = self.param.omega[:,housingSectors]

        U_tilde =  omega * a_mn * U_n[np.newaxis,:]

        for m in Kn.sum(1).nonzero()[0]:        #where there is substitution
            subt_choices = Kn[m,:].nonzero()[0]
            aux = Attractor[subt_choices]*np.exp(-sigma[m]*U_tilde[m,subt_choices])     #n
            s = aux.sum(0)

            S_mn[m, subt_choices] = np.where(aux == 0, 0, aux/s)     
        # Computation of dX/d_omega & Computation of dX/d_delta

        if jac_delta:
            #cococococ
            da = - U_n[np.newaxis, :] * (a_mn - demin)      #mn
            Jac_delta = np.einsum('m, mn, m, mq, q, mq, mn, mq->nmq', X0, a_mn, -sigma, omega, U_n, da, -S_mn, S_mn)
            Jac_delta[range(nHSectors),:,range(nHSectors)] = np.einsum('m, mn, mn->nm', X0, da, S_mn) + \
                                                             np.einsum('m, mn, m, mn, n, mn, mn->nm',X0, a_mn, -sigma, omega, U_n, da, S_mn -S_mn**2 ) 
            prod = ExogDemand + np.dot(X0, a_mn*S_mn)
            return prod, Jac_delta[:, self.param.substitutionSectors,:]

        if jac_sigma:
            #cucucuc
            dS = S_mn * (-omega * a_mn * U_n[np.newaxis,:] + np.einsum('ml,ml,l,ml->m', omega, a_mn, U_n, S_mn)[:, np.newaxis])
            Jac_sigma = np.einsum('m, mn, mn->nm', X0, a_mn, dS)
            prod = ExogDemand + np.dot(X0, a_mn*S_mn)
            
            return prod, Jac_sigma[:, self.param.substitutionSectors]

        if jac_omega:
            Jac_omega = np.einsum('m,mn,mq,q,mn,mq->nmq', X0 * sigma, a_mn, a_mn, U_n, S_mn, S_mn)
            Jac_omega[range(nHSectors),:,range(nHSectors)] = - np.einsum('m,mn,n,mn->nm', X0 * sigma, a_mn**2, U_n, S_mn -S_mn**2)
            prod = ExogDemand + np.dot(X0, a_mn*S_mn)
            
            return prod, Jac_omega[:, self.param.substitutionSectors,:]
        else:
            return ExogDemand + np.dot(X0, a_mn*S_mn)

    def calc_induced_prod_housing_total(self, h_housing):
        X_housing          = np.array(map(self.calc_induced_prod_housing, h_housing.transpose(),range(self.param.nZones))).transpose()  
        return X_housing

    def residual_housing(self, h_i, i=0, jac_omega=False, jac_delta=False, jac_sigma=False, scaled=False): 

        '''residual_housing(h_i,i=0)

        This function returns the residual of the housing sectors
        in the zone i.


        Parameters
        ----------
        h_i : 1-dimensional ndarray
            Shadow prices for land-use sectors for zone i.
        i : interger
            Corresponding zone.
        jac_omega: bool, optional
            If True, returns the jac_omega dX_i / domega_mn along 
            the induce production.

        Returns
        -------
        ret : 1-dimensional ndarray
            The difference between Induced Production and Base year 
            production.
        jac_omega : 3-dimensional ndarray
            The jacobien dX_i / domega_mn

        Examples
        --------
        '''

        InduProd = self.param.InduProd[self.housingSectors,i]
        if scaled:
            scale = np.where(InduProd>0, 1/InduProd, 0) #tranus style tricks...
        else:
            scale = InduProd**0

        if jac_delta:
            res, jac = self.calc_induced_prod_housing(h_i, i, jac_delta=True)
            return (res-InduProd)*scale, jac*scale[:,np.newaxis,np.newaxis]
        if jac_omega:
            res, jac = self.calc_induced_prod_housing(h_i,i, jac_omega=True)
            return (res-InduProd)*scale, jac*scale[:,np.newaxis,np.newaxis]
        if jac_sigma:
            res, jac = self.calc_induced_prod_housing(h_i,i, jac_sigma=True)
            return (res-InduProd)*scale, jac*scale[:,np.newaxis]
        else:
            return (self.calc_induced_prod_housing(h_i,i)-InduProd)*scale


    def residual_housing0(self, i, jac_omega=False, jac_delta=False, jac_sigma=False, scaled=False): 
        h0 = np.zeros(len(self.housingSectors))
        return self.residual_housing(h0, i, jac_omega=jac_omega, jac_delta=jac_delta, jac_sigma=jac_sigma, scaled=scaled)


    def calc_sp_lu(self,i,h0_i = None): 
        '''calc_sp_lu(i,h0_i = None)

        Computes the optimal shadow prices for land used
        sectors for zone i. 

        This is done using leverberg marquandt algorithm 
        scipy.optimize.leastsq, the initial value is h0_i.

        The method will print on screen information for each zone:
        zone, Solution Status (leastsq flag), Residuals


        Parameters
        ----------
        i : interger
            Corresponding zone.

        h0_i : 1-dimensional ndarray, optional
            Initial value of Shadow prices for land-use sectors for zone i.
            If equals to None, the vector of zeros is used as initial value.

        Returns
        -------
        ret : 1-dimensional ndarray
            The vector of optimal land-use shadow prices for zone i.
        '''
        
        if h0_i != None:
            if h0_i.shape !=(len(self.housingSectors),):
                print "Error in shadow price shape, please give a initial vector of shape = (h_sectors,1)"
            v   =leastsq(self.residual_housing,h0_i,args=(i,),full_output=True,ftol=1e-12)
            return v
        else:
            h0_i = np.random.random(len(self.housingSectors))
            h0_i = np.where(self.param.InduProd[self.housingSectors,i]==0,0,h0_i)
            v   =leastsq(self.residual_housing,h0_i,args=(i,),full_output=True,ftol=1e-12)
            if v[-1]>4:
                if v[-1]==5:
                    v = leastsq(self.residual_housing,h0_i,args=(i,),maxfev=1500,full_output=True,ftol=1e-12)
                logging.debug('Optimizing housing zone: %s, Status: %s, Residuals %s   <----check this one, no solution found'%(i,v[-1],v[2]['fvec']))
            else:
                logging.debug( 'Optimizing housing zone: %s, Status: %s, Residuals %03s'%(i,v[-1],v[2]['fvec']))

            if norm(v[2]['fvec'])>0.001:
                v = leastsq(self.residual_housing,h0_i*0,args=(i,),full_output=True,ftol=1e-12)
                logging.debug('Optimizing housing zone: %s, Status: %s, Residuals %s   <----Re-executing with h0=0'%(i,v[-1],v[2]['fvec']))
        return v[0]         #THIS ONE WORKS!!!!!
    
    def calc_sp_lu_all(self, zones):  
        '''calc_sp_lu_all(zones)

        Computes the optimal shadow prices for land used
        sectors for the given zones. 

        This is done with a map() of calc_sp_lu over zones.

        Parameters
        ----------
        zones : 1-dimensional array like
            The zones where the land-use shadow prices need to be computed.

        Returns
        -------
        aux : 2-dimensional ndarray
            The vector of optimal land-use shadow prices for the zones 
            in zones.

        Examples
        --------
        '''
        if self.multicpu == True:
            logging.debug("Calculating X and h for housing-sectors")
            ti      = time()
            aux     = np.array(parmap(self.calc_sp_lu,zones))
            to      = time()
            logging.debug("Multiproc Solver, t=%f [s]"%(to-ti))
        else:
            logging.debug("Calculating X and h for housing-sectors")
            ti      = time()
            aux     = np.array(map(self.calc_sp_lu,zones))
            to      = time()
            logging.debug("Singleproc Solver, t=%f [s]"%(to-ti))
        return aux
            
    def calc_sp_housing(self):
        '''calc_sp_housing()

        Computes the optimal shadow prices for all zones. 
        The method sets the optimal value of land-use shadow prices 
        in the class variable "h_housing", and the computed induce production
        in the class variable "X_housing".
        It also computes the values of the demand function and Sustitution 
        probabilities as they only depend on housing shadow prices.
        This variables are Lcal.a, Lcal.S and Lcal.B
        It also sets the class variable "nonTransportableDone" as True.

        Parameters
        ----------

        Returns
        -------
        None

        Examples
        --------
        >>>filename = '/ExampleC_n/'
        >>>t=TranusConfig(nworkingDirectory = filename) 
        >>>Lcal=LCALn(t)
        >>>Lcal.calc_sp_housing()
        Running Housing SP optimization
        Optimizing housing zone: 0, Status: 2, Residuals [ 0. -0. -0.]
        Optimizing housing zone: 1, Status: 2, Residuals [ 0. -0.  0.]
        Optimizing housing zone: 2, Status: 2, Residuals [ 0. -0.  0.]
        '''
        
        logging.debug('''------------------------------------------------------------\nRunning Housing SP optimization [calc_sp_housing()]:\n------------------------------------------------------------\n''')
        if self.nonTransportableDone:
            return

        self.h_housing = self.calc_sp_lu_all(range(self.param.nZones)).transpose()
        self.X_housing = np.array(map(self.calc_induced_prod_housing,self.h_housing.transpose(),range(self.param.nZones))).transpose()    #INduprod from sectors housingSectors
        self.X[self.housingSectors,:] = self.X_housing
        #Fixing the values of S and a after the computation of housing
        #shadow prices

        self.h[self.housingSectors,:] = self.h_housing
        self.p[self.housingSectors,:] = self.param.Price[self.housingSectors,:]
        self.a = self.calc_a(self.h + self.p)
        self.S = self.calc_subst(self.h + self.p)
        # for n in self.genflux_sectors:
        for n in range(self.param.nSectors):
            self.D[n,:] = self.param.ExogDemand[n,:]+(self.a[:,n,:]* self.S[:,n,:]*(self.X_0+self.param.ExogProd)).sum(0)


        self.nonTransportableDone = True
        print( "  Total housing sectors fitting ||X-X_0|| = %s"%norm(self.X_housing - self.param.InduProd[self.housingSectors,:]))
        print( "  Total housing sectors fitting np.allclose(X, X_0, rtol=0.001) = %s"%np.allclose(self.X_housing,self.param.InduProd[self.housingSectors,:],0.001))
        print( "  Total housing sectors fitting np.allclose(X, X_0, rtol=0.0001) = %s"%np.allclose(self.X_housing,self.param.InduProd[self.housingSectors,:],0.0001))
        return


    #####################################################
    ##Computation of \phi and shadow prices optimisation:
    #####################################################


    def reshape_vec(self, vec):
        '''reshape_vec(vec)
        vec -- > h, p
        Transforms the vector vec = (h, p) containing h and p in 
        1-dimensional form, and returns both arrays h and p in 
        nSectors times nZones. Very useful for optimisation methods
        that takes only 1-dimensional inputs.'''

        nSectors = self.param.nSectors
        nZones = self.param.nZones
        h = vec[0:nSectors*nZones].reshape((nSectors,nZones))
        p = vec[nSectors*nZones:].reshape((nSectors,nZones))
        return h, p

    def deshape_vec(self,h,p):
        '''deshape_vec(h, p)
        h, p -- > vec
        Stacks h and p in an array.'''

        vec = np.hstack((h.reshape(-1),p.reshape(-1)))
        return vec
    ###############################################
    ##Omega optimisation (substitution parameters):
    ###############################################

    def f_omega(self, omega, scale = 1):
        '''f_omega(omega, scale = 1)
        Computes the value of the squared residual of the housing sectors.
        The shadow prices are set to their initial value (normally zero)

        Parameters
        ----------
        omega : 2-dimensional ndarray
            Lcal.param.omega values for housing consuming sectors. 
        scale : double (optional)
            Scale factor, ideally in the order of demax**2.

        Returns
        -------
        ret : double
            (1/scale)*||X-X_0||**2 for housing sectors.

        Examples
        --------
        '''

        n = len(self.housingSectors)
        m = len(self.param.substitutionSectors)
        if omega.shape!=(m,n):
            print 'you have to call f_omega with the right size (m,n) array'
            return

        residuals = np.zeros((n,self.param.nZones))
        h_housing = self.h_housing * 0
        self.param.omega[np.ix_(self.param.substitutionSectors,self.housingSectors)] = omega
        for i in range(self.param.nZones):
            residuals[:,i] = self.residual_housing(h_housing[:,i],i)
        ret = norm(residuals.ravel()) / scale
        return ret

    def f_omega_vec(self, vec):
        '''f_omega_vec(vec)
        Flatten out version of f_omega
        '''
        n = len(self.housingSectors)
        m = len(self.param.substitutionSectors)
        return self.f_omega(vec.reshape((m,n)))

    def set_omega(self, omega):
        n = len(self.housingSectors)
        m = len(self.param.substitutionSectors)
        if omega.shape!=(m,n):
            print 'you have to call f_omega with the right size (m,n) array'
            return
        self.param.omega[np.ix_(self.param.substitutionSectors,self.housingSectors)] = omega
        return 

    def get_omega(self):
        n = len(self.housingSectors)
        m = len(self.param.substitutionSectors)
        return self.param.omega[np.ix_(self.param.substitutionSectors,self.housingSectors)]

    def f_omega_grad(self, omega, scale=1, scaled=False, zones=range(229)):
        '''f_omega_grad(omega, scale = 1)
        Computes the value of the squared residual of the housing sectors, and
        the gradient of this function against omega. The shadow prices are set 
        to their initial value (normally zero)]. This method uses the jac_omega
        that is previously computed in self.residual_housing.

        Parameters
        ----------
        omega : 2-dimensional ndarray
            Lcal.param.omega values for housing consuming sectors. 
        scale : double (optional)
            Scale factor, ideally in the order of demax**2.

        Returns
        -------
        ret : double
            (1/scale)*||X-X_0||**2 for housing sectors.
        jac : 1-dimensional ndarray
            2*(1/scale)*<X-X_0, DX_omega> for housing sectors, the gradient of
            ret. This is done analitically.

        '''
        

        n = len(self.housingSectors)
        m = len(self.param.substitutionSectors)
        self.set_omega(omega)
        residuals = np.zeros((n,self.param.nZones))
        h_housing = self.h_housing*0
        
        DX_omega = np.zeros((n,self.param.nZones,m,n))
        for i in zones:
            residuals[:,i], jac = self.residual_housing(h_housing[:,i], i, jac_omega=True, scaled=scaled)
            DX_omega[:,i,:,:] = jac 
            # print i, jac/scale
        # print 'DX_max, residuals_max: %s,%s'%(DX_omega.max(), residuals.max())
        res_norm = norm(residuals.ravel())
        ret, grad = res_norm / scale, np.einsum('ni,nimq->mq',residuals,DX_omega).ravel() / (res_norm * scale)

        # print omega
        # print ret, grad
        return ret, grad

    def f_omega_grad_vec(self, vec, scaled=False, zones=range(229)):
        '''f_omega_grad_vec(vec)
        Vectorized version of f_omega_grad. Instead of taking a 2-dimensional 
        input, it takes the flatten version. This is done only to be able to
        use the optimisation algorithms with it.

        Parameters
        ----------
        vec: 1-dimensional ndarray
            Lcal.param.omega written in flat form, including values for 
            housing consuming sectors. 

        Returns
        -------
        ret : double
            (1/scale)*||X-X_0||**2 for housing sectors.
        jac : 1-dimensional ndarray
            2*(1/scale)*<X-X_0,DX_omega> for housing sectors, the gradient of
            ret. This is done analitically.

        '''
        n = len(self.housingSectors)
        m = len(self.param.substitutionSectors)

        return self.f_omega_grad(vec.reshape((m,n)),scaled=scaled,zones=zones)

    def f_omega_bounds(self, bound=0.1):
        '''f_omega_bounds(omega_a = 0.001, omega_b = 10)
        Funtion that set's up the bounds for the optimisation method.

        '''

        n = len(self.housingSectors)
        m = len(self.param.substitutionSectors)
        omegas = self.get_omega().ravel()
        if bound ==-1:
            bounds_omega = tuple((1, 3) for x in range(n*m))
        else:
            bounds_omega = tuple(((1-bound)*omegas[x], (1+bound)*omegas[x]) for x in range(n*m))
        return bounds_omega

    def omega_constraints(self):
        cons = ({'type': 'ineq',
                 'fun' : 1 })


    def omega_optimisation(self, bound=0.1, method='TNC', scaled=False, zones = -1):
        '''omega_optimisation()
        Computes the optimal values of omega parameters for the substitution
        model. This is done using bounds to for the omega values, and 
        performing optimisation of the residuals of the housing sectors
        production. This methods asumes that the shadow prices are all equals
        to zero and that self.calc_sp_housing() has not been runned.

        This method is void and updates the values of the self.param.omega
        variable.

        Parameters
        ----------

        Returns
        -------

        '''
        print 'Computing optimal omega (substitution) values:'
        if zones == -1:
            zones = range(self.param.nZones)
        n = len(self.housingSectors)
        m = len(self.param.substitutionSectors)
        if m==0:
            print 'No substitution in this model'
            return
        self.h_housing = self.h_housing*0
        omega_0 = self.param.omega[np.ix_(self.param.substitutionSectors,self.housingSectors)].copy()
        f_bounds = self.f_omega_bounds(bound)
        res = minimize(self.f_omega_grad_vec,
            args=(scaled,zones),    
            x0 = omega_0, 
            jac = True, 
            bounds = f_bounds, 
            method = method, 
            options = {'disp':True})
        print 'Obtained omega values for consuming sectors:'
        print res
        self.nonTransportableDone = False
        self.calc_sp_housing()
        print self.get_omega()
        return res
        
    def h_over_p(self, h):
        '''Computes h/p for housing sectors'''
        nZones = self.param.nZones
        h_p = np.zeros((len(self.housingSectors),nZones))
        if h.shape!= (len(self.housingSectors),nZones):
            print 'h has to be Hsectors by zones'
            return -1
        p = self.param.Price[self.housingSectors,:]
        h_p = np.where(p>0, h/p, 0)
        return h_p

    ##\phi optimisation (Transportable sectors):

    def calc_D_n(self, n):
        '''Computes the demands'''
        if not self.nonTransportableDone:
            self.calc_sp_housing()
        return self.D[n,:]
    
    def calc_Pr(self, ph, n):
        '''calc_Pr(self, ph, n):
        Computers Probabilities of transportable sector n at value ph'''
        nSectors    = self.param.nSectors
        nZones      = self.param.nZones
        beta        = self.param.beta
        thetaLoc    = self.param.thetaLoc
        lamda       = self.param.lamda
        t_nij       = self.param.t_nij
        A_ni        = self.A_ni
        if self.gen_flux(n) and n not in self.housingSectors:

            U_n = lamda[n] * ph[np.newaxis,:] + t_nij[n,:,:])
            Pr_n = np.zeros((nZones,nZones))
           
            if logit:
                aux = A_ni[n,np.newaxis,:] * np.exp( -beta[n] * U_n[:,:])  
            else: #powit
                aux = A_ni[n,np.newaxis,:] * U_n ** (-beta[n]) 
            s = aux.sum(1)
            Pr_n[:,:] = aux / s[:,np.newaxis]                                 

            return Pr_n
        else:
            return np.eye(nZones)

    def calc_X_n(self, ph, n):
        '''calc_X_n(self, ph, n):
        Computer production X for sector n, using demands'''

        D_n = self.calc_D_n(n)
        Pr_n = self.calc_Pr(ph, n)
        X_n = np.dot(Pr_n.transpose(), D_n)
        return X_n
    
    def calc_X(self, ph):
        '''calc_X(self, ph):
        Computer production X, using demands'''
        for n in self.genflux_sectors:
            self.X[n,:] = np.einsum('i,ij->j',self.D[n,:], self.calc_Pr(ph[n,:], n))
        return


    def calc_DX_n(self, ph, n):
        '''calc_DX_n(self, ph, n):
        Computer the gradient of function X in sector n'''

        nSectors    = self.param.nSectors
        nZones      = self.param.nZones
        beta        = self.param.beta
        lamda       = self.param.lamda
        t_nij       = self.param.t_nij
        #lamda       =self.param.lamda

        D_n = self.calc_D_n(n)
        Pr_n = self.calc_Pr(ph, n)
        DX = np.zeros((nZones,nZones))

        U_n = lamda[n] * ph[np.newaxis,:] + t_nij[n,:,:])
        # cython_DX_n(DX, nSectors, nZones, beta[n], lamda[n], D_n, Pr_n)
        cython_DX_n(DX, nSectors, nZones, beta[n], lamda[n], D_n, Pr_n, U_n)
            #cython is doing this:
            # DX2 = np.zeros((nZones,nZones))
            # for j in range(nZones):
            #     for k in range(nZones):
            #         aux=0
            #         for i in range(nZones):
            #             if k==j:
            #                 aux+=(-lamda[n]*beta[n]*(Pr_n[i,j]-Pr_n[i,j]**2))*D_n[i]
            #             else:
            #                 aux+=(lamda[n]*beta[n]*Pr_n[i,j]*Pr_n[i,k])*D_n[i]
            #         DX2[j,k] = aux  
            # print 'norm: %f'%norm(DX2-DX)
        return DX

    def res_X_n(self, ph, n):
        '''res_X_n(self, p,n):
        Computes the residual of production X at sector n'''

        r =  self.calc_X_n(ph, n)-self.param.InduProd[n,:]
        return r

    def norm_res_X_n(self,ph,n):

        r = self.res_X_n(ph, n) 
        DX = self.calc_DX_n(ph, n) 

        return norm(r)**2, 2*np.dot(r, DX)

    def norm_res_X_n2(self,ph,n):
        r = self.res_X_n(ph, n)
        return norm(r)**2


    def calc_ph_n(self, n, ph0, algo = 'leastsq'):
        '''calc_ph_n(self, n, ph0, algo = 'leastsq'):
        Computes the optimal value of ph for sector n, starting from ph0
        optional: algo = 'leastsq' or other to use minimize solver'''

        if not self.gen_flux(n):
            print 'n = %d is not a transportable sector'%n
            return
        if algo!='leastsq':
            res = minimize(self.norm_res_X_n, ph0, args =(n,), jac = True, method = algo)
        else:
            res = leastsq(self.res_X_n, ph0, args = (n,), Dfun = self.calc_DX_n, full_output=True, ftol = 1E-4, xtol = 1E-4 )
        #goodness of fit:
            infodict = res[2]
            ss_err = (infodict['fvec']**2).sum()
            y = self.param.InduProd[n,:]
            ss_tot = ((y-y.mean())**2).sum()
            rsquared = 1-(ss_err/ss_tot)
            logging.debug('  GOF: %f'%rsquared)
            logging.debug( '  Error: %f'%norm(infodict['fvec'] / self.param.InduProd[n,:].max()))

        return res

    def calc_ph(self, ph0):
        '''calc_ph(self, ph0):
        Computes optimal set of ph for all sectors starting from ph0'''

        logging.debug(  '''------------------------------------------------------------\nComputing optimal p+h [calc_ph(ph0)]:\n------------------------------------------------------------\n''')
        conv = 1
        ph = np.random.random((self.param.nSectors, self.param.nZones))
        d={}
        for n in self.genflux_sectors:
            logging.debug("  Computing h+p for sector n = %d"%n)
            res = self.calc_ph_n(n, ph0[n])
            ph[n, :] = res[0]
            d[n] = res[-1]

            if d[n]>4:
                conv = 0
        return ph - ph.mean(1)[:,np.newaxis], conv
        # return ph, conv


    def calc_p_linear(self, ph):
        '''calc_p_linear(self, ph):
        Computes the equilibrium prices solving the corresponding linear 
        system. This is only done for gen_flux type sectors (transportable)'''

        logging.debug(   '''------------------------------------------------------------\nComputing linear p [calc_p_linear(ph)]:\n------------------------------------------------------------\n''')    
        nZones = self.param.nZones  #number of zones: 225   
        nSectors = self.param.nSectors    #number of sectors: 20
        genflux_sectors = self.genflux_sectors
        nGenflux = len(self.genflux_sectors)
        ValueAdded = self.param.ValueAdded
        tm = self.param.tm_nij
        
        logging.debug("Calculating LAMBDA2 and DELTA2")
        LAMBDA2 =np.zeros((nSectors,nZones))
        DELTA2  =np.zeros((nSectors*nZones,nSectors*nZones))

        alpha = self.a
        S = self.S
        # Pr, U = self.calc_prob_loc(ph, 0*ph)
        Pr = np.zeros((nSectors,nZones, nZones))
        for n in range(nSectors):
            Pr[n,:,:] = self.calc_Pr(ph[n,:], n)
        self.Pr = Pr
        for m in range(nSectors): #THIS IS EXPENSIVE TIME (COULD BE REOPTIMIZED)
            
            for i in range(nZones):
                if self.gen_flux(m):
                    #print "Gen Flux: %d"%m
                    LAMBDA2[m,i]    =ValueAdded[m,i]+(alpha[m,:,i]*S[m,:,i]*(Pr[:,i,:]*tm[:,i,:]).sum(1)).sum()
                    DELTA2[m*nZones+i,:]=np.einsum('nj,n->nj',Pr[:,i,:],alpha[m,:,i]*S[m,:,i]).reshape(-1)

        LAMBDA2 =LAMBDA2.reshape(-1)

        logging.debug("SOLVING SPARSE SYSTEM 2")

        DELTA2=np.eye(nSectors*nZones)-DELTA2



        self.DELTA2=DELTA2

        self.LAMBDA2=LAMBDA2



        p = linalg.solve(DELTA2,LAMBDA2)

        logging.debug("Exiting CALC_PRIX, Solve STATUS: norm(dif)= %s"%norm(np.dot(DELTA2, p) - LAMBDA2))

        return p.reshape((nSectors, nZones))#,DELTA2,LAMBDA2

    def compute_shadow_prices(self, ph0):
        '''p, h, conv, lamda_opti = compute_shadow_prices(self,ph0):
        Computes optimal shadow prices
        Only for gen_flux sectors'''

        h = np.zeros((self.param.nSectors, self.param.nZones))
        self.calc_sp_housing()
        ph, conv = self.calc_ph(ph0)
        p = self.calc_p_linear(ph)
        self.calc_X(ph)

        p[self.housingSectors,:] = self.param.Price[self.housingSectors, :]
        #computes optimal lambda values minimising variance of shadow prices
        lamda_opti = self.compute_lamda(ph, p)
        for n in self.genflux_sectors:
            h[n,:] = ph[n,:] / self.param.lamda[n] - p[n,:]
            h[n,:] = h[n,:] - h[n,:].mean()
            self.h[n,:] = h[n,:]
            self.p[n,:] = p[n,:]
        # logging.debug( '  Var        = %s'%h.var(1))
        # logging.debug( '  lamda_opti = %s'%lamda_opti)
        return p, h, conv, lamda_opti

    def compute_lamda(self, ph, p):
        lamda_opti = np.zeros(self.param.nSectors)
        for n in self.genflux_sectors:
            lamda_opti[n] = np.var(ph[n,:])/np.cov(ph[n,:], p[n,:],bias=1)[1,0]
            # logging.debug( 'lamda_opti[%s] = %s'%(n,lamda_opti[n]))
        return lamda_opti
##Extras...

    def coscon(self):
        "Consumption Cost"
        consuming_sectors = self.param.demin.sum(0)
        coscon = np.einsum('nij,nij->ni',self.Pr, (self.p[:,np.newaxis,:]+self.param.tm_nij))
        coscon[consuming_sectors==0]=0
        return coscon
        
    def cospro(self):
        aux = np.einsum('mni, ni->mi', self.a*self.S, self.coscon())+ self.param.ValueAdded
        return aux

    def results_housing(self):
        h = self.h_housing
        cols_names = self.tranusConfig.numberingSectors()
        cols = [cols_names[i] for i in self.housingSectors]
        h_p = DataFrame(columns = cols, data = 100 * self.h_over_p(h).T)
        print h_p.describe()
        return h_p

    # def read_imploc(txtfile, pandas = False):
    #     '''Reads the output of imploc, all info comma delimited and put it in
    #     a matrix of nSectors x nZones x 7, where 7 is the following list:
    #         [TotProd, TotDem, ProdCost, Price, MinRes, MaxRes, Adjust]    
    #     '''
    #     if not pandas:
    #         out = np.loadtxt(txtfile, delimiter = ',', skiprows=1, usecols=(3,4,5,6,7,8,9)).reshape((Lcal.param.nSectors, Lcal.param.nZones,7))
    #     else:
    #         out = pd.read_csv('imploc_nohead.txt')
    #     return out

    def to_Series(self, A):

        n,i = A.shape
        list_zones = self.param.list_zones
        if (n, i) == (self.param.nSectors, self.param.nZones):
            list_sectors = self.param.list_sectors           
        else:
            list_sectors =  (np.array(self.param.list_sectors)[self.housingSectors]).tolist()
        l_sectors = np.array(list_sectors).repeat(i)
        return pd.Series(A.ravel(), index = [l_sectors.tolist(), list_zones*n])

    def to_DataFrame(self):
        X_ser = self.to_Series(self.X_housing)
        h_ser = self.to_Series(self.h_housing)
        p_ser = self.to_Series(self.param.Price[self.housingSectors,:])
        Indu_ser = self.to_Series(self.param.InduProd[self.housingSectors,:])
        adjust = h_ser/p_ser*100
        df = pd.DataFrame([Indu_ser, X_ser, p_ser, adjust, h_ser]).T
        df.columns = ['Production','Demand','Price','Adjust (%)','SPrice']
        return df

    #Grenoble Case model only!    
    def get_consumption(self,sectors, path='', agg=False, percentage=False, sqm=True):

        index = np.ix_(sectors, self.housingSectors, range(self.param.nZones))
        if sqm:
            cons = self.X_0[sectors, np.newaxis, :] * self.a[index] * self.S[index]
        else:
            cons = self.X_0[sectors, np.newaxis, :] * self.S[index]

        cons = cons.sum(2) / 1000
        cons_agg = np.zeros((5,6))
        xticks = [self.param.sectors_names[n] for n in self.housingSectors]
        if agg == True:
            cons_agg[:,0] = cons[:,0]
            cons_agg[:,1] = cons[:,1]+cons[:,2]
            cons_agg[:,2] = cons[:,3]+cons[:,4]
            cons_agg[:,3] = cons[:,5]
            cons_agg[:,4] = cons[:,6]+cons[:,7]
            cons_agg[:,5] = cons[:,8]

            cons = cons_agg
            xticks = ('mp','mm','mg','ap','am','ag')

        if percentage:
            cons = 100*cons/cons.sum(1)[:, np.newaxis]
        df = DataFrame(cons,columns=xticks,index=[self.param.list_sectors[x] for x in sectors])

        return df.T

    def plot_consumptions(self, sectors, path='', agg=False, compare=True, filename='cons_new.pdf'):
        fig = plt.gcf()
        fig.set_size_inches(18, 13)

        cons_ref = np.load(path+'cons_ref.npy')[sectors,:] / 1000
        index = np.ix_(sectors, self.housingSectors, range(self.param.nZones))
        cons = self.X_0[sectors, np.newaxis, :] * self.a[index] * self.S[index] 
        cons = cons.sum(2) / 1000
        cons_agg = np.zeros((5,6))
        cons_ref_agg = np.zeros((5,6))
        xticks = ('mp','mm1','mm2','mg1','mg2', 'ap','am1','am2','ag')
        if agg == True:
            cons_agg[:,0] = cons[:,0]
            cons_agg[:,1] = cons[:,1]+cons[:,2]
            cons_agg[:,2] = cons[:,3]+cons[:,4]
            cons_agg[:,3] = cons[:,5]
            cons_agg[:,4] = cons[:,6]+cons[:,7]
            cons_agg[:,5] = cons[:,8]

            cons_ref_agg[:,0] = cons_ref[:,0]
            cons_ref_agg[:,1] = cons_ref[:,1]+cons_ref[:,2]
            cons_ref_agg[:,2] = cons_ref[:,3]+cons_ref[:,4]
            cons_ref_agg[:,3] = cons_ref[:,5]
            cons_ref_agg[:,4] = cons_ref[:,6]+cons_ref[:,7]
            cons_ref_agg[:,5] = cons_ref[:,8]
            cons = cons_agg
            cons_ref = cons_ref_agg
            xticks = ('mp','mm','mg','ap','am','ag')


        plot_n = len(sectors)*100+ 11
        k=0
        ind = np.arange(len(self.housingSectors))
        if agg:
            ind = np.arange(6)
        width = 0.35
        for n in range(len(sectors)):
            plt.subplot(plot_n+k)
            
            if compare:
                plot_1 = plt.bar(ind , cons[n,:], width, color = 'green')
                plot_2 = plt.bar(ind - width, cons_ref[n,:], width, color = 'blue')
                plt.xticks(ind, xticks)
            else:
                plot_1 = plt.bar(ind , cons[n,:], width*2, color = 'green')
                plt.xticks(ind + width, xticks)
            plt.ylabel('Cons. %d [1000 m^2] '%sectors[n])
            
            k+=1
        if compare:
            plt.legend((plot_1[0], plot_2[0]), ('new', 'ref'), loc=('upper right'),
                bbox_to_anchor=(1.1, 1.0), ncol=1, fancybox=True, shadow=True)
        plt.savefig(filename, bbox_inches='tight',dpi=100)
        plt.show()
        plt.close()
        return  cons



if __name__ == '__main__':
    pass





        