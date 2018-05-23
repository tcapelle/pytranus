from __future__ import division
from cython.parallel cimport prange

import numpy as np
cimport numpy as c_np

import cython



DTYPE = np.float
from libc.math cimport exp


ctypedef c_np.float_t DTYPE_t



@cython.boundscheck(False) # turn of bounds-checking for entire function
def cython_DX_h(c_np.ndarray[DTYPE_t, ndim=3] Pr, 
				c_np.ndarray[DTYPE_t, ndim=1] lamda, 
				c_np.ndarray[DTYPE_t, ndim=1] beta, 
				c_np.ndarray[DTYPE_t, ndim=2] U, 
				c_np.ndarray[DTYPE_t, ndim=3] alpha, 
				c_np.ndarray[DTYPE_t, ndim=2] gap, 
				c_np.ndarray[DTYPE_t, ndim=2] delta, 
				c_np.ndarray[DTYPE_t, ndim=2] X_0, 
				int nSectors, 
				int nZones, 
				c_np.ndarray[c_np.int_t, ndim=1] genflux_sectors):

	assert Pr.dtype == DTYPE and lamda.dtype == DTYPE and \
	beta.dtype == DTYPE and U.dtype == DTYPE and alpha.dtype == DTYPE and \
	gap.dtype == DTYPE and delta.dtype == DTYPE and X_0.dtype == DTYPE and \
	genflux_sectors.dtype == int

	cdef c_np.ndarray[DTYPE_t,ndim=4] DX_h = np.zeros((nSectors,nZones,
													   nSectors,nZones),
													   dtype=DTYPE)
	cdef DTYPE_t sum_mi,aux1,aux2,delta_mn
	cdef unsigned int i,j,k,m,ni,L,n#, zero_entries
	sum_mi=0.0
	aux1=0.0
	aux2=0.0
	n=0
	# zero_entries=0
	L=genflux_sectors.shape[0]
	for ni in prange(L,nogil=True):
	# for ni in xrange(L):
		n=genflux_sectors[ni]
		for j in xrange(nZones):
			for k in xrange(nZones):
				sum_mi=0.0
				
				for m in xrange(nSectors):
					if alpha[m,n,0]==0:#if there is no consume of n by m )cheking just 1 zone, could be any
						pass
					else:
						delta_mn = delta[m,n]
						aux2=lamda[n]*beta[n]
						if gap[m,n]!=0:  #this is always cero! demin=demax
							aux1=gap[m,n]*delta_mn
							for i in xrange(nZones):
								if k==j:
									sum_mi=sum_mi+(-aux1*exp(-delta_mn*U[n,i])*Pr[n,i,k]*Pr[n,i,j]+alpha[m,n,i]*(-aux2*(Pr[n,i,j]-Pr[n,i,j]*Pr[n,i,j])))*X_0[m,i]
								else:
									sum_mi=sum_mi+(-aux1*exp(-delta_mn*U[n,i])*Pr[n,i,k]*Pr[n,i,j]+alpha[m,n,i]*(aux2*Pr[n,i,j]*Pr[n,i,k]))*X_0[m,i]		
						else:
							for i in xrange(nZones):
								if k==j:
									sum_mi=sum_mi+(alpha[m,n,i]*(-aux2*(Pr[n,i,j]-Pr[n,i,j]*Pr[n,i,j])))*X_0[m,i]
								else:
									sum_mi=sum_mi+(alpha[m,n,i]*(aux2*Pr[n,i,j]*Pr[n,i,k]))*X_0[m,i]
				DX_h[n,j,n,k]=sum_mi
				# if sum_mi==0:
				# 	zero_entries=zero_entries+1			
		#print 'finished!'
	# print zero_entries
	return DX_h


@cython.boundscheck(False) # turn of bounds-checking for entire function
def cython_DX_n(c_np.ndarray[DTYPE_t, ndim=2] DX, 
				int nSectors, 
				int nZones, 
				DTYPE_t beta, 
				DTYPE_t lamda,
				c_np.ndarray[DTYPE_t, ndim=1] D_n, 
				c_np.ndarray[DTYPE_t, ndim=2] Pr_n,
				c_np.ndarray[DTYPE_t, ndim=2] U_n,
				bool logit):

	assert DX.dtype == DTYPE and D_n.dtype ==DTYPE and Pr_n.dtype == DTYPE and\
	U_n.dtype == DTYPE

	cdef unsigned int i,j,k  #zero_entries
	cdef DTYPE_t sum_i
	sum_i = 0.0

	for j in prange(nZones,nogil=True):
		for k in xrange(nZones):
			sum_i = 0
			for i in xrange(nZones):
				if logit:
					if k==j:
						sum_i = sum_i + (- lamda * beta * (Pr_n[i,j] - Pr_n[i,j] ** 2)) * D_n[i]
					else:
						sum_i = sum_i + (lamda * beta * Pr_n[i,j] * Pr_n[i,k]) * D_n[i]
				else:
					if k==j:
						sum_i = sum_i + (- (lamda * beta / U_n[i,j]) * (Pr_n[i,j] - Pr_n[i,j] ** 2)) * D_n[i]
					else:
						sum_i = sum_i + ((lamda * beta / U_n[i,j]) * Pr_n[i,j] * Pr_n[i,k]) * D_n[i]
			DX[j,k] = sum_i
	return DX

# @cython.boundscheck(False) # turn of bounds-checking for entire function
# def cython_DX_delta(c_np.ndarray[DTYPE_t, ndim=2] DX, 
# 					int nSectors, 
# 					int nZones, 
# 					c_np.ndarray[DTYPE_t, ndim=1] X0, 
# 					c_np.ndarray[DTYPE_t, ndim=2] delta_mn, 
# 					c_np.ndarray[DTYPE_t, ndim=2] S_mn, 
# 					c_np.ndarray[DTYPE_t, ndim=2] a_mn
# 					c_np.ndarray[DTYPE_t, ndim=2] dS_mn
# 					c_np.ndarray[DTYPE_t, ndim=2] da_mn):

# 	assert DX.dtype == DTYPE and 
# 		S_mn.dtype == DTYPE and 
# 		dS_mn.dtype == DTYPE and
# 		a_mn.dtype == DTYPE and
# 		da_mn.dtype == DTYPE

# 	cdef unsigned int m,n,q  #zero_entries
# 	cdef DTYPE_t sum_i
# 	sum_i = 0.0

# 	for n in prange(nSectors, nogil=True):
# 		for m in xrange(nSectors):
# 			for q in xrange(nSectors):
				
# 					sum_i = sum_i + (-lamda*beta*(Pr_n[i,j]-Pr_n[i,j]**2))*D_n[i]
# 				else:
# 					sum_i = sum_i +(lamda*beta*Pr_n[i,j]*Pr_n[i,k])*D_n[i]
# 			DX[j,k] = sum_i
# 	return DX