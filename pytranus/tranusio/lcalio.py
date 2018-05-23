#! /usr/bin/env python
from pytranus.support import TranusConfig 
import os
import glob
import sys
import numpy as np
import struct
from datetime import datetime



def convert_list_to_float(string_list):

	for e in range(len(string_list)):
		try:
			string_list[e] = float(string_list[e])
		except ValueError: 
			print "This string cannot be converted to float"


	return string_list

def modifyStr(string,lenType):
	
	if len(string) == lenType:
		strModified = string
	else:
		strModified = string+" "*lenType

	return strModified


# def loadFiles(TranusProject,Scenario):
		
# 	CTL_filepath = os.path.join(TranusProject,'W_TRANUS.CTL')
# 	L0E_filepath = glob.glob(os.path.join(TranusProject,'W_*.L0E'))[0]
# 	Z1E_filepath = glob.glob(os.path.join(TranusProject,'W_*.Z1E'))[0]

# 	ScenarioPath = os.path.join(TranusProject,Scenario)
# 	L1E_filepath = glob.glob(os.path.join(ScenarioPath,'W_*.L1E'))[0]

# 	L1S_filepath = os.path.join(ScenarioPath,"NEW_LCAL"+"_"+Scenario+".L1S")

# 	return CTL_filepath,L0E_filepath,Z1E_filepath,L1E_filepath,L1S_filepath

# def numberingZones(Z1E_filepath):

# 	with open(Z1E_filepath,"r") as f :

# 		lines=f.readlines()
# 		numZones = []
# 		numZonesExt= []
# 		n=5  

# 		while n<len(lines)-1:
# 			while lines[n][1:3]!="--":
# 				numZones.append(int(str.split(lines[n])[0]))                   
# 				n+=1             
# 			n+=6		#skiping second level zones
# 			while lines[n][1:3]!="--":
# 				numZonesExt.append(int(str.split(lines[n])[0]))                  
# 				n+=1    
		
# 	return numZones,numZonesExt

# def numberingSectors(L1E_filepath):
	 
# 	with open(L1E_filepath,"r") as f :

# 		lines=f.readlines()
# 		numSectors=[]
# 		n=11  

# 		while n<len(lines):
# 			while lines[n][1:3]!="--":
# 				numSectors.append(int(str.split(lines[n])[0]))                   
# 				n+=1               
# 			n=len(lines)
		
# 		return numSectors

class L1s(object):
	'''Reads the L1S LCAL output file and returns the LCAL computed 
	variables'''
	def __init__(self,file_name,nbSectors,nbZones):

		self.file_name=file_name
		self.iter=0
		self.ian=0
		self.mes=0
		self.idia=0
		self.ihr=0
		self.mins=0
		self.pro = np.zeros((nbSectors,nbZones))
		self.cospro = np.zeros((nbSectors,nbZones))
		self.precio = np.zeros((nbSectors,nbZones))
		self.coscon = np.zeros((nbSectors,nbZones))
		self.utcon = np.zeros((nbSectors,nbZones))
		self.atrac = np.zeros((nbSectors,nbZones))
		self.ajuste = np.zeros((nbSectors,nbZones))
		self.dem = np.zeros((nbSectors,nbZones))


	def read(self):

		offset=0
		list_L1S=[]

		with open(self.file_name, "rb") as f:

			data=f.read()

			header=struct.unpack_from("<3h2i5h3s80s3s80s",data,offset)
			offset=offset+3*2+4+4+5*2+3+80+3+80
			file_major=header[0]
			#print "file_major",file_major
			list_L1S.append(file_major)
			file_minor=header[1]
			#print "file_minor",file_minor
			list_L1S.append(file_minor)
			file_release=header[2]
			#print "file_release",file_release
			list_L1S.append(file_release)
			ifmt_L1S=header[3] 
			#print "ifmt_L1S",ifmt_L1S
			list_L1S.append(ifmt_L1S)
			iter_=header[4]
			
			#print "iter",iter_
			list_L1S.append(iter_)
			ian=header[5]
			self.ian=ian
			#print "ian",ian
			list_L1S.append(ian)
			mes=header[6]
			#print "mes",mes
			self.mes=mes
			list_L1S.append(mes)
			idia=header[7]
			self.idia=idia
			#print "idia",idia
			list_L1S.append(idia)
			ihr=header[8]
			self.ihr=ihr
			#print "ihr",ihr
			list_L1S.append(ihr)
			mins=header[9]
			self.mins=mins
			#print "mins",mins
			list_L1S.append(mins)
			area=header[10]
			#print "area",area
			list_L1S.append(area)
			estudio=header[11]
			#print "estudio",estudio
			list_L1S.append(estudio)
			pol=header[12]
			#print "pol",pol
			list_L1S.append(pol)
			nombre=header[13]
			#print "nombre",nombre
			list_L1S.append(nombre)

			num_policy=struct.unpack_from("<2i",data,offset)
			offset=offset+2*4
			npol=num_policy[0]
			#print "npol",npol
			list_L1S.append(npol)
			npol_neg=num_policy[1]
			#print "npol_neg",npol_neg
			list_L1S.append(npol_neg)

			for i in range(npol):

				info_policy=struct.unpack_from("<2ib5s32s",data,offset)
				offset=offset+2*4+1+5+32
				index_i=info_policy[0]
				#print "index_i",index_i
				list_L1S.append(index_i)
				i_prev_pol=info_policy[1]
				#print "i_prev_pol",i_prev_pol
				list_L1S.append(i_prev_pol)
				prev_pol_type=info_policy[2]
				#print "prev_pol_type",prev_pol_type
				list_L1S.append(prev_pol_type)
				nom_pol=info_policy[3] 
				#print "nom_pol",nom_pol
				list_L1S.append(nom_pol)
				desc_pol=info_policy[4]
				#print "desc_pol",desc_pol
				list_L1S.append(desc_pol)

			npol_ipol_tuple=struct.unpack_from("<2i",data,offset)
			offset=offset+2*4
			npol=npol_ipol_tuple[0] 
			#print "npol",npol
			list_L1S.append(npol)
			ipol=npol_ipol_tuple[1] 
			#print "ipol",ipol
			list_L1S.append(ipol)

			sector_1=struct.unpack_from("<3i",data,offset) 
			offset=offset+3*4
			ns=sector_1[0] 
			#print "ns",ns
			list_L1S.append(ns)
			ns_neg=sector_1[1] 
			#print "ns_neg",ns_neg
			list_L1S.append(ns_neg)
			nflu=sector_1[2] 
			#print "nflu",nflu
			list_L1S.append(nflu)

			for i in range(ns):

				sector_2=struct.unpack_from("<2i32s?5f2i",data,offset) 
				offset=offset+2*4+32+1+5*4+2*4
				index_i=sector_2[0] 
				#print "index_i",index_i
				list_L1S.append(index_i)
				numsec=sector_2[1] 
				#print "numsec",numsec
				list_L1S.append(numsec)
				nomsec=sector_2[2]
				#print "nomsec",nomsec
				list_L1S.append(nomsec)
				lflu=sector_2[3] 
				#print "lflu",lflu
				list_L1S.append(lflu)
				beta_1=sector_2[4] 
				#print "beta_1",beta_1
				list_L1S.append(beta_1) 
				beta_2=sector_2[5]
				#print "beta_2",beta_2
				list_L1S.append(beta_2)
				gama_1=sector_2[6] 
				#print "gama_1",gama_1
				list_L1S.append(gama_1)
				gama_2=sector_2[7] 
				#print "gama_2",gama_2
				list_L1S.append(gama_2)
				min_price_to_cost_ratio=sector_2[8] 
				#print "min_price_to_cost_ratio",min_price_to_cost_ratio
				list_L1S.append(min_price_to_cost_ratio)
				sector_type=sector_2[9] 
				#print "sector_type",sector_type
				list_L1S.append(sector_type)
				target_sector=sector_2[10] 
				#print "target_sector",target_sector
				list_L1S.append(target_sector)

			ns_tuple=struct.unpack_from("<i",data,offset)
			offset=offset+4
			ns=ns_tuple[0]
			#print "ns",ns
			list_L1S.append(ns)

			num_sect=struct.unpack_from("<2i",data,offset) 
			offset=offset+2*4 #[NS,-NS]*
			ns=num_sect[0]
			#print "ns",ns 
			list_L1S.append(ns)
			ns_neg=num_sect[1]
			#print "ns_neg",ns_neg
			list_L1S.append(ns_neg)

			for i in range(ns):

				index_i_tuple=struct.unpack_from("<i",data,offset)
				offset=offset+4
				index_i=index_i_tuple[0]
				#print "index_i",index_i
				list_L1S.append(index_i)

				for j in range(ns):

					demand_functions=struct.unpack_from("<i12fi",data,offset)
					offset=offset+4+12*4+4
					index_j=demand_functions[0]
					#print "index_j",index_j
					list_L1S.append(index_j)
					demin=demand_functions[1] 
					#print "demin",demin
					#print "demin[%s,%s]" %(j,i),demin
					list_L1S.append(demin) 
					demax=demand_functions[2] 
					#print "demax",demax
					#print "demax[%s,%s]" %(j,i),demax
					list_L1S.append(demax) 
					delas=demand_functions[3] 
					#print "delas",delas
					#print "delas[%s,%s]" %(j,i),delas

					list_L1S.append(delas)
					selas=demand_functions[4] 
					#print "selas",selas
					#print "selas[%s,%s]" %(j,i),selas
					list_L1S.append(selas) 
					suslgsc=demand_functions[5] 
					#print "suslgsc",suslgsc
					#print "suslgsc[%s,%s]" %(j,i),suslgsc
					list_L1S.append(suslgsc) 
					xalfa_1=demand_functions[6] 
					#print "xalfa_1",xalfa_1
					#print "xalfa_1[%s,%s]" %(i,j),xalfa_1
					list_L1S.append(xalfa_1) 
					xalfa_2=demand_functions[7] 
					#print "xalfa_2",xalfa_2
					#print "xalfa_2[%s,%s]" %(i,j),xalfa_2
					list_L1S.append(xalfa_2) 
					xalfapro=demand_functions[8]
					#print "xalfapro",xalfapro
					#print "xalfapro[%s,%s]" %(i,j),xalfapro
					list_L1S.append(xalfapro) 
					xalfapre=demand_functions[9] 
					#print "xalfapre",xalfapre
					#print "xalfapre[%s,%s]" %(j,i),xalfapre
					list_L1S.append(xalfapre) 
					xalfacap=demand_functions[10]
					#print "xalfacap",xalfacap
					#print "xalfacap[%s,%s]" %(i,j),xalfacap
					list_L1S.append(xalfacap) 
					alfa_1=demand_functions[11]
					#print "alfa_1[%s,%s]" %(i,j),alfa_1
					#print "alfa_1",alfa_1
					list_L1S.append(alfa_1)
					alfa_2=demand_functions[12]
					#print "alfa_2",alfa_2
					#print "alfa_2[%s,%s]" %(i,j),alfa_2
					list_L1S.append(alfa_2) 
					mxsust=demand_functions[13] 
					#print "mxsust",mxsust
					list_L1S.append(mxsust) 

					for k in range(mxsust):

						nsust_tuple=struct.unpack_from("<i",data,offset) 
						offset=offset+4
						nsust=nsust_tuple[0] 
						#print "nsust",nsust
						"""if nsust !=0 :
							print "nsust[%s,%s,%s]"%(i,j,k),nsust
							if nsust_compare[i,j,k] == nsust:
								print "Ok"""
						list_L1S.append(nsust)
			
			ns_tuple=struct.unpack_from("<i",data,offset) 
			offset=offset+4
			ns=ns_tuple[0] 
			#print "ns",ns
			list_L1S.append(ns)

			param_nzn=struct.unpack_from("<4i",data,offset)
			offset=offset+4*4
			nzn=param_nzn[0] 
			#print "nzn",nzn
			list_L1S.append(nzn)
			nzn_neg=param_nzn[1]
			#print "nzn_neg",nzn_neg
			list_L1S.append(nzn_neg)
			nz_1=param_nzn[2] 
			#print "nz_1",nz_1
			list_L1S.append(nz_1)
			nz_2=param_nzn[3]
			#print "nz_2",nz_2 
			list_L1S.append(nz_2)

			for i in range(nzn):

				param_numzon=struct.unpack_from("<2i32s2i",data,offset)
				offset=offset+2*4+32+2*4
				index_i=param_numzon[0]
				#print "index_i",index_i
				list_L1S.append(index_i)
				numzon=param_numzon[1] 
				#print "numzon",numzon
				list_L1S.append(numzon)
				nomzon=param_numzon[2] 
				#print "nomzon",nomzon
				list_L1S.append(nomzon)
				jer_1=param_numzon[3]
				#print "jer_1",jer_1 
				list_L1S.append(jer_1)
				jer_2=param_numzon[4] 
				#print "jer_2",jer_2
				list_L1S.append(jer_2)

			nzn_tuple=struct.unpack_from("<i",data,offset)
			offset=offset+4
			nzn=nzn_tuple[0] 
			#print "nzn",nzn
			list_L1S.append(nzn)

			nzn_ns=struct.unpack_from("<2i",data,offset)
			offset=offset+2*4
			nzn=nzn_ns[0]
			#print "nzn",nzn
			list_L1S.append(nzn)
			ns=nzn_ns[1] 
			#print "ns",ns
			list_L1S.append(ns)

			for i in range (nzn):

				numzon_tuple=struct.unpack_from("<2i", data, offset) 
				offset=offset+2*4
				index_i=numzon_tuple[0] 
				#print "index_i",index_i
				list_L1S.append(index_i)
				numzon=numzon_tuple[1] 
				#print "numzon",numzon
				list_L1S.append(numzon)

				for n in range(ns):

					fmt="<idf2dfd3fd4fd3f"
					len_fmt=4+8+4+2*8+4+8+3*4+8+4*4+8+3*4
					param_l1s=struct.unpack_from(fmt,data,offset) 
					offset=offset+len_fmt
					index_n=param_l1s[0]
					#print "index_n",index_n
					list_L1S.append(index_n)
					xpro=param_l1s[1] 
					#print "xpro",xpro
					list_L1S.append(xpro)
					probase=param_l1s[2] 
					#print "probase",probase
					list_L1S.append(probase)
					pro=param_l1s[3] 
					self.pro[n,i]=pro
	
					#print "pro",pro
					list_L1S.append(pro)
					cospro=param_l1s[4] 
					self.cospro[n,i]=cospro
					
					#print "cospro",cospro
					list_L1S.append(cospro)
					prebase=param_l1s[5] 
					
					#print "prebase",prebase
					list_L1S.append(prebase)
					precio=param_l1s[6] 
					self.precio[n,i]=precio
					
					#print "precio",precio
					list_L1S.append(precio)
					xdem=param_l1s[7] 
					#print "xdem",xdem
					list_L1S.append(xdem)
					dem_xdem=param_l1s[8] 
					#print "dem_xdem",dem_xdem
					self.dem[n,i]=dem_xdem
					
					list_L1S.append(dem_xdem)
					coscon=param_l1s[9] 
					#print "coscon",coscon
					self.coscon[n,i]=coscon
					
					list_L1S.append(coscon)
					utcon=param_l1s[10] 
					#print "utcon",utcon
					self.utcon[n,i]=utcon
					
					
					list_L1S.append(utcon)
					rmin=param_l1s[11] 
					
					#print "rmin",rmin
					list_L1S.append(rmin)
					rmax=param_l1s[12]
					#print "rmax[%s,%s]" %(n,i),rmax 
					#print "rmax",rmax

					list_L1S.append(rmax)
					atrac=param_l1s[13]
					#print "atrac",atrac
					self.atrac[n,i]=atrac
					
					list_L1S.append(atrac)
					valag=param_l1s[14] 
					
					#print "valag",valag
					list_L1S.append(valag)
					ajuste=param_l1s[15] 
					self.ajuste[n,i]=ajuste
					
					#print "ajuste",ajuste
					list_L1S.append(ajuste)
					atrain=param_l1s[16] 
					
					#print "atrain",atrain
					list_L1S.append(atrain)
					stock=param_l1s[17] 
					
					
					#print "stock",stock
					list_L1S.append(stock)
					unstock=param_l1s[18]
					

					#print "unstock",unstock
					list_L1S.append(unstock)

		
		#return list_L1S
		return [self.pro,self.cospro,self.precio,self.coscon,self.utcon,self.atrac,self.ajuste,self.dem]



class L1sParam:

	def __init__(self,t):
		#tranus files used to extract parameters and L1S file
		self.CTL_file = t.CTL_filepath
		self.L0E_file = t.L0E_filepath
		self.Z1E_file = t.Z1E_filepath
		self.L1E_file = t.L1E_filepath
		self.L1S_file = t.L1S_filepath
		
		#numZones,nbZones,numZonesExt,nbZonesExt,numSectors,nbSectors
		self.numZones,self.numZonesExt = t.numberingZones()
		self.numTotZones = self.numZones+self.numZonesExt
		self.nbZones = len(self.numZones)
		self.nbZonesExt = len(self.numZonesExt)
		self.nbTotZones = self.nbZones+self.nbZonesExt
		self.nbTotZones_neg = -self.nbTotZones
		self.numSectors = t.numberingSectors()
		self.nbSectors = len(self.numSectors)

		#File_parameters
		self.fileMajor = 6
		self.fileMinor = 8
		self.fileRelease = 1
		self.ifmtL1S = 3
		self.nombre = ""

		#Z1E_parameters
		self.nzn = 0
		self.nzn_neg = 0 
		self.nz1 = 0
		self.nz2 = 0
		self.numZon = np.zeros(self.nbTotZones,np.int32)
		self.nomZon = np.empty(self.nbTotZones, dtype = 'S32')
		self.jer1 = np.zeros(self.nbTotZones,np.int32)
		self.jer2 = np.zeros(self.nbTotZones,np.int32)

		#L0E parameters
		self.xpro = np.zeros((self.nbSectors,self.nbTotZones))
		self.probase = np.zeros((self.nbSectors,self.nbTotZones))
		self.xdem = np.zeros((self.nbSectors,self.nbTotZones))
		self.prebase = np.zeros((self.nbSectors,self.nbTotZones))
		self.rmin = np.zeros((self.nbSectors,self.nbTotZones))
		self.rmax = np.zeros((self.nbSectors,self.nbTotZones)) 
		self.rmax.fill(8.99999949E+09)
		self.valag = np.zeros((self.nbSectors,self.nbTotZones))
		self.atrain = np.zeros((self.nbSectors,self.nbTotZones))

		#L1E parameters
		self.nbIterations = 0 # max value finded in L1E
		self.ns = 0
		self.ns_neg = 0
		self.nflu = 0
		self.lflu= np.zeros(self.nbSectors,np.bool_)
		self.numSec = self.numSectors
		self.nomSec = np.empty(self.nbSectors,dtype = 'S32')
		self.beta_1 = np.zeros(self.nbSectors)
		self.beta_2 = np.zeros(self.nbSectors)
		self.gama_1 = np.zeros(self.nbSectors)
		self.gama_2 = np.zeros(self.nbSectors)
		self.minPriceCostToRatio = np.zeros(self.nbSectors)
		self.sectorType = np.zeros(self.nbSectors)
		self.targetSector = np.zeros(self.nbSectors)
		self.demin = np.zeros((self.nbSectors,self.nbSectors))
		self.demax = np.zeros((self.nbSectors,self.nbSectors))
		self.delas = np.zeros((self.nbSectors,self.nbSectors))
		self.selas = np.zeros((self.nbSectors,self.nbSectors))
		self.suslgsc = np.ones((self.nbSectors,self.nbSectors))
		self.xalfa_1 = np.zeros((self.nbSectors,self.nbSectors))
		self.xalfa_2 = np.zeros((self.nbSectors,self.nbSectors))
		self.xalfapro = np.zeros((self.nbSectors,self.nbSectors))
		self.xalfapre = np.zeros((self.nbSectors,self.nbSectors))
		self.xalfacap = np.zeros((self.nbSectors,self.nbSectors))
		self.alfa_1 = np.zeros((self.nbSectors,self.nbSectors))
		self.alfa_2 = np.zeros((self.nbSectors,self.nbSectors))
		self.mxsust = 256
		self.nsust = np.zeros((self.nbSectors,self.nbSectors,self.mxsust))
		self.listSectorsSubst = []

		#CTL parameters
		self.area = ""
		self.estudio = ""
		self.pol = t.scenarioId
		self.npol = 0
		self.npol_neg = 0
		self.iPrevPol = []
		self.prevPolType = []
		self.nomPol = []
		self.descPol = []
		self.iPol = 1
		self.listiPol = []

		#time and date parameters
		self.ian = 0
		self.mes = 0 
		self.idia = 0 
		self.ihr = 0
		self.mins = 0

		#Thomas parameters 
		self.pro = np.zeros((self.nbSectors,self.nbTotZones))
		self.cospro = np.zeros((self.nbSectors,self.nbTotZones))
		self.precio = np.zeros((self.nbSectors,self.nbTotZones))
		self.coscon = np.zeros((self.nbSectors,self.nbTotZones))
		self.utcon = np.zeros((self.nbSectors,self.nbTotZones))
		self.atrac = np.zeros((self.nbSectors,self.nbTotZones))
		self.ajuste = np.zeros((self.nbSectors,self.nbTotZones))
		self.stock = np.zeros((self.nbSectors,self.nbTotZones))
		self.unstock = np.zeros((self.nbSectors,self.nbTotZones))
		self.dem = np.zeros((self.nbSectors,self.nbTotZones))

	def extractionL0Eparam(self):

		with open(self.L0E_file,"r") as f:
			lines = f.readlines()
			length_lines = len(lines)


			for i in range(length_lines):
				lines[i] = str.split(lines[i])

			string = "1.1"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			end_of_section = "*-"
			line+=2

			while lines[line][0][0:2] != end_of_section:

				sector,zone = self.numSectors.index(float(lines[line][0])),self.numZones.index(float(lines[line][1]))
				self.xpro[sector,zone] = lines[line][2]
				self.probase[sector,zone] = lines[line][3]
				self.xdem[sector,zone] = lines[line][4] 
				self.prebase[sector,zone] = lines[line][5]    
				self.valag[sector,zone] = lines[line][6]
				self.atrain[sector,zone] = lines[line][7]
				line+=1

		
			string = "2.1"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			line+=2           
			while lines[line][0][0:2] != end_of_section:
				sector,zone=self.numSectors.index(float(lines[line][0])),self.numZones.index(float(lines[line][1]))
				self.xdem[sector,zone] = lines[line][2]
				line+=1

			string = "2.2"
			for line in range(len(lines)) : 
				if (lines[line][0] == string):
					break
			line+=2           
			while lines[line][0][0:2] != end_of_section:
				sector,zone=self.numSectors.index(float(lines[line][0])),self.numZones.index(float(lines[line][1]))
				self.rmin[sector,zone] = lines[line][2]
				self.rmax[sector,zone] = lines[line][3]
				self.valag[sector,zone] =  lines[line][4]
				self.atrain[sector,zone] = lines[line][5]
				line+=1

			string = "3."
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break
			line+=2          
			while lines[line][0][0:2] != end_of_section:
				sector,zone=self.numSectors.index(float(lines[line][0])),self.numZones.index(float(lines[line][1]))
				self.rmin[sector,zone] = lines[line][2]
				self.rmax[sector,zone] = lines[line][3]
				line+=1

			# to put rmax for external zeros to 0 instead of 8.99...
			for sector in range(self.nbSectors):
				for zone in range(self.nbZones,self.nbTotZones):
					self.rmax[sector,zone] = 0

		return self.xpro,self.probase,self.xdem,self.prebase,self.valag,self.atrain,self.rmin,self.rmax

	def extractionZ1Eparam(self):

		with open (self.Z1E_file,"r") as f :

			lines = f.readlines()
			length_lines = len(lines)
			copyLines=list(lines)

			for i in range(length_lines):
				lines[i] = str.split(lines[i])

			string = "1.0"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			end_of_section = "*-"
			line+=2  

			while lines[line][0][0:2] != end_of_section:

				zone = self.numTotZones.index(float(lines[line][0]))
				self.numZon[zone] = lines[line][0]
				self.nomZon[zone] = copyLines[line].split("'")[1]
				self.nzn+=1
				self.nz1 = self.nzn
				self.nz2 = self.nzn
				self.jer1[zone] = self.nzn
				self.jer2[zone] = self.nzn
				line+=1
		
			string = "3.0" #we suppose that section 2.0 is never used, we jump to section 3.0
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			end_of_section = "*-"
			line+=2

			while lines[line][0][0:2] != end_of_section:

				zone = self.numTotZones.index(float(lines[line][0]))
				self.numZon[zone] = lines[line][0]
				self.nomZon[zone] = lines[line][1].strip("'")
				self.nzn+=1
				self.jer1[zone] = self.nzn
				self.jer2[zone] = self.nzn
				line+=1

		return self.nzn,self.nz1,self.nz2,self.numZon,self.nomZon,self.jer1,self.jer2

	def extractionL1Eparam(self):

		with open(self.L1E_file,"r") as f:
			lines = f.readlines()
			length_lines = len(lines)
			copyLines=list(lines)
			
			for i in range(length_lines):
				lines[i] = str.split(lines[i])

			string = "1.0"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			end_of_section = "*-"
			line+=2

			while lines[line][0][0:2] != end_of_section:
				self.nbIterations = int(lines[line][0])
				line+=1

			string = "2.1"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			end_of_section = "*-"
			line+=3

			
			while lines[line][0][0:2] != end_of_section:
				sector = self.numSectors.index(float(lines[line][0]))
				self.numSec[sector] = lines[line][0]
				self.nomSec[sector] = copyLines[line].split("'")[1]
				self.ns+=1
				self.ns_neg =-self.ns
				self.beta_1[sector] = lines[line][2]
				self.beta_2[sector] = lines[line][3]
				self.gama_1[sector] = lines[line][4]
				self.gama_2[sector] = lines[line][5]
				if self.beta_1[sector] !=0 :
					self.nflu+=1
					self.lflu[sector] = True
				else :
					self.lflu[sector] = False
				self.minPriceCostToRatio[sector] = lines[line][8]
				self.sectorType[sector] = lines[line][9]
				self.targetSector[sector] = 0
				line+=1
			
			self.numSectors = convert_list_to_float(self.numSectors)
			string ="2.2"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break
			line+=2

			while lines[line][0][0:2] != end_of_section:
				
				sector_1,sector_2=self.numSectors.index(float(lines[line][0])),self.numSectors.index(float(lines[line][1]))
				self.demin[sector_1,sector_2] = lines[line][2]
				if float(lines[line][3])-float(lines[line][2])>=0 :
					self.demax[sector_1,sector_2] = float(lines[line][3])-float(lines[line][2]) 
				self.delas[sector_1,sector_2] = lines[line][4]
				line+=1
			
			self.numSectors = convert_list_to_float(self.numSectors)
			string = "2.3"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			line+=2
			copy_listSectorsSubst = []
			
			while lines[line][0][0:2] != end_of_section:
				
				if len(lines[line])==5:
					sector_1 = self.numSectors.index(float(lines[line][0]))
					copy_sector_1=sector_1
					sector_2 = self.numSectors.index(float(lines[line][3]))
					copy_sector_2 = sector_2
					self.selas[sector_1,sector_2] = lines[line][1]
					copy_selas = lines[line][1]
					self.suslgsc[sector_1,sector_2] = lines[line][2]
					copy_suslgsc = lines[line][2]
					self.listSectorsSubst.append(copy_sector_2)
				if len(lines[line]) ==2:
					sector_2 = self.numSectors.index(float(lines[line][0]))
					copy_sector_2 = sector_2
					self.selas[copy_sector_1,sector_2] = copy_selas
					self.suslgsc[copy_sector_1,sector_2] = copy_suslgsc
					self.listSectorsSubst.append(copy_sector_2)
				
				if lines[line][0] == "/":
					for i,value_i in enumerate(self.listSectorsSubst) :
						for j,value_j in enumerate(self.listSectorsSubst):
							if value_i == value_j:
								copy_listSectorsSubst = list(self.listSectorsSubst)
								copy_listSectorsSubst.remove(value_i)
								for k,value_k in enumerate(copy_listSectorsSubst):
									self.nsust[value_i,copy_sector_1,k] = value_k+1
								copy_listSectorsSubst =[]
					self.listSectorsSubst  = []	
				line+=1
				

			self.numSectors = convert_list_to_float(self.numSectors)
			string = "3.1"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			line+=2

			while lines[line][0][0:2] != end_of_section:
				sector = self.numSectors.index(float(lines[line][0]))	
				attrac_sector = self.numSectors.index(float(lines[line][1]))
				self.xalfa_1[sector,attrac_sector] = lines[line][2]
				self.xalfa_2[sector,attrac_sector] = lines[line][3]
				self.xalfapro[sector,attrac_sector] = lines[line][4]
				self.xalfapre[sector,attrac_sector] = lines[line][5]
				self.xalfacap[sector,attrac_sector] = lines[line][6]
				line+=1
			
			self.numSectors=convert_list_to_float(self.numSectors)
			string = "3.2"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break
			line+=2

			while lines[line][0][0:2] != end_of_section:
				sector = self.numSectors.index(float(lines[line][0]))	
				var = self.numSectors.index(float(lines[line][1]))
				self.alfa_1[sector,var] = lines[line][2]
				self.alfa_2[sector,var] = lines[line][3]
				line+=1


		
		return self.ns, self.nflu, self.lflu,self.numSec,self.nomSec,self.beta_1,self.beta_2,self.gama_1,self.gama_2,self.minPriceCostToRatio,self.sectorType,self.targetSector,self.demin,self.demax,self.delas,self.selas,self.suslgsc,self.xalfa_1,self.xalfa_2,self.xalfapro,self.xalfapre,self.xalfacap,self.alfa_1,self.alfa_2

	def extractionCTLparam(self):

		with open(self.CTL_file,"r") as f:
			lines = f.readlines()
			length_lines = len(lines)
			copyLines = list(lines)

			for i in range(length_lines):
				lines[i] = str.split(lines[i])
				

			string = "1.0"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			end_of_section = "*-"
			line+=2


			while lines[line][0][0:2] != end_of_section:
				self.area = copyLines[line].split("'")[1]
				self.estudio = copyLines[line].split("'")[3]
				line+=1
			
			string = "2.0"
			for line in range(len(lines)):
				if (lines[line][0] == string):
					break

			end_of_section = "*-"
			line+=2

			while lines[line][0][0:2] != end_of_section:
				self.nomPol.append(copyLines[line].split("'")[1])
				self.descPol.append(copyLines[line].split("'")[3])
				self.listiPol.append(self.iPol)
				self.npol+=1
				self.npol_neg = -self.npol
				self.iPol+=1
				
				#PrevPoltype
				if copyLines[line].split("'")[5] == ' ':
					self.prevPolType.append(0)
				elif copyLines[line].split("'")[5] != copyLines[line].split("'")[1]:
					self.prevPolType.append(2)
				elif copyLines[line].split("'")[5][:2] == copyLines[line].split("'")[1][:2]:
					self.prevPolType.append(1)

				#iPrevPol
				for i in range(len(self.nomPol)):
					if copyLines[line].split("'")[5] == ' ' :
						self.iPrevPol.append(0)	
					elif copyLines[line].split("'")[5] == self.nomPol[i] :
						self.iPrevPol.append(i+1)
				

				line+=1

			#iPol
			for j in range(len(self.nomPol)):
				if self.pol == self.nomPol[j] :
					self.iPol = self.listiPol[j]
					break

		return self.area,self.estudio,self.pol,self.npol,self.iPrevPol,self.prevPolType,self.nomPol,self.descPol,self.iPol

	def getDateTime(self):

		date_time = datetime.now().strftime('%Y %m %d %H %M')
		self.ian = int(date_time.split(' ')[0])
		self.mes = int(date_time.split(' ')[1])
		self.idia = int(date_time.split(' ')[2])
		self.ihr = int(date_time.split(' ')[3])
		self.mins= int(date_time.split(' ')[4])

		return self.ian,self.mes,self.idia,self.ihr,self.mins

	def extractL1S(self, path_L1S):
		others=L1S_READ(path_L1S,self.nbSectors,self.nbTotZones).read()
		self.pro = others[0]
		self.cospro = others[1]
		self.precio = others[2]
		self.coscon = others[3]
		self.utcon = others[4]
		self.atrac = others[5]
		self.ajuste = others[6]
		self.dem = others[7]

	def extractothers(self, Lcal):
		self.pro[:,:227] = Lcal.X
		self.cospro[:,:227] = Lcal.cospro()
		self.precio[:,:227] = Lcal.p
		self.coscon[:,:227] = Lcal.coscon()
		self.utcon[:,:227] = Lcal.p + Lcal.h
		self.atrac[:,:227] = Lcal.A_ni
		self.ajuste[:,:227] = Lcal.h
		self.dem[:,:227] = Lcal.D






	def runParametersExtraction(self):

		self.extractionL0Eparam()
		self.extractionZ1Eparam()
		self.extractionL1Eparam()
		self.extractionCTLparam()
		self.getDateTime()
		# self.extractothers()

	def WRLPAR(self,file, Lcal):
		'''Writes the L1S file'''

		self.extractothers(Lcal)
		self.getDateTime()

		listHeader = [self.fileMajor,self.fileMinor,self.fileRelease,self.ifmtL1S]
		header = struct.pack("<3hi",*listHeader)
		file.write(header)

		listIter = [self.nbIterations] 
		iterations = struct.pack("<i",*listIter)
		file.write(iterations)

		listTime = [self.idia,self.mes,self.ian,self.ihr,self.mins] # a chercher date et heure !!!!!!!!!!!!!
		time = struct.pack("<5h",*listTime)
		file.write(time)

		listAreaEstudio = [modifyStr(self.area,3),modifyStr(self.estudio,80),self.pol,self.nombre]
		AreaEstudio = struct.pack("<3s80s3s80s",*listAreaEstudio)
		file.write(AreaEstudio)

		self.WritePolInfo(file)

		listSector_1 = [self.ns,self.ns_neg,self.nflu]
		sector_1 = struct.pack("<3i",*listSector_1) 
		file.write(sector_1)

		for i in range(self.ns):

			listSector_2 = [i+1,self.numSec[i],modifyStr(self.nomSec[i],32),self.lflu[i],self.beta_1[i],self.beta_2[i],self.gama_1[i],self.gama_2[i],self.minPriceCostToRatio[i],self.sectorType[i],self.targetSector[i]]
			sector_2 = struct.pack("<2i32s?5f2i",*listSector_2)
			file.write(sector_2)

		listNs = [self.ns]
		ns = struct.pack("<i",*listNs)
		file.write(ns)

		listNumSect = [self.ns,self.ns_neg]
		numSect = struct.pack("<2i",*listNumSect)
		file.write(numSect)

		for i in range(self.ns):

			listIndex = [i+1]
			index = struct.pack("<i",*listIndex)
			file.write(index)

			for j in range(self.ns):

				listDemandFunctions = [j+1,self.demin[j,i],self.demax[j,i],self.delas[j,i],self.selas[j,i],self.suslgsc[j,i],self.xalfa_1[i,j],self.xalfa_2[i,j],self.xalfapro[i,j],self.xalfapre[i,j],self.xalfacap[i,j],self.alfa_1[i,j],self.alfa_2[i,j],self.mxsust]
				demandFunctions = struct.pack("<i12fi",*listDemandFunctions)
				file.write(demandFunctions)

				for k in range(self.mxsust):
					
					listNsust= [self.nsust[i,j,k]]
					nsust = struct.pack("<i",*listNsust)
					file.write(nsust)				
		
		listNs = [self.ns]
		ns = struct.pack("<i",*listNs)
		file.write(ns)

	def WritePolInfo(self,file):

		listNpol = [self.npol,self.npol_neg]
		npol = struct.pack("<2i",*listNpol)
		file.write(npol)

		for i in range(self.npol):

			listInfoPolicy = [i+1,self.iPrevPol[i],self.prevPolType[i],modifyStr(self.nomPol[i],5),modifyStr(self.descPol[i],32)]
			infoPolicy = struct.pack("<2ib5s32s",*listInfoPolicy)
			file.write(infoPolicy)

		listNpolIpol = [self.npol,self.iPol]
		npolIpol=struct.pack("<2i",*listNpolIpol)
		file.write(npolIpol)

	def GRAL1S(self, Lcal):

		with open(self.L1S_file,"wb") as file:

			self.WRLPAR(file, Lcal)

			listParamNzn = [self.nbTotZones,self.nbTotZones_neg,self.nz1,self.nz2]
			paramNzn = struct.pack("<4i",*listParamNzn)
			file.write(paramNzn)

			for i in range(self.nbTotZones):

				listParamZon = [i+1,self.numZon[i],modifyStr(self.nomZon[i],32),self.jer1[i],self.jer2[i]]
				paramZon = struct.pack("<2i32s2i",*listParamZon)
				file.write(paramZon)
			
			listNzn = [self.nbTotZones]
			nzn = struct.pack("<i",*listNzn)
			file.write(nzn)

			listNznNs = [self.nbTotZones,self.ns]
			nznNs = struct.pack("<2i",*listNznNs)
			file.write(nznNs)

			for i in range(self.nbTotZones):

				listINumzon = [i+1,self.numZon[i]]
				INumzon = struct.pack("<2i",*listINumzon)
				file.write(INumzon)

				for n in range(self.ns):

					#print "self.rmax[%s,%s]"%(n,i),self.rmax[n,i]

					listL1S = [n+1,self.xpro[n,i],self.probase[n,i],self.pro[n,i],self.cospro[n,i],self.prebase[n,i],self.precio[n,i],self.xdem[n,i],self.dem[n,i],self.coscon[n,i],self.utcon[n,i],self.rmin[n,i],self.rmax[n,i],self.atrac[n,i],self.valag[n,i],self.ajuste[n,i],self.atrain[n,i],self.stock[n,i],self.unstock[n,i]]
					paramL1S = struct.pack("<idf2dfd3fd4fd3f",*listL1S)
					file.write(paramL1S)
		return

	def RDLPAR(self,data,offset):

		header=struct.unpack_from("<3h2i5h3s80s3s80s",data,offset)
		offset=offset+3*2+4+4+5*2+3+80+3+80

		self.fileMajor = header[0]
		self.fileMinor = header[1]
		self.fileRelease = header[2]
		self.ifmtL1S = header[3] 
		self.nbIterations = header[4]
		self.ian = header[5]
		self.mes = header[6]
		self.idia = header[7]
		self.ihr = header[8]
		self.mins = header[9]
		self.area = header[10]
		self.estudio = header[11]
		self.pol = header[12]
		self.nombre=header[13]
		
		offset = self.skipPolInfo(data,offset)
		
		sector_1 = struct.unpack_from("<3i",data,offset)
		offset=offset+3*4

		self.ns=sector_1[0]
		self.ns_neg=sector_1[1]
		self.nflu=sector_1[2]
	
		for i in range(self.ns):

			sector_2=struct.unpack_from("<2i32s?5f2i",data,offset)
			offset=offset+2*4+32+1+5*4+2*4

			index = sector_2[0]
			self.numSec[i] = sector_2[1] 
			self.nomSec[i] = sector_2[2]
			self.lflu[i] = sector_2[3]
			self.beta_1[i] = sector_2[4]
			self.beta_2[i] = sector_2[5]
			self.gama_1[i] = sector_2[6]
			self.gama_2[i] = sector_2[7]
			self.minPriceCostToRatio[i] = sector_2[8]
			self.sectorType[i] = sector_2[9]
			self.targetSector[i] = sector_2[10]
	
		nsTuple =struct.unpack_from("<i",data,offset)
		offset = offset+4
		self.ns = nsTuple[0]
	
		numSect = struct.unpack_from("<2i",data,offset)
		offset = offset+2*4 
		self.ns = numSect[0]
		self.ns_neg = numSect[1]

		for i in range(self.ns):

			iTuple = struct.unpack_from("<i",data,offset)
			offset = offset+4
			index_i = iTuple[0]

			for j in range(self.ns):

					demandFunctions = struct.unpack_from("<i12fi",data,offset)
					offset=offset+4+12*4+4

					index_j = demandFunctions[0]
					self.demin[j,i] = demandFunctions[1] 
					self.demax[j,i] = demandFunctions[2] 
					self.delas[j,i] = demandFunctions[3] 
					self.selas[j,i] = demandFunctions[4] 
					self.suslgsc[j,i] = demandFunctions[5]
					self.xalfa_1[i,j] = demandFunctions[6] 
					self.xalfa_2[i,j] = demandFunctions[7]
					self.xalfapro[i,j] = demandFunctions[8] 
					self.xalfapre[i,j] = demandFunctions[9]
					self.xalfacap[i,j] = demandFunctions[10] 
					self.alfa_1[i,j] = demandFunctions[11]
					self.alfa_2[i,j] = demandFunctions[12] 
					self.mxsust = demandFunctions[13] 
		
					for k in range(self.mxsust):

						nsustTuple = struct.unpack_from("<i",data,offset) 
						offset = offset+4
						self.nsust[i,j,k] = nsustTuple[0]
						
		nsTuple = struct.unpack_from("<i",data,offset)
		offset = offset+4
		self.ns = nsTuple[0] 

		return offset

	def skipPolInfo(self,data,offset):

		numPolicy = struct.unpack_from("<2i",data,offset)
		offset = offset+2*4

		self.npol = numPolicy[0]
		self.npol_neg = numPolicy[1]

		for i in range(self.npol):

			infoPolicy = struct.unpack_from("<2ib5s32s",data,offset)
			offset=offset+2*4+1+5+32

			index = infoPolicy[0]
			self.iPrevPol[i] = infoPolicy[1]
			self.prevPolType[i] = infoPolicy[2] 
			self.nomPol[i] = infoPolicy[3] 
			self.descPol[i] = infoPolicy[4] 
		
		npolIpol = struct.unpack_from("<2i",data,offset) 
		offset=offset+2*4

		self.npol = npolIpol[0]
		self.ipol = npolIpol[1]

		
		return offset

	def LEEL1S(self):

		offset = 0
	
		with open (self.L1S_file,"rb") as file:

			data = file.read()

			offset = self.RDLPAR(data,offset)

			paramNzn = struct.unpack_from("<4i",data,offset) 
			offset = offset+4*4
			self.nzn = paramNzn[0]
			self.nzn_neg = paramNzn[1]
			self.nz1 = paramNzn[2] 
			self.nz2 = paramNzn[3] 
		
			for i in range(self.nzn):

				paramNumzon = struct.unpack_from("<2i32s2i",data,offset)
				offset = offset+2*4+32+2*4
				index = paramNumzon[0] 
				self.numZon[i] = paramNumzon[1] 
				self.nomZon[i] = paramNumzon[2] 
				self.jer1[i] = paramNumzon[3] 
				self.jer2[i] = paramNumzon[4] 	
		
			nznTuple = struct.unpack_from("<i",data,offset)
			offset = offset+4
			self.nzn = nznTuple[0] 

			nznNs = struct.unpack_from("<2i",data,offset) 
			offset = offset+2*4
			nzn = nznNs[0]
			ns = nznNs[1] 

			for i in range (self.nzn):

				numzonTuple = struct.unpack_from("<2i", data, offset) 
				offset = offset+2*4

				index_i = numzonTuple[0] 
				self.numZon[i] = numzonTuple[1]
			
				for n in range(self.ns):

					fmt = "<idf2dfd3fd4fd3f"
					lenFmt = 4+8+4+2*8+4+8+3*4+8+4*4+8+3*4
					paramL1S = struct.unpack_from(fmt,data,offset)
					offset = offset+lenFmt

					index_n = paramL1S[0]
					self.xpro[n,i] = paramL1S[1]
					self.probase[n,i] = paramL1S[2]
					self.pro[n,i] = paramL1S[3]
					self.cospro[n,i] = paramL1S[4] 
					self.prebase[n,i] = paramL1S[5] 
					self.precio[n,i] = paramL1S[6]
					self.xdem[n,i] = paramL1S[7]
					self.dem[n,i] = paramL1S[8] 
					self.coscon[n,i] = paramL1S[9] 
					self.utcon[n,i] = paramL1S[10] 
					self.rmin[n,i] = paramL1S[11] 
					self.rmax[n,i] = paramL1S[12] 
					self.atrac[n,i] = paramL1S[13] 
					self.valag[n,i] = paramL1S[14] 
					self.ajuste[n,i] = paramL1S[15] 
					self.atrain[n,i] = paramL1S[16] 
					self.stock[n,i] = paramL1S[17] 
					self.unstock[n,i] = paramL1S[18] 


	def run(self):
		
		# parameters extraction
		self.runParametersExtraction()

		# Writing L1S file
		self.GRAL1S()

		#Reading L1S file
		self.LEEL1S()

if __name__=='__main__':

	try :
		TranusProject = sys.argv[1]
	except :
		print "Path for Tranus Project ?"
		TranusProject = raw_input()

	try :
		Scenario = sys.argv[2]
	except :
		print "Scenario ?"
		Scenario = raw_input()

	test = L1sParam(TranusProject,Scenario)
	test.run()