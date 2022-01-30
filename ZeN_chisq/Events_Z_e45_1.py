#!/usr/bin/env python
# coding: utf-8

# Leire Larizgoitia Arcocha

from __future__ import division

#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#from matplotlib import pyplot as plt
#plt.rcParams.update({'font.size': 20})
#import matplotlib.colors as mcolors
#import matplotlib.ticker as mtick
#from matplotlib.ticker import MultipleLocator
#from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math
import scipy.integrate as integrate
import scipy.integrate as quad
import scipy.constants as constants
import pandas as pd
import random
from scipy import stats
from scipy import interpolate

from scipy.stats import norm
from scipy import special
#from scipy.special import gamma, factorial
#import matplotlib.mlab as mlab
import statistics
#from scipy.optimize import curve_fit
#from scipy.stats import poisson
#from scipy.special import gammaln # x! = Gamma(x+1)
from time import time

from derivative import dxdt

#from statsmodels.base.model import GenericLikelihoodModel
from iminuit import Minuit

"Constants"
c = constants.c # speed of light in vacuum
e = constants.e #elementary charge
Na = constants.Avogadro #Avogadro's number
fs = constants.femto # 1.e-15
year = constants.year #one year in seconds
bar = constants.bar

me = constants.value('electron mass energy equivalent in MeV') #electron mass
alpha_em = constants.alpha #electromagnetic fine constant
mu_B = constants.value('Bohr magneton in eV/T') #Bohr magneton

h =  constants.value(u'Planck constant in eV/Hz')
hbar = h/(2*np.pi) # the Planck constant h divided by 2 pi in eV.s
hbar_c = hbar * c* 1E+9 # units MeV fm
hbar_c_ke = 6.58211899*1e-17 * c #KeV cm
Gf00 = constants.physical_constants["Fermi coupling constant"]
Gf_GeV = constants.value(u'Fermi coupling constant') # units: GeV^-2
"1GeV-2 = 0.389379mb" "10mb = 1fm^2"
Gf = Gf_GeV * 1e-12 * (hbar_c_ke)**3 #keV cm^3

"Germanium natural isotopes"
"70Ge, 72Ge, 73Ge, 74Ge, and 76Ge"

Ge70 = 0.2052
Ge72 = 0.2745
Ge73 = 0.0776
Ge74 = 0.3652
Ge76 = 0.0 #0.0775

M70 = 69.9242474 #in u
M72 = 71.9220758
M73 = 72.9234589
M74 = 73.9211778
M76 = 75.9214026

N70 = 38
N72 = 40
N73 = 41
N74 = 42
N76 = 44

N = (N70*Ge70 + N72*Ge72 + N73*Ge73 + N74*Ge74 + N76*Ge76) / (Ge70 + Ge72 + Ge73 + Ge74 + Ge76)

Z =  32

A = N+Z

M_u = (M70*Ge70 + M72*Ge72 + M73*Ge73 + M74*Ge74 + M76*Ge76) / (Ge70 + Ge72 + Ge73 + Ge74 + Ge76)
#print('Standard atomic weight: ', M_u)
M =  M_u * constants.u
M = M*c**2/e *1e-3

"RMS of Germanium"
Rn2 = (4.0495)**2
Rn4 = (4.3765)**4

"Energy range of the incoming neutrino in MeV (more or less)"
Enu_max = 8.01

"Recoil energy range of Ge in KeV"
#T_max = 2* (Enu_max*1e3)**2   / (M + 2*Enu_max*1e3 )

"Approximation values"
sin_theta_w_square = 0.23868 #zero momentum transfer data
Qw = N - (1 - 4*sin_theta_w_square)*Z

#%%

#nsteps = 100

"NORMALIZATION CONSTANT"
Mass_detector = 2.924 #kg

nucleus = Mass_detector/(M_u * constants.u)

nu_on_source = 4.8e13 #nu / cm2 / s

#efficiency = 0.80

normalization = nucleus* 3600*24  *  nu_on_source

"FLUX NORMALIZATION"

"Normalization per isotope"

"Average values of fission fractions during operation"

fracU235 = 0.58
fracPu239 = 0.29
fracU238 = 0.08
fracPu241 = 0.05
fracU238Ncapture = 0.6

def normU235(norm=1):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVU235.txt",unpack=True)
    #Enu = Enu1*norm
    #xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    xdata = np.linspace(0.,8.,num=1000000, endpoint=True)
    fint =  interpolate.interp1d(Enu,rho,fill_value="extrapolate")
    ydata = fint(xdata) #np.interp(xdata,Enu,rho)
    #iint = np.trapz(ydata,x=xdata)
    iint= integrate.simpson(ydata, xdata)
    return iint

def normU238(norm=1):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVU238.txt",unpack=True)
    #Enu = Enu1*norm
    #xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    xdata = np.linspace(0.,8.,num=1000000, endpoint=True)
    fint =  interpolate.interp1d(Enu,rho,fill_value="extrapolate")
    ydata = fint(xdata) #np.interp(xdata,Enu,rho)
    #iint = np.trapz(ydata,x=xdata)
    iint= integrate.simpson(ydata, xdata)
    return iint

def normPu239(norm=1):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVPu239.txt",unpack=True)
    #Enu = Enu1*norm
    xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    #xdata = np.linspace(0.,8.,num=1000000, endpoint=True)
    fint =  interpolate.interp1d(Enu,rho,fill_value="extrapolate")
    ydata = fint(xdata) #np.interp(xdata,Enu,rho)
    iint = np.trapz(ydata,x=xdata)
    iint= integrate.simpson(ydata, xdata)
    return iint

def normPu241(norm=1):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVPu241.txt",unpack=True)
    #Enu = Enu1*norm
    xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    #xdata = np.linspace(0.,8.,num=1000000, endpoint=True)
    fint =  interpolate.interp1d(Enu,rho,fill_value="extrapolate")
    ydata = fint(xdata) #np.interp(xdata,Enu,rho)
    #iint = np.trapz(ydata,x=xdata)
    iint= integrate.simpson(ydata, xdata)
    return iint

def normU238Ncap(norm=1):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVU238Ncapture.txt",unpack=True)
    #Enu = Enu1*norm
    xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    #xdata = np.linspace(0.,8.,num=1000000, endpoint=True)
    fint =  interpolate.interp1d(Enu,rho,fill_value="extrapolate")
    ydata = fint(xdata) #np.interp(xdata,Enu,rho)
    iint = np.trapz(ydata,x=xdata)
    #iint= integrate.simps(ydata, xdata)
    return iint

intnormU235 = normU235()
intnormU238 = normU238()
intnormPu239 = normPu239()
intnormPu241 = normPu241()
#intnormU238Ncap = normU238Ncap()
" "

"FUNCTIONS"

'MVHE - flux per isotope'

def fluxU235(E, norm=1.):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVU235.txt",unpack=True)
    #Enu = Enu1*norm

    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0
        #Enulog=np.log(Enu)
        #rholog=np.log(rho)
        #fint =  interpolate.interp1d(Enulog,rholog,fill_value="extrapolate")
        #xnew = [E]
        #xlog = np.log(xnew)
        #y = np.exp(fint(xlog))
    return y # ve MeV^-1 fission^-1

def fluxU238(E, norm=1.):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVU238.txt",unpack=True)
    #Enu = Enu1*norm
    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0
        #Enulog=np.log(Enu)
        #rholog=np.log(rho)
        #fint =  interpolate.interp1d(Enulog,rholog,fill_value="extrapolate")
        #xnew = [E]
        #xlog = np.log(xnew)
        #y = np.exp(fint(xlog))
    return y # ve MeV^-1 fission^-1

def fluxPu239(E, norm=1.):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVPu239.txt",unpack=True)
    #Enu = Enu1*norm
    Enulog=np.log(Enu)
    rholog=np.log(rho)

    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0
        #Enulog=np.log(Enu)
        #rholog=np.log(rho)
        #fint =  interpolate.interp1d(Enulog,rholog,fill_value="extrapolate")
        #xnew = [E]
        #xlog = np.log(xnew)
        #y = np.exp(fint(xlog))
    return y # ve MeV^-1 fission^-1

def fluxPu241(E, norm=1.):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVPu241.txt",unpack=True)
    #Enu = Enu1*norm

    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0
        #Enulog=np.log(Enu)
        #rholog=np.log(rho)
        #fint =  interpolate.interp1d(Enulog,rholog,fill_value="extrapolate")
        #xnew = [E]
        #xlog = np.log(xnew)
        #y = np.exp(fint(xlog))
    return y # ve MeV^-1 fission^-1

def fluxU238Ncap(E, norm=1.):
    Enu, rho = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/spectraMeVU238Ncapture.txt",unpack=True)
    #Enu = Enu1*norm

    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0
        #Enulog=np.log(Enu)
        #rholog=np.log(rho)
        #fint =  interpolate.interp1d(Enulog,rholog,fill_value="extrapolate")
        #xnew = [E]
        #xlog = np.log(xnew)
        #y = np.exp(fint(xlog))
    return y # ve MeV^-1 fission^-1

"No neutron capture contribution"
def flux_total(E):
    if E>=8.:
        flux = 0.
    else:
        flux = 1/intnormU235*fracU235*fluxU235(E) + 1/intnormU238*fracU238*fluxU238(E) + 1/intnormPu239*fracPu239*fluxPu239(E) + 1/intnormPu241*fracPu241*fluxPu241(E)
    return flux

"Form Factor"

'Klein-Nystrand'
def F(Q2,AtomicNumb):  # form factor of the nucleus
    q = np.sqrt(Q2)
    aa = 0.7 #fm
    r0 = 1.2 #fm
    rho = 3 / (4*np.pi *r0**3)
    R = AtomicNumb**(1/3)*r0
    #    print("A = ", AtomicNumb, "R = ", R)
    FF = 4*np.pi * rho / AtomicNumb / q**3 *hbar_c**3 * (math.sin(q*R/hbar_c) - q*R/hbar_c*math.cos(q*R/hbar_c)) * 1/ (1 + aa**2 *q**2/hbar_c**2)
    return (FF)

'Bohr approximation'
def F_B(Q2):  # from factor of the nucleus
    Fn = N * (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4) #approximation
    Fp = Z * (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4) #approximation
    F = Fn - (1-4*sin_theta_w_square)*Fp
    return (F /Qw)

'Helm'
def F_H(Q2,AtomicNumb):  # form factor of the nucleus
    q = np.sqrt(Q2)
    s = 0.9 #fm
    r0 = 1.2 #fm
    rho = 3 / (4*np.pi *r0**3)
    R = AtomicNumb**(1/3)*r0
    "j1(qR1) = sin(qR1)/(qR1)^2 - cos(qR1)/(qR1)"
    FF = (3 / (q*R/hbar_c)**3 * (math.sin(q*R/hbar_c) - q*R/hbar_c*math.cos(q*R/hbar_c)))  * np.exp(-(q*s/hbar_c)**2 /2)

    return (FF)

"Cross section"
def Z_eff(Te):
    if Te >= 11.11:
        Z=32
    elif 11.11 > Te and Te >= 1.4146:
        Z=30
    elif 1.4146 > Te and Te >= 1.248:
        Z=28
    elif 1.248 > Te and Te >= 1.217:
        Z=26
    elif 1.217 > Te and Te >= 0.1801:
        Z=22
    elif 0.1801 > Te and Te >= 0.1249:
        Z=20
    elif 0.1249 > Te and Te >= 0.1208:
        Z=18
    elif 0.1208 > Te and Te >= 0.0298:
        Z=14
    else:
        Z=4
    return Z


'SM'
def cross_section_SM_N(T,Enu):
    gvn = - 1/2
    gvp = 1/2 - 2*sin_theta_w_square
    Qw2_SM = 4 * (Z*(gvp) + N*(gvn))**2
    Q2= 2 *Enu**2 * M * T *1e-6 /(Enu**2 - Enu*T*1e-3) #MeV ^2
    #dsigmadT = Gf**2 *M /(4*np.pi) * Qw2_SM* (F(Q2,A))**2 / (hbar_c_ke)**4 * (1 -  M*T*1e-6 / (2*Enu**2) - T*1e-3/Enu + T**2/(2*Enu**2*1e6)) #cm^2/keV
    dsigmadT = Gf**2 *M /(4*np.pi) * Qw2_SM* (F(Q2,A))**2 / (hbar_c_ke)**4 * (1 -  M*T*1e-6 / (2*Enu**2)) #cm^2/keV

    return dsigmadT

def cross_section_SM_e(Te,Enu):
    ga = - 1/2
    gv = 1/2 + 2*sin_theta_w_square
    dsigmadT = Z_eff(Te) * Gf**2 *(me*1e3) /(2*np.pi) / (hbar_c_ke)**4 * ( (gv+ga)**2 + (gv-ga)**2*(1 - Te*1e-3/Enu) + (ga**2-gv**2)*me*1e3*Te*1e-6/(Enu**2) ) #cm^2/keV
    return dsigmadT

'Z'
def cross_section_vN(T,Enu, gv ,mv, qvq, qvnu):
    gv2 = gv**2
    mv2 = mv**2
    Q2= 2 *Enu**2 * M * T *1e-6 /(Enu**2 - Enu*T*1e-3) #MeV ^2
    gvn = - 1/2
    gvp = 1/2 - 2*sin_theta_w_square
    Qw = 2*(Z*(gvp) + N*(gvn))
    QvN = 3 * (N+Z) * qvq

    d1 = (gv2 * qvnu * Gf *Qw *QvN *M *(1 -  M*T*1e-6/(2*Enu**2))) / (np.sqrt(2)*np.pi*(2*M*T + mv2) * (hbar_c_ke))
    d2 = ((QvN*gv2)**2 * qvnu**2 * M *(1 -  M*T*1e-6/(2*Enu**2)) * (hbar_c_ke)**2) / (2*np.pi*(2*M*T + mv2)**2)
    dsigmadT = (d2-d1) * (F(Q2,A))**2
    return dsigmadT

def cross_section_ve(Te,Enu, gv ,mv, qve, qvnu):
    gv2 = gv**2
    mv2 = mv**2
    gve = - 1/2 + 2*sin_theta_w_square
    delta = ((gv2**2 *qve**2 *qvnu**2 *(me*1e3)/(2*np.pi*(2*(me*1e3)*Te + mv2)**2) * (hbar_c_ke)**2 )
             + ( (np.sqrt(2) * gv2 * gve * Gf * qve * qvnu * (me*1e3))/(np.pi *(2*(me*1e3)*Te + mv2)) / (hbar_c_ke)))
    dsigmadT = delta * Z_eff(Te)
    return dsigmadT

def cross_section_fact1_ve(Te,Enu, mv, qve, qvnu): #gv2**2 = gv**4
    mv2 = mv**2
    gve = - 1/2 + 2*sin_theta_w_square
    delta = (qve**2 *qvnu**2 *(me*1e3)/(2*np.pi*(2*(me*1e3)*Te + mv2)**2) * (hbar_c_ke)**2 )
    dsigmadT = delta * Z_eff(Te)
    return dsigmadT

def cross_section_fact2_ve(Te,Enu, mv, qve, qvnu): #gv2 = gv**2
    mv2 = mv**2
    gve = - 1/2 + 2*sin_theta_w_square
    delta = (np.sqrt(2) * gve * Gf * qve * qvnu * (me*1e3))/(np.pi *(2*(me*1e3)*Te + mv2)) / (hbar_c_ke)
    dsigmadT = delta * Z_eff(Te)
    return dsigmadT

" dN /dT "
'SM'
def differential_events_flux_SMN(T):
    nsteps = 100
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1e-3 #MeV
    EE = np.linspace(Emin,Enu_max,num=nsteps, endpoint=True)
    iint = []
    for i in range (0,nsteps):
        iint.append(cross_section_SM_N(T,EE[i]) * flux_total(EE[i]))
    return (integrate.simpson(iint,EE))

def differential_events_flux_SMe(T):
    nsteps = 1000
    Emin = 1/2* (T + np.sqrt(T**2 + 2*T*(me*1e3))) * 1e-3 #MeV
    EE = np.linspace(Emin,Enu_max,num=nsteps, endpoint=True)
    iint = []
    for i in range (0,nsteps):
        iint.append( cross_section_SMe(T,EE[i])  * flux_total(EE[i]))
    return (integrate.simpson(iint,EE))


'Vectorial'

def differential_events_flux_vN(T, gz=0 ,mz=0, qvq=1., qvnu=1.):
    nsteps = 100
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1e-3 #MeV
    EE = np.linspace(Emin,Enu_max,num=nsteps, endpoint=True)
    iint = []
    for i in range (0,nsteps):
        #iint.append(cross_section_vN(T,EE[i],gz,mz,qvq,qvnu) * flux_total(EE[i]))
        iint.append(cross_section_fact1_vN(T,EE[i],gz,mz,qvq,qvnu) * flux_total(EE[i]))
        #iint.append(cross_section_fact2_vN(T,EE[i],gz,mz,qvq,qvnu) * flux_total(EE[i]))
    return (integrate.simpson(iint,EE))

def differential_events_flux_ve(T,gz=0 ,mz=0,qve=1.,qvnu=1.):
    nsteps = 1000
    Emin = 1/2* (T + np.sqrt(T**2 + 2*T*(me*1e3))) * 1e-3 #MeV
    EE = np.linspace(Emin,Enu_max,num=nsteps, endpoint=True)
    iint = []
    for i in range (0,nsteps):
        #iint.append(cross_section_ve(T,EE[i],gz,mz,qve,qvnu) * flux_total(EE[i]))
        iint.append(cross_section_fact1_ve(T,EE[i],mz,qve,qvnu) * flux_total(EE[i]))
        #iint.append(cross_section_fact2_ve(T,EE[i],mz,qve,qvnu) * flux_total(EE[i]))
    return (integrate.simpson(iint,EE))

def differential_events_flux_ve2(T,gz=0 ,mz=0,qve=1.,qvnu=1.):
    nsteps = 1000
    Emin = 1/2* (T + np.sqrt(T**2 + 2*T*(me*1e3))) * 1e-3 #MeV
    EE = np.linspace(Emin,Enu_max,num=nsteps, endpoint=True)
    iint = []
    for i in range (0,nsteps):
        #iint.append(cross_section_ve(T,EE[i],gz,mz,qve,qvnu) * flux_total(EE[i]))
        iint.append(cross_section_fact2_ve(T,EE[i],mz,qve,qvnu) * flux_total(EE[i]))
    return (integrate.simpson(iint,EE))


"Binning"
"Bins of 10eV"
def binning(thres, tmax):
    #thres=0.0
    bins = []
    centre = []
    t = thres
    step = 0.01 # 10eV
    bins = np.arange(t, tmax, step)
    for i in range(0,len(bins)-1):
        w = (bins[i+1]+bins[i]) /2
        centre.append(round(w,6))

    return bins,centre

"Energy resolution - Germanium"
def E_resolution(E, ares =1.):
    F = 0.11
    eta = 2.96 *1e-3 #KeV
    sigma_n = 68.5e-3
    sigma = ares * np.sqrt( (sigma_n)**2 + (F*eta*E) )
    return sigma

"Quenching factor"

"Iron filter - Fef model"
def QF(E,ind, aa=1.):
    Enr,QF_Fef,QF_YBe = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/QF_Fef_YBe.txt",unpack=True)
    if ind==0:
        QF = QF_Fef
    elif ind==1:
        QF = QF_YBe
    else:
        print('no QF found')

    QFaa=QF #*aa

    if E<=Enr[len(Enr)-1] and E>=Enr[0]:
        xnew = [E]
        yy =  np.interp(xnew,Enr,QFaa) #* Nve
        y = yy[0]
    else:
        Enr[0]=1e-12
        Enrlog=np.log(Enr)
        QFlog=np.log(QFaa)
        fint =  interpolate.interp1d(Enrlog,QFlog,fill_value="extrapolate")

        xnew = [E]
        xlog = np.log(xnew)
        yy = np.exp(fint(xlog))
        y = yy[0]

    return y # fractional QF

def fncEee_QF(E, ind, aa=1.):
    Tnr,QF_Fef,QF_YBe = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/QF_Fef_YBe.txt",unpack=True)

    if ind==0:
        QF = QF_Fef
    elif ind==1:
        QF = QF_YBe
    else:
        print('no QF found')

    Eee=Tnr*QF #*aa

    xnew = [E]
    yy = np.interp(xnew,Eee,Tnr)
    y = yy[0]

    return y # Enr corresponding to that Eee wih interpolation


def D(xlist,ylist):
    yprime = np.diff(ylist)/np.diff(xlist) #derivative function
    return yprime

def dQdEI(E,ind, aa=1.):
    QF=[]
    Tnr,QF_Fef,QF_YBe = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/QF_Fef_YBe.txt",unpack=True)

    if ind==0:
        QF = QF_Fef
    elif ind==1:
        QF = QF_YBe
    else:
        print('no QF found')

    QFaa=QF #*aa
    E_ee=Tnr*QFaa

    Enr=[]
    E_I=[]
    for i in range(0,len(Tnr)-1):
        Enr.append(Tnr[i])
        E_I.append(E_ee[i])

    dQdE = D(E_ee,QFaa) #derivative

    if E<=Enr[len(Enr)-1] and E>=Enr[0]:
        xnew = [E]
        yy =  np.interp(xnew,E_I,dQdE) #* Nve
        y = yy[0]
    else:
        y = 0.0

    return y

#%%
"MAIN PART"
print(' ')

E_ion, counts_ON, counts_ON_err, counts_OFF, counts_OFF_err = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/data_release.txt",unpack=True)

#cSMe , eSMe = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/Counts_SMe.txt",unpack=True)
#cSMN , eSMN_Fef, eSMN_YBe = np.loadtxt("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/Counts_SMN_Fef_YBe.txt",unpack=True)

def fnc_events_MHVE_N(ind=0, qq=1.,qnu=1., gg=0.,mm=0., parsys=[1.]):
    Enr = []
    Ei = []
    dNdx =[]

    'ind defines the QF, 0-Fef, 1-YBe'

    Eee_thres = 0.003
    Edet_max = 1.51
    #EI_max=15*Edet_max

    T_thres = fncEee_QF(Eee_thres, ind)
    #T_max = 15*Edet_max/QF(EI_max,ind)
    T_max = 2*(8.*1e3)**2/(M+2*(8.*1e3)) #2*(Enu_max*1e3)**2/(M+2*(Enu_max*1e3))
    Enr = np.linspace(T_thres, T_max, num=100, endpoint=True, dtype=float)
    Ei=[]
    for tnr in Enr:
        Ei.append(tnr*QF(tnr,ind))
        dNdx.append(normalization *(differential_events_flux_vN(tnr, gg,mm,qq,qnu)))

    binss , centre = binning(0.0, Edet_max)

    tbin=[]
    centrebin=[]

    events_N=[]
    events_e=[]
    events=[]
    for j in range(0,len(binss)-1):
        c = centre[j]
        centrebin.append(c)
        t1 = binss[j]
        t2 = binss[j+1]

        dNdT_res= []
        for i in range(0,len(Ei)):
            sigma = E_resolution(Ei[i])
            AA = 2/(1 + math.erf((Ei[i])/(np.sqrt(2)*sigma)))
            intgauss_res = AA* 1/2 *(math.erf((t2 - Ei[i])/(np.sqrt(2)*sigma)) - math.erf((t1 - Ei[i])/(np.sqrt(2)*sigma)))

            dNdT_res.append(intgauss_res * dNdx[i])

        events.append(integrate.simpson(dNdT_res, Enr))
        dNdT_res.clear()

    'Above 0.2keVee'
    Eion_bin = [] # > 0.2keVee
    events_NOQw = []  # > 0.2keVee

    for i in range(0, len(centre)):
        if centre[i]>=0.2:
            Eion_bin.append(centrebin[i])
            events_NOQw.append(events[i])

    return Eion_bin,events_NOQw

def fnc_events_MHVE_e(qe=1., qnu=1., gg=0.,mm=0., parsys=[1.]):
    Enr = []
    Ei = []
    dNdx =[]

    Eee_thres = 0.003
    Edet_max = 1.51
    #EI_max=15*Edet_max

    T_max = 15*Edet_max #2*(Enu_max*1e3)**2/(me*1e3)
    Ei = np.linspace(Eee_thres, T_max, num=2000, endpoint=True, dtype=float)
    Enr = Ei
    for x in Ei:
        dNdx.append(normalization *(differential_events_flux_ve(x, gg,mm,qe,qnu)))

    binss , centre = binning(0.0, Edet_max)

    tbin=[]
    centrebin=[]

    events_N=[]
    events_e=[]
    events=[]
    for j in range(0,len(binss)-1):
        c = centre[j]
        centrebin.append(c)
        t1 = binss[j]
        t2 = binss[j+1]

        dNdT_res= []
        for i in range(0,len(Ei)):
            sigma = E_resolution(Ei[i])
            AA = 2/(1 + math.erf((Ei[i])/(np.sqrt(2)*sigma)))
            intgauss_res = AA* 1/2 *(math.erf((t2 - Ei[i])/(np.sqrt(2)*sigma)) - math.erf((t1 - Ei[i])/(np.sqrt(2)*sigma)))

            dNdT_res.append(intgauss_res * dNdx[i])

        events.append(integrate.simpson(dNdT_res, Enr))
        dNdT_res.clear()

    'Above 0.2keVee'
    Eion_bin = [] # > 0.2keVee
    events_NOQw = []  # > 0.2keVee

    for i in range(0, len(centre)):
        if centre[i]>=0.2:
            Eion_bin.append(centrebin[i])
            events_NOQw.append(events[i])

    return Eion_bin,events_NOQw


def fnc_events_MHVE_e2(qe=1., qnu=1., gg=0.,mm=0., parsys=[1.]):
    Enr = []
    Ei = []
    dNdx =[]

    Eee_thres = 0.003
    Edet_max = 1.51
    #EI_max=15*Edet_max

    T_max = 15*Edet_max #2*(Enu_max*1e3)**2/(me*1e3)
    Ei = np.linspace(Eee_thres, T_max, num=2000, endpoint=True, dtype=float)
    Enr = Ei
    for x in Ei:
        dNdx.append(normalization *(differential_events_flux_ve2(x, gg,mm,qe,qnu)))

    binss , centre = binning(0.0, Edet_max)

    tbin=[]
    centrebin=[]

    events_N=[]
    events_e=[]
    events=[]
    for j in range(0,len(binss)-1):
        c = centre[j]
        centrebin.append(c)
        t1 = binss[j]
        t2 = binss[j+1]

        dNdT_res= []
        for i in range(0,len(Ei)):
            sigma = E_resolution(Ei[i])
            AA = 2/(1 + math.erf((Ei[i])/(np.sqrt(2)*sigma)))
            intgauss_res = AA* 1/2 *(math.erf((t2 - Ei[i])/(np.sqrt(2)*sigma)) - math.erf((t1 - Ei[i])/(np.sqrt(2)*sigma)))

            dNdT_res.append(intgauss_res * dNdx[i])

        events.append(integrate.simpson(dNdT_res, Enr))
        dNdT_res.clear()

    'Above 0.2keVee'
    Eion_bin = [] # > 0.2keVee
    events_NOQw = []  # > 0.2keVee

    for i in range(0, len(centre)):
        if centre[i]>=0.2:
            Eion_bin.append(centrebin[i])
            events_NOQw.append(events[i])

    return Eion_bin,events_NOQw


'MODELS' #[B-L, Universal, Leptonic]
# qq = 1/3, qe = qnu = -1 : B-L (N+e, ONLY FOR VECTORIAL!!)
# qq = qe = qnu = 1 : UNIVERSAL (N+e)
# qq = 0, qe = qnu = 1 : LEPTONIC (e)

ind = 0 #Fef (leptonic case has no QF)
#ind = 1 #YBe (leptonic case has no QF)
qq = 1. #[1, 0]
qe = 1. #[1,1]
qnu = 1. #[1,1]

mm = [10000.000000000002, 12589.254117941673,15848.931924611137, 19952.6231496888, 25118.864315095807]
#mm = [31622.7766016838, 39810.717055349737, 50118.72336272724, 63095.73444801934, 79432.82347242816]

m_list=[]
c_list=[]
ese_list=[]
ese_list2=[]
esum=[]
for i in range(0,len(mm)):
    #print(mm[i])
    c_m, ese_m = fnc_events_MHVE_e(qe,qnu, 1.,mm[i])
    c_m2, ese_m2 = fnc_events_MHVE_e2(qe,qnu, 1.,mm[i])
    #c_m, ese_m = fnc_events_MHVE_N(ind,qq,qnu, 1.,mm[i])
    for j in range(0, len(ese_m)):
        m_list.append(mm[i])
        c_list.append(c_m[j])
        ese_list.append(ese_m[j])
        ese_list2.append(ese_m2[j])
        print(mm[i],c_m[j],ese_m[j])
        print(mm[i],c_m2[j],ese_m2[j])

np.savetxt('/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/ZeN_chisq/Events_ve1_m_45_1.txt', np.c_[m_list,c_list,ese_list])
np.savetxt('/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/ZeN_chisq/Events_ve2_m_45_1.txt', np.c_[m_list,c_list,ese_list2])
