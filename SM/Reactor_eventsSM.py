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

from scipy.stats import norm
from scipy import special
from scipy.special import gamma, factorial
import matplotlib.mlab as mlab
import statistics
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.special import gammaln # x! = Gamma(x+1)
from time import time

from derivative import dxdt

from statsmodels.base.model import GenericLikelihoodModel
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
Enu_max = 10.0

"Recoil energy range of Ge in KeV"
#T_max = 2* (Enu_max*1e3)**2   / (M + 2*Enu_max*1e3 )

"Approximation values"
sin_theta_w_square = 0.23868 #zero momentum transfer data
Qw = N - (1 - 4*sin_theta_w_square)*Z

#%%

nsteps = 100

"NORMALIZATION CONSTANT"
Mass_detector = 2.924 #kg

nucleus = Mass_detector/(M_u * constants.u)

nu_on_source = 4.8e13 #nu / cm2 / s

#efficiency = 0.80

normalization = Mass_detector/(M_u * constants.u) * 3600*24  *  nu_on_source

"FLUX NORMALIZATION"

"Normalization per isotope"

"Average values of fission fractions during operation"

fracU235 = 0.58
fracPu239 = 0.29
fracU238 = 0.08
fracPu241 = 0.05
fracU238Ncapture = 0.6

def normU235(norm=1):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVU235.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVU235.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0])*norm)
        rho.append(float(x.split(' ')[1]))
    file.close()

    xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    ydata = np.interp(xdata,Enu,rho)

    int= integrate.simps(ydata, xdata)

    return int

def normU238(norm=1):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVU238.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVU238.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0])*norm)
        rho.append(float(x.split(' ')[1]))
    file.close()

    xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    ydata = np.interp(xdata,Enu,rho)

    int= integrate.simps(ydata, xdata)

    return int

def normPu239(norm=1):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVPu239.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVPu239.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0])*norm)
        rho.append(float(x.split(' ')[1]))
    file.close()

    xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    ydata = np.interp(xdata,Enu,rho)

    int= integrate.simps(ydata, xdata)

    return int

def normPu241(norm=1):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVPu241.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVPu241.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0])*norm)
        rho.append(float(x.split(' ')[1]))
    file.close()

    xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    ydata = np.interp(xdata,Enu,rho)

    int= integrate.simps(ydata, xdata)

    return int

def normU238Ncap(norm=1):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVU238Ncapture.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVU238Ncapture.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0])*norm)
        rho.append(float(x.split(' ')[1]))
    file.close()

    xdata = np.linspace(Enu[0],Enu[len(Enu)-1],num=1000000, endpoint=True)
    ydata = np.interp(xdata,Enu,rho)

    int= integrate.simps(ydata, xdata)

    return int

intnormU235 = normU235()
intnormU238 = normU238()
intnormPu239 = normPu239()
intnormPu241 = normPu241()
intnormU238Ncap = normU238Ncap()

" "

"FUNCTIONS"

'MVHE - flux per isotope'

def fluxU235(E, norm=1.):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVU235.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVU235.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0]))
        rho.append(float(x.split(' ')[1])*norm)
    file.close()

    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0.0

    return y # ve MeV^-1 fission^-1

def fluxU238(E, norm=1.):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVU238.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVU238.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0]))
        rho.append(float(x.split(' ')[1])*norm)
    file.close()

    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0.0

    return y # ve MeV^-1 fission^-1

def fluxPu239(E, norm=1.):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVPu239.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVPu239.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0]))
        rho.append(float(x.split(' ')[1])*norm)
    file.close()

    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0.0

    return y # ve MeV^-1 fission^-1

def fluxPu241(E, norm=1.):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVPu241.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVPu241.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0]))
        rho.append(float(x.split(' ')[1])*norm)
    file.close()

    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0.0

    return y # ve MeV^-1 fission^-1

def fluxU238Ncap(E, norm=1.):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/spectraMeVU238Ncapture.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/spectraMeVU238Ncapture.txt", "r")
    lines=file.readlines()
    Enu=[]
    rho=[]
    for x in lines:
        Enu.append(float(x.split(' ')[0]))
        rho.append(float(x.split(' ')[1])*norm)
    file.close()

    if E<=Enu[len(Enu)-1] and E>=Enu[0]:
        xnew = [E]
        yy = np.interp(xnew,Enu,rho)
        y = yy[0]
    else:
        y = 0.0

    return y # ve MeV^-1 fission^-1


"No neutron capture contribution"
def flux_total_NONcap(E):
    flux = 1/intnormU235*fracU235*fluxU235(E) + 1/intnormU238*fracU238*fluxU238(E) + 1/intnormPu239*fracPu239*fluxPu239(E) + 1/intnormPu241*fracPu241*fluxPu241(E)
    return flux

"U238 U239 neutron capture"
def flux_total(E):
    flux = 1/intnormU235*fracU235*fluxU235(E) + 1/intnormU238*fracU238*fluxU238(E) + 1/intnormPu239*fracPu239*fluxPu239(E) + 1/intnormPu241*fracPu241*fluxPu241(E) +  1/intnormU238Ncap*fracU238Ncapture*fluxU238Ncap(E)
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

def cross_section_SM(T,Enu):
    gvn = - 1/2
    gvp = 1/2 - 2*sin_theta_w_square
    Qw2_SM = 4 * (Z*(gvp) + N*(gvn))**2

    Q2= 2 *Enu**2 * M * T *1e-6 /(Enu**2 - Enu*T*1e-3) #MeV ^2
    dsigmadT = Gf**2 *M /(4*np.pi) * Qw2_SM* (F(Q2,A))**2 / (hbar_c_ke)**4 * (1 -  M*T*1E-6 / (2*Enu**2) - T*1e-3/Enu + T**2/(2*Enu**2*1e6)) #cm^2/keV

    return dsigmadT

def cross_section_mu(T,Enu, mu_muB2=0.): #mu_muB = mu / muB
    Fem = 1. #Fem(Q2) electromagnetic form factor approximated to 1
    new = np.pi*alpha_em**2/(me*1e3)**2 *(hbar_c_ke)**2 * (1/T - 1/(Enu*1e3) + T/(4*Enu**2)*1e-6) * Z**2 * Fem**2 * mu_muB2

    dsigmadT = cross_section_SM(T,Enu)  + new
    return dsigmadT

def cross_section_Z(T,Enu, gz2=0., mz2=0.):
    gvn = - 1/2
    gvp = 1/2 - 2*sin_theta_w_square
    Qw2_SM = 4 * (Z*(gvp) + N*(gvn))**2
    Qw = np.sqrt(Qw2_SM)
    #Qz= 3* A * gz

    Gz = 1 - ((3 * A * gz2) * (hbar_c_ke)**3 / (Qw * np.pi(2) * Gf * (2*M*T + mz2) ) )

    dsigmadT = cross_section_SM(T,Enu) * (Gz)**2
    return dsigmadT

def cross_section_scalar(T,Enu, Qphi2=0.,mphi2=0.):
    Qphi2 = (gphi**2 * (14*N + 15.1*Z))**2

    new = Qphi2 * M**2 * T * (hbar_c_ke)**2 / (4*np.pi * (Enu*1e3)**2 *(2*M*T + mphi2)**2)

    dsigmadT = cross_section_SM(T,Enu) + new
    return dsigmadT

" dN /dT "

def differential_events_flux_MHVE(T):
    nsteps = 50
    iint=  np.zeros((nsteps+1),float)
    EE = np.zeros((nsteps+1),float)
    "Integral Bounds"
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1e-3 #MeV
    for i in range (0,nsteps+1):
        EE[i] = Emin + (Enu_max - Emin)/nsteps * i
        iint[i] = cross_section_SM(T,EE[i]) * flux_total(EE[i])

    return (np.trapz(iint, x=EE))

def differential_events_flux_mu(T, mu2=0.):
    nsteps = 50
    iint=  np.zeros((nsteps+1),float)
    EE = np.zeros((nsteps+1),float)
    "Integral Bounds"
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1e-3 #MeV
    for i in range (0,nsteps+1):
        EE[i] = Emin + (Enu_max - Emin)/nsteps * i
        iint[i] = cross_section_mu(T,EE[i],mu2) * flux_total(EE[i])

    return (np.trapz(iint, x=EE))

def differential_events_flux_Z(T, gz2=0., mz2=0.):
    nsteps = 50
    iint=  np.zeros((nsteps+1),float)
    EE = np.zeros((nsteps+1),float)
    "Integral Bounds"
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1e-3 #MeV
    for i in range (0,nsteps+1):
        EE[i] = Emin + (Enu_max - Emin)/nsteps * i
        iint[i] = cross_section_Z(T,EE[i],gz2,mz2) * flux_total(EE[i])

    return (np.trapz(iint, x=EE))

def differential_events_flux_scalar(T,Qphi2=0.,mphi2=0.):
    nsteps = 50
    iint=  np.zeros((nsteps+1),float)
    EE = np.zeros((nsteps+1),float)
    "Integral Bounds"
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1e-3 #MeV
    for i in range (0,nsteps+1):
        EE[i] = Emin + (Enu_max - Emin)/nsteps * i
        iint[i] = cross_section_Z(T,EE[i],Qphi2,mphi2) * flux_total(EE[i])

    return (np.trapz(iint, x=EE))

def differential_events_identifier(T, mu2=0., gz2=0., mz2=0., Qphi2=0.,mphi2=0.):
    if mu2==0. and gz2== 0. and mz2==0. and Qphi2==0. and mphi2==0.:
        return differential_events_flux_MHVE(T)
    if mu2!=0:
        return differential_events_flux_mu(T, mu2)
    if gz2!= 0. or  mz2!=0.:
        return differential_events_flux_Z(T, gz2, mz2)
    if Qphi2!=0. or mphi2!=0.:
        return differential_events_flux_scalar(T, Qphi2, mphi2)



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
        w = (bins[i+1]+bins[i] ) /2
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

def D(xlist,ylist):
    yprime = np.diff(ylist)/np.diff(xlist) #derivative function
    return yprime

"Iron filter - Fef model"
def QF(E, aa=1.):
    file = open("QF_Fef_YBe.txt", "r")
    lines=file.readlines()
    Enr=[]
    QF=[]
    QF_Fef=[]
    QF_YBe=[]
    for x in lines:
        Enr.append(float(x.split(' ')[0]))
        QF.append(float(x.split(' ')[1]))  #Fef
        #QF.append(float(x.split(' ')[2])) #YBe

    QFaa=[]
    for i in range(0,len(QF)):
        QFaa.append(aa * QF[i])

    if E<=Enr[len(Enr)-1] and E>=Enr[0]:
        xnew = [E]
        yy =  np.interp(xnew,Enr,QFaa) #* Nve
        y = yy[0]
    else:
        y = 1.0

    return y # fractional QF

def dQdEI(E, aa=1.):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/QF_Fef_YBe.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/QF_Fef_YBe.txt", "r")
    lines=file.readlines()
    Tnr=[]
    QF=[]
    QF_Fef=[]
    QF_YBe=[]
    for x in lines:
        Tnr.append(float(x.split(' ')[0]))
        QF.append(float(x.split(' ')[1]))  #Fef
        #QF.append(float(x.split(' ')[2])) #YBe

    QFaa=[]
    E_ee=[]
    for i in range(0,len(QF)):
        QFaa.append(aa * QF[i])
        E_ee.append(Tnr[i]*QFaa[i])

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

def fncEee_QF(E, aa=1.):
    file = open("/scratch/franmon/Reactor_CEnuNS/SM/QF_Fef_YBe.txt", "r")
    #file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/QF_Fef_YBe.txt", "r")
    lines=file.readlines()
    Tnr=[]
    QF=[]
    QF_Fef=[]
    QF_YBe=[]
    for x in lines:
        Tnr.append(float(x.split(' ')[0]))
        QF.append(float(x.split(' ')[1]))  #Fef
        #QF.append(float(x.split(' ')[2])) #YBe
    file.close()

    Eee=[]
    for i in range(0,len(Tnr)):
        Eee.append(Tnr[i]*QF[i]*aa)

    xnew = [E]
    yy = np.interp(xnew,Eee,Tnr)
    y = yy[0]

    return y # Enr corresponding to that Eee wih interpolation



#%%
"MAIN PART"
print(' ')

file = open("/scratch/franmon/Reactor_CEnuNS/SM/data_release.txt", "r")
#file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/data_release.txt", "r")
lines=file.readlines()
E_ion=[] # bin center energy
counts_ON=[] # counts/10 eV·3 kg·day for 96.4 days of Rx- ON operation.
counts_ON_err=[] # error bar (combination of statistical and signal acceptance uncertainties -see Fig. 1 data release-).
counts_OFF=[] # counts/10 eV·3 kg·day for 25 days of Rx-OFF operation.
counts_OFF_err=[] # error bars (same way).
for x in lines:
    E_ion.append(float(x.split(' ')[0]))
    counts_ON.append(float(x.split(' ')[1]))
    counts_ON_err.append(float(x.split(' ')[2]))
    counts_OFF.append(float(x.split(' ')[3]))
    counts_OFF_err.append(float(x.split(' ')[4]))
file.close()


"dNdT"

def fnc_dNdx_MHVE_Fef():
    Eee_QF = []
    dNdxQF = []

    events_interval_obs= []

    dNdT_res= []

    dNdT_res_obs=[]
    Eee_obs= []

    nsteps = 1000

    Eee_thres = 1e-9
    Eee_max = 0.8

    for i in range(0,nsteps+1):
        x = Eee_thres + (Eee_max - Eee_thres)/nsteps * i
        t = fncEee_QF(x)
        Eee_QF.append(x)
        dNdxQF.append(normalization *(differential_events_flux_MHVE(t,N, M)) * 1 / QF(t) * (1- x/QF(t) * dQdEI(x)))

    'proved QF and NOQF same result'

    "Sample T -> Tobs , in the way we want"
    nsteps_obs=100

    binss , centre = binning(0.0, Eee_max) #Eee_thres

    x1 = - float('inf')
    x2 = float('inf')

    tbin=[]
    centrebin=[]
    for j in range(0,len(binss)-1):
        for i in range(0,nsteps_obs):
            t2 = binss[j+1]
            t1 = binss[j]
            tbin.append(t1 + (t2-t1)/nsteps_obs *i)

        c = centre[j] #(tbin[0] + tbin[len(tbin)-1] )/2
        centrebin.append(c)
        for tobs in tbin: #T observed
            for i in range(0,len(Eee_QF)):
                sigma = E_resolution(Eee_QF[i], 1.) #sigma0e * np.sqrt(tobs/Eee_thres)

                var2 = ((Eee_QF[i] - x2) / (np.sqrt(2) * sigma))
                A2 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var2)
                var1 = ((Eee_QF[i] - x1) / (np.sqrt(2) * sigma))
                A1 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var1)
                A = 1/ (A2 - A1)

                gauss_res = A * np.exp(-(Eee_QF[i]-tobs)**2 / (2*sigma**2))
                dNdT_res.append(gauss_res * dNdxQF[i])

            dNdT_res_obs.append(np.trapz(dNdT_res, x=Eee_QF))
            Eee_obs.append(tobs)
            dNdT_res.clear()

        #events_interval_obs.append(np.trapz(dNdT_res_obs, x=tbin))
        #dNdT_res_obs.clear()
        tbin.clear()

    return Eee_obs,dNdT_res_obs


E_MHVE_Fef, dNdx_MHVE_Fef = fnc_dNdx_MHVE_Fef()

"SM prediction"

gvn = - 1/2
gvp = 1/2 - 2*sin_theta_w_square

Qw2 = 4 * (Z*(gvp) + N*(gvn))**2



"Total number of events"

def fnc_events_MHVE_Fef(parsys=[1.,1.]):
    Eee_QF = []
    dNdxQF = []

    events_interval_obs= []

    dNdT_res= []

    dNdT_res_obs=[]
    Eee_obs= []

    nsteps = 100

    Eee_thres = 1e-9
    Eee_max = 1.51

    for i in range(0,nsteps+1):
        x = Eee_thres + (Eee_max - Eee_thres)/nsteps * i
        t = fncEee_QF(x, parsys[1])
        Eee_QF.append(x)
        dNdxQF.append( normalization *(differential_events_flux_MHVE(t)) * 1 / QF(t, parsys[1]) * (1- x/QF(t, parsys[1]) * dQdEI(x, parsys[1])))

    'proved QF and NOQF same result'

    "Sample T -> Tobs"

    binss , centre = binning(0.0, Eee_max) #Eee_thres

    x1 = - float('inf')
    x2 = float('inf')

    tbin=[]
    centrebin=[]
    for j in range(0,len(binss)-1):
        c = centre[j]
        centrebin.append(c)
        t1 = binss[j]
        t2 = binss[j+1]

        for i in range(0,len(Eee_QF)):
            sigma = E_resolution(Eee_QF[i], parsys[0])

            var2 = ((Eee_QF[i] - x2) / (np.sqrt(2) * sigma))
            A2 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var2)
            var1 = ((Eee_QF[i] - x1) / (np.sqrt(2) * sigma))
            A1 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var1)
            A = 1/ (A2 - A1)
            #gauss_res = A * np.exp(-(Eee_QF[i]-tobs)**2 / (2*sigma**2))
            intgauss_res = A * np.sqrt(np.pi/2)* sigma *(math.erf((t2 - Eee_QF[i])/(np.sqrt(2)*sigma)) - math.erf((t1 - Eee_QF[i])/(np.sqrt(2)*sigma)))

            dNdT_res.append(intgauss_res * dNdxQF[i])

        events_interval_obs.append(np.trapz(dNdT_res, x=Eee_QF))
        dNdT_res.clear()

    'Above 0.2keVee'
    Eion_bin = [] # > 0.2keVee
    events_NOQw = []  # > 0.2keVee

    for i in range(0, len(centre)):
        if centre[i]>=0.2:
            Eion_bin.append(centrebin[i])
            events_NOQw.append(events_interval_obs[i])

    return Eion_bin,events_NOQw




centre, events_MHVE_Fef = fnc_events_MHVE_Fef()

#centreYBe, events_MHVE_YBe = fnc_events_MHVE_YBe()

#centre, events_MHVE_Lindhard = fnc_events_MHVE_Lindhard()

n_signal=[]
for i in range(0,len(events_MHVE_Fef)):
    n_signal.append(events_MHVE_Fef[i])

sum_nobs = sum(n_signal)

print('Total number of expected events SM above 0.2keVee: ', sum(n_signal))

#n_signalYBe=[]

#for i in range(0,len(events_MHVE_YBe)):
#    n_signalYBe.append(events_MHVE_YBe[i]*Qw2_SM)




txt_file = open("/scratch/franmon/Reactor_CEnuNS/SM/Counts_MVHE_Fef_SM.txt", "w")
#txt_file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/Counts_MVHE_Fef_SM.txt", "w")
for aa in range(0,len(centre)):
    a_obs = centre[aa]
    e_obs = events_MHVE_Fef[aa]
    content = str(a_obs)+' '+ str(e_obs)
    txt_file.write("".join(content) + "\n")

txt_file.close()


"Using MINUIT"

def gaussian(x, height, center, width, offset):
    return (height*np.exp(-(x - center)**2/(2*width**2)) + offset)

def exponential(x,A,B,C):
    return (A + B * np.exp(- C * x))

def fnc_fitON(xx, par_L1, aM_prior, par_exp, wM, events):
    'Bckg related parameters'
    hL1, cL1, wL1 = par_L1
    An, Bn, Cn = par_exp

    hL2 = 0.008 * hL1
    cL2 = 1.142 #keV
    wL2 = wL1

    AMAL1 = aM_prior * 0.16
    cM = 0.158 #keV
    hM = AMAL1 * hL1 * abs(wL1) / abs(wM)

    fit=[]

    for i in range(0,len(xx)):
        x = xx[i]
        fit.append(events[i] + (gaussian(x, hM, cM, wM, offset=0) + exponential(x,An,Bn,Cn) + gaussian(x, hL2, cL2, wL2, offset=0) +
            gaussian(x, hL1, cL1, wL1, offset=0) ))

    return fit


def fcn_np(par):
    par_L1 = [par[0],par[1],par[2]]
    par_exp = [par[3],par[4],par[5]]
    aM_prior = par[6]

    'NSI'
    Qw2_ud = par[7]

    'Systematics in signal'
    norm_sys = 1. #par[8]
    par_sys = [1.,1.] #[par[9],par[10]]

    'Intrinsic resolution changes'
    wM = sigma_n * par_sys[1]

    #centre, events_NOQw = fnc_events_MHVE_Fef(par_sys)

    events = []
    for i in range(0,len(events_MHVE_Fef)):
        events.append(events_MHVE_Fef[i]*norm_sys)  #events_NOQw

    mu_est = []
    mu_est =fnc_fitON(E_ion,par_L1, aM_prior, par_exp, wM, events)

    sum_tot=0
    for i in range(0,len(n_obs)): #sum over bins
        sum_tot = sum_tot + ((n_obs[i]- mu_est[i])**2 / (counts_ON_err[i])**2 )

    chi2 = sum_tot + ((aM_prior - 1.)**2 / sigma_aM**2) #+ ((norm_sys - alpha_BF[0])**2 / sigma_a[0]**2) + ((par_sys[0] - alpha_BF[1])**2 / sigma_a[1]**2) + ((par_sys[1] - alpha_BF[2])**2 / sigma_a[2]**2)
    return chi2


'Alternative minimizations. Not valid for this case. In this exercise LS method must be used in order to account for the statistic and systematic errors of the measurements'

def fcn_np_others(par):
    par_L1 = [par[0],par[1],par[2]]
    par_exp = [par[3],par[4],par[5]]
    aM_prior = par[6]

    'NSI'
    Qw2_ud = par[7]

    'Systematics in signal'
    norm_sys = 1. #par[8]
    par_sys = [1.,1.] #[par[9],par[10]]

    'Intrinsic resolution changes'
    wM = sigma_n * par_sys[1]

    #centre, events_NOQw = fnc_events_MHVE_Fef(par_sys)

    events = []
    for i in range(0,len(events_MHVE_Fef)):
        events.append(events_MHVE_Fef[i]*Qw2_ud*norm_sys)  #events_NOQw

    mu_est = []
    mu_est =fnc_fitON(E_ion,par_L1, aM_prior, par_exp, wM, events)

    sum_tot=0
    for i in range(0,len(n_obs)): #sum over bins
        sum_tot = sum_tot + (mu_est[i] - n_obs[i] + n_obs[i]*np.log(n_obs[i] / mu_est[i] )) # this is lnL / lnL_max (EML). Poisson errors
        sum_tot = sum_tot + ((n_obs[i]- mu_est[i])**2 / (mu_est[i]) ) # Test Newman
        sum_tot = sum_tot + ((n_obs[i]- mu_est[i])**2 / (n_obs[i]) ) # Test Pearson

    chi2 = sum_tot + ((aM_prior - 1.)**2 / sigma_aM**2) #+ ((norm_sys - alpha_BF[0])**2 / sigma_a[0]**2) + ((par_sys[0] - alpha_BF[1])**2 / sigma_a[1]**2) + ((par_sys[1] - alpha_BF[2])**2 / sigma_a[2]**2)
    return chi2


#Qw2_ud = 4 * (Z*(gvp + 2*ee_u +ee_d) + N*(gvn + ee_u + 2*ee_d))**2

#R = (Z*gvp + N*gvn) / (2*Z+N)

#f = 1 + (N-Z)/(2*Z+N)

#Qw2_ud = 4 * (2*Z+N)**2 *( R + ee_u + f * ee_d))**2


'Systematics'

"M-Shell"
aM_prior_BF = 1.
sigma_aM = 0.03 #3%

'Flux, Resolution, QF'

norm_BF = 1.
sigma_norm = 0.05 #5%

res_BF = 1. #factor multipled to energy resolution
'note that this changes the intrinsic resolution too by the same factor'
sigma_res = 0.05 #5%

QF_BF = 1. #factor multipled to QF at each energy
sigma_QF = 0.05 #5%

alpha_BF = [norm_BF,res_BF, QF_BF]

sigma_a = [sigma_norm,sigma_res,sigma_QF]

start_1 = 1.

sigma_n = 68.5e-3 #keV intrinsic noise

'Observed data for reactor ON'
n_obs = counts_ON

fcn_np.errordef = 1. #Minuit.LIKELIHOOD

#print(describe(fcn_np_nosyst_NSI))

m = Minuit(fcn_np, (100, 1.297, 0.1, 20., 150, 4, 1.,start_1) ,
           name=('hL1', 'cL1', 'wL1', 'An', 'Bn', 'Cn', 'aM_prior','a_NSI')) #,'a_norm','a_res','a_QF')) #start_1,start_1,start_1)

m.limits['hL1'] = (75.0,150)
m.limits['cL1'] = (0.5,1.5)
m.limits['wL1'] = (0.0,4)
m.limits['An'] = (0.0,None)
m.limits['Bn'] = (0.0,None)
m.limits['Cn'] = (0.0,None)
m.limits['aM_prior'] = (0.0,1.5)

#m.limits['a_NSI'] = (0.0,None)

m.fixed['a_NSI'] = True

#m.limits['a_norm'] = (0.0,5)
#m.limits['a_res'] = (0.0,None)
#m.limits['a_QF'] = (0.1,None)

#m.fixed['a_norm'] = True
#m.fixed['a_res'] = True
#m.fixed['a_QF'] = True

print('MIGRAD Run')
#print(m.migrad() )  # run optimiser

resum = m.simplex().migrad() #m.migrad()

#print(resum)

print(' ')
print('MINOS Run')
# Run MINOS to compute asymmetric confidence intervals.
resume_minos = m.minos(cl = 0.9) #cl = 0.9
#print(resume_minos)
print('Valid: ', m.valid)
#print(repr(m.params)) # parameter info
#print(m.errors) #symmetric uncertainties


'Optimal parameters'
hL1_ML = m.values[0]
cL1_ML = m.values[1]
wL1_ML = m.values[2]
An_ML = m.values[3]
Bn_ML = m.values[4]
Cn_ML = m.values[5]

aM_prior_ML = m.values[6]

parL1_ML = [hL1_ML, cL1_ML, wL1_ML]
parexp_ML = [An_ML, Bn_ML, Cn_ML]

aNSI_ML = m.values[7]

'systematics'
#a_norm_ML = m.values[8]
#a_res_ML = m.values[9]
#a_QF_ML = m.values[10]

#par_sys_ML = [a_res_ML,a_QF_ML]

'Minos errors'
hL1_err = m.errors[0]
cL1_err = m.errors[1]
wL1_err = m.errors[2]
An_err = m.errors[3]
Bn_err = m.errors[4]
Cn_err = m.errors[5]
aM_prior_err = m.errors[6]

#aNSI_err = m.merrors[7]

#print(aNSI_err)

#a_norm_err = m.errors[8]
#a_res_err = m.errors[9]
#a_QF_err = m.errors[10]

print(m.params)
print(m.errors)

print(m.values)

chisq_ML = fcn_np(m.values)
chisqndf = fcn_np(m.values) / (len(centre)-7)
print('chi2/ndf  min: ' , chisqndf)


'Minuit'
gvn = - 1/2
gvp = 1/2 - 2*sin_theta_w_square

R = (Z*gvp + N*gvn) / (2*Z+N)

f = 1 + (N-Z)/(2*Z+N)

Qw2_ud = 1. #(R + eps_ML)**2

centre, events_NOQw = fnc_events_MHVE_Fef() #par_sys_ML

eventsNSI = []

for i in range(0,len(events_NOQw)):
        eventsNSI.append(events_NOQw[i]) #*a_norm_ML

wM_ML = sigma_n #* a_res_ML

fitcurve_ON = fnc_fitON(E_ion, parL1_ML, aM_prior_ML, parexp_ML, wM_ML,  eventsNSI)

txt_file = open("/scratch/franmon/Reactor_CEnuNS/SM/MHVE_Fef_SM_mparams90cl.txt", "w")
#txt_file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/MHVE_Fef_SM_mparams90cl.txt", "w")

content = str(m.params)
txt_file.write("".join(content) + "\n")

txt_file.close()

chisqndf = fcn_np(m.values) / (len(centre)-7)

txt_file = open("/scratch/franmon/Reactor_CEnuNS/SM/MHVE_Fef_SM_chisq90cl.txt", "w")
#txt_file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/MHVE_Fef_SM_chisq90cl.txt", "w")
for i in range(0,len(m.values)):
    content = str(m.values[i])
    txt_file.write("".join(content) + "\n")
txt_file.write('chi2ndf'+' '+str(chisqndf))
txt_file.write('chi2'+' '+str(fcn_np(m.values)))

txt_file.close()

txt_file = open("/scratch/franmon/Reactor_CEnuNS/SM/MHVE_Fef_SM_fit90cl.txt", "w")
#txt_file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/MHVE_Fef_SM_fit90cl.txt", "w")
for aa in range(0,len(E_ion)):
    a_obs = E_ion[aa]
    e_obs = fitcurve_ON[aa]
    content = str(a_obs)+' '+ str(e_obs)
    txt_file.write("".join(content) + "\n")
txt_file.close()


txt_file = open("/scratch/franmon/Reactor_CEnuNS/SM/MHVE_Fef_SM_events90cl.txt", "w")
#txt_file = open("/scratch/llarizgoitia/Reactor/Reactor_CEnuNS/SM/MHVE_Fef_SM_events90cl.txt", "w")
for aa in range(0,len(E_ion)):
    a_obs = E_ion[aa]
    e_obs = eventsNSI[aa]
    content = str(a_obs)+' '+ str(e_obs)
    txt_file.write("".join(content) + "\n")
txt_file.close()
