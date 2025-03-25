# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 08:42:09 2024

@author: amart
"""




import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from scipy.stats import ttest_rel,ttest_ind, shapiro, f_oneway, tukey_hsd,chisquare
from iapws import IAPWS97 
from scipy.optimize import curve_fit

from sklearn.metrics import r2_score 
from scipy.stats import linregress

os.chdir(r"XXXXXXXX") #Input Working Directory

sns.set_theme()
sns.set_style("ticks")
sns.set_context("paper",font_scale=1.5)
sns.set_palette("bright")



def diffusion(tau,g0,td):
    
    w =  0.23961
    
    z0 = 3.964
    
    y = g0/((1+(tau/td))*(1+(w**2/z0**2)*(tau/td))**(1/2))
    
    return y


def anomalous_D(tau,g0,td,a,):
    
    w =  0.23961
    
    z0 = 3.964
    
    y = g0/((1+(tau/td)**a)*(1+(w**2/z0**2)*(tau/td)**a)**(1/2))
    
    return y


def Diffusion_fit(x,a,b):
    
    y = a*x+b
    
    return y

w = 0.23961

kb = 1.38e-23


T = 273.15+26

D_values = pd.DataFrame()
k=0


Viscosities = {"1":0.00087273,"2":0.0022965,"3":0.0059057,"4":0.010703,"5":0.030085}


for file in os.listdir():
    
    if file.endswith(".csv")==True:
        
        

        Dil = file.split(" ")[1]
        
        Rep = int(file.strip(".csv").split(" ")[-1])
        
        data = pd.read_csv(file,skiprows=1,sep="\t",skipinitialspace=True)
        for col in data.columns:
            if "Correlation Channel" in col:
                
                n = col.split(" ")[-1]
                
                G_exp = data.loc[:,col]
                
                t = data.loc[:,f"Time [ms]{n[1:]}"]
                
                pars,cov = curve_fit(diffusion,t,G_exp)
                
                (g0,tau_d) = pars
                
                D = w**2/(4*tau_d*1e-3)
                
                Viscosity = Viscosities[Dil]
                
                d = 2*kb*T/(6*Viscosity*np.pi*D)*1e12*1e9

                


                D_values.loc[k,"D ($\mu$m$^2$/s)"] = D
               
                D_values.loc[k,"Repetition"] = Rep
                D_values.loc[k,"Viscosity, $\mu$ (Pa·s)"] = Viscosity
                D_values.loc[k,"Size (nm)"] = d
                D_values.loc[k,"G$_0$"] = g0
                
                k+=1
                
                #plt.plot(t,G_exp,"b-")
                
                #plt.plot(t,diffusion(t,*pars),"r-")
                
                
        """        
        plt.title(file)
        plt.xscale("log")
        plt.xlabel("Time (ms)")
        plt.ylabel(r"G($\tau$)")
        plt.savefig(f"Figures/{file[:-4]}_fit.tif",bbox_inches="tight")
        plt.show()
        plt.close()
        """
                
                

mus = np.array(list(Viscosities.values()))

mus_theo = np.arange(0.00087273,0.030085,0.0000001)

D_theo = kb*T/(6*mus_theo*np.pi*(54.193584/2))*1e12*1e9


means = D_values.groupby(by=["Viscosity, $\mu$ (Pa·s)","Repetition"]).mean()
means2 = means.groupby(by=["Viscosity, $\mu$ (Pa·s)"]).mean()
stds = D_values.groupby(by=["Viscosity, $\mu$ (Pa·s)","Repetition"]).std()
stds2 = means.groupby(by=["Viscosity, $\mu$ (Pa·s)"]).std()

print(means)
plt.plot(mus_theo,D_theo,"k")
sns.scatterplot(data = means2,x = "Viscosity, $\mu$ (Pa·s)",y = "D ($\mu$m$^2$/s)",palette ="bright")
sns.lineplot(data = D_values,x = "Viscosity, $\mu$ (Pa·s)",y = "D ($\mu$m$^2$/s)",palette ="bright",errorbar="sd",err_style='bars',)

plt.show()
plt.close()

Ds = np.array(means2.loc[:,"D ($\mu$m$^2$/s)"])

errors =  np.array(stds2.loc[:,"D ($\mu$m$^2$/s)"])

plt.errorbar(mus*1000,Ds,yerr=errors,fmt="k.",barsabove=True,capsize=2.5,label="Experimental")
plt.plot(mus_theo*1000,D_theo,color="r",label="Theoretical")
pars,cov = curve_fit(Diffusion_fit,np.log10(mus),np.log10(Ds))
plt.plot(mus_theo*1000,10**(Diffusion_fit(np.log10(mus_theo),*pars)),"g",label="Fit")



plt.legend()
plt.xlabel("Viscosity, $\mu$ (cP)")
plt.ylabel("Diffusion Coefficient D ($\mu$m$^2$/s)")
plt.xscale("log")
plt.yscale("log")
plt.savefig("Figures/Viscosities Calibration Log Scale.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()


Ds = np.array(means2.loc[:,"D ($\mu$m$^2$/s)"])

errors =  np.array(stds2.loc[:,"D ($\mu$m$^2$/s)"])

plt.errorbar(mus*1000,Ds,yerr=errors,fmt="k.",barsabove=True,capsize=2.5,label="Experimental")
plt.plot(mus_theo*1000,D_theo,color="#e8000b",label="Theoretical")
pars,cov = curve_fit(Diffusion_fit,np.log10(mus),np.log10(Ds))
plt.plot(mus_theo*1000,10**(Diffusion_fit(np.log10(mus_theo),*pars)),"#1ac938",label="Fit")



plt.legend()
plt.xlabel("Viscosity, $\mu$ (cP)")
plt.ylabel("Diffusion Coefficient D ($\mu$m$^2$/s)")

plt.savefig("Figures/Viscosities Calibration.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()


g0s = np.array(means2.loc[:,"G$_0$"])
errors_g0 =  np.array(stds2.loc[:,"G$_0$"])


plt.errorbar(mus*1000,g0s,yerr=errors_g0,fmt="k.",barsabove=True,capsize=2.5,label="Experimental")
plt.xlabel("Viscosity, $\mu$ (cP)")
plt.ylabel("G$_0$")

regression_result = linregress(np.log10(mus),np.log10(g0s))

m = regression_result.slope
b = regression_result.intercept
rvalue = regression_result.rvalue
pvalue = regression_result.pvalue
m_err = regression_result.stderr
b_err = regression_result.intercept_stderr

plt.plot(mus_theo*1000,10**(m*np.log10(mus_theo)+b),"r",label="Fit")

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("Figures/g0 vs viscosity log.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()

g0s = np.array(means2.loc[:,"G$_0$"])
errors_g0 =  np.array(stds2.loc[:,"G$_0$"])


plt.errorbar(mus*1000,g0s,yerr=errors_g0,fmt="k.",barsabove=True,capsize=2.5,label="Experimental")
plt.xlabel("Viscosity, $\mu$ (cP)")
plt.ylabel("G$_0$")
plt.plot(mus_theo*1000,10**(m*np.log10(mus_theo)+b),"r",label="Fit")

plt.legend()
plt.savefig("Figures/g0 vs viscosity.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()


plt.errorbar(Ds,g0s,yerr=errors_g0,xerr = errors, fmt="k.",barsabove=True,capsize=2.5,label="Experimental")
plt.xlabel("D ($\mu$m$^2$/s)")
plt.ylabel("G$_0$")

regression_result = linregress(np.log10(Ds),np.log10(g0s))

m = regression_result.slope
b = regression_result.intercept
rvalue = regression_result.rvalue
pvalue = regression_result.pvalue
m_err = regression_result.stderr
b_err = regression_result.intercept_stderr

plt.plot(Ds,10**(m*np.log10(Ds)+b),"r",label="Fit")

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("Figures/g0 vs D log.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()


plt.errorbar(Ds,g0s,yerr=errors_g0,xerr = errors,fmt="k.",barsabove=True,capsize=2.5,label="Experimental")
plt.xlabel("D ($\mu$m$^2$/s)")
plt.ylabel("G$_0$")
plt.plot(Ds,10**(m*np.log10(Ds)+b),"r",label="Fit")

plt.legend()
plt.savefig("Figures/g0 vs D.tif",bbox_inches="tight",dpi=300)
plt.show()
plt.close()









results = pd.DataFrame()

results.loc[:,"Viscosity (cP)"] = mus*1000

for i in range(len(means2.index)):
    results.loc[i,"D"] = means2.loc[means2.index[i],"D ($\mu$m$^2$/s)"]
    results.loc[i,"D err"] = stds2.loc[means2.index[i],"D ($\mu$m$^2$/s)"]
    results.loc[i,"D Theo"] = D_theo[i]
    results.loc[i,"Size (nm)"] = means2.loc[means2.index[i],"Size (nm)"]
    results.loc[i,"Size err"] = stds2.loc[means2.index[i],"Size (nm)"]
    results.loc[i,"G$_0$"] = means2.loc[means2.index[i],"G$_0$"]
    results.loc[i,"G$_0$_err"] = stds2.loc[means2.index[i],"G$_0$"]

print(results)

results.to_csv("results.txt",sep="\t",decimal=",")


