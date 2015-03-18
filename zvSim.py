#!/usr/bin/python
# -*- encoding: UTF-8 -*-
'''z(V) Modeling and Simulation

	List of Classes: -none-
	List of Functions:
			quickRun
			scqZV
			rtcZV
'''

# built-in modules
import sys
sys.path.append('/home/alex/Dropbox/ampPy')

# authored modules
import rtc
import ampNum as an

# third-party modules
from scipy import *
from scipy.optimize import brenth
from scipy.integrate import quad
from pylab import *

#===============================================================================
def main(*args):
	pass
# END main

#===============================================================================
def quickRun():
	V = linspace(0.5, 4.5, 200)
	params = {}
	Z = rtcZV(V, 10.0, 24, **params)
	
	f = open('quickRun results.csv', 'w')
	for i in range(0,len(V)):
		f.write('{0:0.8f}, {1:0.8f}\n'.format(V[i], Z[i]))
	f.close()
	
	figure()
	plot(V, Z)
	xlabel('Gap Bias (V)')
	ylabel('z(V) (V)')
	title('Simulated cc-STS Results, z')
	grid(True)
	figure()
	plot(V, an.deriv(V, Z))
	xlabel('Gap Bias (V)')
	ylabel('dz/dV(V) (V)')
	title('Simulated cc-STS Reults, dz/dV')
	grid(True)
	show()
# END quickRun

#===============================================================================
def scqZV(V, I, T, **params):
	# dkargs: Default values for keyword arguments
	dparams = {
		'phi': 5.0, 'tipR': 1.0, 'rhot': 1.0, 'rhoMax': 1.0E-3, 'E0': 2.5,
		'w': 0.5, 'rho0': 1.0E-3
	}
	# Handle keyword arguments
	for k in dparams.keys():
		# Set all missing keyword arguments to their default values
		if k not in params.keys():
			params[k] = dparams[k]
		elif type(params[k]).__name__ != type(dparams[k]).__name__:
			raise TypeError(
				'Keyword argument "{0}" should be {1}'.format(
				k, type(dparams[k]).__name__
				)
			)
	# END for
	
	# cE: conversion factor for energy
	cE = 27.211 #eV/Hartree
	# cL: conversion factor for length
	cL = 0.0529 #nm/Bohr Radii
	# cI: conversion factor for current
	cI = 6.62361782E9 #pA/auI
	# cDOS: conversion factor for DOS
	cDOS = cL**3 * cE #(eV*nm^3)/(Hartree*BohrRadii^3)
	
	# convert input current from pA to a.u.
	I /= cI
	# convert input voltage from V to Hartree
	V = V/cE
	
	# Set up physical parameters
	# phi: apparent tunneling barrier height in eV (-> a.u.)
	phi = params['phi'] / cE
	# A: area of the tunneling junction in nm^2 (-> a.u.)
	A = pi*(params['tipR']/cL)**2
	# rhot: constant value approximation for DOS of tip in 1/eV/nm^3
	#       (-> a.u.)
	rhot = params['rhot'] * cDOS
	# Sample DOS parameters
	rhoMax = params['rhoMax'] * cDOS
	E0 = params['E0'] / cE
	w = params['w'] / cE
	rho0 = params['rho0'] * cDOS
	
	# rhosF: DOS function consisting of a flat background and a single
	#        gaussian peak
	rhosF = lambda E: rhoMax*exp( -log(16.0)*((E-E0)/w)**2 ) + rho0
	
	# TF: Transmission function
	# square barrier approximation
	if T == 'sqr':
		TF = lambda z,v,E: exp( -z * sqrt( 8*(phi + 0.5*v - E) ) )
	# trapezoid barrier approximation
	elif T == 'trap':
		TF = lambda z,v,E: exp(
			-sqrt(32.0/9.0) * z * ((phi+v-E)**1.5 - (phi-E)**1.5) / v
		)
	else:
		print '**transmission function label not reconized!**'
		raise
	
	# tunCurrF: Tunneling current functions
	def tunCurrF(z, v):
		integralVal = quad(lambda E: rhosF(E)*TF(z,v,E), 0, v)[0]
		return A*0.5*pi*rhot*integralVal
	# END tunCurr
	
	Z = zeros(len(V))
	for i, v in enumerate(V):
		if i == 0:
			z1 = 0.001/cL
			z2 = 0.5/cL
		else:
			z1 = Z[i-1]
			z2 = z1 + 0.05/cL
		
		try:
			Z[i] = brenth(lambda z: tunCurrF(z, v)-I, z1, z2)
		except ValueError:
			print 'V[{0}] = {1:0.3f}V'.format(i, v*cE)
			print '(z1 = {0:0.5f}nm, I-10 = {1:0.5e}pA)'.format(
				z1*cL, (tunCurrF(z1, v)-I)*cI
			)
			print '(z2 = {0:0.5f}nm, I-10 = {1:0.5e}pA)'.format(
				z2*cL, (tunCurrF(z2, v)-I)*cI
			)
			quit()
	# END for
	
	return Z*cL
# END scqZV

#===============================================================================
def rtcZV(V, I, K, **params):
	# dkargs: Default values for keyword arguments
	dparams = {
		'phi': 5.0, 'tipR': 1.0, 'rhot': 1.0, 'rhoMax': 1.0E-3, 'E0': 2.5,
		'w': 0.5, 'rho0': 1.0E-3
	}
	# Handle keyword arguments
	for k in dparams.keys():
		# Set all missing keyword arguments to their default values
		if k not in params.keys():
			params[k] = dparams[k]
		elif type(params[k]).__name__ != type(dparams[k]).__name__:
			raise TypeError(
				'Keyword argument "{0}" should be {1}'.format(
				k, type(dparams[k]).__name__
				)
			)
	# END for
	
	# cE: conversion factor for energy
	cE = 27.211 #eV/Hartree
	# cL: conversion factor for length
	cL = 0.0529 #nm/Bohr Radii
	# cI: conversion factor for current
	cI = 6.62361782E9 #pA/auI
	# cDOS: conversion factor for DOS
	cDOS = cL**3 * cE #(eV*nm^3)/(Hartree*BohrRadii^3)
	
	# number of points
	N = len(V)
	
	# convert input current from pA to a.u.
	I /= cI
	# convert input voltage from V to Hartree
	V = V/cE
	
	# Set up physical parameters
	# phi: apparent tunneling barrier height in eV (-> a.u.)
	phi = params['phi'] / cE
	# A: area of the tunneling junction in nm^2 (-> a.u.)
	A = pi*(params['tipR']/cL)**2
	# rhot: constant value approximation for DOS of tip in 1/eV/nm^3
	#       (-> a.u.)
	rhot = params['rhot'] * cDOS
	# Sample DOS parameters
	rhoMax = params['rhoMax'] * cDOS
	E0 = params['E0'] / cE
	w = params['w'] / cE
	rho0 = params['rho0'] * cDOS
	
	# rhosF: DOS function consisting of a flat background and a single
	#        gaussian peak
	rhosF = lambda E: rhoMax*exp( -log(16.0)*((E-E0)/w)**2 ) + rho0
	
	# TF: Transmission function, square barrier approximation
	TF = lambda z,E: exp( -z * sqrt( 8*(phi + 0.5*V[0] - E) ) )
	
	# tunCurrF: Tunneling current functions
	def tunCurrF(z):
		integralVal = quad(lambda E: rhosF(E)*TF(z,E), 0, V[0])[0]
		return A*0.5*pi*rhot*integralVal
	
	# Initial z value
	z0 = brenth(lambda z: tunCurrF(z)-I, 0.2/cL, 0.3/cL)
	
	# Z: empty vector for storing z values
	Z = zeros(N)
	Z[0] = z0
	
	# Set up diff eq prefactors
	# Cst: ... for the "rhos*T" term
	Cst = pi * A * rhot / (I * sqrt(32*phi))
	# Cz: ... for the "z" term
	Cz = 1/(4.0 * phi)
	
	# pre-loop computation
	Q = zeros( (K+1,K+1) )
	for k in range(1,K):
		for j in range(0,k):
			Q[j,k] = 1.0 - float(j)/float(k)
	
	# z(V)-solve loop
	# set up loop variables
	# taylor coeff array of unitary constant
	c = rtc.const(K)
	# rv: vector containing taylor coefficients of v
	rv = zeros(K+1)
	rv[1] = 1.0
	# rz: vector containing taylor coefficients of z
	rz = zeros(K+1)
	# iteratively solve for Z across V, starting at V[1]
	for i in range(1,N):
		# h: step size
		h = V[i] - V[i-1]
		# previous V value
		rv[0] = V[i-1]
		# previous Z value
		rz[0] = Z[i-1]
		
		# W1: working variable 1, taylor coeff array
		W1 = rtc.sqrt( 8*(phi*c - 0.5*rv), approx='lin' )
		
		# rhos: sample DOS, taylor coeff array
		rhos = rhoMax*rtc.gauss(rv, x0=E0, w=w) + rho0*c
		
		W2 = zeros(K+1)
		T = zeros(K+1)
		# iteratively solve for all orders of taylor coefficients for z
		for k in range(0, K):
			W2[k] = rtc.prod(-rz, W1, k)
			
			# T = e^W2
			if k == 0:
				T[0] = exp(W2[0])
			else:
				for j in range(0,k):
					T[k] += Q[j,k] * W2[k-j] * T[j]
			
			# z' = Cst*rhos*T - Cz*z
			rz[k+1] = ( Cst*rtc.prod(rhos, T, k) - Cz*rz[k] )/(k+1)
		
		# build up z at the next V point as a Taylor approximation of the
		#  function out from the previous points
		for k in range(0,K+1):
			Z[i] += rz[k]*(h**k)
	# END z(V)-solve loop
	
	return Z*cL
# END rtcZV

#===============================================================================
if __name__ == "__main__":
	quickRun()















