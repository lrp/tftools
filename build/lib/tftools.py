#   tftools.py: Utilities for optimizing transfer function excitation signals
#   Copyright (C) 2013  Larry Price
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import division

import numpy as np
import scipy.signal as sig
from matplotlib.mlab import csd, psd
import sys

def tukeywin(m, a=0.5):
	'''
	Produces a tukey window

	a = overlap parameter between 0 returns a square window and 1 returns a Hann(ing) window
	m = number of points in the window

	see, e.g., https://en.wikipedia.org/wiki/Window_function#Tukey_window
	'''

	if a <= 0:
		return np.ones(m)
	elif a >= 1:
		return np.hanning(m)


	x = np.linspace(0, 1, m)
	w = np.ones_like(x)
	w[x < a/2] = (1 + np.cos(2*np.pi/a * (x[x < a/2] - a/2) )) / 2
	w[x >= (1 - a/2)] = (1 + np.cos(2*np.pi/a * (x[x >= (1 - a/2)] - 1 + a/2))) / 2

	return w

def getTF(exc,resp,tmeas,Fs,Nfft,padto=None):
	"""
	compute transfer function along with coherence and SNR. uses PSD/CSD method with 50% overlapping windows
	returns f, TF, Coherence, SNR

	exc = excitation signal
	resp = system response
	tmeas = duration of mesurement in seconds
	Fs = sample rate (Hz)
	Nfft = number of data points to be used in each block for fft
	padto = pad to this many points
	"""

	N = 1.89 * tmeas / (Nfft / Fs)

	Sx, f = psd(exc,NFFT=Nfft,Fs=Fs,noverlap=int(Nfft/2),pad_to=padto)
	Sy = psd(resp,NFFT=Nfft,Fs=Fs,noverlap=int(Nfft/2),pad_to=padto)[0]
	Sxy = csd(exc,resp,NFFT=Nfft,Fs=Fs,noverlap=int(Nfft/2),pad_to=padto)[0]
	Cxy = (Sxy * np.conj(Sxy)) / (Sx * Sy)
	snr = np.sqrt(Cxy * 2 * N / (1 - Cxy) )

	return f, Sxy / Sx, Cxy, snr

def fishersfzpk(ltisys,w,Sn):
	"""
	create the single-frequency fisher matrix for a transfer function in zpk form, i.e.
	           __
			   ||_i (w - z_i)

	H(w) = k --------------------
	           __
			   ||_i (w - p_i)

	*** the excitation signal is assumed to be a sum of sines ***
	*** the denominator must be monic ***

	arguments:

	ltisys = a scipy.signal.lti instance of the transfer function
	w = (angular) frequencies of interest (in rad/s)
	Sn = PSD of the noise as an array (same length as w)


	returns:
		an NxN numpy array with each value being an array of len(w) (the collection of all single
		frequency Fisher matrices at frequencies w)
	"""
	###FIXME: add some basic error handling

	#tf is in terms of iw, not w
	s = 1j*w

	#create the transfer function
	#use the lower case w because scipy puts the i in the transfer function for us
	tf = ltisys.freqresp(w)[1]

	#do the magnitude squared here once and for all
	tfmagsq = tf * np.conj(tf)

	#get the number of parameters
	if ltisys.gain == 1:
		N = len(ltisys.zeros) + len(ltisys.poles)
	else:
		N = len(ltisys.zeros) + len(ltisys.poles) + 1

	#take all the derivatives
	Dz = np.zeros([len(ltisys.zeros),len(s)],dtype=np.complex128)
	Dp = np.zeros([len(ltisys.poles),len(s)],dtype=np.complex128)

	for i,z in enumerate(ltisys.zeros):
		Dz[i] = -1 / (s - z)
	for i,p in enumerate(ltisys.poles):
		Dp[i] = 1 / (s - p)

	#check for unity gain and the absence of zeros
	if ltisys.gain == 1 and ltisys.zeros.size:
		deez = np.vstack((Dz,Dp))
	elif ltisys.gain == 1 and not ltisys.zeros.size:
		deez = Dp
	elif ltisys.gain != 1 and ltisys.zeros.size:
		deez = np.vstack((np.vstack((Dz,Dp)), 1/ltisys.gain[0] * np.ones(len(s))))
	else:
		deez = np.vstack((Dp,1/ltisys.gain[0] * np.ones(len(s))))

	#put it together to make the fisher matrix
	fisher = np.zeros([N,N,len(w)],dtype=np.float64)
	for i in range(N):
		for j in range(N):
			fisher[i][j] = 0.5 * tfmagsq * np.real(np.conj(deez[i])*deez[j]) / Sn

	#all done
	return fisher

def fishersfab(ltisys,w,Sn):
	"""
	create the single-frequency fisher matrix for a transfer function as a rational function, i.e.

			  Sum_0^N b_i s^i
	H(w) =  --------------------
			1 + Sum_1^M a_i s^i

	*** the excitation signal is assumed to be a sum of sines ***
	*** the denominator must be monic (it's enforced, so no worries)***

	arguments:

	ltisys = instance of scipy.signal.lti
	w = frequencies of interest
	Sn = PSD of the noise as an array (same length as w)

	returns:
		an NxN numpy array with each value being an array of len(w) (the collection of all single
		frequency Fisher matrices at frequencies w)
		NB: you have to take the transpose of the result if you want to, say compute the determinant via numpy.linalg.det
	"""
	###FIXME: add some basic error handling

	#tf is in terms of iw, not w
	s = 1j*w

	#get the tf in the right form
	a,b = lti2ab(ltisys)

	#create the numerator and denominator of the tf
	if b.size:
		numer = np.sum(np.array([ b[i] * s**i for i in range(len(b))]),axis=0)
	else: #i don't think this is even possible unless the tf is just a pass-through
		numer = np.ones(len(s))

	denom = np.sum(np.array([ a[i] * s**(i+1) for i in range(len(a))]),axis=0) + np.ones(len(s))

	#number of parameters
	N = len(a) + len(b)

	#take all the derivatives
	deez = np.zeros([N,len(w)],dtype=np.complex128)

	for i in range(N):
		#derivative wrt denominator
		#funky numbering because denom is monic (demonic?)
		if i < len(a):
			deez[i] = - s**(i+1) * numer / denom**2
		#derivative wrt numerator
		else:
			deez[i] = s**(i-len(a)) / denom

	#put it together to make the fisher matrix
	fisher = np.zeros([N,N,len(w)],dtype=np.float64)
	for i in range(N):
		for j in range(N):
			fisher[i][j] = 0.5 * np.real(np.conj(deez[i])*deez[j]) / Sn

	#all done
	return fisher

def fishersf(ltisys,w,Sn,usezpk=False):
	"""
	convenience function to select between zpk (default) and ab form for computing the fisher matrix
	"""
	if usezpk is True:
		return fishersfzpk(ltisys,w,Sn)
	else:
		return fishersfab(ltisys,w,Sn)

def fisherdesign(fmsf,Sx):
	"""
	compute the fisher matrix associated with the design Sx
	uses the Sx and the single frequency fisher matrix
	"""

	return np.sum(fmsf*Sx,axis=2)

def dispersion(fmdesign,fmsf):
	"""
	compute the dispersion from the single frequency and design fisher matrices
	"""

	return np.trace(np.dot(np.linalg.inv(fmdesign),fmsf))

def lti2ab(ltisys):
	"""
	convenience function to convert scipy.signal.lti instance to a,b suitable for fisher matrix calculation
	ltisys is an instance of scipy.signal.lti

	returns a,b
	"""

	b = ltisys.num
	a = ltisys.den

	#fancy array slicing to reverse the order (::-1) and remove the first element of a (1:)
	return a[::-1][1:] / a[-1], b[::-1] / a[-1]


def findfreqs(ltisys,Sn,w,nfreqs=None,usezpk=False):
	"""
	find best frequencies for optimal design (brute force method)

	arguments:

	ltisys = instance of scipy.signal.lti
	w = (angular) frequencies of interest
	nfreqs = # of frequencies to return.  default is 3 x #parameters
	usezpk = boolean for indicating form of the transfer function

	returns:

	wopt = array of optimal frequencies to use
	fisherf = single-frequency fisher matrix evaluated at wopt (basically input for design optimization)
	"""
	#get the number of parameters and put the transfer function in the right form
	if usezpk is True:
		#number of parameters for zpk representation
		if ltisys.gain == 1:
			nparm = len(ltisys.zeros) + len(ltisys.poles)
		else:
			nparm = len(ltisys.zeros) + len(ltisys.poles) + 1
	else:
		#using ab form
		a,b = lti2ab(ltisys)

		#number of parameters
		nparm = len(a) + len(b)

	#set the number of frequencies
	if nfreqs is None:
		nfreqs = 3 * nparm
	if nfreqs < 2 * nparm:
		raise ValueError('Must specify an nfreqs at least twice as large as the number of parameters!')
		sys.exit(0)

	fmsf = fishersf(ltisys,w,Sn,usezpk=usezpk)
	thesefreqs = np.sort(np.argsort(np.linalg.det(fmsf.T))[-nfreqs:])

	return w[thesefreqs], fmsf.T[thesefreqs].T


def optdesign(ltisys,w,usezpk=False,fmsf=None,Sn=None,tol=None,maxit=10000):
	"""
	compute the optimal design, Sx

	arguments:

	ltisys = instance of scipy.signal.lti
	w = the frequencies to optimize over
	tol = if max(dispersion - nparam) < tol, then iteration ceases.  if tol isn't specified then iteration continues until maxit
	maxit = maximum number of iterations to perform

	returns a tuple containing:

	Sx = optimal design as a numpy array
	max(dispersion - nparam)
	"""
	#FIXME: add some error handling
	if fmsf is None and Sn is None:
		raise ValueError('Must specify Sn to compute Fisher!')
		sys.exit(1)

	#get the number of parameters and put the transfer function in the right form
	if usezpk is True:
		#number of parameters for zpk representation
		if ltisys.gain == 1:
			nparm = len(ltisys.zeros) + len(ltisys.poles)
		else:
			nparm = len(ltisys.zeros) + len(ltisys.poles) + 1
	else:
		#using ab form
		a,b = lti2ab(ltisys)

		#number of parameters
		nparm = len(a) + len(b)

	#compute the single frequency fisher matrix
	if fmsf is None:
		fmsf = fishersf(ltisys,w,Sn,usezpk=usezpk)

	#initial design
	#normalized to one with all the power evenly distributed
	#don't worry about phases for now...FIXME: optimize phases
	Sx = np.ones(len(w)) / len (w)

	#compute the dispersion
	disp = dispersion(fisherdesign(fmsf,Sx),fmsf)

	for i in range(maxit):
		if tol is not None:
			if np.max(disp - nparm) < tol:
				break
		else:
			Sx *= (disp / nparm)
			disp = dispersion(fisherdesign(fmsf,Sx),fmsf)

	return Sx, np.max(disp - nparm)
