#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:08:16 2016

@author: Alban Siffer 
@company: Amossys
@license: GNU GPLv3
"""

from math import log, floor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy.optimize import minimize

# colors for plot
deep_saffron = '#FF9933'
air_force_blue = '#5D8AA8'

"""
================================= MAIN CLASS ==================================
"""


class SPOT:
    """
    This class allows to run SPOT algorithm on univariate dataset (upper-bound)
    
    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        
    data : numpy.array
        stream
    
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
    
    init_threshold : float
        initial threshold computed during the calibration step
    
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
    
    n : int
        number of observed values
    
    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor

	    Parameters
	    ----------
	    q
		    Detection level (risk)
	
	    Returns
	    ----------
    	SPOT object
        """
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to SPOT object
        
        Parameters
	    ----------
	    init_data : list, numpy.array or pandas.Series
		    initial batch to calibrate the algorithm
            
        data : numpy.array
		    data for the run (list, np.array or pd.series)
	
        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data
        
        Parameters
	    ----------
	    data : list, numpy.array, pandas.Series
		    data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        """
        Run the calibration (initialization) step
        
        Parameters
	    ----------
        level : float
            (default 0.98) Probability associated with the initial threshold t 
	    verbose : bool
		    (default = True) If True, gives details about the batch initialization
        verbose: bool
            (default True) If True, prints log
        min_extrema bool
            (default False) If True, find min extrema instead of max extrema
        """
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level

        level = level - floor(level)

        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(level * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function
        
        Parameters
        ----------
        fun : function
		    scalar function
        jac : function
            first order derivative of the function  
        bounds : tuple
            (min,max) interval for the roots search    
        npoints : int
            maximum number of roots to output      
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval
        
        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)
        
        Parameters
        ----------
        Y : numpy.array
		    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)   

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick
        
        Parameters
        ----------
        epsilon : float
		    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q
        
        Parameters
        ----------
        gamma : float
		    GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True, dynamic=True):
        """
        Run SPOT on the stream
        
        Parameters
        ----------
        with_alarm : bool
		    (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'thresholds' and 'alarms'
            
            'thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms
            
        """
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):

            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                # If the observed value exceeds the current threshold (alarm case)
                if self.data[i] > self.extreme_quantile:
                    # if we want to alarm, we put it in the alarm list
                    if with_alarm:
                        alarm.append(i)
                    # otherwise we add it in the peaks
                    else:
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        self.Nt += 1
                        self.n += 1
                        # and we update the thresholds

                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)

                # case where the value exceeds the initial threshold but not the alarm ones
                elif self.data[i] > self.init_threshold:
                    # we add it in the peaks
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1

            th.append(self.extreme_quantile)  # thresholds record

        return {'thresholds': th, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results of given by the run
        
        Parameters
        ----------
        run_results : dict
            results given by the 'run' method
        with_alarm : bool
		    (default = True) If True, alarms are plotted.


        Returns
        ----------
        list
            list of the plots
            
        """
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'thresholds' in K:
            th = run_results['thresholds']
            th_fig, = plt.plot(x, th, color=deep_saffron, lw=2, ls='dashed')
            fig.append(th_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            al_fig = plt.scatter(alarm, self.data[alarm], color='red')
            fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig


"""
============================ UPPER & LOWER BOUNDS =============================
"""


class biSPOT:
    """
    This class allows to run biSPOT algorithm on univariate dataset (upper and lower bounds)
    
    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        
    data : numpy.array
        stream
    
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
    
    init_threshold : float
        initial threshold computed during the calibration step
    
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
    
    n : int
        number of observed values
    
    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4):
        """
        Constructor

	    Parameters
	    ----------
	    q
		    Detection level (risk)
	
	    Returns
	    ----------
        biSPOT object
        """
        self.proba = q
        self.data = None
        self.init_data = None
        self.n = 0
        nonedict = {'up': None, 'down': None}

        self.extreme_quantile = dict.copy(nonedict)
        self.init_threshold = dict.copy(nonedict)
        self.peaks = dict.copy(nonedict)
        self.gamma = dict.copy(nonedict)
        self.sigma = dict.copy(nonedict)
        self.Nt = {'up': 0, 'down': 0}

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
                s += '\t triggered alarms : %s (%.2f %%)\n' % (len(self.alarm), 100 * len(self.alarm) / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t upper extreme quantile : %s\n' % self.extreme_quantile['up']
                s += '\t lower extreme quantile : %s\n' % self.extreme_quantile['down']
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to biSPOT object
        
        Parameters
	    ----------
	    init_data : list, numpy.array or pandas.Series
		    initial batch to calibrate the algorithm ()
            
        data : numpy.array
		    data for the run (list, np.array or pd.series)
	
        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data
        
        Parameters
	    ----------
	    data : list, numpy.array, pandas.Series
		    data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, verbose=True):
        """
        Run the calibration (initialization) step
        
        Parameters
	    ----------
	    verbose : bool
		    (default = True) If True, gives details about the batch initialization
        """
        n_init = self.init_data.size

        S = np.sort(self.init_data)  # we sort X to get the empirical quantile
        self.init_threshold['up'] = S[int(0.98 * n_init)]  # t is fixed for the whole algorithm
        self.init_threshold['down'] = S[int(0.02 * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks['up'] = self.init_data[self.init_data > self.init_threshold['up']] - self.init_threshold['up']
        self.peaks['down'] = -(
                self.init_data[self.init_data < self.init_threshold['down']] - self.init_threshold['down'])
        self.Nt['up'] = self.peaks['up'].size
        self.Nt['down'] = self.peaks['down'].size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        l = {'up': None, 'down': None}
        for side in ['up', 'down']:
            g, s, l[side] = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)
            self.gamma[side] = g
            self.sigma[side] = s

        ltab = 20
        form = ('\t' + '%20s' + '%20.2f' + '%20.2f')
        if verbose:
            print('[done]')
            print('\t' + 'Parameters'.rjust(ltab) + 'Upper'.rjust(ltab) + 'Lower'.rjust(ltab))
            print('\t' + '-' * ltab * 3)
            print(form % (chr(0x03B3), self.gamma['up'], self.gamma['down']))
            print(form % (chr(0x03C3), self.sigma['up'], self.sigma['down']))
            print(form % ('likelihood', l['up'], l['down']))
            print(form % ('Extreme quantile', self.extreme_quantile['up'], self.extreme_quantile['down']))
            print('\t' + '-' * ltab * 3)
        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function
        
        Parameters
        ----------
        fun : function
		    scalar function 
        jac : function
            first order derivative of the function  
        bounds : tuple
            (min,max) interval for the roots search    
        npoints : int
            maximum number of roots to output      
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval
        
        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)
        
        Parameters
        ----------
        Y : numpy.array
		    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)   

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, side, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick
        
        Parameters
        ----------
        epsilon : float
		    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks[side].min()
        YM = self.peaks[side].max()
        Ymean = self.peaks[side].mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = biSPOT._rootsFinder(lambda t: w(self.peaks[side], t),
                                         lambda t: jac_w(self.peaks[side], t),
                                         (a + epsilon, -epsilon),
                                         n_points, 'regular')

        right_zeros = biSPOT._rootsFinder(lambda t: w(self.peaks[side], t),
                                          lambda t: jac_w(self.peaks[side], t),
                                          (b, c),
                                          n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = biSPOT._log_likelihood(self.peaks[side], gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks[side]) - 1
            sigma = gamma / z
            ll = biSPOT._log_likelihood(self.peaks[side], gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, side, gamma, sigma):
        """
        Compute the quantile at level 1-q for a given side
        
        Parameters
        ----------
        side : str
            'up' or 'down'
        gamma : float
		    GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        if side == 'up':
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold['up'] + (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold['up'] - sigma * log(r)
        elif side == 'down':
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold['down'] - (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold['down'] + sigma * log(r)
        else:
            print('error : the side is not right')

    def run(self, with_alarm=True):
        """
        Run biSPOT on the stream
        
        Parameters
        ----------
        with_alarm : bool
		    (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'upper_thresholds', 'lower_thresholds' and 'alarms'
            
            '***-thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms
            
        """
        if (self.n > self.init_data.size):
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # list of the thresholds
        thup = []
        thdown = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):

            # If the observed value exceeds the current threshold (alarm case)
            if self.data[i] > self.extreme_quantile['up']:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks['up'] = np.append(self.peaks['up'], self.data[i] - self.init_threshold['up'])
                    self.Nt['up'] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw('up')
                    self.extreme_quantile['up'] = self._quantile('up', g, s)

            # case where the value exceeds the initial threshold but not the alarm ones
            elif self.data[i] > self.init_threshold['up']:
                # we add it in the peaks
                self.peaks['up'] = np.append(self.peaks['up'], self.data[i] - self.init_threshold['up'])
                self.Nt['up'] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw('up')
                self.extreme_quantile['up'] = self._quantile('up', g, s)

            elif self.data[i] < self.extreme_quantile['down']:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks['down'] = np.append(self.peaks['down'], -(self.data[i] - self.init_threshold['down']))
                    self.Nt['down'] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw('down')
                    self.extreme_quantile['down'] = self._quantile('down', g, s)

            # case where the value exceeds the initial threshold but not the alarm ones
            elif self.data[i] < self.init_threshold['down']:
                # we add it in the peaks
                self.peaks['down'] = np.append(self.peaks['down'], -(self.data[i] - self.init_threshold['down']))
                self.Nt['down'] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw('down')
                self.extreme_quantile['down'] = self._quantile('down', g, s)
            else:
                self.n += 1

            thup.append(self.extreme_quantile['up'])  # thresholds record
            thdown.append(self.extreme_quantile['down'])  # thresholds record

        return {'upper_thresholds': thup, 'lower_thresholds': thdown, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results of given by the run
        
        Parameters
        ----------
        run_results : dict
            results given by the 'run' method
        with_alarm : bool
		    (default = True) If True, alarms are plotted.


        Returns
        ----------
        list
            list of the plots
            
        """
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'upper_thresholds' in K:
            thup = run_results['upper_thresholds']
            uth_fig, = plt.plot(x, thup, color=deep_saffron, lw=2, ls='dashed')
            fig.append(uth_fig)

        if 'lower_thresholds' in K:
            thdown = run_results['lower_thresholds']
            lth_fig, = plt.plot(x, thdown, color=deep_saffron, lw=2, ls='dashed')
            fig.append(lth_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            al_fig = plt.scatter(alarm, self.data[alarm], color='red')
            fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig


"""
================================= WITH DRIFT ==================================
"""


def backMean(X, d):
    M = []
    w = X[:d].sum()
    M.append(w / d)
    for i in range(d, len(X)):
        w = w - X[i - d] + X[i]
        M.append(w / d)
    return np.array(M)


class dSPOT:
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper-bound)
    
    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        
    depth : int
        Number of observations to compute the moving average
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        
    data : numpy.array
        stream
    
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
    
    init_threshold : float
        initial threshold computed during the calibration step
    
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
    
    n : int
        number of observed values
    
    Nt : int
        number of observed peaks
    """

    def __init__(self, q, depth):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0
        self.depth = depth

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
                s += '\t triggered alarms : %s (%.2f %%)\n' % (len(self.alarm), 100 * len(self.alarm) / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t extreme quantile : %s\n' % self.extreme_quantile
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to DSPOT object
        
        Parameters
	    ----------
	    init_data : list, numpy.array or pandas.Series
		    initial batch to calibrate the algorithm
            
        data : numpy.array
		    data for the run (list, np.array or pd.series)
	
        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data
        
        Parameters
	    ----------
	    data : list, numpy.array, pandas.Series
		    data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, verbose=True):
        """
        Run the calibration (initialization) step
        
        Parameters
	    ----------
	    verbose : bool
		    (default = True) If True, gives details about the batch initialization
        """
        n_init = self.init_data.size - self.depth

        M = backMean(self.init_data, self.depth)
        T = self.init_data[self.depth:] - M[:-1]  # new variable

        S = np.sort(T)  # we sort X to get the empirical quantile
        self.init_threshold = S[int(0.98 * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks = T[T > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)

        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function
        
        Parameters
        ----------
        fun : function
		    scalar function 
        jac : function
            first order derivative of the function  
        bounds : tuple
            (min,max) interval for the roots search    
        npoints : int
            maximum number of roots to output      
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval
        
        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)
        
        Parameters
        ----------
        Y : numpy.array
		    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)   

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick
        
        Parameters
        ----------
        epsilon : float
		    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')

        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = dSPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        """
        Compute the quantile at level 1-q
        
        Parameters
        ----------
        gamma : float
		    GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, with_alarm=True):
        """
        Run biSPOT on the stream
        
        Parameters
        ----------
        with_alarm : bool
		    (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'upper_thresholds', 'lower_thresholds' and 'alarms'
            
            '***-thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms
            
        """
        if (self.n > self.init_data.size):
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # actual normal window
        W = self.init_data[-self.depth:]

        # list of the thresholds
        th = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):
            Mi = W.mean()
            # If the observed value exceeds the current threshold (alarm case)
            if (self.data[i] - Mi) > self.extreme_quantile:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks = np.append(self.peaks, self.data[i] - Mi - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)  # + Mi
                    W = np.append(W[1:], self.data[i])

            # case where the value exceeds the initial threshold but not the alarm ones
            elif (self.data[i] - Mi) > self.init_threshold:
                # we add it in the peaks
                self.peaks = np.append(self.peaks, self.data[i] - Mi - self.init_threshold)
                self.Nt += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw()
                self.extreme_quantile = self._quantile(g, s)  # + Mi
                W = np.append(W[1:], self.data[i])
            else:
                self.n += 1
                W = np.append(W[1:], self.data[i])

            th.append(self.extreme_quantile + Mi)  # thresholds record

        return {'thresholds': th, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results given by the run
        
        Parameters
        ----------
        run_results : dict
            results given by the 'run' method
        with_alarm : bool
		    (default = True) If True, alarms are plotted.


        Returns
        ----------
        list
            list of the plots
            
        """
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        #        if 'upper_thresholds' in K:
        #            thup = run_results['upper_thresholds']
        #            uth_fig, = plt.plot(x,thup,color=deep_saffron,lw=2,ls='dashed')
        #            fig.append(uth_fig)
        #
        #        if 'lower_thresholds' in K:
        #            thdown = run_results['lower_thresholds']
        #            lth_fig, = plt.plot(x,thdown,color=deep_saffron,lw=2,ls='dashed')
        #            fig.append(lth_fig)

        if 'thresholds' in K:
            th = run_results['thresholds']
            th_fig, = plt.plot(x, th, color=deep_saffron, lw=2, ls='dashed')
            fig.append(th_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            if len(alarm) > 0:
                plt.scatter(alarm, self.data[alarm], color='red')

        plt.xlim((0, self.data.size))

        return fig


"""
=========================== DRIFT & DOUBLE BOUNDS =============================
"""


class bidSPOT:
    """
    This class allows to run DSPOT algorithm on univariate dataset (upper and lower bounds)
    
    Attributes
    ----------
    proba : float
        Detection level (risk), chosen by the user
        
    depth : int
        Number of observations to compute the moving average
        
    extreme_quantile : float
        current threshold (bound between normal and abnormal events)
        
    data : numpy.array
        stream
    
    init_data : numpy.array
        initial batch of observations (for the calibration/initialization step)
    
    init_threshold : float
        initial threshold computed during the calibration step
    
    peaks : numpy.array
        array of peaks (excesses above the initial threshold)
    
    n : int
        number of observed values
    
    Nt : int
        number of observed peaks
    """

    def __init__(self, q=1e-4, depth=10):
        self.proba = q
        self.data = None
        self.init_data = None
        self.n = 0
        self.depth = depth

        nonedict = {'up': None, 'down': None}

        self.extreme_quantile = dict.copy(nonedict)
        self.init_threshold = dict.copy(nonedict)
        self.peaks = dict.copy(nonedict)
        self.gamma = dict.copy(nonedict)
        self.sigma = dict.copy(nonedict)
        self.Nt = {'up': 0, 'down': 0}

    def __str__(self):
        s = ''
        s += 'Streaming Peaks-Over-Threshold Object\n'
        s += 'Detection level q = %s\n' % self.proba
        if self.data is not None:
            s += 'Data imported : Yes\n'
            s += '\t initialization  : %s values\n' % self.init_data.size
            s += '\t stream : %s values\n' % self.data.size
        else:
            s += 'Data imported : No\n'
            return s

        if self.n == 0:
            s += 'Algorithm initialized : No\n'
        else:
            s += 'Algorithm initialized : Yes\n'
            s += '\t initial threshold : %s\n' % self.init_threshold

            r = self.n - self.init_data.size
            if r > 0:
                s += 'Algorithm run : Yes\n'
                s += '\t number of observations : %s (%.2f %%)\n' % (r, 100 * r / self.n)
                s += '\t triggered alarms : %s (%.2f %%)\n' % (len(self.alarm), 100 * len(self.alarm) / self.n)
            else:
                s += '\t number of peaks  : %s\n' % self.Nt
                s += '\t upper extreme quantile : %s\n' % self.extreme_quantile['up']
                s += '\t lower extreme quantile : %s\n' % self.extreme_quantile['down']
                s += 'Algorithm run : No\n'
        return s

    def fit(self, init_data, data):
        """
        Import data to biDSPOT object
        
        Parameters
	    ----------
	    init_data : list, numpy.array or pandas.Series
		    initial batch to calibrate the algorithm
            
        data : numpy.array
		    data for the run (list, np.array or pd.series)
	
        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        elif isinstance(init_data, pd.Series):
            self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) & (init_data < 1) & (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else:
            print('The initial data cannot be set')
            return

    def add(self, data):
        """
        This function allows to append data to the already fitted data
        
        Parameters
	    ----------
	    data : list, numpy.array, pandas.Series
		    data to append
        """
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, np.ndarray):
            data = data
        elif isinstance(data, pd.Series):
            data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            return

        self.data = np.append(self.data, data)
        return

    def initialize(self, verbose=True):
        """
        Run the calibration (initialization) step
        
        Parameters
	    ----------
	    verbose : bool
		    (default = True) If True, gives details about the batch initialization
        """
        n_init = self.init_data.size - self.depth

        M = backMean(self.init_data, self.depth)
        T = self.init_data[self.depth:] - M[:-1]  # new variable

        S = np.sort(T)  # we sort T to get the empirical quantile
        self.init_threshold['up'] = S[int(0.98 * n_init)]  # t is fixed for the whole algorithm
        self.init_threshold['down'] = S[int(0.02 * n_init)]  # t is fixed for the whole algorithm

        # initial peaks
        self.peaks['up'] = T[T > self.init_threshold['up']] - self.init_threshold['up']
        self.peaks['down'] = -(T[T < self.init_threshold['down']] - self.init_threshold['down'])
        self.Nt['up'] = self.peaks['up'].size
        self.Nt['down'] = self.peaks['down'].size
        self.n = n_init

        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')

        l = {'up': None, 'down': None}
        for side in ['up', 'down']:
            g, s, l[side] = self._grimshaw(side)
            self.extreme_quantile[side] = self._quantile(side, g, s)
            self.gamma[side] = g
            self.sigma[side] = s

        ltab = 20
        form = ('\t' + '%20s' + '%20.2f' + '%20.2f')
        if verbose:
            print('[done]')
            print('\t' + 'Parameters'.rjust(ltab) + 'Upper'.rjust(ltab) + 'Lower'.rjust(ltab))
            print('\t' + '-' * ltab * 3)
            print(form % (chr(0x03B3), self.gamma['up'], self.gamma['down']))
            print(form % (chr(0x03C3), self.sigma['up'], self.sigma['down']))
            print(form % ('likelihood', l['up'], l['down']))
            print(form % ('Extreme quantile', self.extreme_quantile['up'], self.extreme_quantile['down']))
            print('\t' + '-' * ltab * 3)
        return

    def _rootsFinder(fun, jac, bounds, npoints, method):
        """
        Find possible roots of a scalar function
        
        Parameters
        ----------
        fun : function
		    scalar function 
        jac : function
            first order derivative of the function  
        bounds : tuple
            (min,max) interval for the roots search    
        npoints : int
            maximum number of roots to output      
        method : str
            'regular' : regular sample of the search interval, 'random' : uniform (distribution) sample of the search interval
        
        Returns
        ----------
        numpy.array
            possible roots of the function
        """
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)

        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j

        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))

        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        """
        Compute the log-likelihood for the Generalized Pareto Distribution (μ=0)
        
        Parameters
        ----------
        Y : numpy.array
		    observations
        gamma : float
            GPD index parameter
        sigma : float
            GPD scale parameter (>0)   

        Returns
        ----------
        float
            log-likelihood of the sample Y to be drawn from a GPD(γ,σ,μ=0)
        """
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _grimshaw(self, side, epsilon=1e-8, n_points=8):
        """
        Compute the GPD parameters estimation with the Grimshaw's trick
        
        Parameters
        ----------
        epsilon : float
		    numerical parameter to perform (default : 1e-8)
        n_points : int
            maximum number of candidates for maximum likelihood (default : 10)

        Returns
        ----------
        gamma_best,sigma_best,ll_best
            gamma estimates, sigma estimates and corresponding log-likelihood
        """

        def u(s):
            return 1 + np.log(s).mean()

        def v(s):
            return np.mean(1 / s)

        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1

        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks[side].min()
        YM = self.peaks[side].max()
        Ymean = self.peaks[side].mean()

        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points

        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        # We look for possible roots
        left_zeros = bidSPOT._rootsFinder(lambda t: w(self.peaks[side], t),
                                          lambda t: jac_w(self.peaks[side], t),
                                          (a + epsilon, -epsilon),
                                          n_points, 'regular')

        right_zeros = bidSPOT._rootsFinder(lambda t: w(self.peaks[side], t),
                                           lambda t: jac_w(self.peaks[side], t),
                                           (b, c),
                                           n_points, 'regular')

        # all the possible roots
        zeros = np.concatenate((left_zeros, right_zeros))

        # 0 is always a solution so we initialize with it
        gamma_best = 0
        sigma_best = Ymean
        ll_best = bidSPOT._log_likelihood(self.peaks[side], gamma_best, sigma_best)

        # we look for better candidates
        for z in zeros:
            gamma = u(1 + z * self.peaks[side]) - 1
            sigma = gamma / z
            ll = bidSPOT._log_likelihood(self.peaks[side], gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll

        return gamma_best, sigma_best, ll_best

    def _quantile(self, side, gamma, sigma):
        """
        Compute the quantile at level 1-q for a given side
        
        Parameters
        ----------
        side : str
            'up' or 'down'
        gamma : float
		    GPD parameter
        sigma : float
            GPD parameter

        Returns
        ----------
        float
            quantile at level 1-q for the GPD(γ,σ,μ=0)
        """
        if side == 'up':
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold['up'] + (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold['up'] - sigma * log(r)
        elif side == 'down':
            r = self.n * self.proba / self.Nt[side]
            if gamma != 0:
                return self.init_threshold['down'] - (sigma / gamma) * (pow(r, -gamma) - 1)
            else:
                return self.init_threshold['down'] + sigma * log(r)
        else:
            print('error : the side is not right')

    def run(self, with_alarm=True, plot=True):
        """
        Run biDSPOT on the stream
        
        Parameters
        ----------
        with_alarm : bool
		    (default = True) If False, SPOT will adapt the threshold assuming \
            there is no abnormal values


        Returns
        ----------
        dict
            keys : 'upper_thresholds', 'lower_thresholds' and 'alarms'
            
            '***-thresholds' contains the extreme quantiles and 'alarms' contains \
            the indexes of the values which have triggered alarms
            
        """
        if (self.n > self.init_data.size):
            print('Warning : the algorithm seems to have already been run, you \
            should initialize before running again')
            return {}

        # actual normal window
        W = self.init_data[-self.depth:]

        # list of the thresholds
        thup = []
        thdown = []
        alarm = []
        # Loop over the stream
        for i in tqdm.tqdm(range(self.data.size)):
            Mi = W.mean()
            Ni = self.data[i] - Mi
            # If the observed value exceeds the current threshold (alarm case)
            if Ni > self.extreme_quantile['up']:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks['up'] = np.append(self.peaks['up'], Ni - self.init_threshold['up'])
                    self.Nt['up'] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw('up')
                    self.extreme_quantile['up'] = self._quantile('up', g, s)
                    W = np.append(W[1:], self.data[i])

            # case where the value exceeds the initial threshold but not the alarm ones
            elif Ni > self.init_threshold['up']:
                # we add it in the peaks
                self.peaks['up'] = np.append(self.peaks['up'], Ni - self.init_threshold['up'])
                self.Nt['up'] += 1
                self.n += 1
                # and we update the thresholds
                g, s, l = self._grimshaw('up')
                self.extreme_quantile['up'] = self._quantile('up', g, s)
                W = np.append(W[1:], self.data[i])

            elif Ni < self.extreme_quantile['down']:
                # if we want to alarm, we put it in the alarm list
                if with_alarm:
                    alarm.append(i)
                # otherwise we add it in the peaks
                else:
                    self.peaks['down'] = np.append(self.peaks['down'], -(Ni - self.init_threshold['down']))
                    self.Nt['down'] += 1
                    self.n += 1
                    # and we update the thresholds

                    g, s, l = self._grimshaw('down')
                    self.extreme_quantile['down'] = self._quantile('down', g, s)
                    W = np.append(W[1:], self.data[i])

            # case where the value exceeds the initial threshold but not the alarm ones
            elif Ni < self.init_threshold['down']:
                # we add it in the peaks
                self.peaks['down'] = np.append(self.peaks['down'], -(Ni - self.init_threshold['down']))
                self.Nt['down'] += 1
                self.n += 1
                # and we update the thresholds

                g, s, l = self._grimshaw('down')
                self.extreme_quantile['down'] = self._quantile('down', g, s)
                W = np.append(W[1:], self.data[i])
            else:
                self.n += 1
                W = np.append(W[1:], self.data[i])

            thup.append(self.extreme_quantile['up'] + Mi)  # upper thresholds record
            thdown.append(self.extreme_quantile['down'] + Mi)  # lower thresholds record

        return {'upper_thresholds': thup, 'lower_thresholds': thdown, 'alarms': alarm}

    def plot(self, run_results, with_alarm=True):
        """
        Plot the results given by the run
        
        Parameters
        ----------
        run_results : dict
            results given by the 'run' method
        with_alarm : bool
		    (default = True) If True, alarms are plotted.


        Returns
        ----------
        list
            list of the plots
            
        """
        x = range(self.data.size)
        K = run_results.keys()

        ts_fig, = plt.plot(x, self.data, color=air_force_blue)
        fig = [ts_fig]

        if 'upper_thresholds' in K:
            thup = run_results['upper_thresholds']
            uth_fig, = plt.plot(x, thup, color=deep_saffron, lw=2, ls='dashed')
            fig.append(uth_fig)

        if 'lower_thresholds' in K:
            thdown = run_results['lower_thresholds']
            lth_fig, = plt.plot(x, thdown, color=deep_saffron, lw=2, ls='dashed')
            fig.append(lth_fig)

        if with_alarm and ('alarms' in K):
            alarm = run_results['alarms']
            if len(alarm) > 0:
                al_fig = plt.scatter(alarm, self.data[alarm], color='red')
                fig.append(al_fig)

        plt.xlim((0, self.data.size))

        return fig
