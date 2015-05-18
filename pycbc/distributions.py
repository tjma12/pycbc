# Copyright (C) 2015  Collin Capano
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

"""
This module provides classes to create injection distributions, and to convert
from one distribution to another.
"""

import lal
import numpy
from scipy import stats
import ConfigParser
from glue import segments


class CBCDistribution(object):
    """
    Base class to store basic information about a distribution of CBCs.
    """
    distribution_id = None
    name = None
    inspinj_name = None
    description = None
    _parameters = None
    _optional_parameters = None
    _bounds = None
    _norm = None

    @property
    def parameters(self):
        return self._parameters

    @property
    def optional_parameters(self):
        return self._optional_parameters

    @property
    def bounds(self):
        return self._bounds

    def min_bound(self, param):
        return self._bounds[param][0]

    def max_bound(self, param):
        return self._bounds[param][1]

    def set_bounds(self, **kwargs):
        self._bounds = {}
        for param in self._parameters:
            try:
                these_bounds = kwargs.pop(param)
            except KeyError:
                raise ValueError("Bound on parameter %s not specified!" %(
                    param))
            try:
                left, right = these_bounds
            except:
                raise ValueError("Must provide a minimum and a maximum for " +\
                    "parameter %s" %(param))
            if left is None:
                raise ValueError("No minimum specified for parameter %s" %(
                    param))
            if right is None:
                raise ValueError("No maximum specified for parameter %s" %(
                    param))
            self._bounds[param] = segments.segment(left, right)
        # set optional parameters
        for param in self._optional_parameters:
            try:
                these_bounds = kwargs.pop(param)
            except KeyError:
                continue
            try:
                left, right = these_bounds
            except:
                raise ValueError("Must provide a minimum and a maximum for " +\
                    "parameter %s" %(param))
            self._bounds[param] = segments.segment(left, right)
        if kwargs != {}:
            raise ValueError("Parameters %s " %(', '.join(kwargs.keys())) +\
                "are not parameters for this distribution")

    @property
    def norm(self):
        """
        Returns the normalization of this distribution.
        """
        return self._norm

    def _pdf(self, *args):
        raise ValueError("Density function not set!")

    def pdf(self, *args):
        """
        Returns the value of this distribution's density_func evaluated
        at the given arguments.
        """
        return self._pdf(*args)

    def pdf_from_result(self, result):
        """"
        Returns the value of this distribution's probability density function
        evaluated at the given result's parameters.

        Parameters
        ----------
        result: plot.Result instance
            An instance of a pycbc.plot.Result populated with the needed
            parameters.

        Returns
        -------
        pdf: float
            Value of the pdf at the given parameters.
        """
        args = [getattr(result, p) for p in self._parameters]
        return self.pdf(*args)



class UniformComponent(CBCDistribution):
    """
    A distribution that is uniform in component masses. Can optionally specify
    a cut on total mass.
    """
    name = 'uniform_component'
    inspinj_name = 'componentMass'
    description = 'uniform in component masses'
    _parameters = ['mass1', 'mass2']
    _optional_parameters = ['mtotal']

    def __init__(self, min_mass1, max_mass1, min_mass2, max_mass2,
            min_mtotal=None, max_mtotal=None):
        # ensure data types are correct
        min_mass1 = float(min_mass1)
        max_mass1 = float(max_mass1)
        min_mass2 = float(min_mass2)
        max_mass2 = float(max_mass2)
        if min_mtotal is not None:
            min_mtotal = float(min_mtotal)
        if max_mtotal is not None:
            max_mtotal = float(max_mtotal)
        bounds = {'mass1': (min_mass1, max_mass1),
            'mass2': (min_mass2, max_mass2)}
        if max_mtotal is not None or min_mtotal is not None:
            if min_mtotal is None:
                min_mtotal = min_mass1+min_mass2
            if max_mtotal is None:
                max_mtotal = max_mass1+max_mass2
            bounds.update({'mtotal': (min_mtotal, max_mtotal)})
        self.set_bounds(**bounds)
        # set the norm
        self.set_norm()

    def set_norm(self):
        """
        Calculates the norm of self given the boundaries; the result is
        stored to self._norm.
        """
        # norm depends on whether or not there is a total mass cut
        try:
            min_mtotal, max_mtotal = self.bounds['mtotal']
        except KeyError:
            min_mtotal = -numpy.inf
            max_mtotal = numpy.inf
        min_m1, max_m1 = self.bounds['mass1']
        min_m2, max_m2 = self.bounds['mass2']
        if max_mtotal < max_m1+max_m2:
            invnorm = _uniform_invnorm_mtotal_cut(min_m1, max_m1, min_m2,
                max_m2, max_mtotal)
        else:
            invnorm = abs(self.bounds['mass1'])*abs(self.bounds['mass2']) 
        if min_mtotal > min_m1+min_m2:
            # subtract out the area below the cut line
            invnorm -= _uniform_invnorm_mtotal_cut(min_m1, max_m1, min_m2,
                max_m2, min_mtotal)
        self._norm = 1./invnorm
        return self._norm

    def _pdf(self, mass1, mass2):
        if mass1 not in self.bounds['mass1']:
            return 0.
        if mass2 not in self.bounds['mass2']:
            return 0.
        try:
            if mass1+mass2 not in self.bounds['mtotal']:
                return 0.
        except KeyError:
            pass
        return self.norm

    def _rand_mass_func(self, min_val, max_val, num=1):
        """
        Returns a mass according to self's pdf. No cut bounds are applied.
        """
        return numpy.random.uniform(min_val, max_val, size=num)

    def rvs(self, num=1):
        """
        Returns 1 or more random variates drawn from this distribution. The
        bounds specified in self.bounds are applied.

        Parameters
        ----------
        num: {1 | int}
            Number of draws to do.
        
        Returns
        -------
        mass1: numpy.array
            The list of mass 1s.

        mass2: numpy.array
            The list of mass 2s. Note: may be larger than mass1.
        """
        min_m1, max_m1 = self.bounds['mass1']
        min_m2, max_m2 = self.bounds['mass2']
        masses = numpy.zeros((num, 2))
        # if there aren't any cuts, just create all the points at once
        if 'mtotal' not in self.bounds:
            masses[:,0] = self._rand_mass_func(min_m1, max_m1, num)
            masses[:,1] = self._rand_mass_func(min_m2, max_m2, num)
        else:
            ii = 0
            while ii < num:
                these_ms = self._rand_mass_func(min_m1, max_m1, 2)
                if these_ms.sum() not in self.bounds['mtotal']:
                    continue
                masses[ii,:] = these_ms
                ii += 1
        return masses[:,0], masses[:,1]

    @classmethod
    def load_from_database(cls, connection, process_id):
        """
        Given a connection to a database and a process_id of an inspinj job,
        loads all of the needed parameters.
        """
        cursor = connection.cursor()
        # first, check that the process_id belongs to an inspinj job with the
        # correct mass distribution
        mdistr = get_mdistr_from_database(connection, process_id)
        if mdistr != cls.inspinj_name:
            raise ValueError("given process_id does not match this " +
                "distribution")
        # now get everything we need for this distribution: min,max component
        # masses, plus any cuts on mass ratio and/or total mass
        relevant_parameters = ["--min-mass1", "--max-mass1", "--min-mass2",
            "--max-mass2", "--min-mtotal", "--max-mtotal", "--min-mratio",
            "--max-mratio"]
        sqlquery = """
            SELECT
                pp.param, pp.value
            FROM
                process_params AS pp
            WHERE
                pp.process_id == ? AND (
                %s)""" %(' OR\n'.join(['pp.param == "%s"' %(param) \
                    for param in relevant_parameters]))
        param_values = dict([ [param, None] for param in relevant_parameters])
        for param, value in cursor.execute(sqlquery, (process_id,)):
            param_values[param] = float(value)
        # create the class
        clsinst = cls(
            param_values["--min-mass1"], param_values["--max-mass1"],
            param_values["--min-mass2"], param_values["--max-mass2"],
            min_mtotal=param_values["--min-mtotal"],
            max_mtotal=param_values["--max-mtotal"])
        # check that no other parameters conflict
        if (param_values["--min-mratio"] is not None or 
                param_values["--max-mratio"] is not None):
            raise ValueError("Sorry, a cut on mass ratio is currently not " +\
                "supported.")

        return clsinst

#
#   Helper functions for finding the normalization of UniformComponent
#
def _uniform_rect_invnorm(x0, x1, y0, y1):
    """
    Returns the inverse of the normalization of a distribution that is uniform
    in the component masses, with bounds such that the shape is rectangular.
    """
    return (x1 - x0)*(y1 - y0)

def _uniform_triangle_invnorm(x0, x1, y0, y1):
    """
    Returns the inverse of the normalization of a distribution that is uniform
    in the component masses, with bounds such that the shape is triangular.
    """
    return _uniform_rect_invnorm(x0, x1, y0, y1)/2.

def _uniform_invnorm_mtotal_cut(min_m1, max_m1, min_m2, max_m2, mtotal_cut):
    """
    Computes the inverse norm of a distribution that is uniform in the
    component masses with a cut on total mass.
    """
    # aliases to make equations shorter
    a, b, c, d, M = min_m1, max_m1, min_m2, max_m2, mtotal_cut
    # there are four possible scenarios, depending on where the cut in total
    # mass is
    if M > d and M > b:
        # in this case, there are three regions, two rectangles and a triangle
        # Region I: rectangle
        invnorm = _uniform_rect_invnorm(a, M-d, c, d)
        # Region II: rectangle
        invnorm += _uniform_rect_invnorm(M-d, b, c, M-b)
        # Region III: triangle
        invnorm += _uniform_triangle_invnorm(M-d, b, M-b, d)
    if M <= d and M <= b:
        # in this case, there is only one region, a triangle
        invnorm = _uniform_triangle_invnorm(a, M-a, c, M-c)
    if d < M and M <= b:
        # in this case, there are two regions, a rectangle and a triangle
        # Region I: rectangle
        invnorm = _uniform_rect_invnorm(a, M-d, c, d)
        # Region II: triangle
        invnorm += _uniform_triangle_invnorm(M-d, M-a, c, d)
    if b < M and M <= d:
        # in this case, there is a rectangle and a triangle
        # Region I: rectangle
        invnorm = _uniform_rect_invnorm(a, b, c, M-b)
        # Region II: triangle
        invnorm += _uniform_triangle_invnorm(a, b, M-b, M-c)
    return invnorm



class LogComponent(UniformComponent):
    """
    A distribution that is uniform in the log of the component masses. This is
    the same as UniformComponent, except that the normalization and
    _rand_mass_funcs are changed.
    """
    name = 'log_component'
    inspinj_name = 'log'
    description = 'uniform in the log of the component masses'

    def __init__(self, min_mass1, max_mass1, min_mass2, max_mass2,
            min_mtotal=None, max_mtotal=None):
        super(LogComponent, self).__init__(min_mass1, max_mass1, min_mass2,
            max_mass2, min_mtotal=min_mtotal, max_mtotal=max_mtotal)
        # override the normalization
        self.set_norm()

    def set_norm(self):
        """
        Calculates the norm of self given the boundaries; the result is
        stored to self._norm.
        """
        try:
            min_mtotal, max_mtotal = self.bounds['mtotal']
        except KeyError:
            min_mtotal = -numpy.inf
            max_mtotal = numpy.inf
        min_m1, max_m1 = self.bounds['mass1']
        min_m2, max_m2 = self.bounds['mass2']
        if max_mtotal < max_m1+max_m2:
            invnorm = _loguniform_invnorm_mtotal_cut(min_m1, max_m1, min_m2,
                max_m2, max_mtotal)
        else:
            invnorm = numpy.log(max_m1/min_m1)*numpy.log(max_m2/min_m2)
        if min_mtotal > min_m1+min_m2:
            # subtract out the area below the cut line
            invnorm -= _loguniform_invnorm_mtotal_cut(min_m1, max_m1, min_m2,
                max_m2, min_mtotal)
        self._norm = 1./invnorm
        return self._norm

    def _rand_mass_func(self, min_val, max_val, num=1):
        """
        Returns a mass according to self's pdf. No cut bounds are applied.
        """
        return min_val * (float(max_val)/min_val)**numpy.random.uniform(0., 1.,
            size=num)


#
#   Helper functions for finding the normalization of LogComponents
#

def polylog_term(z, s, k):
    """
    Returns the kth term in the polylog power series.
    """
    return float(z)**k / k**s

def polylog(z, s, Nterms=1e4):
    """
    Returns the polylogarithm of order s and argument z. This is given by the
    power series \sum_{k=1}^{\infty} z^k/k^s, where |z| <= 1. See
    en.wikipedia.org/wiki/Polylogarithm for more details.

    Parameters
    ----------
    z: float
        The argument of the polylogarithm; abs(z) must be <= 1.
    s: float
        The order of the polylogarithm.
    Nterms: {1e4 | int}
        The number of terms to use in the power series. Default is 1e4.

    Returns
    -------
    x: float
        The value of the polylog.
    """
    if abs(z) > 1:
        raise ValueError("z must be <= 1.")
    ks = numpy.arange(Nterms)+1
    return polylog_term(z, s, ks).sum()


def _loguniform_rect_invnorm(x0, x1, y0, y1):
    """
    Returns the inverse of the normalization of a distribution that is uniform
    in the log of the component masses, with bounds such that the shape is
    rectangular.
    """
    return numpy.log(x1/x0)*numpy.log(y1/y0)

def _loguniform_triangle_invnorm(x0, x1, y0, y1):
    """
    Returns the inverse of the normalization of a distribution that is uniform
    in the log of the component masses, with bounds such that the shape is
    a right triangle with the equation for the hypotenuse = x + y = constant.
    """
    # the hyptoneuse
    M = x0 + y1
    if M != x1 + y0:
        raise ValueError("x0 + y1 != x0 + y1")
    return numpy.log(M)*numpy.log(x1/x0) - numpy.log(y0)*numpy.log(x1/x0) \
        + polylog(x0/M, 2) - polylog(x1/M, 2)

def _loguniform_invnorm_mtotal_cut(min_m1, max_m1, min_m2, max_m2, mtotal_cut):
    """
    Computes the inverse norm of a distribution that is uniform in the log of
    the component masses with a cut on total mass.
    """
    # aliases to make equations shorter
    a, b, c, d, M = min_m1, max_m1, min_m2, max_m2, mtotal_cut
    # there are four possible scenarios, depending on where the cut in total
    # mass is
    if M > d and M > b:
        # in this case, there are three regions, two rectangles and a triangle
        # Region I: rectangle
        invnorm = _loguniform_rect_invnorm(a, M-d, c, d)
        # Region II: rectangle
        invnorm += _loguniform_rect_invnorm(M-d, b, c, M-b)
        # Region III: triangle
        invnorm += _loguniform_triangle_invnorm(M-d, b, M-b, d)
    if M <= d and M <= b:
        # in this case, there is only one region, a triangle
        invnorm = _loguniform_triangle_invnorm(a, M-a, c, M-c)
    if d < M and M <= b:
        # in this case, there are two regions, a rectangle and a triangle
        # Region I: rectangle
        invnorm = _loguniform_rect_invnorm(a, M-d, c, d)
        # Region II: triangle
        invnorm += _loguniform_triangle_invnorm(M-d, M-a, c, d)
    if b < M and M <= d:
        # in this case, there is a rectangle and a triangle
        # Region I: rectangle
        invnorm = _loguniform_rect_invnorm(a, b, c, M-b)
        # Region II: triangle
        invnorm += _loguniform_triangle_invnorm(a, b, M-b, M-c)
    return invnorm


class UniformMq(CBCDistribution):
    """
    A distribution that is uniform in total mass and mass ratio, with cuts in
    m1 and m2.
    """
    name = 'uniform_Mq'
    inspinj_name = None
    description = 'uniform in M,q (q >= 1) with a cut on component masses'
    _parameters = ['mtotal', 'q', 'mass1', 'mass2']
    _optional_parameters = []

    def __init__(self, min_mtotal, max_mtotal, min_q, max_q, min_mass1,
            max_mass1, min_mass2, max_mass2):
        # ensure data types are correct
        min_mass1 = float(min_mass1)
        max_mass1 = float(max_mass1)
        min_mass2 = float(min_mass2)
        max_mass2 = float(max_mass2)
        min_mtotal = float(min_mtotal)
        max_mtotal = float(max_mtotal)
        min_q = float(min_q)
        max_q = float(max_q)
        bounds = {
            'mtotal': (min_mtotal, max_mtotal),
            'q': (min_q, max_q),
            'mass1': (min_mass1, max_mass1),
            'mass2': (min_mass2, max_mass2)}
        self.set_bounds(**bounds)
        # set the norm
        self.set_norm()

    def set_norm(self, Ntestpts=1e6):
        """
        Calculates the norm of self given the boundaries; the result is
        stored to self._norm.
        """
        # we'll just do this numerically
        # draw M,q
        Ms = numpy.random.uniform(self.min_bound('mtotal'),
            self.max_bound('mtotal'), size=Ntestpts)
        qs = numpy.random.uniform(self.min_bound('q'),
            self.max_bound('q'), size=Ntestpts)
        m2s = Ms/(1.+qs)
        m1s = m2s * Ms
        outside_idx = numpy.where(numpy.logical_or(
            numpy.logical_or(m1s < self.min_bound('mass1'), 
                             m1s >= self.max_bound('mass1')),
            numpy.logical_or(m2s < self.min_bound('mass2'),
                             m2s >= self.max_bound('mass2')))
            )[0]
        frac_in = (Ntestpts - len(outside_idx))/float(Ntestpts)
        inv_norm = abs(self.bounds['mtotal'])*abs(self.bounds['q'])*frac_in
        self._norm = 1./inv_norm
        return self._norm

    def _pdf(self, M, q, mass1, mass2):
        if M not in self.bounds['mtotal']:
            return 0.
        if q not in self.bounds['q']:
            return 0.
        #mass2 = M/(1.+q)
        #mass1 = q*mass2
        if mass1 not in self.bounds['mass1']:
            return 0.
        if mass2 not in self.bounds['mass2']:
            return 0.
        return self.norm


class DominikEtAl2012(CBCDistribution):
    """
    A distribution derived from one of the priors in Dominik et al. 2012.

    These priors are defined in the rest-frame mass, rather than the observed
    mass. For this reason, when using pdf_from_result, the masses are
    are automatically de-red shifted to get the correct result.
    """
    #_parameters = ['mchirp', 'eta']
    _parameters = ['mass1', 'mass2']
    _optional_parameters = []
    name = 'dominik_etal_2012'
    inspinj_name = None

    def __init__(self, filename, description):
        # set the name and description
        self.description = description
        # load the filename and the kernel
        self.filename = filename
        self._kernel, m1s, m2s = _get_data_and_kernel(filename)
        m1s_min, m1s_max = m1s.min(), m1s.max()
        m2s_min, m2s_max = m2s.min(), m2s.max()
        #self._kernel, mchirps, etas = _get_data_and_kernel(filename)
        #mchirp_min, mchirp_max = mchirps.min(), mchirps.max()
        #eta_min, eta_max = etas.min(), 0.25
        # we'll bump up/down the chirp mass bounds slightly
        bounds = {'mass1': (0.8*m1s_min, 1.2*m1s_max),
            'mass2': (0.8*m2s_min, 1.2*m2s_max)}
        #bounds = {'mchirp': (0.8*mchirp_min, 1.2*mchirp_max),
        #          'eta': (0., 0.25)
        #          }
        self.set_bounds(**bounds)
        # calculate the norm
        #norm = self._kernel.integrate_box(
        #    (self.bounds['mchirp'][0], self.bounds['eta'][0]),
        #    (self.bounds['mchirp'][1], self.bounds['eta'][1]))
        norm = self._kernel.integrate_box(
            (self.bounds['mass1'][0], self.bounds['mass2'][0]),
            (self.bounds['mass1'][1], self.bounds['mass2'][1]))
        self._norm = norm

    #def _pdf(self, mchirp, eta):
    #    return self._norm*self._kernel.evaluate((mchirp, eta))[0]
    def _pdf(self, mass1, mass2):
        return self._norm*self._kernel.evaluate((mass1, mass2))[0]

    def pdf_from_result(self, result):
        """"
        Returns the value of this distribution's probability density function
        evaluated at the given result's parameters. Overrides parent's class
        so that the observed chirp mass can be de-redshifted prior to calling
        self's _pdf.

        Parameters
        ----------
        result: plot.Result instance
            An instance of a pycbc.plot.Result populated with the needed
            parameters.

        Returns
        -------
        pdf: float
            Value of the pdf at the given parameters.
        """
        source_mass1 = get_source_mass(result.mass1,
            get_proper_distance(result.distance))
        source_mass2 = get_source_mass(result.mass2,
            get_proper_distance(result.distance))

        return (1.+get_redshift(result.distance))**2. * \
            self._pdf(source_mass1, source_mass2)
        #source_mchirp = get_source_mass(result.mchirp,
        #    get_proper_distance(result.distance))
        #return (1.+get_redshift(result.distance))**2. * \
        #    self._pdf(source_mchirp, result.eta)


# helper functions
def _load_dominik2012_model(filename):
    """
    Loads masses from a Dominik et al. 2012 .dat files.
    """
    # we just want the masses, which are stored in the first two columns
    data = numpy.loadtxt(filename, usecols=(0,1))
    m1s = data[:,0]
    m2s = data[:,1]
    m1s, m2s = _enforce_m1gtm2(m1s, m2s)
    return m1s, m2s

def _enforce_m1gtm2(m1s, m2s):
    """
    Given two lists of masses, ensures that m1 > m2.
    This is done in place.
    """
    replace_idx = numpy.where(m1s < m2s)
    smaller_masses = m1s[replace_idx]
    m1s[replace_idx] = m2s[replace_idx]
    m2s[replace_idx] = smaller_masses
    return m1s, m2s

def _get_data_and_kernel(filename):
    """
    Loads masses from a Dominik et al. 2012 .dat file and applies a Gaussian
    kernel density estimator (KDE) on the chirp mass and eta.
    """
    m1s, m2s = _load_dominik2012_model(filename)
    dataset = numpy.vstack((m1s, m2s))
    #Ms = m1s + m2s
    #etas = m1s*m2s/Ms**2.
    #mchirps = etas**(3./5)*Ms
    #dataset = numpy.vstack((mchirps, etas))
    kernel = stats.gaussian_kde(dataset)
    return kernel, m1s, m2s#mchirps, etas

# default hubble constant comes from arXiv:1212.5225
default_H0 = 69.32 # km/s/Mpc

def get_redshift(distance, hubble_const = None):
    """
    Gives the redshift corresponding
    to the distance. Distance should be in Mpc.
    The hubble_const should be in km/s/Mpc, if None
    the default_H0 will be used.
    """
    if hubble_const is None:
        hubble_const = default_H0
    return distance * hubble_const * 1e3 / lal.C_SI

def get_source_mass(redshifted_mass, distance, hubble_const = None):
    """
    Distance should be in Mpc.
    """
    return redshifted_mass/(1 + get_redshift(distance, hubble_const))

def get_proper_distance(distance, hubble_const = None):
    """
    Distance should be in Mpc.
    """
    return distance / (1 + get_redshift(distance, hubble_const))


class SubmodelAOCEFit(CBCDistribution):
    """
    A distribution based on a fit to the Submodel A with optimisitic common
    envelope evolution in Dominik et al. 2012, 2014. The distribution in
    chirp mass is given by:
        p(M) = A * M**alpha * exp(-beta * M),
    where:
        A = 6.58166e-6
        alpha = 5.25954242624
        beta = 0.343162134711
    In mass ratio (here, defined on (0, 1]):
       p(q) = a * exp(b*q),
    where:
       a = 4.4748490112078417e-5
       b = 12.5436728009
    """
    name = 'submodel_a_ocefit'
    inspinj_name = None
    description = 'Fitted Submodel A with optimisitc common envelope evolution'
    _parameters = ['mchirp', 'inv_q']
    _optional_parameters = []

    def __init__(self):
        bounds = {'mchirp': (0., numpy.inf),
            'inv_q': (0., 1.)}
        self.set_bounds(**bounds)
        # mchirp parameters
        self.A = 6.58166e-6
        self.alpha = 5.25954242624
        self.beta = 0.343162134711
        # q parameters
        self.a = 4.4748490112078417e-5
        self.b = 12.5436728009
        # set the norm
        self.set_norm()

    def set_norm(self):
        """
        Calculates the norm of self; the result is
        stored to self._norm.
        """
        self._norm = self.A * self.a
        return self._norm

    def _pdf(self, mchirp, inv_q):
        return self.norm * mchirp**self.alpha * numpy.exp(-self.beta * mchirp)\
            * numpy.exp(self.b * inv_q)

    def pdf_from_result(self, result):
        """"
        Returns the value of this distribution's probability density function
        evaluated at the given result's parameters. Overrides parent's class
        so that it uses mass1, mass2 from the result.

        Parameters
        ----------
        result: plot.Result instance
            An instance of a pycbc.plot.Result populated with the needed
            parameters.

        Returns
        -------
        pdf: float
            Value of the pdf at the given parameters.
        """
        return self._pdf(result.mchirp, 1./result.q)

        

# The known distributions
distributions = {
    UniformComponent.name: UniformComponent,
    LogComponent.name: LogComponent,
    UniformMq.name: UniformMq,
    SubmodelAOCEFit.name: SubmodelAOCEFit,
    DominikEtAl2012.name: DominikEtAl2012
    }

# Mapping from names used by inspinj to distributions defined here
inspinj_map = {
    UniformComponent.inspinj_name: UniformComponent,
    LogComponent.inspinj_name: LogComponent
    }

#
#
#   Utilities for retrieving a distribution from files
#
def get_mdistr_from_database(connection, process_id):
    """
    Gets the mass distribution of the given process_id.
    """
    cursor = connection.cursor()
    # first, check that the process_id belongs to an inspinj job with the
    # correct mass distribution
    sqlquery = """
        SELECT
            pp.value
        FROM
            process_params AS pp
        WHERE
            pp.process_id == ? AND
            pp.param == "--m-distr"
        """
    return cursor.execute(sqlquery, (process_id,)).fetchone()[0]


def get_inspinj_distribution(connection, process_id):
    """
    Given the process_id of an inspinj job, retrieves all the information
    from the process params to create a distribution.
    """
    # get the distribution type
    mdistr = get_mdistr_from_database(connection, process_id)
    try:
        distribution = inspinj_map[mdistr]
    except KeyError:
        raise ValueError("Sorry, mass distribution %s " %(mdistr) +\
            "is currently unsupported.")
    return distribution.load_from_database(connection, process_id)


def load_distribution_from_config(config_file):
    """
    Load a distribution from the given configuration file. The configuration
    file should contain exactly one section, the name of which is the name
    of the distribution to create. The options in the section should give
    the arguments needed to initialize the distribution. For example, the 
    following would create a UniformComponent distribution:
    ::
        [uniform_component]
        min_mass1 = 2.
        max_mass1 = 48.
        min_mass2 = 2.
        max_mass2 = 48.
        max_mtotal = 50.

    Parameters
    ----------
    config_file: str
        Name of the configuration file.

    Returns
    -------
    distribution: CBCDistribution instance
        An instance of the desired distribution.
    """
    cp = ConfigParser.ConfigParser()
    cp.read(config_file)
    if len(cp.sections()) != 1:
        raise ValueError("distribution config files should only specify " +\
            "one distribution; found %i" %(len(cp.sections())))
    dist_name = cp.sections()[0]
    if dist_name not in distributions:
        raise ValueError("unrecognized distribution %s; " %(dist_name) +\
            "options are %s" %(', '.join(distributions.keys())))
    # get the parameters specified
    parameters = dict([ [param, cp.get(dist_name, param)] \
        for param in cp.options(dist_name)])
    # create the distribution
    return distributions[dist_name](**parameters)


#
#
#   Utilities for converting between distributions
#
#
def identity(result):
    """
    Returns 1., which is the Jacobian when converting between the same
    distribution.
    """
    return 1.

def logComponent_to_uniformComponent(result):
    """
    Returns the Jacobian needed to convert from a distribution that is uniform
    in the log of the component masses to a distribution that is uniform
    in the component masses.
    """
    return result.m1 * result.m2

def uniformComponent_to_uniformMq(result):
    """
    Returns the Jacobian needed to convert from uniform in m1,m2 to uniform
    in M,q.
    """
    return (result.m1**2. + result.m2**2.) / (result.m1*result.m2**2.)

def uniformComponent_to_uniformMchirpEta(result):
    """
    Returns the Jacobian needed to convert from a distribution that is
    uniform in the component masses to a distribution that is uniform in
    chirp mass and eta.
    """
    return result.mass2**(3./5)/(result.mtotal**(1./5) * result.mass1**(7./5))

def uniformComponent_to_MchirpEta(result):
    """
    Returns the Jacobian needed to convert from a distribution that is uniform
    in the component masses to a distribution that is uniform in chirp mass and
    symmetric mass ratio (eta).
    """
    return result.mass2**(3./5)/(result.mtotal**(1./5) * result.mass1**(7./5))

def logComponent_to_MchirpEta(result):
    """
    Returns the Jacobian needed to convert from a distribution that is log in
    the component masses to one that is uniform in chirp mass and eta.
    """
    return logComponent_to_uniformComponent(result)*\
            uniformComponent_to_MchirpEta(result)

# the known Jacobians
Jacobians = {
    ("log_component", "uniform_component"): logComponent_to_uniformComponent,
    ("uniform_component", "uniform_Mq"): uniformComponent_to_uniformMq,
    ("uniform_component", "dominik_etal_2012"): identity,#uniformComponent_to_MchirpEta,
    ("log_component", "dominik_etal_2012"): logComponent_to_uniformComponent,#MchirpEta,
    # identities
    ("log_component", "log_component"): identity,
    ("uniform_component", "uniform_component"): identity,
    ("dominik_etal_2012", "dominik_etal_2012"): identity,
    }

def convert_distribution(result, from_distr, to_distr):
    """
    Given a result instance, converts from one distribution to another.
    """
    # get the jacobian weight
    try:
        jacobian = Jacobians[from_distr.name, to_distr.name](result)
    except KeyError:
        raise ValueError("I don't know how to convert from %s to %s" %(
            from_distr.name, to_distr.name))
    return jacobian * to_distr.pdf_from_result(result) / \
            from_distr.pdf_from_result(result)
