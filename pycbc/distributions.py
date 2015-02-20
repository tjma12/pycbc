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

import numpy
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
        
        

# The known distributions
distributions = {
    UniformComponent.name: UniformComponent,
    LogComponent.name: LogComponent
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
    the bounds of the parameters of the distribution. For example, the 
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
    parameters = dict([ [param, float(cp.get(dist_name, param))] \
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
    in the log of the component masses to the a distribution that is uniform
    in the component masses.
    """
    return result.m1 * result.m2

# the known Jacobians
Jacobians = {
    ("log_component", "uniform_component"): logComponent_to_uniformComponent,
    # identities
    ("log_component", "log_component"): identity,
    ("uniform_component", "uniform_component"): identity,
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
