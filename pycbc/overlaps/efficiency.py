#! /usr/bin/env python

import numpy
import operator
import copy
import ConfigParser
from glue import segments

from pycbc.plot import plot_utils

#
#
#   Utilities for calculating single-detector efficiency and sensitivity
#

# for conversions
MpcToGpc = 1e-3

def isfound(result, ranking_stat, compare_operator, threshold):
    """
    Determins whether an injection is found.

    Parameters
    ----------
    result: plot_utils.Result
        Result class instance representing an injection.
    ranking_stat: str
        What to use for the ranking. Must be a parameter of plot_utils.Result.
    compare_operator: operator function
        The comparison operator to use. For example, if this is operator.ge,
        then result's ranking_stat must be >= threshold to be found.
    threshold: float
        The threshold to use.

    Returns
    -------
    isfound: bool
    """
    return compare_operator(getattr(result, ranking_stat), threshold)


class PHyperCube:
    def __init__(self, bounds={}):
        self.bounds = {}
        self.set_bounds(**bounds)
        # values unique to each set
        self.data = None
        self.threshold = None
        self.efficiencies = []
        self.eff_err_low = []
        self.eff_err_high = []
        self.nsamples = []
        self.integrated_eff = []
        self.integrated_err_low = []
        self.integrated_err_high = []
        self.distances = []
        self.d_distr = None

    def set_bound(self, param, lower, upper):
        self.bounds[param] = segments.segment(lower, upper)

    def get_bound(self, param):
        return self.bounds[param]

    def set_bounds(self, **kwargs):
        for param,(lower, upper) in kwargs.values():
            self.set_bound(param, lower, upper)

    def get_bounds(self):
        return self.bounds

    def central_value(self, param):
        param_range = self.get_bound(param)
        return param_range[0] + abs(param_range)/2.

    def set_data(self, data):
        """
        Given some data, extracts the points that are relevant to this cube.
        """
        self.data = plot_utils.slice_results(data, self.bounds)

    def calculate_efficiency(self, threshold,
            ranking_stat='new_snr', rank_by='max', 
            dist_func='dist', nbins=20, d_distr='log10',
            min_dist=None, max_dist=None):
        if rank_by == 'max':
            compare_operator = operator.ge
        elif rank_by == 'min':
            compare_operator = operator.lt
        else:
            raise ValueError("unrecognized rank_by argument %s; " %(rank_by) +\
                "options are 'max' or 'min'")
        if max_dist is None:
            max_dist = getattr(max(self.data,
                key=lambda x: getattr(x, dist_func)), dist_func)
        if min_dist is None:
            min_dist = getattr(min(self.data,
                key=lambda x: getattr(x, dist_func)), dist_func)
        if d_distr == 'log10':
            dist_bins = numpy.logspace(numpy.log10(min_dist),
                numpy.log10(max_dist), num=nbins)
        elif d_distr == 'linear':
            dist_bins = numpy.linspace(min_dist, max_dist, num=nbins)
        else:
            raise ValueError, "unrecognized distribution %s" % d_distr

        dist_bins[-1] *= 1.01
        
        # calculate the efficiency in each distance bin
        numFound = numpy.zeros(len(dist_bins)-1)
        numMissed = numpy.zeros(len(dist_bins)-1)
        for kk,dbin in enumerate(dist_bins[:-1]):
            nextbin = dist_bins[kk+1]
            thisDgroup = plotUtils.apply_cut(self.data[group],
                {dist_func: (dbin, nextbin)})
            numFound[kk] = float(len([thisRes for thisRes in thisDgroup \
                if isfound(thisRes, ranking_stat, compare_operator, threshold)
                ]))
            numMissed[kk] = float(len(thisDgroup) - numFound[kk])
        
        numTotal = numFound + numMissed
        # if some bins have no injections in them
        # we'll combine with adjacent bins
        nzidx = numpy.nonzero(numTotal)
        numTotal = numTotal[nzidx]
        numFound = numFound[nzidx]
        numMissed = numMissed[nzidx]

        thisEff = numFound / numTotal 

        # the error in the efficiency comes from eqn. 1.9 in J.T. Whalen's 
        # note cbc/protected/review/notes/poisson_errors.pdf in the 
        # ligo/virgo cvs
        a = numTotal*(2.*numFound + 1.)
        b = numpy.sqrt(4.*numTotal*numFound*(numTotal - numFound) + \
                numTotal**2)
        c = 2*numTotal*(numTotal + 1)
        effPlus = (a + b)/c
        effMinus = (a - b)/c
        
        effPlus = effPlus - thisEff
        effMinus = thisEff - effMinus

        # save
        self.nsamples = numTotal
        self.threshold = threshold
        self.efficiencies = thisEff
        self.eff_err_low = effMinus
        self.eff_err_high = effPlus
        self.distances = numpy.array(dist_bins[nzidx].tolist() + \
            [dist_bins[-1]])
        self.d_distr = d_distr


    def integrate_efficiencies(self):
        if self.efficiencies == []:
            raise ValueError("self.efficiencies is empty! " +\
                "Run self.calculate_efficiency first")
        if self.d_distr == 'linear':
            this_dist = self.distances[:-1] + numpy.diff(self.distances)/2.
            dr = numpy.diff(self.distances)
            self.integrated_eff = 4*numpy.pi*(
                    self.efficiencies * this_dist**2 * dr).sum() + \
                (4./3)*numpy.pi*self.distances[0]**3
            # error
            self.integrated_err_low = 4*numpy.pi*numpy.sqrt(((
                self.eff_err_low * this_dist**2 * dr)**2).sum())
            self.integrated_err_high[group] = 4*numpy.pi*numpy.sqrt(((
                self.eff_err_high * this_dist**2 * dr)**2).sum())
        elif self.d_distr == 'log10':
            this_dist = 10**(numpy.log10(self.distances[:-1]) + \
                numpy.diff(numpy.log10(self.distances))/2)
            # in this case we use:
            # int(eff) = \int eff(r) r**2 \dr =
            #   ln10 \int eff(log10 r) r**3 \dlog10r
            dr = numpy.diff(numpy.log10(self.distances))
            self.integrated_eff = numpy.log(10)*(4.*numpy.pi * \
                self.efficiencies * this_dist**3 * dr).sum() + \
                4*numpy.pi*(1./3)*self.distances[0]**3
            # error
            # note that in the following we're assuming the error in the
            # efficiency below and above the measured region is 0
            self.integrated_err_low = 4.*numpy.pi*numpy.sqrt((( \
                self.eff_err_low * this_dist**3 * dr)**2).sum())
            self.integrated_err_high = 4.*numpy.pi*numpy.sqrt((( \
                self.eff_err_high * this_dist**3 * dr)**2).sum())
        else:
            raise ValueError('unrecognized distance distribution %s' %(
                self.d_distr))


class PHyperCubeGain:
    """
    Class to store information about a "tile" in parameter space. A tile is
    a two-dimensional slice in a higher dimensional parameter space that is
    bounded by some values. The class stores information about a "reference"
    and "test" set of results. When taking the ratio of sensitive volumes
    in the tile, the "reference" is the denominator and the "test" is the
    numerator.
    """
    def __init__(self, bounds={}):
        self.bounds = {}
        self.set_bounds(**bounds)
        # values unique to each set
        self.reference_cube = PHyperCube(bounds)
        self.test_cube = PHyperCube(bounds)
        # values common to both
        self.fractional_gain = 0.
        self.gain_err_low = 0.
        self.gain_err_high = 0.

    def set_bound(self, param, lower, upper):
        self.bound[param] = segments.segment(lower, upper)
        self.reference_cube.set_bound(param, lower, upper)
        self.test_cube.set_bound(param, lower, upper)

    def get_bound(self, param):
        return self.bounds[param]

    def set_bounds(self, **kwargs):
        for param,(lower, upper) in kwargs.values():
            self.set_bound(param, lower, upper)

    def get_bounds(self):
        return self.bounds

    def central_value(self, param):
        param_range = self.get_bound(param)
        return param_range[0] + abs(param_range)/2.

    def set_reference_data(self, data):
        self.reference_cube.set_data(data)

    def set_test_data(self, data):
        self.test_cube.set_data(data)

    def calculate_efficiencies(self, dist_func='dist', ranking_stat='new_snr',
            nbins=20, d_distr='log10'):
        max_dist = min(getattr(max(self.reference_cube.data,
                key=lambda x: getattr(x, dist_func)), dist_func),
            getattr(max(self.test_cube.data,
                key=lambda x: getattr(x, dist_func)), dist_func))
        min_dist = max(getattr(min(self.reference_cube.data,
                key=lambda x: getattr(x, dist_func)), dist_func),
            getattr(min(self.test_cube.data,
                key=lambda x: getattr(x, dist_func)), dist_func))
        self.reference_cube.calculate_efficiencies(dist_func=dist_func,
            ranking_stat=ranking_stat, nbins=nbins, d_distr=d_distr)
        self.test_cube.calculate_efficiencies(dist_func=dist_func,
            ranking_stat=ranking_stat, nbins=nbins, d_distr=d_distr)

    def integrate_efficiencies(self, d_distr='log10'):
        self.reference_cube.integrate_efficiences(d_distr=d_distr)
        self.test_cube.integrate_efficiences(d_distr=d_distr)
        
    def calculate_fractional_gain(self):
        self.fractional_gain = self.test_cube.integrated_eff / \
            self.reference_cube.integrated_eff
        # error
        self.gain_err_low = self.fractional_gain * \
            numpy.sqrt((self.test_cube.integrated_err_low /\
                self.test_cube.integrated_eff)**2. + \
            (self.reference_cube.integrated_err_high /\
                self.reference_cube.integrated_eff)**2.)
        self.gain_err_high = self.fractional_gain * \
            numpy.sqrt((self.test_cube.integrated_err_high /\
                self.test_cube.integrated_eff)**2. + \
            (self.reference_cube.integrated_err_low / \
                self.reference_cube.integrated_eff)**2.)
        return self.fractional_gain, self.gain_err_high, self.gain_err_low


class Layer:
    """
    Class to store and organize PHyperCubes, as well as other layers.
    """
    def __init__(self, level, x_arg, y_arg, x_label='', y_label='',
            cube_type='single'):
        self.level = level
        self.x_arg = x_arg
        self.y_arg = y_arg
        self.x_label = x_label
        self.y_label = y_label
        self._x_bins = segments.segmentlist([])
        self._y_bins = segments.segmentlist([])
        self._x_distr = None
        self._y_distr = None
        self._super_layer = None
        self._sub_layer = None
        self._phyper_cubes = []
        self.set_cube_type(cube_type)

    def create_bins(self, x_or_y, min, max, num, distr):
        """
        Creates the bins to use.

        Parameters
        ----------
        x_or_y: str {'x'|'y'}
            Which axis to set the bins for.
        min: float
            The minimum value to use.
        max: float
            The maximum value to use.
        num: int
            The number of bins to create.
        distr: str {'linear'|'log10'}
            How to distribute the bins between the min and maximum values.
        """
        if distr == 'log10':
            bins = numpy.logspace(numpy.log10(min), numpy.log10(max), num=num)
        elif distr == 'linear':
            bins = numpy.linspace(min, max, num=num)
        else:
            raise ValueError("unrecognized distribution %s; " %(distr) +\
                "options are 'linear' or 'log10'")
        # the actual bins will be a segments list
        boundslist = segments.segmentlist([])
        for ii,lower in enumerate(bins[:-1]):
            upper = bins[ii+1]
            boundslist.append(segments.segment(lower, upper))
        # set
        setattr(self, '_%s_bins' % x_or_y, boundslist)
        setattr(self, '_%s_distr' % x_or_y, distr)

    @property
    def x_lims(self):
        return self._x_bins.extent()

    @property
    def y_lims(self):
        return self._y_bins.extent()

    @property
    def x_nbins(self):
        return len(self._x_bins)

    @property
    def y_nbins(self):
        return len(self._y_bins)

    @property
    def x_distr(self):
        return self._x_distr

    @property
    def y_distr(self):
        return self._y_distr

    @property
    def cube_type(self):
        return self._cube_type

    def set_cube_type(self, cube_type):
        if cube_type == 'single':
            self._cube_type = PHyperCube
        elif cube_type == 'gain':
            self._cube_type = PHyperCubeGain
        else:
            raise ValueError("unrecognized cube_type %s; " %(cube_type) +\
                "must be either 'gain' or 'single'")

    @property
    def super_layer(self):
        return self._super_layer

    def set_super_layer(self, layer):
        """
        Sets the layer that is above this one.

        Parameters
        ----------
        layer: Layer instance
            The layer to set as being above this one. The level number must
            of layer must be > this layer's level.
        """
        if layer.level >= self.level:
            raise ValueError("given layer has a larger level number than self")
        if layer._cube_type != self._cube_type:
            raise ValueError("given layer has a different cube type than self")
        self._super_layer = layer


    @property
    def sub_layer(self):
        return self._sub_layer

    def set_sub_layer(self, layer):
        """
        Sets the layer that is below this one.

        Parameters
        ----------
        layer: Layer instance
            The layer to set as being above this one. The level number must
            of layer must be < this layer's level.
        """
        if layer.level <= self.level:
            raise ValueError("given layer has a smaller level number")
        if layer._cube_type != self._cube_type:
            raise ValueError("given layer has a different cube type than self")
        self._sub_layer = layer


    def create_phyper_cubes(self):
        """
        Creates all parameter hypercubes at this level. If this level is > 0,
        super_layers must be set.
        """
        # first, tile up this layer
        this_layer_tiles = [(xbin, ybin) for ybin in self._y_bins \
            for xbin in self._x_bins]
        # now, if there is a super layer, create a phyper cube for every tile
        # for every phyper cube in super layer
        if self.level > 0:
            if self._super_layer is None:
                raise ValueError("self.level is %i but super layer not set" %(
                    self.level))
            if self._super_layer._phyper_cubes is None:
                raise ValueError("super layer's phyper cubes have not been " +\
                    "created yet")
            for super_cube in self._super_layer._phyper_cubes:
                for (xbounds, ybounds) in this_layer_tiles:
                    these_bounds = copy.copy(super_cube.get_bounds())
                    these_bounds[self.x_arg] = xbounds
                    these_bounds[self.y_arg] = ybounds
                    this_cube = self._cube_type(these_bounds)
                    self._phyper_cubes.append(this_cube)
        else:
            # just create cubes out of the tiles on this layer
            for (xbounds, ybounds) in this_layer_tiles:
                these_bounds = {self.x_arg: xbounds, self.y_arg: ybounds}
                this_cube = self._cube_type(these_bounds)
                self._phyper_cubes.append(this_cube)

    @property
    def phyper_cubes(self):
        return self._phyper_cubes


    def set_cube_data(self, data, ref_or_test=None):
        """
        Bins the given data into self's phyper cubes. If self's _cube_type is
        PHyperCubeGain, must also specify whether the data is test or
        reference.

        Parameters
        ----------
        data: dict
            The data to slice.
        ref_or_test: {None, 'reference'|'test'}
            If self's _cube_type is PHyperCubeGain, specifies whether the given
            data is for the reference cubes or the test cubes.
        """
        if self._phyper_cubes == []:
            raise ValueError("phyper cubes not set; run create_phyper_cubes")
        if self._cube_type == PHyperCubeGain:
            if ref_or_test is None:
                raise ValueError("_cube_type is PHyperCubeGain, but " +\
                    "ref_or_test not set")
            if not (ref_or_test == 'reference' or ref_or_test == 'test'):
                raise ValueError("unrecognized option for ref_or_test %s" %(
                    ref_or_test))
            funcstr = '_' + ref_or_test

        else:
            if ref_or_test is not None:
                raise ValueError("ref_or_test is not None, but _cube_type "+\
                    "is PHyperCube")
            funcstr = ''
                 
        for this_cube in self._phyper_cubes:
            setattr(this_cube, 'set%s_data' %(funcstr), data)


    def set_cube_data_using_super(self):
        """
        Uses self's super layers to slice the data for each cube in self.
        """
        if self.level == 0:
            raise ValueError("This is level 0: no super-layers!")
        if self._cube_type == PHyperCubeGain:
            data_sets = ['reference_', 'test_']
        else:
            data_sets = ['']
        for dset in data_sets:
            [setattr(this_cube, 'set_%sdata' %(dset),
                getattr(super_cube, '%sdata'%(dset))) \
                for this_cube in self._phyper_cubes \
                # find the super cube that this_cube is a sub-space of
                for super_cube in self._super_layer._phyper_cubes \
                if all([super_cube.get_bound(param) == this_bound \
                        for (param, this_bound) in this_cube.bounds.values()])]
            # check that all cubes have been populated
            if any([getattr(this_cube, '%sdata' %(dset)) is None for \
                    this_cube in self._phyper_cubes]):
                raise ValueError("Not all cubes were populated; check bounds")


    def integrate_volumes(self):
        """
        Integrates the sensitive volumes of each phyper cube in self. When
        integrating, self's Jacobian is applied.
        """
        

def create_layers_from_config(config_file, cube_type='single'):
    """
    Constructs a hierarchy of layers from the given configuration file.

    The configuration file should have a section called 'layers'. Each option
    in this section is a layer; the details for that layer are provided by
    another section in the ini file with the same name as the option. The
    option should be set to the level number for the layer. The section with
    the details about the layer must have an x-arg, x-label, x-nbins, x-min,
    x-max, x-distr, and the same for y. For example:

    ::
        [layers]
        mtotal-q = 0

        [mtotal-q]
        ; x parameters
        x-arg = mtotal
        x-label = '$M (\mathrm{M}_\odot)$'
        x-min = 6.
        x-max = 50.
        x-nbins = 4
        x-distr = linear
        ; y parameters
        y-arg = q
        y-label = '$q$'
        y-min = 1.
        y-max = 16.
        y-nbins = 4
        y-distr = log10

    Parameters
    ----------
    config_file: str
        File name of the configuration file.
    cube_type: str {'single'|'gain'}
        What cube type to create in the layers. If 'single', PHyperCubes will
        be created in each layer. If 'gain', PHyperCubeGains will be created
        in each layer.

    Returns
    -------
    layers: list
        The list of layers that are created. The order of the layers in the
        list is the same as their level numbers.
    """
    # load the layers
    cp = ConfigParser.ConfigParser()
    cp.read(opts.layer_config_file)

    # construct the layers and parameter hyper cubes (phyper cubes)
    layer_names = dict([ [int(cp.get('layers', lname)), lname] for lname in \
        cp.options('layers')])

    layers = []
    for level, layer_name in sorted(layer_names.items()):
        x_arg = cp.get(layer_name, 'x-arg')
        y_arg = cp.get(layer_name, 'y-arg')
        x_label = cp.get(layer_name, 'x-label')
        y_label = cp.get(layer_name, 'y-label')
        this_layer = Layer(level, x_arg, y_arg, x_label, y_label, cube_type)
        # set x bins
        x_min = cp.get(layer_name, 'x-min')
        x_max = cp.get(layer_name, 'x-max')
        x_nbins = cp.get(layer_name, 'x-nbins')
        x_distr = cp.get(layer_name, 'x-distr')
        this_layer.create_bins('x', x_min, x_max, x_nbins, x_distr)
        # ditto y
        y_min = cp.get(layer_name, 'y-min')
        y_max = cp.get(layer_name, 'y-max')
        y_nbins = cp.get(layer_name, 'y-nbins')
        y_distr = cp.get(layer_name, 'y-distr')
        this_layer.create_bins('y', y_min, y_max, y_nbins, y_distr)
        # set super layer and create phyper cubes
        if level > 0:
            this_layer.set_super_layer(layers[level-1])
        this_layer.create_phyper_cubes()
        # add to the list of layers
        layers.append(this_layer)

    # set sub layers
    for this_layer in layers[:-1]:
        this_layer.set_sub_layer(layers[this_layer.level+1])

    return layers




def create_html_page(self, out_dir, htmlName):
    htmlMsun = 'M<sub>&#9737;</sub>'
    f = open('%s/%s' % (out_dir, htmlName), 'w')
    print >> f, "<html><body><h1>"
    if self.x_arg == 'mtotal' and self.y_arg == 'q':
        print >> f, "M<sub>total</sub> &isin; [%.2f, %.2f) %s, q &isin; [%.2f, %.2f) %s" %(self.m1range[0], self.m1range[1], htmlMsun, self.m2range[0], self.m2range[1], htmlMsun),
    else:
        print >> f, "m<sub>1</sub> &isin; [%.2f, %.2f) %s, m<sub>2</sub> &isin; [%.2f, %.2f) %s" %(self.m1range[0], self.m1range[1], htmlMsun, self.m2range[0], self.m2range[1], htmlMsun),
    if abs(self.s1zrange) != 0 and abs(self.s1zrange) != numpy.inf:
        print >> f, ", &chi;<sub>1</sub> &isin; [%.2f, %.2f)" %(self.s1zrange[0], self.s1zrange[1]),
    if abs(self.s2zrange) != 0 and abs(self.s2zrange) != numpy.inf:
        print >> f, ", &chi;<sub>2</sub> &isin; [%.2f, %.2f)" %(self.s2zrange[0], self.s2zrange[1]),
    print >> f, "</h1><hr>"
    print >> f, "<p>"
    print >> f, "Total number of injections: %i<br>" % sum(self.nsamples['reference'])
    print >> f, "Total sensitive volume (Gpc<sup>3</sup>):<br>"
    V = plot_efficiency.get_signum(self.integrated_eff['test']*MpcToGpc**3., self.integrated_err_low['test']*MpcToGpc**3.)
    Verr = plot_efficiency.get_signum(self.integrated_err_low['test']*MpcToGpc**3., self.integrated_err_low['test']*MpcToGpc**3.)
    print >> f, "%s: %s +/- %s <br>" %('test', V, Verr)
    V = plot_efficiency.get_signum(self.integrated_eff['reference']*MpcToGpc**3., self.integrated_err_low['reference']*MpcToGpc**3.)
    Verr = plot_efficiency.get_signum(self.integrated_err_low['reference']*MpcToGpc**3., self.integrated_err_low['reference']*MpcToGpc**3.)
    print >> f, "%s: %s +/- %s <br>" %('reference', V, Verr)
    print >> f, "</p><hr>"
    for plotfn in self.plots:
        print >> f, '<a href="%s"><img src="%s" width="500" /></a>' % (plotfn, plotfn)
    print >> f, "</body></html>"
    f.close()
    self.htmlpage = htmlName

    return htmlName
