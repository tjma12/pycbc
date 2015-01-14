#! /usr/bin/env python

import os
import numpy
import operator
import copy
import ConfigParser
from glue import segments

from pycbc.plot import plot_utils

#
#
#   HTML utilities
#
#
def mathjax_html_header():
    """
    Standard header to use for html pages to display latex math.

    Returns
    -------
    header: str
        The necessary html head needed to use latex on an html page.
    """
    return """
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}});
</script>
<script type="text/javascript"
    src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
"""

def mapper_header(mapper_jsfile):
    """
    Standard header for mapper js file.
    """
    return """
<script type="text/javascript" src="%s"></script>\n""" %(mapper_jsfile)

#
#
#   Data organization classes
#
#
class PHyperCube:
    def __init__(self, bounds=None):
        self._bounds = {}
        if bounds is not None:
            self.set_bounds(**bounds)
        self.data = None
        self._ranking_stat = None
        self._rank_by = None
        self.nsamples = None
        self._cached_volumes = {}
        # for grouping together phyper cubes
        self.parent = None
        self.children = []
        # for storing plots and pages
        self.volumes_vs_stat_plot = None
        self.tiles_plot = None
        self.subtiles_plot = None
        self.additional_plots = []
        self.html_page = None

    def set_bound(self, param, lower, upper):
        self._bounds[param] = segments.segment(lower, upper)

    def get_bound(self, param):
        try:
            return self._bounds[param]
        except KeyError:
            raise ValueError("parameter %s not in bounds; " %(param) + 
                "bound parameters are %s" %(', '.join([self._bounds.keys()])))

    def set_bounds(self, **kwargs):
        for param,(lower, upper) in kwargs.items():
            self.set_bound(param, lower, upper)

    @property
    def bounds(self):
        return self._bounds

    @property
    def params(self):
        return self._bounds.keys()

    def __contains__(self, other):
        """
        Determines whether or not other's bounds are in self's bounds. Is the
        same as "other in self". Since both self and other may be
        multi-dimensional, 3 possible values may be returned. They are:
        
          * 1: Means that both self and other's bounds have the same
            parameters. The segment bound of every parameter in other is in
            the segment bound of the same parameter in self, as determined
            by glue.segments.segment.__contains__.
          * -1: Means that self and other share some of the same parameters.
            The segment bound of every common parameter in other is in the
            segment bound of the same parameter in self.
          * 0: Means that some or all of the parameters of self and other are
            the same, but that one or more segment bounds in other are not in
            self.

        If none of the parameters in other's bounds match the parameters in
        self, a KeyError is raised.

        Parameters
        ----------
        other: PHyperCube
            The PHyperCube who's bounds are checked to see if they are in self.

        Returns
        -------
        int {-1, 0, 1}
            The result of the test.
        """
        common_params = set(other.params).intersection(set(self.params))
        if common_params == set():
            raise KeyError("other and self have no common parameters")
        result = int(all(other.get_bound(param) in self.get_bound(param) \
            for param in common_params))
        if len(self.params) == len(other.params) == len(common_params):
            result *= -1
        return result

    def central_value(self, param):
        param_range = self.get_bound(param)
        return param_range[0] + abs(param_range)/2.

    def set_data(self, data):
        """
        Given some data, extracts the points that are relevant to this cube.
        """
        self.data = plot_utils.slice_results(data, self.bounds)
        self.nsamples = len(self.data)

    def set_ranking_params(self, ranking_stat, rank_by):
        """
        Set self's ranking_stat and rank_by. These are needed to compute
        sensitive volumes.

        Parameters
        ----------
        ranking_stat: str
            The stat to use when determining whether or injections are found
            or missed. Must be an attribute, or a math operation on a
            combination of attributes, of the elements in self.data
        rank_by: str {'max'|'min'}
            How the ranking stat scales with signficance. If 'max' ('min'),
            larger (smaller) values of the ranking stat will be considered
            more significant.
        """
        self._ranking_stat = ranking_stat
        if not (rank_by == 'max' or rank_by == 'min'):
            raise ValueError("unrecognized rank_by argument %s; " %(rank_by) +\
                "options are 'max' or 'min'")
        self._rank_by = rank_by

    @property
    def ranking_stat(self):
        return self._ranking_stat

    @property
    def rank_by(self):
        return self._rank_by
        
    def get_volume(self, threshold):
        """
        Retrieves the sensitive volume using the given threshold of self's
        ranking_stat. This is done by integrating the efficiency in self
        using self._integrate_efficiency. Volumes are cached to
        self._cached_volumes for faster retrieval later on. Self's
        _ranking_stat and _rank_by parameters must be set (see
        set_ranking_params).

        Parameters
        ----------
        threshold: float
            Value to use for considering what is a found injection and what is
            missed.

        Returns
        -------
        volume: float
            The sensitive volume.
        err: float
            The error on the sensitive volume measurement, found from the
            standard deviation of Monte Carlo integral.
        """
        try:
            return self._cached_volumes[threshold, self._ranking_stat, 
                self._rank_by]
        except KeyError:
            if self._ranking_stat is None:
                raise ValueError("must set ranking_stat and rank_by first; " +
                    "see set_ranking_params()")
            volume, err = self._integrate_efficiency(threshold,
                self._ranking_stat, self._rank_by)
            self._cached_volumes[threshold,self._ranking_stat,self._rank_by] =\
                (volume, err)
            return volume, err

    def _integrate_efficiency(self, threshold, ranking_stat, rank_by):
        """
        Integrates the efficiency in self's bounds to get a measure of the
        sensitive volumes.
        """
        if rank_by == 'max':
            compare_operator = operator.ge
        elif rank_by == 'min':
            compare_operator = operator.lt
        else:
            raise ValueError("unrecognized rank_by argument %s; " %(rank_by) +\
                "options are 'max' or 'min'")
        if self.nsamples <= 1:
            raise ValueError("there must be more than one sample to " +
                "compute efficiency")
        integrand = numpy.array([x.inj_weight * \
            float(_isfound(x, ranking_stat, compare_operator, threshold)) \
            for x in self.data])
        return integrand.mean(), integrand.std()

    def clear_cache(self):
        """
        Clears self's _cached_volumes.
        """
        self._cached_volumes.clear()

    def create_html_page(self, out_dir, html_name, mapper=None,
            mainplots_widths=1000):
        """
        Create's self html page. See _create_html_page for details.
        """
        _create_html_page(self, out_dir, html_name, mapper=mapper)
        self.html_page = '%s/%s' %(out_dir, html_name)


class PHyperCubeGain:
    """
    Class to store information about a "tile" in parameter space. A tile is
    a two-dimensional slice in a higher dimensional parameter space that is
    bounded by some values. The class stores information about a "reference"
    and "test" set of results. When taking the ratio of sensitive volumes
    in the tile, the "reference" is the denominator and the "test" is the
    numerator.
    """
    def __init__(self, bounds=None):
        self._bounds = {}
        if bounds is not None:
            self.set_bounds(**bounds)
        # values unique to each set
        self._reference_cube = PHyperCube(bounds)
        self._test_cube = PHyperCube(bounds)
        # values common to both
        self.fractional_gain = 0.
        self.gain_err_low = 0.
        self.gain_err_high = 0.
        # for grouping together phyper cubes
        self.parent = None
        self.children = []
        # for storing plots and pages
        self.volumes_vs_stat_plot = None
        self.tiles_plot = None
        self.subtiles_plot = None
        self.additional_plots = []
        self.html_page = None

    @property
    def reference_cube(self):
        return self._reference_cube

    @property
    def test_cube(self):
        return self._test_cube

    def set_bound(self, param, lower, upper):
        self._bounds[param] = segments.segment(lower, upper)
        self._reference_cube.set_bound(param, lower, upper)
        self._test_cube.set_bound(param, lower, upper)

    def get_bound(self, param):
        return self._bounds[param]

    def set_bounds(self, **kwargs):
        for param,(lower, upper) in kwargs.items():
            self.set_bound(param, lower, upper)

    @property
    def bounds(self):
        return self._bounds

    @property
    def params(self):
        return self._bounds.keys()

    def __contains__(self, other):
        """
        Since the reference cube's bounds are the same as self's and 
        self.test_cube's bounds, uses the reference_cube  __contains__ function
        to do the comparison between self and other. See
        PHyperCubeGain.__contains__ for details.
        """
        return other._reference_cube in self._reference_cube

    def central_value(self, param):
        param_range = self.get_bound(param)
        return param_range[0] + abs(param_range)/2.

    def set_reference_data(self, data):
        self._reference_cube.set_data(data)

    def set_test_data(self, data):
        self._test_cube.set_data(data)

    def _set_ranking_params(ref_or_test, ranking_stat, rank_by):
        """
        Sets the reference or test cube's ranking_stat and rank_by. See
        PHyperCube.set_ranking_params for details.
        """
        getattr(self, '_%s_cube' %(ref_or_test)).set_ranking_params(
            ranking_stat, rank_by)

    def set_reference_ranking_params(ranking_stat, rank_by):
        """
        Set's the reference cube's ranking params. See
        PHyperCube.set_ranking_params for details.
        """
        self._set_ranking_params('reference', ranking_stat, rank_by)

    def set_test_ranking_params(ranking_stat, rank_by):
        """
        Set's the test cube's ranking params.
        """
        self._set_ranking_params('test', ranking_stat, rank_by)

    def get_fractional_gain(self, ref_threshold, test_threshold):
        """
        Gets the fractional gain and error.

        Parameters
        ----------
        ref_threshold: float
            The threshold to use for the reference data. The reference
            ranking params must be set; see set_reference_ranking_params.
        test_threshold: float
            Same, but for the test data/cube.
        """
        if self._reference_cube.ranking_stat is None:
            raise ValueError("reference ranking params not set")
        if self._test_cube.ranking_stat is None:
            raise ValueError("test ranking params not set")
        ref_volume, ref_err = self._reference_cube.get_volume(ref_threshold)
        test_volume, test_err = self._test_cube.get_volume(test_threshold)
        fractional_gain = test_volume / ref_volume
        gain_err = fractional_gain * \
            numpy.sqrt((test_err / test_volume)**2. + \
                       (ref_err / ref_volume)**2.)
        return fractional_gain, gain_err

    def create_html_page(self, out_dir, html_name, mapper=None,
            mainplots_widths=1000):
        """
        Create's self html page. See _create_html_page for details.
        """
        _create_html_page(self, out_dir, html_name, mapper=mapper)
        self.html_page = '%s/%s' %(out_dir, html_name)


#
#
#   Utilities for PHyperCube(Gain)s
#
#
def _isfound(result, ranking_stat, compare_operator, threshold):
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
    _isfound: bool
    """
    return compare_operator(getattr(result, ranking_stat), threshold)


def ischild(parent, child):
    """
    Checks if the parent's bounds contains the childs bounds.
    False is returned if parent.__contains__(child) is 0. If
    parent.__contains__(child) raises a KeyError, False is returned. 
    """
    try:
        return abs(child in parent)
    except KeyError:
        return False

def _construct_links(plotted_cubes, tiles):
    """
    Constructs a list of links from a list of phyper cubes and their
    corresponding tiles in a plot so that links can be passed to
    plot_utils.plot2imgmap.
    """
    links = []
    tags = []
    for cube, tile in zip(plotted_cubes, tiles):
        # if the plotted cube doesn't have an html page to link to, skip it
        if cube.html_page is None:
            continue
        # tile's coordinates go clockwise around the tile, starting from the
        # bottom left
        bl, tl, tr, br, _ = tile.xy
        # to create an image map, we need the bottom left and top right
        # corners
        links.append((bl, tr, './'+cube.html_page))
        # for the tags, we'll include the x, y values
        tags.append('x: [%f, %f)\ny: [%f, %f)' %(bl[0], br[0], bl[1], tl[1]))
    return links, tags


def _create_html_page(phyper_cube, out_dir, html_name, mainplots_widths=1000,
        mapper=None):
    """
    Creates an html page for the given PHyperCube.
    """
    #
    #   Open the html page and write the header
    #
    html_page = '%s/%s' % (out_dir, html_name)
    f = open(html_page, 'w')
    print >> f, "<!DOCTYPE html>\n<html>"
    # we'll use MathJax to be able to display latex math
    print >> f, "<head>"
    print >> f, mathjax_html_header()
    if mapper is not None:
        print >> f, mapper_header(mapper)
    print >> f, "</head>"
    #
    # Now the body
    #
    print >> f, "<body>"
    # we'll put the bounds in the heading
    print >> f, "<h1>"
    for (param, (lower, upper)) in sorted(phyper_cube.bounds.items()):
        # if the param is a Parameter instance with a label, we can
        # just get the label to use from it; otherwise, we'll just use
        # the parameter name
        try:
            plbl = param.label
        except AttributeError:
            plbl = param
        print >> f, r"%s $\in [%.2f, %.2f)$<br />" %(plbl, lower, upper)
    print >> f, "</h1>"
    # print some basic info about this cube
    print >> f, "<h2>"
    print >> f, "Total number of injections: %i<br />" %(
        phyper_cube.nsamples)
    print >> f, "</h2>"
    print >> f, "<hr />"
    # put the V vs stat plot
    if phyper_cube.volumes_vs_stat_plot is not None:
        mfig = phyper_cube.volumes_vs_stat_plot
        figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
                os.path.dirname(os.path.abspath(html_page)))
        print >> f, '<img src="%s" width="%i" />' %(figname,
            mainplots_widths)
        print >> f, "<hr />"
    # put the tiles plot
    if phyper_cube.tiles_plot is not None:
        # if no clickable elements, just add the plot
        mfig = phyper_cube.tiles_plot
        # ensure links are set appropriately
        [setattr(clickable, 'link', clickable.data.html_page) \
            for clickable in mfig.clickable_elements]
        if mfig.clickable_elements == [] or not \
                all([clickable.link is not None \
                for clickable in mfig.clickable_elements]):
            figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
                os.path.dirname(os.path.abspath(html_page)))
            print >> f, '<img src="%s" width="%i" />' %(figname,
                mainplots_widths)
        else:
            print >> f, "Click on a tile to go to the next layer for that " +\
                "tile. <br />"
            print >> f, mfig.create_image_map(html_page,
                view_width=mainplots_widths)
    # put the subtiles plot
    if phyper_cube.subtiles_plot is not None:
        print >> f, "<hr />"
        mfig = phyper_cube.subtiles_plot
        [setattr(clickable, 'link', clickable.data.html_page) \
            for clickable in mfig.clickable_elements]
        if mfig.clickable_elements == [] or not \
                all([clickable.link is not None \
                for clickable in mfig.clickable_elements]):
            figname = os.path.relpath(os.path.abspath(mfig.saved_filename),
                os.path.dirname(os.path.abspath(html_page)))
            print >> f, '<img src="%s" width="%i" />' %(figname,
                mainplots_widths)
        else:
            print >> f, "Click on a tile to go to the next layer for that " +\
                "tile. <br />"
            print >> f, mfig.create_image_map(html_page,
                view_width=mainplots_widths)
    # close out
    print >> f, '</body>\n</html>'
    f.close()


class Parameter(str):
    """
    Class to store information about a parameter. The class inherits from str.
    The string itself is the parameter "argument"; this is what you would use
    for retrieving the parameter from, say, a Result class. This class adds a
    label attribute to the string. The label is itself a string, but it is
    what you might put onto a plot axis. The label is optional; if none
    provided, the label is set to None. For example:
    ::
        >>> p = Parameter('mchirp', '$\mathcal{M}~(\mathrm{M}_\odot)$')

        >>> p
            'mchirp'

        >>> p.label
            '$\\mathcal{M}~(\\mathrm{M}_\\odot)$'

        >>> p = Parameter('mchirp')

        >>> p
            'mchirp'

        >>> p.label

    """
    def __new__(cls, arg, label=None):
        obj = str.__new__(cls, arg)
        obj.label = label
        return obj


class Layer:
    """
    Class to store and organize parents PHyperCubes, as well as other layers.
    """
    def __init__(self, level, x_arg, y_arg, x_label=None, y_label=None,
            cube_type=None):
        self.level = level
        if x_label is None:
            x_label = x_arg
        self.x_param = Parameter(x_arg, x_label)
        if y_label is None:
            y_label = y_arg
        self.y_param = Parameter(y_arg, y_label)
        self._x_bins = segments.segmentlist([])
        self._y_bins = segments.segmentlist([])
        self._x_distr = None
        self._y_distr = None
        self._super_layer = None
        self._sub_layer = None
        self._parents = []
        self._cube_type = 'single'
        if cube_type is not None:
            self.set_cube_type(cube_type)
        # plot options
        self.plot_x_min = None
        self.plot_x_max = None
        self.plot_y_min = None
        self.plot_y_max = None
        # attributes for organizing html pages, plots
        self.root_dir = None
        self.web_dir = None
        self.images_dir = None

    def create_bins(self, x_or_y, minval, maxval, num, distr):
        """
        Creates the bins to use.

        Parameters
        ----------
        x_or_y: str {'x'|'y'}
            Which axis to set the bins for.
        minval: float
            The minimum value to use.
        maxval: float
            The maximum value to use.
        num: int
            The number of bins to create.
        distr: str {'linear'|'log10'}
            How to distribute the bins between the minimum and maximum values.
        """
        if distr == 'log10':
            bins = numpy.logspace(numpy.log10(minval), numpy.log10(maxval),
                num=num+1)
        elif distr == 'linear':
            bins = numpy.linspace(minval, maxval, num=num+1)
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


    def create_cubes(self):
        """
        Creates all parameter hypercubes at this layer, along with the lineage
        linking cubes at this level to the children cubes of the super layer.
        If this level is > 0, super_layer must be set.
        """
        # first, tile up this layer
        this_layer_tiles = [(xbin, ybin) for ybin in self._y_bins \
            for xbin in self._x_bins]
        # now, if there is a super layer, create a child phyper cube for every
        # tile for every parent phyper cube in super layer
        if self.level > 0:
            if self._super_layer is None:
                raise ValueError("self.level is %i but super layer not set" %(
                    self.level))
            if self._super_layer._parents == []:
                raise ValueError("super layer's parents have not been " +\
                    "created yet")
            # the parents of this layer are the children of super_layer's
            self._parents = self._super_layer.all_children
        else:
            # the level 0 layer just has a single parent, which is the cube
            # that contains all of the data
            self._parents = [self._cube_type({self.x_param: self.x_lims,
                self.y_param: self.y_lims})]
        for parent in self._parents:
            # the children are all of the tiles in self
            for (xbounds, ybounds) in this_layer_tiles:
                these_bounds = copy.copy(parent.bounds)
                these_bounds[self.x_param] = xbounds
                these_bounds[self.y_param] = ybounds
                child = self._cube_type(these_bounds)
                parent.children.append(child)
                child.parent = parent


    @property
    def parents(self):
        return self._parents

    @property
    def all_children(self):
        return [child for parent in self._parents \
            for child in parent.children]

    def set_cube_data(self, data, ref_or_test=None):
        """
        Bins the given data into self's parents of phyper cubes. If self's
        _cube_type is PHyperCubeGain, must also specify whether the data is
        test or reference.

        Parameters
        ----------
        data: dict
            The data to slice.
        ref_or_test: {None, 'reference'|'test'}
            If self's _cube_type is PHyperCubeGain, specifies whether the given
            data is for the reference cubes or the test cubes.
        """
        if self._parents == []:
            raise ValueError("parents have not been created yet; " +
                "run create_cubes")
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
        # set the parent, child data
        for parent in self._parents:
            getattr(parent, 'set%s_data' %(funcstr))(data)
            # now set the children
            for child in parent.children:
                getattr(child, 'set%s_data' %(funcstr))(data)


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
            for parent in self._parents:
                inherited_data = getattr(parent, '%sdata' %(dset))
                if inherited_data is None:
                    raise ValueError("Parent data not set! This is probably " +
                        "because the data of this layer's super layer " +
                        "has not been set yet.")
                [getattr(child, 'set_%sdata' %(dset))(inherited_data) \
                    for child in parent.children]

    def set_ranking_params(self, ranking_stat, rank_by, ref_or_test=None):
        """
        Sets the ranking_stat and rank_by for all parent and children cubes
        in self. If self's cube_type is PHyperCubeGain, ref_or_test must
        be either 'reference' or 'test' to specify whehther to set the ranking
        parameters of the reference or test cubes.
        """
        if self._cube_type == PHyperCubeGain:
            if not (ref_or_test == 'test' or ref_or_test == 'reference'):
                raise ValueError("_cube_type is PHyperCubeGain, but " +
                    "ref_or_test not set to 'reference' or 'test' (got %s)" %(
                    str(ref_or_test)))
            ref_or_test = '_%s' %(ref_or_test)
        if self._cube_type == PHyperCube:
            if ref_or_test != None:
                raise ValueError("ref_or_test is not None, but _cube_type "+\
                    "is PHyperCube")
            ref_or_test = ''

        [getattr(parent, 'set_%sranking_params' %(ref_or_test))(ranking_stat,
            rank_by) for parent in self.parents]
        [getattr(child, 'set_%sranking_params' %(ref_or_test))(ranking_stat,
            rank_by) for child in self.all_children]




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
        x-label = $M (\mathrm{M}_\odot)$
        x-min = 6.
        x-max = 50.
        x-nbins = 4
        x-distr = linear
        plot-x-min = 4.
        plot-x-max = 52.
        ; y parameters
        y-arg = q
        y-label = $q$
        y-min = 1.
        y-max = 16.
        y-nbins = 4
        y-distr = log10
        plot-y-min = 0.9
        plot-y-max = 20.

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
    cp.read(config_file)

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
        x_min = float(cp.get(layer_name, 'x-min'))
        x_max = float(cp.get(layer_name, 'x-max'))
        x_nbins = int(cp.get(layer_name, 'x-nbins'))
        x_distr = cp.get(layer_name, 'x-distr')
        this_layer.create_bins('x', x_min, x_max, x_nbins, x_distr)
        # ditto y
        y_min = float(cp.get(layer_name, 'y-min'))
        y_max = float(cp.get(layer_name, 'y-max'))
        y_nbins = int(cp.get(layer_name, 'y-nbins'))
        y_distr = cp.get(layer_name, 'y-distr')
        this_layer.create_bins('y', y_min, y_max, y_nbins, y_distr)
        # set plotting options
        if cp.has_option(layer_name, 'plot-x-min'):
            this_layer.plot_x_min = float(cp.get(layer_name, 'plot-x-min'))
        if cp.has_option(layer_name, 'plot-x-max'):
            this_layer.plot_x_max = float(cp.get(layer_name, 'plot-x-max'))
        if cp.has_option(layer_name, 'plot-y-min'):
            this_layer.plot_y_min = float(cp.get(layer_name, 'plot-y-min'))
        if cp.has_option(layer_name, 'plot-y-max'):
            this_layer.plot_y_max = float(cp.get(layer_name, 'plot-y-max'))
        # set super layer and create phyper cubes
        if level > 0:
            this_layer.set_super_layer(layers[level-1])
        this_layer.create_cubes()
        # add to the list of layers
        layers.append(this_layer)

    # set sub layers
    for this_layer in layers[:-1]:
        this_layer.set_sub_layer(layers[this_layer.level+1])

    return layers
