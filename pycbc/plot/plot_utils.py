#! /usr/bin/env python

import os, sys
import math
import sqlite3
import numpy
import lal

def get_signum(val, err, max_sig = numpy.inf):
    """
    Given an error, returns a string for val to formated the appropriate
    number of significant figures.
    """
    coeff, pwr = ('%e' % err).split('e')
    if pwr.startswith('-'):
        pwr = int(pwr[1:])
        if round(float(coeff)) == 10.:
            pwr -= 1
        pwr = min(pwr, max_sig)
        tmplt = '%.' + str(pwr) + 'f'
        return tmplt % val
    else:
        pwr = int(pwr[1:])
        if round(float(coeff)) == 10.:
            pwr += 1
        return '%i' %(round(val, -pwr))


def ColorBarLog10Formatter(y, pos):
    """
    Formats y=log10(x) values for a colorbar so that they appear as 10^y.
    """
    return "$10^{%.1f}$" % y


#############################################
#
#   Data Storage and Slicing tools
#
#############################################

class Result:
    """
    Class to store results for plot plotting.
    To see the arguments that are stored and what can be plotted,
    run Result.plottable_arguments()
    """
    def __init__(self):
        self.apprx = None
        self.unique_id = None
        self.m1 = None
        self.m2 = None
        self.s1z = None
        self.s2z = None
        self.distance = None
        self.inclination = None
        self.inj_sigma = {}
        self.inj_min_vol = None
        self.inj_weight = None
        self.effectualness = None
        self.weight_function = None
        self.weight = None
        self.snr = None
        self.snr_std = None
        self.chisq = None
        self.chisq_std = None
        self.chisq_dof = None
        self.new_snr = None
        self.new_snr_std = None
        self.num_samples = None
        self.database = None
        self.coinc_event_id = None
        self.tmplt_id = None
        self.tmplt_m1 = None
        self.tmplt_m2 = None
        self.tmplt_s1z = None
        self.tmplt_s2z = None

    @property
    def mtotal(self):
        return self.m1 + self.m2

    @property
    def mtotal_s(self):
        return lal.MTSUN_SI*self.mtotal

    @property
    def q(self):
        return self.m1 / self.m2

    @property
    def eta(self):
        return self.m1*self.m2 / self.mtotal**2.

    @property
    def mchirp(self):
        return self.eta**(3./5)*self.mtotal

    @property
    def chi(self):
        return (self.m1*self.s1z + self.m2*self.s2z) / self.mtotal

    @property
    def tmplt_mtotal(self):
        return self.tmplt_m1 + self.tmplt_m2

    @property
    def tmplt_q(self):
        return self.tmplt_m1/self.tmplt_m2

    @property
    def tmplt_eta(self):
        return self.tmplt_m1*self.tmplt_m2 / self.tmplt_mtotal**2.

    @property
    def tmplt_mchirp(self):
        return self.tmplt_eta**(3./5)*self.tmplt_mtotal

    @property
    def tmplt_chi(self):
        return (self.tmplt_m1*self.tmplt_s1z + self.tmplt_m2*self.tmplt_s2z) \
            / self.tmplt_mtotal
        
    def tau0(self, f0 = 40):
        return (5./(256*numpy.pi*f0*self.eta))*(numpy.pi*self.mtotal*lal.MTSUN_SI*f0)**(-5./3.)
    
    def v0(self, f0 = 40):
        return (2*numpy.pi*f0*self.mtotal*lal.MTSUN_SI)**(1./3)

    @property
    def plottable_arguments(self):
        """
        Returns a string listing all plottable arguments.
        """
        return """
m1: Injected mass of the larger object, in solar masses.
m2: Injected mass of the larger object, in solar masses.
s1z: The larger object's projection of the dimensionless spin along the orbital angular momentum.
s2z: Same as s1z, but for the smaller body.
mtotal: The injected total mass, in solar masses.
q: The injected mass ratio.
mchirp: The injected chirp mass, in solar masses.
eta: The injected symmetric-mass ratio.
tmplt_{m1|m2|...|eta}: The equivalent physical parameters of the best-matching template.
eff_dist: The effective distance of the injection, in Mpc.
dist: The injection's distance, in Mpc.
inclination: The injection's inclination.
effectualness: The injection's effectualness.
snr: The measured expectation value of the SNR, as found by calc_exval.
snr_std: The standard deviation of the expectation value of the SNR, as found by calc_exval.
chisq: The measured expectation value of chisq, as found by calc_exval.
chisq_dof: The number of degress of freedom of the chisq.
chisq_std: The standard deviation of the expectation value of chisq, as found by calc_exval.
new_snr: The measured expectation value of new SNR, as found by calc_exval.
new_snr_std: The standard deviation of the expectation value of new_snr, as found by calc_exval.
"""

def parse_results_cache(cache_file):
    filenames = []
    f = open(cache_file, 'r')
    for line in f:
        thisfile = line.split('\n')[0]
        if os.path.exists(thisfile):
            filenames.append(thisfile)
    f.close()
    return filenames

def get_injection_results(filenames, weight_function='uniform',
        verbose=False):
    sqlquery = """
        SELECT
            sim.waveform, sim.simulation_id,
            sim.mass1, sim.mass2, sim.spin1z, sim.spin2z,
            sim.distance, sim.inclination, tmplt.event_id,
            tmplt.mass1, tmplt.mass2, tmplt.spin1z, tmplt.spin2z,
            res.effectualness, res.snr, res.snr_std,
            tw.weight, res.chisq, res.chisq_std, res.chisq_dof, res.new_snr,
            res.new_snr_std, res.num_successes, res.sample_rate,
            res.coinc_event_id
        FROM
            overlap_results AS res
        JOIN
            sim_inspiral as sim, coinc_event_map as map
        ON
            sim.simulation_id == map.event_id AND
            map.coinc_event_id == res.coinc_event_id
        JOIN
            sngl_inspiral AS tmplt, coinc_event_map AS mapB
        ON
            mapB.coinc_event_id == map.coinc_event_id AND
            mapB.event_id == tmplt.event_id
        JOIN
            coinc_definer AS cdef, coinc_event AS cev
        ON
            cev.coinc_event_id == res.coinc_event_id AND
            cdef.coinc_def_id == cev.coinc_def_id
        JOIN
            tmplt_weights as tw
        ON
            tw.tmplt_id == tmplt.event_id AND
            tw.weight_function == cdef.description
        WHERE
            cdef.description == ?
    """
    results = []
    id_map = {}
    idx = 0
    for ii,thisfile in enumerate(filenames):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(filenames)),
            sys.stdout.flush()
        if not thisfile.endswith('.sqlite'):
            continue
        connection = sqlite3.connect(thisfile)
        cursor = connection.cursor()
        try:
            for (apprx, sim_id, m1, m2, s1z, s2z, dist, inc,
                    tmplt_evid, tmplt_m1, tmplt_m2, tmplt_s1z,
                    tmplt_s2z, ff, snr,
                    snr_std, weight, chisq, chisq_std, chisq_dof, new_snr,
                    new_snr_std, nsamp, sample_rate, ceid) in \
                    cursor.execute(sqlquery, (weight_function,)):
                thisRes = Result()
                thisRes.unique_id = idx
                id_map[thisfile, sim_id] = idx
                idx += 1
                thisRes.apprx = apprx
                # ensure that m1 is always > m2
                if m2 > m1:
                    thisRes.m1 = m2
                    thisRes.m2 = m1
                    thisRes.s1z = s2z
                    thisRes.s2z = s1z
                else:
                    thisRes.m1 = m1
                    thisRes.m2 = m2
                    thisRes.s1z = s1z
                    thisRes.s2z = s2z
                thisRes.mtotal = m1+m2
                thisRes.distance = dist
                thisRes.inclination = inc
                thisRes.tmplt_id = tmplt_evid
                thisRes.tmplt_m1 = tmplt_m1
                thisRes.tmplt_m2 = tmplt_m2
                thisRes.tmplt_s1z = tmplt_s1z
                thisRes.tmplt_s2z = tmplt_s2z
                thisRes.effectualness = ff
                thisRes.snr = snr
                thisRes.snr_std = snr_std
                thisRes.weight_function = weight_function
                thisRes.weight = weight
                thisRes.chisq = chisq
                thisRes.chisq_sts = chisq_std
                thisRes.chisq_dof = chisq_dof
                thisRes.new_snr = new_snr
                thisRes.new_snr_std = new_snr_std
                thisRes.num_samples = nsamp
                thisRes.sample_rate = sample_rate
                thisRes.database = thisfile
                thisRes.coinc_event_id = ceid
                results.append(thisRes)

            # try to get the sim_inspiral_params table
            tables = cursor.execute(
                'SELECT name FROM sqlite_master WHERE type == "table"'
                ).fetchall()
            if ('sim_inspiral_params',) in tables:
                sipquery = """
                    SELECT
                        sip.simulation_id, sip.ifo, sip.sigmasq,
                        sip.min_vol, sip.weight
                    FROM
                        sim_inspiral_params AS sip
                    """
                for simid, ifo, sigmasq, min_vol, w in cursor.execute(
                        sipquery):
                    thisRes = results[id_map[thisfile, simid]]
                    thisRes.inj_sigma[ifo] = numpy.sqrt(sigmasq)
                    thisRes.inj_min_vol = min_vol
                    thisRes.inj_weight = weight

        except sqlite3.OperationalError:
            cursor.close()
            connection.close()
            continue
        except sqlite3.DatabaseError:
            cursor.close()
            connection.close()
            print "Database Error: %s" % thisfile
            continue

        connection.close()

    if verbose:
        print >> sys.stdout, ""

    return results, id_map


def get_templates(filename, old_format = False):
    templates = []
    connection = sqlite3.connect(filename)
    if old_format:
        sqlquery = 'select sngl.mass1, sngl.mass2, sngl.alpha3, sngl.alpha6 from sngl_inspiral as sngl'
    else:
        sqlquery = 'select sngl.mass1, sngl.mass2, sngl.spin1z, sngl.spin2z from sngl_inspiral as sngl'
    for m1, m2, s1z, s2z in connection.cursor().execute(sqlquery):
        thisRes = Result()
        thisRes.m1 = m1
        thisRes.m2 = m2
        thisRes.mtotal = m1+m2
        thisRes.s1z = s1z
        thisRes.s2z = s2z
        templates.append(thisRes)
    connection.close()
    return templates


def get_arg(row, arg):
    """
    Retrieves an arbitrary argument from the given row object. For speed, the
    argument will first try to be retrieved using getattr. If this fails (this
    can happen if a function of several attributes of row are requested),
    then Python's eval command is used to retrieve the argument. The argument
    can be any attribute of row, or functions of attributes of the row
    (assuming the relevant attributes are floats or ints). Allowed functions
    are anything in Python's math library. No other functions (including
    Python builtins) are allowed.

    Parameters
    ----------
    row: any instance of a Python class
        The object from which to apply the given argument to.
    arg: string
        The argument to apply.

    Returns
    -------
    value: unknown type
        The result of evaluating arg on row. The type of the returned value
        is whatever the type of the data element being retreived is.
    """
    try:
        return getattr(row, arg)
    except AttributeError:
        row_dict = dict([ [name, getattr(row,name)] for name in dir(row)])
        safe_dict = dict([ [name,val] for name,val in \
            row_dict.items()+math.__dict__.items() \
            if not name.startswith('__')])
        return eval(arg, {"__builtins__":None}, safe_dict)


def result_in_range(result, test_dict):
    cutvals = [(get_arg(result, criteria), low, high) \
        for criteria,(low,high) in test_dict.items()] 
    return not any(x < low or x >= high for (x, low, high) in cutvals)


def result_is_match(result, test_dict):
    try:
        matchvals = [(getattr(result, criteria), targetval) 
            for criteria,targetval in test_dict.items()]
    except:
        matchvals = [(get_arg(result, criteria), targetval) \
            for criteria,targetval in test_dict.items()]
    return not any(x != targetval for (x, tagetval) in matchvals)


def apply_cut(results, test_dict):
    return [x for x in results if result_in_range(x, test_dict)]

def slice_results(results, test_dict):
    return apply_cut(results, test_dict)


#
#
#   Utilities for creating html image maps
#
#
class ClickableElement:
    """
    Class to store additional information about an element in a matplotlib
    plot that is needed to create an image map area of it in an html page.

    Parameters
    ----------
    element: matplotlib element
        The element (e.g., a polygon) in a matplotlib axis overwhich we will
        create an image map area.
    shape: str
        The shape of the image map area. Valid shapes are given by the
        validshapes attribute.
    link: str
        The location of the file that the clickable area will link to.
    tag: str
        Title to appear over the clickable area during mouseover. This is
        added to the title and alt attributes of the area.
    data: Python object
        The object containing the data that the plot element was created from.
        For instance, if the plot element came from some data stored in an
        LSC Table row, data will be that row. This is useful if one needs
        to get additional information about the element, such as when
        building the html page that the clickable area points to.
    pixelcoords: numpy array
        The coordinates of the area, in display units. This should be a 2D
        array of x,y pairs. The x-values are measured from the left edge
        of the figure, the y-values from the top edge of the figure. (Note that
        this is different from matplotlib, in which the y-values are measured
        from the bottom edge.) To ensure proper formatting, pixelcoords can
        can only be set using self.set_pixelcoords().
    radius: int
        The radius of the clickable area, in display units. This should only
        be set if self's shape is circle.
    """
    element = None
    _shape = None
    _radius = None
    _pixelcoords = None
    data = None
    link = None
    tag = None
    validshapes = ['circle', 'rect', 'poly']
    
    def __init__(self, element, shape, data=None, link=None, tag=None,
        pixelcoords=None, radius=None):
        self.element = element
        self.set_shape(shape)
        if radius is not None:
            self.set_radius(radius)
        self.data = data
        self.link = link
        self.tag = tag
        if self.pixelcoords is not None:
            self.set_pixelcoords(pixelcoords)
        else:
            self._pixelcoords = None

    @property
    def shape(self):
        return self._shape

    def set_shape(self, shape):
        if not shape in self.validshapes:
            raise ValueError("unrecognized shape %s; options are %s" %(
                shape, ', '.join(self.validshapes)))
        self._shape = shape

    @property
    def radius(self):
        return self._radius

    def set_radius(self, radius):
        if self._shape != 'circle':
            raise ValueError("a radius should only be set if self's shape " +\
                "is a circle")
        self._radius = radius

    @property
    def pixelcoords(self):
        return self._pixelcoords

    def set_pixelcoords(self, coords):
        """
        Sets the pixelcoords attribute. The given coords must be a 2D array
        of x,y pairs corresponding to the pixel coordinates of the element.
        If self's shape is a rect, coords should be a 2x2 array in which the
        first row is the bottom left corner and the second row is the top
        right corner of the rectangle. If self's shape is circle, coords
        should be a 2x1 array giving the x,y location of the point. If self's
        shape is poly, coords should bw a 2xN array giving the vertices of
        the polygon, where N > 2.
        """
        # check that we have sane options
        if len(coords.shape) != 2:
            raise ValueError('coords should be a 2D array of x,y pairs')
        if self._shape == 'rect' and coords.shape != (2,2):
            raise ValueError('rect shape should have two x and two y ' +
                'coordinates, corresponding to the location of the vertices.')
        if self._shape == 'circle' and coords.shape != (1,2):
            raise ValueError('circle shape should only have a ' +
                'single x,y coordinate')
        if self._shape == 'poly' and coords.shape[0] < 3:
            raise ValueError('poly shape should have at least three x,y ' +
                'coordinates')
        # passed, set the coordinates
        self._pixelcoords = coords


class MappableFigure:
    """
    Class to store various elements of a matplotlib figure to make it easier
    to create an image map for it. The class has a get_pixelcoordinates
    function which will automatically get the pixel coordinates of all desired
    clickable elements formatted in the correct way for an image map. The class
    also has a savefig function. This should be used instead of the figure's
    savefig, as it ensures that the proper coordinate transformations are
    carried out to get the pixel coordinates when the figure is renderred.

    Parameters
    ----------
    figure: matplotlib.figure.Figure instance
        The figure for which an image map may be created.
    figure_filename: str
        The name of the file the figure is saved to.
    clickable_elements: list
        A list of ClickableElement instances for which clickable areas will
        be created in an image map. All of the elements must be in figure.
        To ensure this, a list may only be added via the set_clickable_elements
        function. To add more elements, use the add_clickable function.
    """
    figure = None
    _figure_filename = None
    _clickable_elements = []

    def __init__(self, figure, clickable_elements=None):
        self.figure = figure
        self._figure_filename = None
        if clickable_elements is None:
            self._clickable_elements = []
        else:
            self.set_clickable_elements(clickable_elements)

    def check_clickable_element(self, clickable):
        """
        Checks that the given clickable's element is in self's figure.
        If it is not, a ValueError is raised.
        """
        if clickable.element.figure != self.figure:
            raise ValueError("clickable is not an element in self's figure")

    def add_clickable(self, clickable):
        self.check_clickable_element(clickable)
        self._clickable_elements.append(clickable)

    def set_clickable_elements(self, clickable_elements):
        map(self.check_clickable_element, clickable_elements)
        self._clickable_elements = clickable_elements

    @property
    def clickable_elements(self):
        return self._clickable_elements

    def get_pixel_coordinates(self, event):
        """
        Function to get the pixel coordinates of the clickable elements
        when the figure is saved.
        """
        #fig = event.canvas.figure
        #ax = fig.axes[0]
        for clickable in self._clickable_elements:
            ax = clickable.element.axes
            pixel_coordinates = ax.transData.transform(clickable.element.xy)
            # pixel cooredinates are a 2D array; the first column is the
            # x coordinates, the second, the y coordinates
            # In html, the y-coordinate of an area is measured from the top
            # of the image, whereas matplotlib measures y from the bottom
            # of this image. For this reason, we have to adjust the y
            pixel_coordinates[:,1] *= -1.
            pixel_coordinates[:,1] += self.figure.bbox.height
            clickable.set_pixelcoords(pixel_coordinates)

    def savefig(self, filename, **kwargs):
        """
        A wrapper around pyplot.savefig that will get the area coordinates
        of the clickable elements while saving. This needs to be done
        when the figure is rendered for saving to ensure the proper
        coordinates are retrieved. For more information, see:
        http://stackoverflow.com/a/4672015
        """
        cid = self.figure.canvas.mpl_connect('draw_event',
            self.get_pixel_coordinates)
        self._figure_filename = filename
        self.figure.savefig(filename, **kwargs)
        self.figure.canvas.mpl_disconnect(cid)

    @property
    def saved_filename(self):
        """
        The filename of the saved figure. Only set when self.savefig is called.
        """
        return self._figure_filename

    def create_image_map(self, html_filename, view_width=750,
            view_height=None):
        """
        Creates an image map of self. The pixelcoords and links of every
        element in self.clickable_elements must be populated, and
        self.saved_filename must not be None, meaning that the figure
        was saved to an image file.

        Parameters
        ----------
        html_filename: str
            The file name of the html page that the image map will be placed
            in. This is needed so that the proper relative path to the figure's
            file name can be constructed.
        view_width: {750, int}
            The width, in pixels, of the displayed image. To use the original
            size of the image, pass None.
        view_height: {None, int}
            The height, in pixels, of the displayed image. Default is None,
            in which case the height will automatically be set by the
            browser based on the view_width, such that the aspect ratio is
            preserved.

        Returns
        -------
        html: str
            HTML code describing the image link. This can be added to an
            html file to show the image map.
        """
        return _plot2imgmap(self, html_filename, view_width, view_height)
            

def _plot2imgmap(mappable_figure, html_filename, view_width=750,
        view_height=None):
    """
    Creates the necessary html code to display a figure with an image map
    on an html page.

    Parameters
    ----------
    mappable_figure: MappableFigure instance
        A instance of a MappableFigure. The clickable_elements list must
        be populated, and the figure must have been saved to a file via
        the mappable_figure's savefig function.
    html_filename: str
        The file name of the html page that the image map will be placed
        in. This is needed so that the proper relative path to the figure's
        file name and the links can be constructed.
    view_width: {750, int}
        The width, in pixels, of the displayed image. To use the original
        size of the image, pass None.
    view_height: {None, int}
        The height, in pixels, of the displayed image. Default is None,
        in which case the height will automatically be set by the
        browser based on the view_width, such that the aspect ratio is
        preserved.
    
    Returns
    -------
    html: str
        HTML code describing the image link. This can be added to an
        html file to show the image map.
    """
    if mappable_figure.saved_filename is None:
        raise ValueError("mappable_figure's saved_filename is None! Run " +
            "mappable_figure's savefig to save the figure to disk.")
    figname = mappable_figure.saved_filename
    # we need the fig filename relative to the html page's path
    figname = os.path.relpath(os.path.abspath(figname),
        os.path.dirname(os.path.abspath(html_filename)))
    fig = mappable_figure.figure
    img_height = int(fig.bbox.height)
    img_width = int(fig.bbox.width)
    # get the scale factors we need for the given view width and height
    scale_x = 1.
    scale_y = 1.
    if view_width is not None:
        scale_x = float(view_width)/img_width
        width_str = 'width="%i"' %(view_width)
        if view_height is None:
            scale_y = scale_x
    else:
        width_str = ''
    if view_height is not None:
        scale_y = float(view_height)/img_height
        if view_width is None:
            scale_x = scale_y
        height_str = 'height="%i"' %(view_height)
    else:
        height_str = ''
    scale = numpy.array([scale_x, scale_y])

    # construct the areas
    areas = []
    area_tmplt = '<area shape="%s" coords="%s" href="./%s" %s />'
    for clickable in mappable_figure.clickable_elements:
        if clickable.shape == 'rect' or clickable.shape == 'poly':
            # for rectangle or polygon, the coordinates are just
            # x1,y1,x2,y2,...,xn,yn, where n=4 for rect
            coords = ','.join(
                (clickable.pixelcoords*scale).flatten().astype('|S32'))
        elif clickable.shape == 'circle':
            if clickable.radius is None:
                raise ValueError("shape is circle, but no radius set!")
            coords = ','.join(
                (clickable.pixelcoords*scale).flatten().astype('|S32'))
            coords += ',%i' %(clickable.radius)
        else:
            raise ValueError("unrecognized shape %s" %(clickable.shape))
        # format the link so that it is a relative path w.r.t. the html_page
        if clickable.link is None:
            raise ValueError("no link set!")
        link = os.path.relpath(os.path.abspath(clickable.link),
            os.path.dirname(os.path.abspath(html_filename)))
        # get any tags
        if clickable.tag is not None:
            tag_str = 'alt="%s" title="%s"' %(clickable.tag, clickable.tag)
        else:
            tag_str = ''
        areas.append(area_tmplt %(clickable.shape, coords, link, tag_str))

    # construct the map name from the figure's file name
    mapname = os.path.basename(figname)[:-4]

    # the map template
    tmplt = """
<div>
<img src="%s" class="mapper" usemap="#%s" border="0" %s %s />
</div>
<map name="%s">
%s
</map>
"""
    return tmplt %(figname, mapname, width_str, height_str, mapname,
        '\n'.join(areas))


def plot2imgmap(ax, links, shape, figname, view_width=750, tags=[]):
    """
    Given a matplotlib plot, creates the necessary html to make the figure
    an image map.

    Parameters
    ----------
    ax: matplotlib axes
        The axes in containing the data which we want to create links for.
    links: list of tuples
        The elements of the links list should be tuples containing the data
        coordinates that the link should cover, and the page to link to. If
        shape is 'rect' the coordinates should be of the top left and bottom
        right corners of the link. If shape is 'circle', the coordinates should
        be the x,y coordinates. For example, if shape='rect', links would be:
        ::
            links = [((x1_tl, y1_tl), (x1_br, y1_br), link1),
                     ((x2_tl, y2_tl), (x2_br, y2_br), lin2), ...]
        If shape='circle', links would be:
            links = [(x1, y1, link1), (x2, y2, link2), ...]
    shape: {'rect' | 'circle'}
        The shape of the link to create. Options are either 'circle' or 'rect'.
    figname: str
        The filename of the plot that the figure was saved to.
    view_width: {int, 1000}
        How wide the saved figure should be displayed on the html page.
    tags: {list, []}
        Optional add tags that will appear when each image link is hovered over
        with the mouse. If specified, the length of tags must be the same as
        links.

    Returns
    -------
    html: str
        A block of html code containing the image map and the plot. This can
        be inserted into a file to display the figure.
    """
    fig = ax.get_figure()
    dpi = fig.get_dpi()
    print dpi, fig.get_figwidth()
    img_height = fig.get_figheight() * dpi
    img_width = fig.get_figwidth() * dpi
    if view_width is None:
        view_width = img_width
    scalefac = float(view_width)/img_width
    print view_width, img_width, scalefac

    if tags == []:
        tags = ['' for link in links]

    # transform the coordinates of the data into display coordinates
    trans = ax.transData
    if shape == 'rect':
        img_coords = [(trans.transform(coord1), trans.transform(coord2), link)
            for (coord1, coord2, link) in links]
    elif shape == 'circle':
        img_coords = [(trans.transform([x,y]), link) for (x, y, link) in links]
    else:
        raise ValueError, 'Unrecognized shape %s' % shape

    tmplt = """
<div>
<img src="%s" class="mapper" usemap="#%s" border="0" width="%i">
</div>
<map name="%s">
%s
</map>
"""
    if shape == 'rect':
        maptmplt = '<area shape="rect" coords="%i,%i,%i,%i", href="%s" ' +\
            'alt="%s" title="%s">'
        maps = [maptmplt  %(scalefac*ix1, scalefac*(img_height - iy1),
            scalefac*ix2, scalefac*(img_height - iy2), link, tag, tag) \
            for (((ix1,iy1), (ix2, iy2),link),tag) in zip(img_coords,tags)]
    else:
        maptmplt = '<area shape="circle" coords="%i,%i,5" href="%s" ' +\
            'alt="%s" title="%s">'
        maps = [maptmplt %(scalefac*ix, scalefac*(img_height - iy),
            link, tag, tag) for (((ix,iy), link), tag) in \
            zip(img_coords, tags)]
    # construct the map name from the figname
    mapname = os.path.basename(figname)[:-4]

    return tmplt %(figname, mapname, view_width, mapname, '\n'.join(maps))
