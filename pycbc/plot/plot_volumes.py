#! /usr/bin/env python

columnwidth = 3.4
width = 2*columnwidth
height = 3.5
from matplotlib import pyplot
pyplot.rcParams.update({
    "text.usetex": True,
    "text.verticalalignment": "center",
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.weight": "bold",
    "figure.figsize": (width, height),
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    })
pyplot.rcParams['text.latex.preamble'].append(r'\usepackage{amsmath}')

from pycbc.plot import plot_utils
import numpy
import cPickle
from optparse import OptionParser

# for conversions
MpcToGpc = 1.e-3

def get_log10_ticks(ticks):
    """
    Given some ticks from a log10 axis, converts them into 10^{**} form.

    Parameters
    ----------
    ticks: output of ax.get_(x|y)ticks()

    Returns
    -------
    new_ticks: list
        new tick values; can set the appropriate axis using ax.set_(x|y)ticks
    ticklabels: list
        new tick labels; can set the appropriate axis using
        ax.set_(x|y)ticklabels
    """
    min_tick = numpy.ceil(min(ticks))
    max_tick = numpy.floor(max(ticks))
    major_ticks = numpy.arange(min_tick, max_tick+1)
    new_ticks = numpy.array([coef*10**pwr for pwr in major_ticks[:-1] \
        for coef in range(1,10)] + [10**major_ticks[-1]])
    ticklabels = [tck % 1. == 0 and '$10^%i$' % tck or '' \
        for tck in numpy.log10(new_ticks)]
    return numpy.log10(new_ticks), ticklabels


def empty_plot(ax):
    ax.set_axis_bgcolor('w')
    return ax.annotate("Nothing to plot", (0.5, 0.5))


def plot_volumes(phyper_cubes, xarg, xlabel, yarg, ylabel, min_ninj=2,
        tmplt_label = '', inj_label='', add_title=True,
        colormap='hot', maxvol=None, minvol=None, add_colorbar=False,
        annotate=True, fontsize=8, 
        logx=False, logy=False, logz=False,
        xmin=None, xmax=None, ymin=None, ymax=None, fig=None,
        ax=None, add_clickables=True, dpi=300):

    if fig is None:
        mfig = plot_utils.figure(dpi=dpi)
    else:
        mfig = fig
    if ax is None:
        ax = mfig.add_subplot(111, axisbg='k')
    else:
        # check that the given axes is in fig
        if ax not in mfig.axes:
            raise ValueError("given ax is not in the figure")

    # only use cubes that have enough injections
    phyper_cubes = [this_cube for this_cube in phyper_cubes if \
        this_cube.nsamples >= min_ninj]

    # if there is nothing to plot, just create an empty plot and return
    if len(phyper_cubes) == 0:
        empty_plot(ax)
        return mfig


    Vs = numpy.array([this_cube.integrated_eff for this_cube in \
        phyper_cubes])
    # we'll divide by the closest power of 10 of the smallest value
    conversion_factor = numpy.floor(numpy.log10(Vs.min()))
    Vs *= 10**(-conversion_factor)
    if minvol is not None:
        minvol *= 10**(-conversion_factor)
    if maxvol is not None:
        maxvol *= 10**(-conversion_factor)
    # if the conversion factor is > 10^9, we'll change the label to Gpc
    # (V's are assumed to be in Mpc^3)
    if conversion_factor >= 9:
        conversion_factor -= 9.
        units = 'Gpc'
    else:
        units = 'Mpc'
    if conversion_factor == 0.:
        prefactor = ''
    elif conversion_factor == 1.:
        prefactor = '10'
    else:
        prefactor = '10^{%i}' %(int(conversion_factor))
    units_label = '%s\,\mathrm{%s}^3' %(prefactor, units)

    if minvol is None:
        minvol = Vs.min()
    if maxvol is None:
        maxvol = Vs.max()
    if logz:
        Vs = numpy.log10(Vs)
        minvol = numpy.log10(minvol)
        maxvol = numpy.log10(maxvol)

    # now cycle over the cubes and create each plot tile
    for V,this_cube in zip(Vs, phyper_cubes):

        # get the x, y corners of the tile from the bounds
        xlow, xhigh = this_cube.get_bound(xarg)
        ylow, yhigh = this_cube.get_bound(yarg)
        x = numpy.array([xlow, xlow, xhigh, xhigh])
        if logx:
            x = numpy.log10(x)
        y = numpy.array([ylow, yhigh, yhigh, ylow])
        if logy:
            y = numpy.log10(y)

        # get the color to use for this cube
        clrfac = (V - minvol)/(maxvol - minvol)
        clr = getattr(pyplot.cm, colormap)(clrfac)

        # get the color to use for the text
        # convert the tile color to grayscale; following equation comes from
        # http://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
        r, g, b, a = clr
        clr_grayscale = 0.299*r + 0.587*g + 0.114*b
        if annotate:
            if clr_grayscale < 0.5:
                txt_clr = 'w'
            else:
                txt_clr = 'k'
            # get the text to print
            err = max(this_cube.integrated_err_low,
                this_cube.integrated_err_high) * 10**(-conversion_factor)
            if units == 'Gpc':
                err *= 1e-9
            if logz:
                voltxt = plot_utils.get_signum(10**V, err)
            else:
                voltxt = plot_utils.get_signum(V, err)
            errtxt = plot_utils.get_signum(err, err)
            txt_str = r'$\mathsf{\underset{\pm %s}{%s}}$' %(errtxt, voltxt)
            if logx:
                txtx = numpy.log10(10**min(x) + (10**max(x)-10**min(x))/2.)
            else:
                txtx = min(x) + (max(x) - min(x))/2.
            if logy:
                txty = numpy.log10(10**min(y) + (10**max(y)-10**min(y))/2.)
            else:
                txty = min(y) + (max(y) - min(y))/2.

        # plot
        tile = ax.fill(x, y, color=clr, zorder=1)[0]
        # the tiles will be clickable
        if add_clickables:
            # we'll make the tag be the x, y values of tile
            tag = 'x: [%f, %f)\ny: [%f, %f)' %(xlow, xhigh, ylow, yhigh)
            clickable = plot_utils.ClickableElement(tile, 'poly',
                data=this_cube, tag=tag, link=this_cube.html_page)
            mfig.add_clickable(clickable) 
        if annotate:
            pts = ax.annotate(txt_str, (txtx, txty), ha='center',
                va='center', color=txt_clr, zorder=3, fontsize=fontsize)
    if add_colorbar:
        sm = pyplot.cm.ScalarMappable(cmap=colormap,
            norm=pyplot.Normalize(vmin=minvol, vmax=maxvol))
        sm.set_array(Vs)
        if logz:
            cbformat = pyplot.FuncFormatter(plot_utils.ColorBarLog10Formatter)
        else:
            cbformat = None
        cb = pyplot.colorbar(sm, format=cbformat)

    # set plot parameters
    plot_xmin, plot_xmax = ax.get_xlim()
    plot_ymin, plot_ymax = ax.get_ylim()
    if xmin is not None:
        plot_xmin = xmin
        if logx:
            plot_xmin = numpy.log10(plot_xmin)
    if xmax is not None:
        plot_xmax = xmax
        if logx:
            plot_xmax = numpy.log10(plot_xmax)
    if ymin is not None:
        plot_ymin = ymin
        if logy:
            plot_ymin = numpy.log10(plot_ymin)
    if ymax is not None:
        plot_ymax = ymax
        if logy:
            plot_ymax = numpy.log10(plot_ymax)

    # have to do log scales manually
    if logx:
        # make the limits slightly larger; to get the ticks right; we'll
        # fix it to be correct later on
        ax.set_xlim(plot_xmin - 1., plot_xmax + 1.)
        new_ticks, ticklabels = get_log10_ticks(ax.get_xticks())
        ax.set_xticks(new_ticks)
        ax.set_xticklabels(ticklabels)
    if logy:
        ax.set_ylim(plot_ymin - 1., plot_ymax + 1.)
        new_ticks, ticklabels = get_log10_ticks(ax.get_yticks())
        ax.set_yticks(new_ticks)
        ax.set_yticklabels(ticklabels)

    ax.set_xlim(plot_xmin, plot_xmax)
    ax.set_ylim(plot_ymin, plot_ymax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # set axis color schemes for black background
    ax.tick_params(axis='x', color='w')
    ax.tick_params(axis='y', color='w')
    for side in ['top', 'left', 'right', 'bottom']:
        ax.spines[side].set_color('w')

    if add_title:
        title = r'$V_{%s%s}~(%s)$' %(tmplt_label, inj_label, units_label)
        # if we are adding a colorbar, put the title on its axis
        if add_colorbar:
            cb.ax.set_ylabel(title)
        else:
            ax.set_title(title)

    return mfig


def plot_volumes_from_layer(layer, user_tag='', min_ninj=1,
        tmplt_label='', inj_label='',
        colormap='hot', maxvol=None, minvol=None, fontsize=8, 
        logz=False, dpi=300):
    """
    Wrapper around plot_volumes that creates a volume plot for every
    parent in the given layer. X and Y arguments/labels of the volume
    plot are retrieved from the given layer's x and y arguments. The resulting
    figures are saved to disk, with the filenames saved to each parent's
    tiles_plot attribute.
    """
    # check that we have the necessary directory information
    if not layer.root_dir:
        raise ValueError('no root_dir set for this layer!')
    if not layer.web_dir:
        raise ValueError('no web_dir set for this layer!')
    if not layer.images_dir:
        raise ValueError('no images_dir set for this layer!')
    # Plot file template is:
    # layer.imagesdir/plot_volumes{-user_tag}-{layer_level}-{parent_index}.png
    fnametmplt = '%s/plot_volumes%s-%i-%i.png'
    if user_tag != '':
        user_tag = '-%s' % user_tag
    for ii,parent in enumerate(layer.parents):
        mfig = plot_volumes(
            parent.children, layer.x_param, layer.x_param.label,
            layer.y_param, layer.y_param.label, min_ninj=min_ninj,
            tmplt_label=tmplt_label, inj_label=inj_label, annotate=True,
            add_colorbar=False, colormap=colormap, maxvol=maxvol,
            minvol=minvol, fontsize=fontsize,
            logx=layer.x_distr == 'log10', logy=layer.y_distr == 'log10',
            logz=logz, xmin=layer.plot_x_min, xmax=layer.plot_x_max,
            ymin=layer.plot_y_min, ymax=layer.plot_y_max)
        # save the figure
        plotname = fnametmplt %(layer.images_dir, user_tag, layer.level, ii) 
        print "saving %i-%i..." %(layer.level, ii),
        mfig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
            dpi=dpi)
        print "done"
        parent.tiles_plot = mfig


def plot_subvolumes(phyper_cubes, xarg, xlabel, yarg, ylabel,
        sub_xarg, sub_xlabel, sub_yarg, sub_ylabel, min_ninj=2,
        tmplt_label='', inj_label='',
        colormap='hot', maxvol=None, minvol=None,
        logx=False, logy=False, sub_logx=False, sub_logy=False, logz=False,
        xmin=None, xmax=None, ymin=None, ymax=None, dpi=300):
    """
    Similar to plot_volumes, but each tile shows the volumes of a sub-layer
    of tiles.
    """
    # only use phyper_cubes that have enough injections
    phyper_cubes = [parent for parent in phyper_cubes \
        if parent.nsamples >= min_ninj]

    # if nothing to plot, just create an empty plot and return
    if phyper_cubes == []:
        mfig = plot_utils.figure(dpi=dpi)
        ax = mfig.add_subplot(111)
        empty_plot(ax)
        return mfig

    # to ensure we get properly normalized colors, find the largest
    # and smallest Vs across all of the children of all of the phyper_cubes
    Vs = numpy.array([child.integrated_eff for parent in phyper_cubes \
        for child in parent.children if child.nsamples >= min_ninj])
    minvol = Vs.min()
    maxvol = Vs.max()

    # create the master plot with clickable elements
    mfig = plot_volumes(
        phyper_cubes, xarg, xlabel, yarg, ylabel, min_ninj,
        tmplt_label=tmplt_label, inj_label=inj_label,
        add_title=True, colormap=colormap, maxvol=maxvol, minvol=minvol,
        add_colorbar=True, annotate=False, logx=logx, logy=logy, logz=logz,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, dpi=dpi,
        add_clickables=True)
      
    master_ax = mfig.axes[0]

    # we need to ensure that the master x and y-limits don't change, to ensure
    # that the axes we will put on top of the master plot are in the right
    # places
    master_xlims = master_ax.get_xlim()
    master_ylims = master_ax.get_ylim()
    # we'll want to know which tile is in the furthest lower-left of the plot
    # to determine which inset axis to put labels on
    tile_xvals = numpy.array([parent.get_bound(xarg)[0] \
        for parent in phyper_cubes])
    if logx:
        tile_xvals = numpy.log10(tile_xvals)
    tile_yvals = numpy.array([parent.get_bound(yarg)[0] \
        for parent in phyper_cubes])
    if logy:
        tile_yvals = numpy.log10(tile_yvals)
    label_idx = numpy.intersect1d(
        numpy.where(tile_xvals == tile_xvals.min())[0],
        numpy.where(tile_yvals == tile_yvals.min())[0])[0]
    # the transformation to go from display coordinates to figure coordinates;
    # we'll need this to properly place the inset axes
    invtrans = mfig.transFigure.inverted()
    # we also need to adjust the x and y limits of the master plot 
    # to make room for the inset axes labels
    masterax_lbcoords_fig = numpy.array([master_ax.get_position().xmin,
        master_ax.get_position().ymin])
    tile_leftbottom_coords = numpy.array([tile_xvals.min(), tile_yvals.min()])
    tile_lbcoords_fig = invtrans.transform(
        master_ax.transData.transform(tile_leftbottom_coords))
    axes_dx = tile_lbcoords_fig[0] - masterax_lbcoords_fig[0]
    axes_dy = tile_lbcoords_fig[1] - masterax_lbcoords_fig[1]
    # FIXME: just guessing at how much space we need for the axis labels
    xlabel_height = 0.085
    ylabel_width = 0.06
    transx = 0
    if axes_dx < ylabel_width:
        # get where the left edge of the tile needs to be in figure coords
        transx = ylabel_width - axes_dx
    transy = 0
    if axes_dy < xlabel_height:
        transy = xlabel_height - axes_dy
    if transx != 0 or transy != 0:
        new_tile_coords = tile_lbcoords_fig + numpy.array([transx, transy])
        # convert to data coordinates: first, transform from fig to display
        new_tile_coords = mfig.transFigure.transform(new_tile_coords)
        # now from display to data coords
        new_tile_coords = master_ax.transData.inverted().transform(
            new_tile_coords)
        # the difference between the new_tile_coords and the old tile coords
        # is how much we need to shift the lower limit of x and y back by
        padx, pady = new_tile_coords - tile_leftbottom_coords
        master_xlims = (master_xlims[0]-padx, master_xlims[1])
        master_ylims = (master_ylims[0]-pady, master_ylims[1])
        master_ax.set_xlim(master_xlims)
        master_ax.set_ylim(master_ylims)
        
    # cycle over phyper_cubes that were plotted, creating sub-plots from their
    # children
    for ii,clickable in enumerate(mfig.clickable_elements):
        tile = clickable.element
        parent = clickable.data
        # create an axis to plot this family on; this axis will cover the tile
        # exactly
        tile_fig_coords = invtrans.transform(tile.axes.transData.transform(
            tile.get_xy()))
        # tile fig coords are a 2D array, in which the first column is the
        # x values of the vertices of the tile, the second column are the y;
        # we need [left, bottom, width, height] for the inset axes
        tile_figx = tile_fig_coords[:,0]
        tile_figy = tile_fig_coords[:,1]
        inset_coords = [tile_figx.min(), tile_figy.min(),
            tile_figx.max()-tile_figx.min(), tile_figy.max()-tile_figy.min()]
        inset_axes = mfig.add_axes(inset_coords)
        # now create the sub plot in the inset_axes
        # note that we turn off the clickables in the inset_axes
        plot_volumes(
            parent.children, sub_xarg, sub_xlabel, sub_yarg, sub_ylabel,
            min_ninj=min_ninj, add_title=False, colormap=colormap,
            maxvol=maxvol, minvol=minvol, add_colorbar=False, annotate=False,
            logx=sub_logx, logy=sub_logy, logz=logz, fig=mfig, ax=inset_axes,
            add_clickables=False)
        # make the inset axis limits cover exactly the space of the tiles
        sub_xmin = min([child.get_bound(sub_xarg)[0] \
            for child in parent.children])
        sub_xmax = max([child.get_bound(sub_xarg)[1] \
            for child in parent.children])
        sub_ymin = min([child.get_bound(sub_yarg)[0] \
            for child in parent.children])
        sub_ymax = max([child.get_bound(sub_yarg)[1] \
            for child in parent.children])
        inset_axes.set_xlim(sub_xmin, sub_xmax)
        inset_axes.set_ylim(sub_ymin, sub_ymax)
        if ii == label_idx:
            # make the ticklabels white
            for xy in ['x', 'y']:
                ticklabels = numpy.array(
                    map(str, getattr(inset_axes, 'get_%sticks'%(xy))()))
                getattr(inset_axes,
                    'set_%sticklabels'%(xy))(ticklabels, color='w', fontsize=6)
            # make the axis labels smaller
            inset_axes.set_xlabel(inset_axes.get_xlabel(), color='w',
                fontsize=8, labelpad=2)
            inset_axes.set_ylabel(inset_axes.get_ylabel(), color='w',
                fontsize=8, labelpad=2)
        else:
            # remove tick and axis labels otherwise
            inset_axes.set_xticklabels([])
            inset_axes.set_yticklabels([])
            inset_axes.set_xlabel('')
            inset_axes.set_ylabel('')

    return mfig


def plot_subvolumes_from_layer(layer, user_tag='', min_ninj=1,
        tmplt_label='', inj_label='',
        colormap='hot', maxvol=None, minvol=None, logz=False, dpi=300):
    """
    Wrapper around plot_subvolumes that creates a subvolume plot for every
    parent in the given layer. X and Y arguments/labels of the volume
    plot are retrieved from the given layer's x and y arguments. X and Y
    arguments/labels for the sub-tiles are retrieved from the given layer's
    sub-layer. The resulting figures are saved to disk, with the filenames
    saved to each parent's subtiles_plot attribute.
    """
    # check that we have the necessary directory information
    if not layer.root_dir:
        raise ValueError('no root_dir set for this layer!')
    if not layer.web_dir:
        raise ValueError('no web_dir set for this layer!')
    if not layer.images_dir:
        raise ValueError('no images_dir set for this layer!')
    if not layer.sub_layer:
        raise ValueError('no sub_layer set for this layer!')
    # Plot file template is:
    # layer.imagesdir/plot_subvolumes{-user_tag}-{layer_level}-{parent_index}
    fnametmplt = '%s/plot_subvolumes%s-%i-%i.png'
    if user_tag != '':
        user_tag = '-%s' % user_tag
    for ii,parent in enumerate(layer.parents):
        mfig = plot_subvolumes(
            parent.children, layer.x_param, layer.x_param.label,
            layer.y_param, layer.y_param.label,
            layer.sub_layer.x_param, layer.sub_layer.x_param.label,
            layer.sub_layer.y_param, layer.sub_layer.y_param.label,
            min_ninj=min_ninj, tmplt_label=tmplt_label, inj_label=inj_label,
            colormap=colormap, maxvol=maxvol, minvol=minvol,
            logx=layer.x_distr == 'log10', logy=layer.y_distr == 'log10',
            sub_logx=layer.sub_layer.x_distr == 'log10',
            sub_logy=layer.sub_layer.y_distr == 'log10',
            logz=logz, xmin=layer.plot_x_min, xmax=layer.plot_x_max,
            ymin=layer.plot_y_min, ymax=layer.plot_y_max)

        # save the figure
        plotname = fnametmplt %(layer.images_dir, user_tag, layer.level, ii) 
        print "saving %i-%i..." %(layer.level, ii),
        mfig.savefig('%s%s/%s' %(layer.root_dir, layer.web_dir, plotname),
            dpi=dpi)
        print "done"
        parent.subtiles_plot = mfig
