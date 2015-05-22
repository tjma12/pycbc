import os
from matplotlib import pyplot
pyplot.rcParams.update({
    "text.usetex": True,
    #"lines.markersize": 12,
    #"lines.markeredgewidth": 2,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.weight": "bold",
    "font.size": 16,
    "axes.titlesize": 24,
    "axes.labelsize": 26,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    })
import numpy
from pycbc.plot import plot_utils

def get_chisqr_from_snr_newsnr(snr, newsnr):
    chisqr = (2*(snr/newsnr)**6. - 1)**(1./3)
    if type(chisqr) == numpy.ndarray:
        numpy.putmask(chisqr, chisqr < 1, 1+numpy.zeros(len(chisqr)))
    elif chisqr < 1:
        chisqr = 1.
    return chisqr

def get_snr_from_chisqr_newsnr(chisqr, newsnr):
    snr = newsnr * ((1+(chisqr)**3.)/2.)**(1./6)
    if type(snr) == numpy.ndarray:
        numpy.putmask(snr, chisqr < 1, newsnr)
    elif chisqr < 1:
        snr = newsnr
    return snr


def plot_snrchi(results, ifo=None, labels=[], colors=[], plot_newsnrs=[],
    newsnr_cut=None, plot_reduced=False, xmin=None, xmax=None, ymin=None,
    ymax=None):
    """
    Makes a scatter plot of chisq values versus snr.

    Parameters
    ----------
    results: list of lists
        The data to plot. Each element should be a list of plot_utils.Result
        instances from which the snr and chisq values can be retrieved. The
        z-order of results will be the order in which they are listed.
    ifo: string
        Specify which ifo the results are from; this will be added to the
        axes label. Default is None, in which case no ifo label will be added.
    labels: {[], list}
        Optionally specify the labels to use for each list of results. Length
        should be the same as the number of lists in results. If none provided,
        each result list will be labelled "group x" where x is the index of
        the result group in results. If there is only one group, no legend
        will be created if no label is provided.
    colors: {[], list}
        Optionally specify the colors to use for each result group. Length
        should be the same as the number of lists in results. If none provided,
        pyplot's cmap.jet_r colormap will be used, with the color value given
        by the relative order of a result list in results. This will also be
        used if any of the elements in colors is set to None.
    plot_newsnrs: {[], list of floats}
        If provided, curves will be plotted at the specified new snrs. Curves
        will be black dashed lines. These can only be plotted if plot_reduced
        is True.
    newsnr_cut: {None, float}
        If specified, an additional bold solid line will be plotted at the
        given newsnr. This can be used to show where a threshold is applied,
        for example.
    plot_reduced: {False, bool}
        Whether or not to plot reduced chisq. Default is False.
    xmin: {None, float}
        Specify the minimum SNR to plot.
    xmax: {None, float}
        Specify the maximum SNR to plot.
    ymin: {None, float}
        Specify the minimum (reduced) chisq to plot.
    ymax: {None, float}
        Specify the maximum (reduced) chisq to plot.

    Returns
    -------
    fig: pyplot.figure
        The figure with the plot axes on it.
    plot_data: dict
        A dictionary of all the data that was plotted.
    """
    if newsnr_cut is not None and newsnr_cut not in plot_newsnrs:
        plot_newsnrs.append(newsnr_cut)
    if plot_newsnrs != [] and not plot_reduced:
        raise ValueError("plot_reduced must be True to plot new snrs")
    if ifo is not None:
        ifo = '%s ' %(ifo)
    else:
        ifo = ''
    fig = pyplot.figure(dpi=300)
    ax = fig.add_subplot(111)
    min_chisq = numpy.inf
    plot_data = {} 
    for ii,result_group in enumerate(results):
        snrs = numpy.array([x.snr for x in result_group])
        if snrs.shape[0] == 0:
            # nothing to plot, continue
            continue
        chisqs = numpy.array([x.chisq for x in result_group])
        ndofs = numpy.array([x.chisq_dof for x in result_group])
        if plot_reduced:
            chisqs /= ndofs
        min_chisq = min(min_chisq, chisqs.min())
        if labels != []:
            lbl = labels[ii]
        else:
            lbl = 'group %i' %(ii)
        if colors != []:
            clr = colors[ii]
        else:
            clr = None
        if clr is None:
            if len(results) == 1:
                clrval = 0. 
            else:
                clrval = ii/float(len(results)-1)
            clr = pyplot.cm.jet_r(clrval)
        ax.scatter(snrs, chisqs, marker='x', s=10, c=clr, edgecolors=clr,
            label=lbl, zorder=ii, alpha=0.8)
        plot_data[lbl] = {}
        plot_data[lbl]['snrs'] = snrs
        plot_data[lbl]['chisqs'] = chisqs

    # if nothing was plotted, just create an empty plot and exit
    if plot_data == {}:
        plot_utils.empty_plot(ax) 
        return fig, plot_data

    plt_xmin, plt_xmax = ax.get_xlim()
    plt_ymin, plt_ymax = ax.get_ylim()
    if xmin:
        plt_xmin = xmin
    if xmax:
        plot_xmax = xmax
    if ymin:
        plt_ymin = ymin
    elif plt_ymin <= 0:
        plt_ymin = 10**(numpy.log10(min_chisq)-0.2)
    if ymax:
        plt_ymax = ymax
    
    yrange = numpy.logspace(numpy.log10(plt_ymin), numpy.log10(plt_ymax),
        num=100)
    plot_data['newsnrs'] = []
    for newsnr in plot_newsnrs:
        # figure out the x-values
        zrange = numpy.zeros(len(yrange))+newsnr
        snr = get_snr_from_chisqr_newsnr(yrange, zrange) 
        if newsnr == newsnr_cut:
            ls = '-'
            lw = 2
        else:
            ls = '--'
            lw = 1
        ax.plot(snr, yrange, color='k', linestyle=ls, linewidth=lw,
            label='_nolegend_', zorder=ii+1)
        plot_data['newsnrs'].append((newsnr, snr, yrange))
        # I tried annotating each curve, but it made the plot kind of messy.
        # Still, leaving this here as a commented out in case someone wants
        # to do this in the future.
        #ax.annotate(r'$\hat{\rho} = %.1f$' % newsnr, (newsnr, newsnr),
        #    rotation=90, ha='left', va='top', fontsize='xx-small')

    ax.set_xlabel(r'$\rho$')
    if plot_reduced:
        ax.set_ylabel(r'$\chi_r^2$')
    else:
        ax.set_ylabel(r'$\chi^2$')
    ax.set_title(ifo)
    ax.loglog()
    ax.set_xlim(plt_xmin, plt_xmax)
    ax.set_ylim(plt_ymin, plt_ymax)
    ax.grid(zorder = -1)
    if labels != [] or len(results) > 1:
        ax.legend(loc='upper left')

    return fig, plot_data


#def plot_snrchi_density(results, ref_apprx, test_apprx, num_snr_bins, num_chisq_bins, plot_newsnrs = [], newsnr_cut = None, plot_reduced = False, ndof = None, xmin = None, xmax = None, ymin = None, ymax = None, dpi = 300):

#    return None
