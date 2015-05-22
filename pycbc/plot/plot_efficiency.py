#! /usr/bin/env python

import os
import sqlite3
import matplotlib
matplotlib.use('Agg')
import pylab
pylab.rcParams.update({
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
import plotUtils


def plot_efficiency(tile):
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    xmin = xmax = None
    for clr,apprx in zip(['r', 'b'], [tile.ref_apprx, tile.test_apprx]):
        dist = tile.distances[apprx][:-1] + numpy.diff(tile.distances[apprx])/2.
        if dist.tolist() == []:
            continue
        if xmin == None:
            xmin = min(dist)
        else:
            xmin = min(xmin, min(dist))
        if xmax == None:
            xmax = max(dist)
        else:
            xmax = max(xmax, max(dist))
        eff = tile.efficiencies[apprx]
        err_low = tile.eff_err_low[apprx]
        err_high = tile.eff_err_high[apprx]
        yerr = [err_low, err_high]
        if eff.any() and err_low.any() and err_high.any():
            ax.errorbar(dist, eff, yerr = yerr, marker = 'o', label = apprx)
    ax.semilogx()
    ax.set_xlabel('distance (Mpc)')
    ax.set_ylabel('efficiency')
    xavg = tile.m1range[0] + abs(tile.m1range)/2.
    yavg = tile.m2range[0] + abs(tile.m2range)/2.
    if tile.x_arg == 'mtotal' and tile.y_arg == 'q':
        title_str = '$M = %.1f \pm %.1f \,\mathrm{M}_\odot, ~ q = %.1f \pm %.1f \,\mathrm{M}_\odot' %(xavg, abs(tile.m1range)/2., yavg, abs(tile.m2range)/2.)
    else:
        title_str = '$m_1 = %.1f \pm %.1f \,\mathrm{M}_\odot, ~ m_2 = %.1f \pm %.1f \,\mathrm{M}_\odot' %(xavg, abs(tile.m1range)/2., yavg, abs(tile.m2range)/2.)
    title_size = 24 
    if abs(tile.s1zrange) != 0 and abs(tile.s1zrange) != numpy.inf:
        s1z = tile.s1zrange[0] + abs(tile.s1zrange)/2.
        title_str += ', ~ \chi_1 = %.2f \pm %.2f' % (s1z, abs(tile.s1zrange)/2.)
        title_size = 'small'
    if abs(tile.s2zrange) != 0 and abs(tile.s2zrange) != numpy.inf:
        s2z = tile.s2zrange[0] + abs(tile.s2zrange)/2.
        title_str += ', ~ \chi_2 = %.2f \pm %.2f' % (s2z, abs(tile.s2zrange)/2.)
        title_size = 'small'
    title_str += '$'
    ax.set_title(title_str, fontsize = title_size)
    if xmin is not None and xmax is not None:
        ax.set_xlim(xmin, xmax)
    ax.legend(loc = 'lower left')

    return fig

def get_signum(val, err):
    """
    Given an error, returns a string for val
    to formated the appropriate number of
    significan figures.
    """
    pwr = ('%e' % err).split('e')[1]
    if pwr.startswith('-'):
        pwr = int(pwr[1:])
        tmplt = '%.' + str(pwr) + 'f'
        return tmplt % val
    else:
        pwr = int(pwr[1:])
        return '%i' %(round(val, -pwr))

def plot_sensitive_volume(tiles, apprx, vmin = None, vmax = None, colormap = 'hot', xmin = None, xmax = None, ymin = None, ymax = None, dpi = 300):
    MpcToGpc = 1e-3
    fig = pylab.figure(dpi = dpi)
    ax = fig.add_subplot(111)
    if vmin is not None:
        minvol = vmin
    else:
        minvol = min(tiles, key = lambda x: x.integrated_eff[apprx]).integrated_eff[apprx] * MpcToGpc**3
    if vmax is not None:
        maxvol = vmax
    else:
        maxvol = max(tiles, key = lambda x: x.integrated_eff[apprx]).integrated_eff[apprx] * MpcToGpc**3
    pts = None
    plot_data = {'volume': [], 'nInj': []}
    for tile in tiles:
        m1Low, m1High = tile.m1range
        m2Low, m2High = tile.m2range
        x = numpy.array([m1Low, m1Low, m1High, m1High])
        y = numpy.array([m2Low, m2High, m2High, m2Low])
        if tile.y_arg == 'q':
            y = numpy.log10(y)
        V = tile.integrated_eff[apprx]*MpcToGpc**3.
        VerrLow = tile.integrated_err_low[apprx]*MpcToGpc**3.
        VerrHigh = tile.integrated_err_high[apprx]*MpcToGpc**3.
        if numpy.isnan(V):
            continue
        clrfac = (V - minvol)/(maxvol - minvol)
        clr = getattr(pylab.cm, colormap)(clrfac)
        ax.fill(x, y, color = clr, zorder = 1)
        plot_data['volume'].append((x, y, V, VerrLow, VerrHigh))
        plot_data['nInj'].append(tile.nsamples[apprx])

    # XXX: fix me
    if tile.y_arg == 'q':
        Ms = numpy.linspace(6, 200, num = 50)
        maxqs = (Ms - 3.)/3.
        ax.plot(Ms, numpy.log10(maxqs), 'k--', lw = 1, zorder = 10)
        Ms = numpy.linspace(200, 360, num = 50)
        maxqs = numpy.array([200./(m2 < 3. and 3. or m2) for m2 in Ms-200.])
        ax.plot(Ms, numpy.log10(maxqs), 'k--', lw = 1, zorder = 10)
    plt_xmin, plt_xmax = ax.get_xlim()
    plt_ymin, plt_ymax = ax.get_ylim()
    if xmin is not None:
        plt_xmin = xmin
    if xmax is not None:
        plt_xmax = xmax
    if ymin is not None:
        plt_ymin = ymin
        if tile.y_arg == 'q':
            plt_ymin = numpy.log10(plt_ymin)
    if ymax is not None:
        plt_ymax = ymax
        if tile.y_arg == 'q':
            plt_ymax = numpy.log10(plt_ymax)
    ax.set_xlim(plt_xmin, plt_xmax)
    ax.set_ylim(plt_ymin, plt_ymax)
    # if y axis is q, have to do log scale manually
    if tile.y_arg == 'q':
        ax.set_ylabel('$q$')
        ticks = ax.get_yticks()
        min_tick = numpy.ceil(min(ticks)*10)/10.
        max_tick = numpy.floor(max(ticks)*10)/10.
        major_ticks = numpy.arange(min_tick, max_tick+1)
        new_ticks = numpy.array([coef*10**pwr for pwr in major_ticks[:-1] for coef in range(1,10)] + [10**major_ticks[-1]])
        ticklabels = [tck % 1. == 0 and '$10^%i$' % tck or '' for tck in numpy.log10(new_ticks)]
        ax.set_yticks(numpy.log10(new_ticks))
        ax.set_yticklabels(ticklabels)
    else:
        ax.set_ylabel('$m_2 ~(\mathrm{M}_\odot)$')
    if tile.x_arg == 'mtotal':
        ax.set_xlabel('Total Mass $(\mathrm{M}_\odot)$')
    else:
        ax.set_xlabel('$m_1 ~(\mathrm{M}_\odot)$')

    return fig, ax, plot_data


def plot_fractional_gain(tiles, vmin = None, vmax = None, annotate_spins = False, colormap = 'jet', xmin = None, xmax = None, ymin = None, ymax = None, dpi = 300):
    fig = pylab.figure(dpi = dpi)
    ax = fig.add_subplot(111)
    if vmin is not None:
        mingain = vmin
    else:
        mingain = min(tiles, key = lambda x: x.fractional_gain).fractional_gain
    if vmax is not None:
        maxgain = vmax
    else:
        maxgain = max(tiles, key = lambda x: x.fractional_gain).fractional_gain
    pts = None
    plot_data = {'tiles': [], 'annotation': [], 'nInj': []}
    for tile in tiles:
        m1Low, m1High = tile.m1range
        m2Low, m2High = tile.m2range
        x = numpy.array([m1Low, m1Low, m1High, m1High])
        y = numpy.array([m2Low, m2High, m2High, m2Low])
        if tile.y_arg == 'q':
            y = numpy.log10(y)
        gain = tile.fractional_gain
        err_low = tile.gain_err_low
        err_high = tile.gain_err_high
        if numpy.isnan(gain):
            continue
        clrfac = (gain - mingain)/(maxgain - mingain)
        clr = getattr(pylab.cm, colormap)(clrfac)
        ax.fill(x, y, color = clr, zorder = 1)
        plot_data['tiles'].append((x, y, clrfac))
        if get_signum(err_low, err_low) != get_signum(err_high, err_high):
            annt_str = ''.join(['', get_signum(gain, min(err_low, err_high)), '\n $\,^{+', get_signum(err_high, err_high), '}_{-', get_signum(err_low, err_low), '}$'])
        else:
            annt_str = ''.join(['', get_signum(gain, err_low), '\n ${\pm', get_signum(err_low, err_low), '}$'])
        fs = 7
        if annotate_spins:
            annt_str += ' \n \\footnotesize{[%.2f, %.2f)} \n \\footnotesize{[%.2f, %.2f)}' %(gain, tile.s1zrange[0], tile.s1zrange[1], tile.s2zrange[0], tile.s2zrange[1])
        # set the text color
        if False:#gain/maxgain <= 0.2 or gain/maxgain >= 0.9:
            clr = 'white' 
        else:
            clr = 'k'
        if tile.y_arg == 'q':
            anntx = m1Low + (m1High-m1Low)/2.
            annty = numpy.log10(m2Low + (m2High-m2Low)/2.)
        else:
            anntx = m1Low + (m1High-m1Low)/2.
            annty = m2Low + (m2High-m2Low)/2.
        pts = ax.annotate(annt_str, (anntx, annty), ha = 'center', va = 'center', color = clr, zorder = 2, fontsize = fs)
        plot_data['annotation'].append((anntx, annty, gain, err_low, err_high))
        plot_data['nInj'].append(tile.nsamples[tile.ref_apprx])

    # XXX: fix me
    if tile.y_arg == 'q':
        Ms = numpy.linspace(6, 200, num = 50)
        maxqs = (Ms - 3.)/3.
        ax.plot(Ms, numpy.log10(maxqs), 'k--', lw = 1, zorder = 10)
        Ms = numpy.linspace(200, 360, num = 50)
        maxqs = numpy.array([200./(m2 < 3. and 3. or m2) for m2 in Ms-200.])
        ax.plot(Ms, numpy.log10(maxqs), 'k--', lw = 1, zorder = 10)
    plt_xmin, plt_xmax = ax.get_xlim()
    plt_ymin, plt_ymax = ax.get_ylim()
    if xmin is not None:
        plt_xmin = xmin
    if xmax is not None:
        plt_xmax = xmax
    if ymin is not None:
        plt_ymin = ymin
        if tile.y_arg == 'q':
            plt_ymin = numpy.log10(plt_ymin)
    if ymax is not None:
        plt_ymax = ymax
        if tile.y_arg == 'q':
            plt_ymax = numpy.log10(plt_ymax)
    if pts is None:
        pts = ax.annotate('Nothing to plot', (plt_xmin+(plt_xmax-plt_xmin)/2., plt_ymin+(plt_ymax-plt_ymin)/2.))
    ax.set_xlim(plt_xmin, plt_xmax)
    ax.set_ylim(plt_ymin, plt_ymax)
    # if y axis is q, have to do log scale manually
    if tile.y_arg == 'q':
        ax.set_ylabel('$q$')
        ticks = ax.get_yticks()
        min_tick = numpy.ceil(min(ticks)*10)/10.
        max_tick = numpy.floor(max(ticks)*10)/10.
        major_ticks = numpy.arange(min_tick, max_tick+1)
        new_ticks = numpy.array([coef*10**pwr for pwr in major_ticks[:-1] for coef in range(1,10)] + [10**major_ticks[-1]])
        ticklabels = [tck % 1. == 0 and '$10^%i$' % tck or '' for tck in numpy.log10(new_ticks)]
        ax.set_yticks(numpy.log10(new_ticks))
        ax.set_yticklabels(ticklabels)
    else:
        ax.set_ylabel('$m_2 ~(\mathrm{M}_\odot)$')
    if tile.x_arg == 'mtotal':
        ax.set_xlabel('Total Mass $(\mathrm{M}_\odot)$')
    else:
        ax.set_xlabel('$m_1 ~(\mathrm{M}_\odot)$')

    return fig, ax, plot_data

def plot_injections_using_ideal(inj_dict, x_arg, y_arg, tmplt_dict = {}, xmin = None, xmax = None, ymin = None, ymax = None, ptsize = 3, dpi = 300):
    """
    Plots which injections are using ideal templates.
    @inj_dict: dictionary containing injection masses. Keys should be 'orig' and 'ideal', with values being
     2-d numpy array in which the first column is mass1 and the second is mass2
    """
    fig = pylab.figure(dpi = dpi)
    ax = fig.add_subplot(111)
    clrs = ['gray', 'orange']
    for nn, (tmplt_type, clr) in enumerate(zip(['orig', 'ideal'], clrs)):
        masses = inj_dict[tmplt_type]
        if masses.tolist() == []:
            continue
        if x_arg == 'mtotal':
            xvals =  masses[:,0] + masses[:,1]
            ax.set_xlabel('Total Mass $(\mathrm{M}_\odot)$')
        else:
            xvals = masses[:,0]
            ax.set_xlabel('$m_1 ~(\mathrm{M}_\odot)$')
        if y_arg == 'q':
            yvals = masses[:,0] / masses[:,1]
            ax.set_ylabel('$q$')
        else:
            yvals = masses[:,1]
            ax.set_ylabel('$m_2 ~(\mathrm{M}_\odot)$')
        ax.scatter(xvals, yvals, c = clr, edgecolors = 'none', s = ptsize, zorder = nn) 
    if tmplt_dict:
        clrs = ['k', (0.5, 0, 0)]
        for nn, (tmplt_type, clr) in enumerate(zip(['orig', 'ideal'], clrs)):
            masses = tmplt_dict[tmplt_type]
            if x_arg == 'mtotal':
                xvals =  masses[:,0] + masses[:,1]
            else:
                xvals = masses[:,0]
            if y_arg == 'q':
                yvals = masses[:,0] / masses[:,1]
            else:
                yvals = masses[:,1]
            ax.scatter(xvals, yvals, edgecolors = clr, s = 20, marker = 'x', zorder = nn+2)

    if y_arg == 'q':
        ax.semilogy()

    plt_xmin, plt_xmax = ax.get_xlim()
    plt_ymin, plt_ymax = ax.get_ylim()
    if xmin is not None:
        plt_xmin = xmin
    if xmax is not None:
        plt_xmax = xmax
    if ymin is not None:
        plt_ymin = ymin
    if ymax is not None:
        plt_ymax = ymax
    ax.set_xlim(plt_xmin, plt_xmax)
    ax.set_ylim(plt_ymin, plt_ymax)

    return fig, ax


def plot_thresholds(tiles, apprx, xmin = None, xmax = None, ymin = None, ymax = None, dpi = 300):
    fig = pylab.figure(dpi = dpi)
    ax = fig.add_subplot(111)
    maxthresh = max(tiles, key = lambda x: x.threshold[apprx]).threshold[apprx]
    minthresh = min(tiles, key = lambda x: x.threshold[apprx]).threshold[apprx]
    pts = None
    for tile in tiles:
        m1Low, m1High = tile.m1range
        m2Low, m2High = tile.m2range
        x = [m1Low, m1Low, m1High, m1High]
        y = [m2Low, m2High, m2High, m2Low]
        if tile.y_arg == 'q':
            y = numpy.log10(y)
        threshold = tile.threshold[apprx]
        if threshold is None:
            continue
        clrfac = (threshold - minthresh)/(maxthresh - minthresh)
        clr = pylab.cm.PuBu(clrfac)
        ax.fill(x, y, color = clr, zorder = 1)
        annt_str = '%.2f' % threshold
        fs = 10
        # set the text color
        if clrfac >= 0.85:
            clr = 'white' 
        else:
            clr = 'k'
        if tile.y_arg == 'q':
            pts = ax.annotate(annt_str, (m1Low + (m1High-m1Low)/2., numpy.log10(m2Low + (m2High-m2Low)/2.)), ha = 'center', va = 'center', color = clr, zorder = 2, fontsize = fs)
        else:
            pts = ax.annotate(annt_str, (m1Low + (m1High-m1Low)/2., m2Low + (m2High-m2Low)/2.), ha = 'center', va = 'center', color = clr, zorder = 2, fontsize = fs)

    plt_xmin, plt_xmax = ax.get_xlim()
    plt_ymin, plt_ymax = ax.get_ylim()
    if xmin:
        plt_xmin = xmin
    if xmax:
        plt_xmax = xmax
    if ymin:
        plt_ymin = ymin
        if tile.y_arg == 'q':
            plt_ymin = numpy.log10(plt_ymin)
    if ymax:
        plt_ymax = ymax
        if tile.y_arg == 'q':
            plt_ymax = numpy.log10(plt_ymax)
    if pts is None:
        pts = ax.annotate('Nothing to plot', (xmin+(xmax-xmin)/2., ymin+(ymax-ymin)/2.))
    ax.set_xlim(plt_xmin, plt_xmax)
    ax.set_ylim(plt_ymin, plt_ymax)
    # if y axis is q, have to do log scale manually
    if tile.y_arg == 'q':
        ax.set_ylabel('$q$')
        ticks = ax.get_yticks()
        min_tick = numpy.ceil(min(ticks)*10)/10.
        max_tick = numpy.floor(max(ticks)*10)/10.
        major_ticks = numpy.arange(min_tick, max_tick+1)
        new_ticks = numpy.array([coef*10**pwr for pwr in major_ticks[:-1] for coef in range(1,10)] + [10**major_ticks[-1]])
        ticklabels = [tck % 1. == 0 and '$10^%i$' % tck or '' for tck in numpy.log10(new_ticks)]
        ax.set_yticks(numpy.log10(new_ticks))
        ax.set_yticklabels(ticklabels)
    else:
        ax.set_ylabel('$m_2 ~(\mathrm{M}_\odot)$')
    if tile.x_arg == 'mtotal':
        ax.set_xlabel('Total Mass $(\mathrm{M}_\odot)$')
    else:
        ax.set_xlabel('$m_1 ~(\mathrm{M}_\odot)$')

    return fig, ax
