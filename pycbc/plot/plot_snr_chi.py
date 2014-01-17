import os
import matplotlib
matplotlib.use('Agg')
import pylab
pylab.rcParams.update({
    "text.usetex": True,
    "text.verticalalignment": "center",
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
#import overlapUtils
import plotUtils

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


def plot_snr_chi(results, ref_apprx, test_apprx, plot_newsnrs = [], newsnr_cut = None, plot_reduced = False, plot_trend = False, num_snr_bins = 20, num_chisq_bins = 20, reftest_map = None, xmin = None, xmax = None, ymin = None, ymax = None):

    # XXX: figure size for poster
    fig = pylab.figure(dpi = 300)#, figsize = (16, 12))
    colors = ['b', 'gray']
    ndof = 30# results.values()[0][0].chisq_dof
    min_chisq = numpy.inf
    plot_data = {test_apprx: {}, ref_apprx: {}} 
    for nn, (clr, apprx) in enumerate(zip(colors, [test_apprx, ref_apprx])):
        snrs = numpy.array([x.snr for x in results[apprx]])
        chisqs = numpy.array([x.chisq for x in results[apprx]])
        # check degrees of freedom
        #if any(x.chisq_dof != ndof for x in results[apprx]):
        #    raise ValueError, 'number of degrees of freedom not the same for all triggers'
        if plot_reduced:
            chisqs /= ndof
        min_chisq = min(min_chisq, min(chisqs))
        snr_stds = numpy.array([x.snr_std for x in results[apprx]])
        chisq_stds = numpy.array([x.chisq_std for x in results[apprx]])
        num_samples = numpy.array([x.num_samples for x in results[apprx]])
        #pylab.errorbar(snrs[apprx], chisqs[apprx]/chisq_dofs[apprx], xerr = snr_errs[apprx], yerr = chisq_errs[apprx]/chisq_dofs[apprx], marker = 'o', ms = 5, c = clr, ecolor = clr, mew = 1, linestyle = 'None', label = apprx, zorder = nn)#, alpha = 0.7)
        #pylab.scatter(snrs[apprx], chisqs[apprx]/chisq_dofs[apprx], marker = 'o', s = 5, c = clr, edgecolors = 'none', label = apprx, zorder = nn+1)#, alpha = 0.7)
        pylab.scatter(snrs, chisqs, marker = '+', s = 10, c = clr, edgecolors = clr, label = apprx, zorder = nn+1, alpha = 0.5)
        plot_data[apprx]['snrs'] = snrs
        plot_data[apprx]['chisqs'] = chisqs

    plt_xmin, plt_xmax = pylab.gca().get_xlim()
    plt_ymin, plt_ymax = pylab.gca().get_ylim()
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
    
    yrange = numpy.logspace(numpy.log10(plt_ymin), numpy.log10(plt_ymax), num=100)
    plot_data['newsnrs'] = []
    for newsnr in plot_newsnrs:
        # figure out the x-values
        zrange = numpy.zeros(len(yrange))+newsnr
        if plot_reduced:
            snr = get_snr_from_chisqr_newsnr(yrange, zrange) 
        else:
            snr = get_snr_from_chisqr_newsnr(yrange/results.values()[0][0].chisq_dof, zrange) 
        if newsnr == newsnr_cut:
            ls = '-'
            lw = 2
        else:
            ls = '--'
            lw = 1
        pylab.plot(snr, yrange, color = 'k', linestyle = ls, linewidth = lw, label = '_nolegend_', zorder = len(results.keys())+3) 
        plot_data['newsnrs'].append((newsnr, snr, yrange))
        #pylab.annotate(r'$\hat{\rho} = %.1f$' % newsnr, (newsnr, newsnr), rotation = 90, ha = 'left', va = 'top', fontsize = 'xx-small')

    if plot_trend:
        if reftest_map is None:
            raise ValueError, "must provide a mapping between injections"
        # create an index of the test approximants
        test_idx_map = dict([ [inj.unique_id, inj] for inj in results[test_apprx]])
        
        #use_results = [res for res in results[ref_apprx]+results[test_apprx] if res.snr > 0]
        use_results = [res for res in results[ref_apprx] if res.snr > 0]
        minsnr = min(use_results, key = lambda x: x.snr).snr
        maxsnr = max(use_results, key = lambda x: x.snr).snr + 0.01
        snrbins = numpy.logspace(numpy.log10(minsnr), numpy.log10(maxsnr), num = num_snr_bins+1)

        minchisq = min(use_results, key = lambda x: x.chisq).chisq
        maxchisq = max(use_results, key = lambda x: x.chisq).chisq + 0.01
        chisqbins = numpy.logspace(numpy.log10(minchisq), numpy.log10(maxchisq), num = num_chisq_bins+1)
        if plot_reduced:
            chisqbins /= ndof
        X = []
        Y = []
        U = []
        V = []
        for ii,snrlow in enumerate(snrbins[:-1]):
            snrhigh = snrbins[ii+1]
            thisrow = plotUtils.apply_cut(results[ref_apprx], {'snr':(snrlow, snrhigh)})
            for jj,chisqlow in enumerate(chisqbins[:-1]):
                chisqhigh = chisqbins[jj+1]
                thistile = plotUtils.apply_cut(thisrow, {'chisq/chisq_dof':(chisqlow, chisqhigh)}) 
                if thistile:
                    # get the test injections
                    test_locations = [test_idx_map[reftest_map[inj.unique_id]] for inj in thistile] 
                    
                    # get the mean snr, chisq
                    insnr = numpy.mean([inj.snr for inj in thistile])
                    inchisq = numpy.mean([inj.chisq for inj in thistile]) 
                    foundsnr = numpy.mean([inj.snr for inj in test_locations])
                    foundchisq = numpy.mean([inj.chisq for inj in test_locations])
                    if plot_reduced:
                        inchisq /= ndof
                        foundchisq /= ndof
                    X.append(insnr) #snrlow + (snrhigh - snrlow)/2.)
                    Y.append(inchisq)# + (chisqhigh - chisqlow)/2.)
                    U.append(foundsnr-insnr)
                    V.append(foundchisq-inchisq)
        pylab.quiver(X, Y, U, V, zorder = 10)
        plot_data['trend'] = (X, Y, U, V)

    pylab.xlabel(r'$\rho$')
    if plot_reduced:
        pylab.ylabel(r'$\chi_r^2$')
    else:
        pylab.ylabel(r'$\chi^2$')
    pylab.gca().loglog()
    pylab.xlim(plt_xmin, plt_xmax)
    pylab.ylim(plt_ymin, plt_ymax)
    pylab.grid(zorder = -1)
    pylab.legend(loc = 'upper left')

    return fig, plot_data


def plot_snrchi_density(results, ref_apprx, test_apprx, num_snr_bins, num_chisq_bins, plot_newsnrs = [], newsnr_cut = None, plot_reduced = False, ndof = None, xmin = None, xmax = None, ymin = None, ymax = None, dpi = 300):

    minsnr = min(results[ref_apprx]+results[test_apprx], key = lambda x: x.snr).snr
    maxsnr = max(results[ref_apprx]+results[test_apprx], key = lambda x: x.snr).snr + 0.01
    #snrbins = numpy.logspace(numpy.log10(minsnr), numpy.log10(maxsnr), num = num_snr_bins)
    snrbins = numpy.linspace(minsnr, maxsnr, num = num_snr_bins)

    minchisq = min(results[ref_apprx]+results[test_apprx], key = lambda x: x.chisq).chisq
    maxchisq = max(results[ref_apprx]+results[test_apprx], key = lambda x: x.chisq).chisq + 0.01
    #chisqbins = numpy.logspace(numpy.log10(minchisq), numpy.log10(maxchisq), num = num_snr_bins)
    chisqbins = numpy.linspace(minchisq, maxchisq, num = num_snr_bins)
    if plot_reduced:
        chisqbins /= ndof
    
    fig = pylab.figure()
    
    levels = None # [0.68, 0.95, 0.99]
    #xgrid, ygrid = numpy.meshgrid(snrbins[:-1], chisqbins[:-1])
    xgrid, ygrid = numpy.meshgrid(snrbins[:-1] + numpy.diff(snrbins)/2., chisqbins[:-1] + numpy.diff(chisqbins)/2.)
    zgrid = numpy.zeros(xgrid.shape)
    cmaps = [pylab.cm.Reds, pylab.cm.Blues]
    for nn, (apprx, cmap) in enumerate(zip([ref_apprx, test_apprx], cmaps)):
        for ii,snrlow in enumerate(snrbins[:-1]):
            snrhigh = snrbins[ii+1]
            thisrow = plotUtils.apply_cut(results[apprx], {'snr':(snrlow, snrhigh)})
            for jj,chisqlow in enumerate(chisqbins[:-1]):
                chisqhigh = chisqbins[ii+1]
                thistile = plotUtils.apply_cut(thisrow, {'chisq/chisq_dof':(chisqlow, chisqhigh)}) 
                try:
                    zgrid[ii, jj] = len(thistile) / float(len(thisrow))
                except:
                    continue

        #xvals = numpy.array([x.snr for x in results[apprx]])
        #yvals = numpy.array([x.chisq/ndof for x in results[apprx]])
        #if plot_reduced:
        #    yvals /= ndof
        #zgrid, xgrid, ygrid = numpy.histogram2d(xvals, yvals, bins = [snrbins, chisqbins])
        #xgrid = xgrid[:-1]# + 10**(numpy.diff(numpy.log10(xgrid))/2.)
        #ygrid = ygrid[:-1]# + 10**(numpy.diff(numpy.log10(ygrid))/2.)
        #xgrid, ygrid = numpy.meshgrid(xgrid, ygrid)
        # normalize by the number in snr bin
        #norm = zgrid.sum(axis = 0)
        #for ii,thisnorm in enumerate(norm):
        #    zgrid[ii,:] /= thisnorm
        # create the contour
        cs = pylab.contour(xgrid, ygrid, zgrid, levels = levels, cmap = cmap, label = apprx, alpha = 0.5, zorder = nn, legend = apprx)
        pylab.clabel(cs, inline = 1, fontsize = 10)

    plt_xmin, plt_xmax = pylab.gca().get_xlim()
    plt_ymin, plt_ymax = pylab.gca().get_ylim()

    # plot newsnrs
    yrange = numpy.logspace(numpy.log10(plt_ymin), numpy.log10(plt_ymax), num=100)
    for newsnr in plot_newsnrs:
        # figure out the x-values
        zrange = numpy.zeros(len(yrange))+newsnr
        if plot_reduced:
            snr = get_snr_from_chisqr_newsnr(yrange, zrange) 
        else:
            snr = get_snr_from_chisqr_newsnr(yrange/results.values()[0][0].chisq_dof, zrange) 
        if newsnr == newsnr_cut:
            ls = '-'
            lw = 2
        else:
            ls = '--'
            lw = 1
        pylab.plot(snr, yrange, color = 'k', linestyle = ls, linewidth = lw, label = '_nolegend_', zorder = len(results.keys())+3) 

    if xmin:
        plt_xmin = xmin
    if xmax:
        plot_xmax = xmax
    if ymin:
        plt_ymin = ymin
    if ymax:
        plt_ymax = ymax

    pylab.xlabel(r'$\rho$')
    if plot_reduced:
        pylab.ylabel(r'$\chi_r^2$')
    else:
        pylab.ylabel(r'$\chi^2$')

    #pylab.gca().loglog()
    pylab.xlim(plt_xmin, plt_xmax)
    pylab.ylim(plt_ymin, plt_ymax)
    pylab.grid(zorder = -1)
    #pylab.legend()

    return fig

def plot_snr_compare(results, reftest_map, xarg, xlabel, zarg, zlabel, ref_apprx, test_apprx, ref_apprx_lbl = None, test_apprx_lbl = None, frac_diff = False, xrad = False, invertz = False, plot_mean = False, mean_min_bin = None, mean_max_bin = None, xmin = None, xmax = None, ymin = None, ymax = None, zmin = None, zmax = None):

    # XXX: figure size for poster
    fig = pylab.figure(dpi = 300, figsize = (16, 12))

    if ref_apprx_lbl is None:
        ref_apprx_lbl = ref_apprx

    if test_apprx_lbl is None:
        test_apprx_lbl = test_apprx

    if invertz:
        cmap = pylab.cm.jet_r
    else:
        cmap = pylab.cm.jet

    # create an index of the test approximants
    test_idx_map = dict([ [inj.unique_id, inj] for inj in results[test_apprx]])
    use_results = [res for res in results[ref_apprx] if res.snr > 0]
    
    # get difference in snr
    snrdiff = numpy.array([test_idx_map[reftest_map[inj.unique_id]].snr - inj.snr for inj in use_results])
    if frac_diff:
        snrdiff /= numpy.array([inj.snr for inj in use_results])
        ylabel = r'$(\rho_{\mathrm{%s}} - \rho_{\mathrm{%s}})/\rho_{\mathrm{%s}}$' %(test_apprx_lbl, ref_apprx_lbl, ref_apprx_lbl)
    else:
        ylabel = r'$\rho_{\mathrm{%s}} - \rho_{\mathrm{%s}}$' %(test_apprx_lbl, ref_apprx_lbl)
    xvals = numpy.array([plotUtils.get_arg(test_idx_map[reftest_map[inj.unique_id]], xarg) for inj in use_results])
    if xrad:
        xvals /= numpy.pi
    zvals = [plotUtils.get_arg(test_idx_map[reftest_map[inj.unique_id]], zarg) for inj in use_results]

    if plot_mean:
        nbins = 20
        if mean_min_bin is not None:
            min_xval = mean_min_bin
        else:
            min_xval = xvals.min()
        if mean_max_bin is not None:
            max_xval = mean_max_bin
        else:
            max_xval = xvals.max()
        xbins = numpy.linspace(min_xval, max_xval, num = nbins+1)
        mean_vals = numpy.zeros(nbins)
        mean_err = numpy.zeros(nbins)
        for nn,min_val in enumerate(xbins[:-1]):
            max_val = xbins[nn+1]
            bin_idx = (numpy.intersect1d(numpy.where(xvals >= min_val)[0], numpy.where(xvals < max_val)[0]),)
            mean_vals[nn] = snrdiff[bin_idx].mean()
            mean_err[nn] = snrdiff[bin_idx].std()/numpy.sqrt(float(len(bin_idx[0])))
        xbins = xbins[:-1] + numpy.diff(xbins)/2.

    pylab.scatter(xvals, snrdiff, c = zvals, s = 4, edgecolors = 'none', vmin = zmin, vmax = zmax, cmap = cmap)
    cb = pylab.colorbar()
    cb.ax.set_ylabel(zlabel)

    if plot_mean:
        #pylab.errorbar(xbins, mean_vals, yerr = mean_err, linestyle = '--', c = 'k', linewidth = 2, elinewidth = 2, zorder = 10)
        pylab.plot(xbins, mean_vals, 'k--', linewidth = 3, zorder = 10)

    pylab.ylabel(ylabel)
    pylab.xlabel(xlabel)

    plt_xmin, plt_xmax = pylab.gca().get_xlim()
    plt_ymin, plt_ymax = pylab.gca().get_ylim()
    if xmin is not None:
        plt_xmin = xmin
    if xmax is not None:
        plt_xmax = xmax
    if ymin is not None:
        plt_ymin = ymin
    if ymax is not None:
        plt_ymax = ymax
    pylab.xlim(plt_xmin, plt_xmax)
    pylab.ylim(plt_ymin, plt_ymax)

    pylab.grid(zorder = -1)

    return fig
