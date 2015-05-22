import os
from matplotlib import pyplot
width = 3.4
height = 3.0
dpi = 300
pyplot.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.weight": "bold",
    "figure.dpi": dpi,
    "figure.figsize": (width, height),
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
    })
import math
import numpy
from pycbc.plot import plot_utils
import lal
import lalsimulation as lalsim
from pycbc.overlaps import waveform_utils
from pycbc.overlaps import overlap_utils

def ColorFormatter(y, pos):
    return "$10^{%.1f}$" % y

def plot_waveforms(h, hprime, dpi = 300):
    fig = pyplot.figure()
    # plot the full waveforms
    ax = fig.add_subplot(111)
    time = hprime.deltaT * numpy.arange(hprime.data.length)# + hprime.epoch.gpsSeconds + 1e-9*h.epoch.gpsNanoSeconds
    ax.plot(time, hprime.data.data, c = 'r', label = 'Injection', zorder = 2)
    time = h.deltaT * numpy.arange(h.data.length) - (hprime.epoch.gpsSeconds + 1e-9*hprime.epoch.gpsNanoSeconds - (h.epoch.gpsSeconds + 1e-9*h.epoch.gpsNanoSeconds))
    ax.plot(time, h.data.data, c = 'k', label = 'Template', zorder = 1)

    ax.set_xlabel('Time')
    ax.set_ylabel('Strain')

    return fig

def get_phase_diff(h, hprime, overlap_fmin, psd_fmin, psd_model = 'aLIGOZeroDetHighPower', asd_file = None):
    import lal
    import lalsimulation as lalsim
    
    workSpace = overlap_utils.WorkSpace()

    # get the psd model to use
    if asd_file is None:
        min_length = 0
    # if using an asd file, the df must be greater than what's in the file
    else:
        min_length = 1. / overlap_utils.get_asd_file_df(asd_file)
   
    # calculate N
    sample_rate = 1./h.deltaT
    N = 2*int(2**numpy.ceil(numpy.log2(max(h.data.length, hprime.data.length, min_length * sample_rate))))
    df = sample_rate / N
    fftplan = workSpace.get_fftplan(N)

    hpad = waveform_utils.zero_pad_h(h, N, 0, overwrite = False)
    htilde = waveform_utils.convert_lalFS_to_qm(waveform_utils.get_htilde(hpad, N, df, fftplan), scale = qm.DYN_RANGE_FAC*sample_rate)

    hpad = waveform_utils.zero_pad_h(hprime, N, N/2, overwrite = False)
    htildeprime = waveform_utils.convert_lalFS_to_qm(waveform_utils.get_htilde(hpad, N, df, fftplan), scale = qm.DYN_RANGE_FAC*sample_rate)

    # get the psd
    if asd_file is not None:
        psd = workSpace.get_psd_from_file(N, sample_rate, psd_fmin, asd_file)
    else:
        psd = workSpace.get_psd(df, psd_fmin, sample_rate, psd_model)

    cplx_fitfac = qm.cplx_filter(None, htildeprime, htilde, psd, overlap_fmin, None, None).array()
    fitfac = abs(cplx_fitfac)
    maxidx = numpy.where(fitfac == fitfac.max())[0][0]
    phase = numpy.arctan2(cplx_fitfac[maxidx].imag, cplx_fitfac[maxidx].real)

    return phase


def plot_overlap(h, hprime, overlap_fmin, psd_fmin, psd_model = 'aLIGOZeroDetHighPower', asd_file = None, dpi = 300):
    
    workSpace = overlap_utils.WorkSpace()

    # get the psd model to use
    if asd_file is None:
        psd_model = psds.noise_models[psd_model]
        min_length = 0
    # if using an asd file, the df must be greater than what's in the file
    else:
        min_length = 1. / overlap_utils.get_asd_file_df(asd_file)
   
    # calculate N
    sample_rate = 1./h.deltaT
    N = 2*int(2**numpy.ceil(numpy.log2(max(h.data.length, hprime.data.length, min_length * sample_rate))))
    df = sample_rate / N
    fftplan = workSpace.get_fftplan(N)

    hpad = waveform_utils.zero_pad_h(h, N, 0, overwrite = False)
    htilde = waveform_utils.convert_lalFS_to_qm(waveform_utils.get_htilde(hpad, N, df, fftplan), scale = qm.DYN_RANGE_FAC*sample_rate)

    hpad = waveform_utils.zero_pad_h(hprime, N, N/2, overwrite = False)
    htildeprime = waveform_utils.convert_lalFS_to_qm(waveform_utils.get_htilde(hpad, N, df, fftplan), scale = qm.DYN_RANGE_FAC*sample_rate)

    # get the psd
    if asd_file is not None:
        psd = workSpace.get_psd_from_file(N, sample_rate, psd_fmin, asd_file)
    else:
        psd = workSpace.get_psd(df, psd_fmin, sample_rate, psd_model)

    fitfac = qm.fd_fd_match(None, htildeprime, htilde, psd, overlap_fmin, None, None, None).array()
    maxidx = numpy.where(fitfac == fitfac.max())[0][0]

    fig = pyplot.figure()
    ax1 = fig.add_subplot(211)
    time = h.deltaT * (numpy.arange(N) + numpy.nonzero(h.data.data)[0][-1])
    ax1.plot(time, fitfac)
    ax1.set_ylabel('Overlap')
   
    time = h.deltaT * numpy.arange(N)
    ax2 = fig.add_subplot(212)
    ax2.plot(time, hpad.data.data, c = 'r', zorder = 2)

    time = h.deltaT * (numpy.arange(h.data.length) + maxidx)
    ax2.plot(time, h.data.data, c = 'k', zorder = 1)
    ax2.set_ylabel('Strain')
    ax2.set_xlabel('Time (s)')

    injnz = numpy.nonzero(hpad.data.data)[0]
    tmpltnz = numpy.nonzero(h.data.data)[0] + maxidx
    xmin = min(injnz[0], tmpltnz[0]) * h.deltaT
    xmax = max(injnz[-1], tmpltnz[-1]) * h.deltaT
    ax1.set_xlim(xmin, xmax)
    ax2.set_xlim(xmin, xmax)

    return fig

def plot_param_bias(injected_values, recovered_values, param_arg, param_label,
    c_array=None, log_c=False, c_label=None, param_windows=None,
    missed_inj=None, missed_rec=None, xmin=None, xmax=None, ymin=None,
    ymax=None, plot_zoom=False, zoom_xlims=None, zoom_ylims=None):
    """
    Plots (found - injected)/found of given parameter.

    @c_array: array of values to use for the color bar; if None specified,
     will use the fitting factors.
    @log_c: make the color bar logarithmic
    """
    # get injected values
    inj_vals = numpy.array([getattr(inj, param_arg) for inj in injected_values])
    found_vals = numpy.array([getattr(tmplt, param_arg) for tmplt in recovered_values])
    if plot_zoom:
        pyplot.rcParams.update({
            "figure.figsize": (7., 3.)})
        fig = pyplot.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
    if c_array is not None:
        if log_c:
            c_array = numpy.log10(c_array)
        sc = ax.scatter(inj_vals, 100.*(found_vals - inj_vals)/inj_vals,
            c=c_array, edgecolors='none')
        if plot_zoom:
            sc = ax2.scatter(inj_vals, 100*(found_vals - inj_vals)/inj_vals,
                c=c_array, edgecolors = 'none')
        cb = fig.colorbar(sc, format=log_c and pyplot.FuncFormatter(
            ColorFormatter) or None)
        if c_label:
            cb.ax.set_ylabel(c_label)
    else:
        sc = ax.scatter(inj_vals, 100*(found_vals - inj_vals)/inj_vals,
            edgecolors='none')
        if plot_zoom:
            sc = ax2.scatter(inj_vals, 100*(found_vals - inj_vals)/inj_vals,
                edgecolors = 'none')

    if missed_inj is not None and missed_rec is not None:
        xvals = missed_inj
        yvals = 100.*(missed_rec - missed_inj)/missed_inj
        ax.scatter(xvals, yvals, c='k', marker='x', s=30, zorder=10)
        if plot_zoom:
            ax2.scatter(xvals, yvals, c='k', marker='x', s=30, zorder=10)

    if param_windows is not None:
        inj_vals = sorted(inj_vals)
        # the lower bound
        winidx = [param_windows.find(x) for x in inj_vals]
        recbnd = numpy.array([param_windows[ii].min_recovered(x) for ii,x in zip(winidx, inj_vals)])
        yvals = 100.*(recbnd - inj_vals) / inj_vals
        ax.plot(inj_vals, yvals, 'k--', linewidth = 2, zorder = 10)
        if plot_zoom:
            ax2.plot(inj_vals, yvals, 'k--', linewidth = 2, zorder = 10)
        # the upper bound
        recbnd = numpy.array([param_windows[ii].max_recovered(x) for ii,x in zip(winidx, inj_vals)])
        yvals = 100.*(recbnd - inj_vals) / inj_vals
        ax.plot(inj_vals, yvals, 'k--', linewidth = 2, zorder = 10)
        if plot_zoom:
            ax2.plot(inj_vals, yvals, 'k--', linewidth = 2, zorder = 10)

    ax.set_xlabel(param_label)
    ax.set_ylabel(r'(Rec. - Inj.) / Inj. (\%)')
    if plot_zoom:
        ax2.set_xlabel(param_label)
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
    if plot_zoom and zoom_xlims is not None:
        ax2.set_xlim(zoom_xlims)
    if plot_zoom and zoom_ylims is not None:
        ax2.set_ylim(zoom_ylims)
    
    return fig

def plot_recovered_injected(injected_values, recovered_values, param_label,
    param_windows=None, pwin_missed_indices=None, xmin=None, xmax=None,
    ymin=None, ymax=None, plot_zoom=False, zoom_xlims=None, zoom_ylims=None,
    point_size=3, effectualness=[]):
    """
    Plots recovered vs injected of given parameter.

    pwin_missed_indices: list
        List of injections that would be missed if the given parameter window
        list were used.
    """
    if plot_zoom:
        pyplot.rcParams.update({
            "figure.figsize": (7., 3.)})
        fig = pyplot.figure()
        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
    else:
        fig = pyplot.figure()
        ax = fig.add_subplot(111)

    if effectualness != []:
        effectualness = numpy.array(effectualness)
        sort_idx = numpy.argsort(effectualness)[::-1]
        injected_values = injected_values[sort_idx]
        recovered_values = recovered_values[sort_idx]
        effectualness = effectualness[sort_idx]
        clrs = effectualness
    else:
        clrs = 'b'
    sc = ax.scatter(injected_values, recovered_values, s=point_size, c=clrs,
        edgecolors='none', zorder=1)

    if effectualness != []:
        cb = fig.colorbar(sc) 
        cb.set_label('$\mathcal{E}$')

    if plot_zoom:
        sc = ax2.scatter(injected_values, recovered_values, s=point_size, c=clrs,
            edgecolors='none', zorder=1)

    if param_windows is not None:
        if pwin_missed_indices:
            pwin_missed_indices = numpy.array(pwin_missed_indices)
            ax.scatter(injected_values[pwin_missed_indices],
                recovered_values[pwin_missed_indices],
                edgecolors='r', marker='x', s=30, zorder=2)
            if plot_zoom:
                ax2.scatter(injected_values[pwin_missed_indices],
                    recovered_values[pwin_missed_indices],
                    edgecolors='r', marker='x', s=30, zorder=2)

        # plot the windows
        injected_values = sorted(injected_values)

        # the lower bound
        winidx = [param_windows.find(x) for x in injected_values]
        recbnd = numpy.array([param_windows[ii].min_recovered(x) for ii,x in \
            zip(winidx, injected_values)])
        ax.plot(injected_values, recbnd, 'k--', linewidth=2, zorder=3)
        if plot_zoom:
            ax2.plot(injected_values, recbnd, 'k--', linewidth=2, zorder=3)

        # the upper bound
        recbnd = numpy.array([param_windows[ii].max_recovered(x) for ii,x in \
            zip(winidx, injected_values)])
        ax.plot(injected_values, recbnd, 'k--', linewidth=2, zorder=3)
        if plot_zoom:
            ax2.plot(injected_values, recbnd, 'k--', linewidth=2, zorder=3)

    ax.set_xlabel('Injected %s' % param_label)
    ax.set_ylabel('Recovered %s' % param_label)
    if plot_zoom:
        ax2.set_xlabel('Injected %s' % param_label)
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
    if plot_zoom and zoom_xlims is not None:
        ax2.set_xlim(zoom_xlims)
    if plot_zoom and zoom_ylims is not None:
        ax2.set_ylim(zoom_ylims)
    
    return fig
