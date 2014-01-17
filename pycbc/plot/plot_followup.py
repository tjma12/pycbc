import os
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
import math
import numpy
import plotUtils
import lal
import lalsimulation as lalsim
import waveformUtils
import overlapUtils
import qm

def ColorFormatter(y, pos):
    return "$10^{%.1f}$" % y

def plot_waveforms(h, hprime, dpi = 300):
    fig = pylab.figure()
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
    import qm
    import waveformUtils
    import overlapUtils
    
    workSpace = overlapUtils.WorkSpace()

    # get the psd model to use
    if asd_file is None:
        min_length = 0
    # if using an asd file, the df must be greater than what's in the file
    else:
        min_length = 1. / overlapUtils.get_asd_file_df(asd_file)
   
    # calculate N
    sample_rate = 1./h.deltaT
    N = 2*int(2**numpy.ceil(numpy.log2(max(h.data.length, hprime.data.length, min_length * sample_rate))))
    df = sample_rate / N
    fftplan = workSpace.get_fftplan(N)

    hpad = waveformUtils.zero_pad_h(h, N, 0, overwrite = False)
    htilde = waveformUtils.convert_lalFS_to_qm(waveformUtils.get_htilde(hpad, N, df, fftplan), scale = qm.DYN_RANGE_FAC*sample_rate)

    hpad = waveformUtils.zero_pad_h(hprime, N, N/2, overwrite = False)
    htildeprime = waveformUtils.convert_lalFS_to_qm(waveformUtils.get_htilde(hpad, N, df, fftplan), scale = qm.DYN_RANGE_FAC*sample_rate)

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
    
    workSpace = overlapUtils.WorkSpace()

    # get the psd model to use
    if asd_file is None:
        psd_model = psds.noise_models[psd_model]
        min_length = 0
    # if using an asd file, the df must be greater than what's in the file
    else:
        min_length = 1. / overlapUtils.get_asd_file_df(asd_file)
   
    # calculate N
    sample_rate = 1./h.deltaT
    N = 2*int(2**numpy.ceil(numpy.log2(max(h.data.length, hprime.data.length, min_length * sample_rate))))
    df = sample_rate / N
    fftplan = workSpace.get_fftplan(N)

    hpad = waveformUtils.zero_pad_h(h, N, 0, overwrite = False)
    htilde = waveformUtils.convert_lalFS_to_qm(waveformUtils.get_htilde(hpad, N, df, fftplan), scale = qm.DYN_RANGE_FAC*sample_rate)

    hpad = waveformUtils.zero_pad_h(hprime, N, N/2, overwrite = False)
    htildeprime = waveformUtils.convert_lalFS_to_qm(waveformUtils.get_htilde(hpad, N, df, fftplan), scale = qm.DYN_RANGE_FAC*sample_rate)

    # get the psd
    if asd_file is not None:
        psd = workSpace.get_psd_from_file(N, sample_rate, psd_fmin, asd_file)
    else:
        psd = workSpace.get_psd(df, psd_fmin, sample_rate, psd_model)

    fitfac = qm.fd_fd_match(None, htildeprime, htilde, psd, overlap_fmin, None, None, None).array()
    maxidx = numpy.where(fitfac == fitfac.max())[0][0]

    fig = pylab.figure()
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

def plot_param_bias(injected_values, recovered_values, param_arg, param_label, c_array = None, log_c = False, c_label = None, param_windows = None, missed_inj = None, missed_rec = None, xmin = None, xmax = None, ymin = None, ymax = None):
    """
    Plots (found - injected)/found of given parameter.

    @c_array: array of values to use for the color bar; if None specified,
     will use the fitting factors.
    @log_c: make the color bar logarithmic
    """
    fig = pylab.figure()
    # get injected values
    inj_vals = numpy.array([getattr(inj, param_arg) for inj in injected_values])
    found_vals = numpy.array([getattr(tmplt, param_arg) for tmplt in recovered_values])
    if c_array is None:
        c_array = [result.fitting_factor for result in injected_values]
        c_label = 'Fitting Factor'
    if log_c:
        c_array = numpy.log10(c_array)
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    sc = ax.scatter(inj_vals, (found_vals - inj_vals)/inj_vals, c = c_array, edgecolors = 'none')
    sc = ax2.scatter(inj_vals, (found_vals - inj_vals)/inj_vals, c = c_array, edgecolors = 'none')
    cb = fig.colorbar(sc, format = log_c and pylab.FuncFormatter(ColorFormatter) or None)
    if c_label:
        cb.ax.set_ylabel(c_label)
    if param_windows is not None:
        inj_vals = sorted(inj_vals)
        # the lower bound
        winidx = [param_windows.find(x) for x in inj_vals]
        recbnd = numpy.array([param_windows[ii].min_recovered(x) for ii,x in zip(winidx, inj_vals)])
        yvals = (recbnd - inj_vals) / inj_vals
        ax.plot(inj_vals, yvals, 'k--', linewidth = 2, zorder = 10)
        ax2.plot(inj_vals, yvals, 'k--', linewidth = 2, zorder = 10)
        # the upper bound
        recbnd = numpy.array([param_windows[ii].max_recovered(x) for ii,x in zip(winidx, inj_vals)])
        yvals = (recbnd - inj_vals) / inj_vals
        ax.plot(inj_vals, yvals, 'k--', linewidth = 2, zorder = 10)
        ax2.plot(inj_vals, yvals, 'k--', linewidth = 2, zorder = 10)

    if missed_inj is not None and len(missed_inj) > 0:
        ax.scatter(missed_inj, (missed_rec - missed_inj)/missed_inj, c = 'k', marker = 'x', s = 40, zorder = 10)
        ax2.scatter(missed_inj, (missed_rec - missed_inj)/missed_inj, c = 'k', marker = 'x', s = 40, zorder = 10)
    ax.set_xlabel(param_label)
    ax.set_ylabel(r'$\Delta$%s / %s' %(param_label, param_label))
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
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.2, 0.2)#plt_ymin, plt_ymax)
    
    return fig

def plot_recovered_injected(injected_values, recovered_values, param_arg, param_label, c_array = None, log_c = False, c_label = None, param_windows = None, missed_inj = None, missed_rec = None, xmin = None, xmax = None, ymin = None, ymax = None):
    """
    Plots recovered vs injected of given parameter.
    """
    fig = pylab.figure()
    # get injected values
    inj_vals = numpy.array([getattr(inj, param_arg) for inj in injected_values])
    found_vals = numpy.array([getattr(tmplt, param_arg) for tmplt in recovered_values])
    if c_array is None:
        c_array = [result.fitting_factor for result in injected_values]
        c_label = 'Fitting Factor'

    ax1 = fig.add_subplot(121)
    sc = ax1.scatter(inj_vals, found_vals, c = c_array, edgecolors = 'none')
    ax2 = fig.add_subplot(122)
    sc = ax2.scatter(inj_vals, found_vals, c = c_array, edgecolors = 'none')
    cb = fig.colorbar(sc, format = log_c and pylab.FuncFormatter(ColorFormatter) or None)
    if c_label:
        cb.ax.set_ylabel(c_label)
    if param_windows is not None:
        inj_vals = sorted(inj_vals)
        # the lower bound
        winidx = [param_windows.find(x) for x in inj_vals]
        recbnd = numpy.array([param_windows[ii].min_recovered(x) for ii,x in zip(winidx, inj_vals)])
        ax1.plot(inj_vals, recbnd, 'k--', linewidth = 2, zorder = 10)
        ax2.plot(inj_vals, recbnd, 'k--', linewidth = 2, zorder = 10)
        # the upper bound
        recbnd = numpy.array([param_windows[ii].max_recovered(x) for ii,x in zip(winidx, inj_vals)])
        ax1.plot(inj_vals, recbnd, 'k--', linewidth = 2, zorder = 10)
        ax2.plot(inj_vals, recbnd, 'k--', linewidth = 2, zorder = 10)
    if missed_inj is not None and len(missed_inj) > 0:
        ax1.scatter(missed_inj, missed_rec, c = 'k', marker = 'x', s = 40, zorder = 10)
        ax2.scatter(missed_inj, missed_rec, c = 'k', marker = 'x', s = 40, zorder = 10)

    ax1.set_xlabel('Injected %s' % param_label)
    ax1.set_ylabel('Recovered %s' % param_label)
    ax1.set_xlim(0,100)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    return fig
