#! /usr/bin/env python

import sys
import numpy
import copy
import os, shutil
import h5py
import operator

import lal
import lalsimulation as lalsim

from pycbc import types as pyTypes
from pycbc import filter
from pycbc import waveform

from glue import segments

from pylal import ligolw_sqlutils as sqlutils
import scratch
import tempfile

#
#       Utility Functions
#
def get_taper_string(taper_str):
    if taper_str == "start":
        taper = "TAPER_START"
    elif taper_str == "end":
        taper = "TAPER_END"
    elif taper_str == "start_end":
        taper == "TAPER_STARTEND"
    elif taper_str is None:
        taper = "TAPER_NONE"
    else:
        raise ValueError, "unrecognized taper option %s" % taper_str
    return taper

def calculate_eff_dist(htilde, psd, fmin, snr, fmax = None):
    """
    Calculates the effective distance of the given htilde.

    Note: htilde and psd must be pycbc frequency series
    """
    sigmasq = filter.sigmasq(htilde, psd, fmin, fmax)
    return numpy.sqrt(sigmasq) / snr


def zero_pad_h(h, N, position):
    # check that desired position is in bounds
    if position > N:
        raise ValueError, "position must be <= N"
    # the resizing is done using lal to allow for finer control
    newh = h.lal()
    lal.ResizeREAL8TimeSeries(newh, 0, N-position)
    lal.ResizeREAL8TimeSeries(newh, -1*position, N)
    return pyTypes.TimeSeries(newh.data.data, newh.deltaT, epoch = newh.epoch)


def position_td_template(h, N):
    """
    Zero-pads and postions a time-domain template with segment-length N.  The
    template is placed such that the peak amplitude occurs at the end of the
    segment, with anything afterward wrapped around to the start.

    Parameters
    ----------
    h: pycbc TimeSeries
        The waveform time series to re-position.
    N: int
        The number of points in the segment.

    Returns
    -------
    pycbc TimeSeries
        New time series of length N with the correctly positioned waveform.
    """
    if N < len(h):
        raise ValueError("N must be >= the length of h")

    new_h = pyTypes.TimeSeries(numpy.zeros(N), delta_t=h.delta_t,
        epoch=h.start_time)

    peakidx = (h.data**2).argmax()
    new_h[N-peakidx:] = h[:peakidx]
    new_h[:len(h)-peakidx] = h[peakidx:]

    return new_h


def ligotimegps_from_float(time):
    s, ns = int(numpy.floor(time)), int(round((time % 1)*1e9))
    return lal.LIGOTimeGPS(s, ns)
    

def position_td_injection(h, segment_start, segment_length):
    """
    Zero pads and positions a time-domain injection such that it would
    fit in segment with length segment_length and GPS starting time
    segment_start. Any part of the injection that occurs before the start
    of the segment or after the end of the segment will be clipped. If the
    segment and the injection do not overlap at all, a ValueError is raised.

    Parameters
    ----------
    h: pycbc TimeSeries
        The injection time series. The epoch of the time series should be set
        to the GPS time of the first sample of the injection. This is used to
        determine the placement of the injection.
    segment_start: float or LIGOTimeGPS
        Start time of the segment.
    segment_length: int
        Duration of the segment, in seconds.

    Returns
    -------
    new_h: pycbc TimeSeries
        A new time-series with length equal to that of the segment and
        start time the same as the segment with the injection placed
        appropriately in it. This time series can be added to a data segment
        time series.
    """
    if not isinstance(segment_start, lal.LIGOTimeGPS):
        segment_start = ligotimegps_from_float(segment_start)
    data_segment = segments.segment(segment_start,
        segment_start+segment_length)
    last_sample_idx = numpy.nonzero(h)[0][-1]
    h_segment = segments.segment(h.start_time,
        h.start_time + last_sample_idx*h.delta_t) 
    try:
        common_seg = data_segment & h_segment
        common_dur = float(abs(common_seg))
        common_start, common_end = common_seg 
    except ValueError:
        raise ValueError, "injection starts after the end of the segment!"
    sample_rate = int(1./h.delta_t)
    # if the waveform start/end times or the segment duration is not a
    # multiple of the sample rate, we'll get non-integers for indices below
    # we deal with this by taking the floor of the start index and the ceil
    # of the stop index
    data_start_idx = int(numpy.floor(
        float(common_start - data_segment[0]) * sample_rate))
    h_start_idx = int(numpy.floor(
        float(common_start - h_segment[0]) * sample_rate))
    data_end_idx = data_start_idx + int(numpy.ceil(common_dur * sample_rate))
    h_end_idx = h_start_idx + int(numpy.ceil(common_dur * sample_rate))
    new_h = pyTypes.TimeSeries(numpy.zeros(sample_rate*segment_length),
        delta_t=h.delta_t, epoch=segment_start)
    new_h[data_start_idx:data_end_idx] = h[h_start_idx:h_end_idx]

    return new_h


#
#   Archive Utilities
#
def construct_archive_key(sample_rate, segment_length, dtype, tag,
    hcross=False, hdf5=False):
    """
    Constructs a group key used in an the archive.
    """
    if hcross:
        tag += 'CROSS'
    key = (sample_rate, segment_length, dtype, tag)
    if hdf5:
        key = ','.join(map(str, key))
    return key


def get_scratch_archive(tmp_path, suffix='', start_archive=None):
    # get the temp archive name
    tmp_archive = scratch.get_temp_filename(filename=start_archive,
        suffix=suffix+'.hdf5', tmp_path=tmp_path,
        copy=start_archive is not None, replace_file=False)
    # load it
    if start_archive is not None:
        mode = 'r+'
    else:
        mode = 'w'
    tmp_archive = h5py.File(tmp_archive, mode)

    return tmp_archive


def close_scratch_archive(tmp_archive):
    tmp_fn = tmp_archive.filename
    tmp_archive.close()
    scratch.discard_temp_filename(tmp_fn)


#
#   Functions to estimate PN expansion
#
def schwarzschild_fisco(m1, m2):
    return lal.LAL_C_SI**3. / (6.**(3./2) * numpy.pi * (m1+m2)*lal.LAL_MSUN_SI
            * lal.LAL_G_SI)

# The following are from Poisson & Will
def eta_from_m1m2(m1, m2):
    return m1*m2 / (m1+m2)**2.

def mchirp_from_m1m2(m1, m2):
    return eta_from_m1m2(m1, m2)**(3./5)*(m1+m2)

def so_coupling(m1, m2, s1z, s2z):
    m1 = m1 * lal.LAL_MTSUN_SI
    m2 = m2 * lal.LAL_MTSUN_SI
    eta = eta_from_m1m2(m1, m2)
    M = m1+m2
    return sum([(113 * (m_i/M)**2. + 75 * eta)*s_i for m_i, s_i in
        [(m1, s1z), (m2, s2z)]])/12.

def ss_coupling(m1, m2, s1, s2):
    eta = eta_from_m1m2(m1*lal.LAL_MTSUN_SI, m2*lal.LAL_MTSUN_SI)
    return (eta/48.) * (-247 * numpy.dot(s1, s2) + 721 * s1[2] * s2[2])

def t0PN(m1, m2, f):
    """
    Gives the time-to-coalesence using Newtonian estimate. Also known as tau0.

    Parameters
    ----------
    m1: float
        Mass of the larger object in solar masses
    m2: float
        Mass of the smaller object in solar masses
    f: float
        Starting frequency in Hz

    Returns
    -------
    tau0: float
        The time-to-coalesence from the starting frequency, in seconds.
    """
    mchirp = mchirp_from_m1m2(m1, m2) * lal.LAL_MTSUN_SI
    return (5./256)* mchirp * (numpy.pi * mchirp * f)**(-8./3)

def t1PN(m1, m2, f):
    """
    Gives the 1PN correction of the time-to-coalesence.

    Parameters
    ----------
    m1: float
        Mass of the larger object in solar masses
    m2: float
        Mass of the smaller object in solar masses
    f: float
        Starting frequency in Hz

    Returns
    -------
    t1PN: float
        The 1PN correction of the time-to-coalesence in seconds
    """
    mchirp = mchirp_from_m1m2(m1, m2) * lal.LAL_MTSUN_SI
    eta = eta_from_m1m2(m1, m2)
    M = (m1+m2) * lal.LAL_MTSUN_SI
    return  (4./3) * (743./336 + (11./4)*eta) * (numpy.pi * M * f)**(2./3)

def t1_5PN(m1, m2, s1, s2, f):
    """
    Gives the 1.5PN correction of the time-to-coalesence.

    Parameters
    ----------
    m1: float
        Mass of the larger object in solar masses
    m2: float
        Mass of the smaller object in solar masses
    s1: array of floats
        Spin components of the larger object
    s2: array of floats
        Spin components of the smaller object
    f: float
        Starting frequency in Hz

    Returns
    -------
    t1_5PN: float
        The 1.5PN correction of the time-to-coalesence in seconds
    """
    M = (m1+m2) * lal.LAL_MTSUN_SI
    beta = so_coupling(m1, m2, s1[2], s2[2])
    return -(8./5) * (4*numpy.pi - beta) * (numpy.pi * M * f)

def t2PN(m1, m2, s1, s2, f):
    """
    Gives the 2PN correction of the time-to-coalesence.

    Parameters
    ----------
    m1: float
        Mass of the larger object in solar masses
    m2: float
        Mass of the smaller object in solar masses
    s1: array of floats
        Spin components of the larger object
    s2: array of floats
        Spin components of the smaller object
    f: float
        Starting frequency in Hz

    Returns
    -------
    t2PN: float
        The 2PN correction of the time-to-coalesence in seconds
    """
    eta = eta_from_m1m2(m1, m2)
    sigma = ss_coupling(m1, m2, s1, s2)
    M = (m1+m2) * lal.LAL_MTSUN_SI
    return 2*(3058673./1016064 + (5429./1008)*eta +
        (617./144)*eta**2. - sigma) * (numpy.pi * M * f)**(4./3)

def t_of_F(m1, m2, s1, s2, f, tc=0., order=4):
    """
    Gives the time-to-coalesence at the given frequency using the PN
    approximation to the given order.

    Parameters
    ----------
    m1: float
        Mass of the larger object in solar masses
    m2: float
        Mass of the smaller object in solar masses
    s1: array of floats
        Spin components of the larger object
    s2: array of floats
        Spin components of the smaller object
    f: float
        Starting frequency in Hz
    tc: float
        Time at coalesence in seconds. Default is 0.
    order: int
        Order of largest correction to use (xPN = order/2). Currently supports
        up to 4 (= 2PN).

    Returns
    -------
    tc - t: float
        The amount of time from the given frequency until coalesence,
        in seconds.
    """
    PNs = numpy.array([
            1., 0., t1PN(m1, m2, f), t1_5PN(m1, m2, s1, s2, f),
        t2PN(m1, m2, s1, s2, f)])
    if order > (len(PNs)-1):
        raise ValueError("order must be <= %i" %(len(PNs)-1))
    return tc - t0PN(m1, m2, f)*PNs[:order+1].sum()

def estimate_duration(m1, m2, s1, s2, f0, f, order=4):
    """
    Estimates the inspiral duration between an initial frequency f0 and a final
    frequency f using post-Newtonian approximation to the given order.

    Parameters
    ----------
    m1: float
        Mass of the larger object in solar masses
    m2: float
        Mass of the smaller object in solar masses
    s1: array of floats
        Spin components of the larger object
    s2: array of floats
        Spin components of the smaller object
    f0: float
        Starting frequency in Hz
    f: float
        Ending frequency in Hz
    order: int
        Order of largest correction to use (xPN = order/2). Currently supports
        up to 4 (= 2PN).

    Returns
    -------
    duration: float
        An estimate of the duration of an inspiral waveform between f0 and f,
        in seconds.
    """
    # we do t(f0) - t(f) since t_of_F with tc = 0 returns the number
    # of seconds before coalesence
    return t_of_F(m1, m2, s1, s2, f, order=order) - \
        t_of_F(m1, m2, s1, s2, f0, order=order)
    

def lambda0(m1, m2, f0):
    m1 *= lal.LAL_MTSUN_SI
    m2 *= lal.LAL_MTSUN_SI
    mchirp = mchirp_from_m1m2(m1, m2)
    return 3 * (numpy.pi * mchirp * f0)**(-5./3) / 128.

def lambda2(m1, m2, f0):
    m1 *= lal.LAL_MTSUN_SI
    m2 *= lal.LAL_MTSUN_SI
    mchirp = mchirp_from_m1m2(m1, m2)
    eta = eta_from_m1m2(m1, m2)
    return (5. * (743./336 + 11*eta/4.))/(96 * eta**(2./5)* numpy.pi * mchirp * f0)

def lambda3(m1, m2, s1z, s2z, f0):
    m1 *= lal.LAL_MTSUN_SI
    m2 *= lal.LAL_MTSUN_SI
    mchirp = mchirp_from_m1m2(m1, m2)
    eta = eta_from_m1m2(m1, m2)
    beta = so_coupling(m1, m2, s1z, s2z)
    return -(3 * numpy.pi / (8 * eta**(3./5))) * (1 - beta/(4*numpy.pi)) * \
        (numpy.pi * mchirp * f0)**(-2./3)


#
#   Waveform Base-class
#
class Waveform(object):
    """
    Holds all information needed to generate a waveform.  Also provides
    methods to easily call a waveform from an archive. The archive can be a
    dictionary in memory, or an hdf5 file.
    """

    __slots__ = [
        # intrinsic parameters
        'mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
        'spin2x', 'spin2y', 'spin2z',
        # extrinsic parameters
        'phi0', 'inclination', 'distance',
        # waveform generation parameters
        'sample_rate', 'segment_length', 'f_min', 'f_ref', 'f_max',
        'lambda1', 'lambda2', 'axis_choice', 'modes_flag',
        'amp_order', 'phase_order', 'spin_order', 'tidal_order',
        'approximant', 'taper', 'eccentricity',
        # derived parameters
        'sigma', '_duration', '_f_final',
        # archive parameters
        '_archive', '_archive_id']

    def __init__(self, **kwargs):
        default = None
        [setattr(self, param, kwargs.pop(param, default)) for param in \
            self.__slots__]
        # check for needed args
        required_args = ['mass1', 'mass2', 'approximant', 'f_min']
        for arg in required_args:
            if getattr(self, arg) is None:
                raise ValueError, "please specify %s" % arg
        # set default parameters necessary for waveform generation
        default_zero_args = ['spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y',
            'spin2z', 'phi0', 'inclination', 'f_ref', 'f_max',
            'lambda1', 'lambda2']
        for arg in default_zero_args:
            if getattr(self, arg) is None:
                setattr(self, arg, 0.)
        if self.distance is None:
            self.distance = 1.
        if self.amp_order is None:
            self.amp_order = 0
        if self.phase_order is None:
            self.phase_order = 7
        if self.taper is None:
            self.taper = 'TAPER_NONE'
   
    # derived parameters 
    @property
    def q(self):
        return self.mass1 / self.mass2

    @property
    def mtotal(self):
        return self.mass1 + self.mass2

    @property
    def eta(self):
        return eta_from_m1m2(self.mass1, self.mass2)

    @property
    def mchirp(self):
        return mchirp_from_m1m2(self.mass1, self.mass2)

    @property
    def tau0(self):
        return t0PN(self.mass1, self.mass2, self.f_min)

    def set_f_final(self, f_final=None):
        """
        Sets self._f_final to given value. If None, SimInspiralGetFinalFreq
        in lalsimulation is used. Note: this can result in an error as not
        all waveform approximants are supported by SimInspiralGetFinalFreq.

        Parameters
        ----------
        f_final: float
            What to set self._f_final to. See above.

        Returns
        -------
        f_final: float
            The value that self.f_final is set to.
        """
        if f_final is None:
            f_final = lalsim.SimInspiralGetFinalFreq(
                self.mass1*lal.LAL_MSUN_SI, self.mass2*lal.LAL_MSUN_SI,
                self.spin1x, self.spin1y, self.spin1z,
                self.spin2x, self.spin2y, self.spin2z,
                getattr(lalsim, self.approximant))
        self._f_final = f_final
        return self._f_final 

    def get_f_final(self, cache=False):
        """
        Returns whatever self._f_final is set to.
        """
        return self._f_final 

    def estimate_duration(self, f_min, f_final=None, order=4):
        """
        Uses PN approximation to the given order to estimate the duration of
        the waveform in the time domain.

        Parameters
        ----------
        f_final: float
            Terminating frequency, in Hz. If None, will use Swarzschild ISCO.
        order: int
            Order to use to estimate = twice the PN order. Currently supports
            up to 4 (= 2PN).

        Returns
        -------
        duration: float
            The duration estimate.
        """
        if f_final is None:
            f_final = schwarzschild_fisco(self.mass1, self.mass2)
        return estimate_duration(self.mass1,
            self.mass2, numpy.array([self.spin1x, self.spin1y,
            self.spin1z]), numpy.array([self.spin2x, self.spin2y,
            self.spin2z]), f_min, f_final, order=order)

    def set_duration(self, duration):
        """
        Sets the duration of the waveform to the specified value. Since the
        duration is a derived value, it is up to the user to figure out how to
        determine. For PN approximation, see self.estimate_duration.
        """
        self._duration = duration

    def get_duration(self):
        """
        Returns whatever self._duration is set to.
        """
        return self._duration

    # archive related
    def set_archive(self, archive={}):
        """
        Sets the archive to use for waveform storage and retrieval. 
        If no archive is given, a new dictionary is used.

        Parameters
        ----------
        archive: dict or hdf5 file
            The archive to use. If none specified, a new dictionary is used.

        Returns
        -------
        archive: dict or hdf5 file
            The archive that will be used.
        """
        if not (isinstance(archive, dict) or \
                isinstance(archive, h5py.highlevel.File)):
            raise ValueError, "archive must be either dict or hdf5 file"
        self._archive = archive

    def get_archive(self):
        return self._archive

    def set_archive_id(self, archive_id=None):
        """
        Sets the archive id to use for saving and retrieving this waveform.

        Parameters
        ----------
        archive_id: int or str
            The id to use for this waveform in the archive. Should be unique
            to this instance of Waveform. If no archive_id given, the output
            of python's id() function on self (which is the memory address of
            self) is used.

        Returns
        -------
        archive_id: int or str
            The id that will be used in the archive.
        """
        if archive_id is None:
            archive_id = id(self)
        if not (isinstance(archive_id, int) or isinstance(archive_id, str) or
                isinstance(archive_id, unicode)):
            raise ValueError, "archive_id must be int or str"
        self._archive_id = str(archive_id)
        return self._archive_id

    def get_archive_id(self):
        return self._archive_id

    def store_waveform(self, h, sample_rate, segment_length,
            hcross=None, tag=''):
        # ensure that we have an archive and id set
        if self._archive is None:
            raise ValueError, "No archive set! Run set_archive()"
        if self._archive_id is None:
            raise ValueError, "No archive_id set! Run set_archive_id()"
        # figure out if h is time domain or frequency domain
        if isinstance(h, pyTypes.timeseries.TimeSeries):
            dtype = 'TD'
        elif isinstance(h, pyTypes.frequencyseries.FrequencySeries):
            dtype = 'FD'
        else:
            raise ValueError, "unrecognized waveform datatype"
        if hcross is not None and type(hcross) != type(hplus):
            raise ValueError, "hcross is a different datatype than h"

        archive = self._archive
        if isinstance(archive, h5py.highlevel.File):
            archive_key = construct_archive_key(sample_rate, segment_length,
                dtype, tag, hcross=False, hdf5=True)
            if archive_key not in archive:
                archive.create_group(archive_key)
            archive[archive_key].create_dataset(self._archive_id, data=h.data)
            if hcross is not None:
                archive_key = construct_archive_key(sample_rate,
                    segment_length, dtype, tag, hcross=True, hdf5=True)
                if archive_key not in archive:
                    archive.create_group(archive_key)
                archive[archive_key].create_dataset(self._archive_id,
                    data=hcross.data)

        elif isinstance(archive, dict):
            archive_key = construct_archive_key(sample_rate, segment_length,
                dtype, tag, hcross=False, hdf5=False)
            archive.setdefault(archive_key, {})
            archive[archive_key][self._archive_id] = h
            if hcross is not None:
                archive_key = construct_archive_key(sample_rate,
                    segment_length, dtype, tag, hcross=True, hdf5=False)
                archive[archive_key][self._archive_id] = hcross
        else:
            raise ValueError, "unrecognized archive format: " +\
                "got %s, expected dict or h5py file" % type(archive)


    def waveform_from_archive(self, sample_rate, segment_length,
            dtype, tag=''):
        # ensure that we have an archive and id set
        # note that if none are set a KeyError is raised instead of a
        # ValueError. This is so that no archive set can be treated the
        # same as is the waveform not existing in the archive, allowing
        # get_(td|fd)_waveform to quickly treat both cases in the same manner
        if self._archive is None:
            raise KeyError, "No archive set! Run set_archive()"
        if self._archive_id is None:
            raise KeyError, "No archive_id set! Run set_archive_id()"
        archive = self._archive
        if isinstance(archive, h5py.highlevel.File):
            archive_key = construct_archive_key(sample_rate, segment_length,
                dtype, tag, hcross=False, hdf5=True)
            data = archive[archive_key][self._archive_id]
            if dtype == 'TD':
                h = pyTypes.TimeSeries(data, delta_t=1./sample_rate)
            else:
                h = pyTypes.FrequencySeries(data, delta_f=1./segment_length)
            # get hcross if it exists
            archive_key = construct_archive_key(sample_rate, segment_length,
                dtype, tag, hcross=True, hdf5=True)
            try:
                data = archive[archive_key][self._archive_id]
                if dtype == 'TD':
                    hcross = pyTypes.TimeSeries(data, delta_t=1./sample_rate)
                else:
                    hcross = pyTypes.FrequencySeries(data,
                        delta_f=1./segment_length)
            except KeyError:
                hcross = None

        elif isinstance(archive, dict):
            archive_key = construct_archive_key(sample_rate, segment_length,
                dtype, tag, hcross=False, hdf5=False)
            h = archive[archive_key][self._archive_id]
            # get hcross if it exists
            archive_key = construct_archive_key(sample_rate, segment_length,
                dtype, tag, hcross=True, hdf5=False)
            try:
                hcross = archive[archive_key][self._archive_id]
            except KeyError:
                hcross = None
        else:
            raise ValueError, "unrecognized archive format: " +\
                "got %s, expected dict or h5py file" % type(archive)

        return h, hcross

    
    def del_from_archive(self, sample_rate, segment_length, dtype, tag=''):
        '''
        Deletes an entry with the given arguments from the archive.
        '''
        archive_key = construct_archive_key(sample_rate, segment_length,
            dtype, tag, hcross=False, hdf5=isinstance(archive,
            h5py.highlevel.File))
        try:
            del archive[archive_key][self._archive_id]
        except KeyError:
            pass
        archive_key = construct_archive_key(sample_rate, segment_length,
            dtype, tag, hcross=True, hdf5=isinstance(archive,
            h5py.highlevel.File))
        try:
            del archive[archive_key][self._archive_id]
        except KeyError:
            pass



    def get_td_waveform(self, sample_rate, segment_length=None, position=0,
            store=False):

        try:
            hplus, hcross = self.waveform_from_archive(sample_rate,
                segment_length, 'TD')

        except KeyError:
            
            # if we're going to store, make sure that archive and archive_id
            # have been set before trying to generate the waveform
            if store and (self._archive is None or self._archive_id is None):
                raise ValueError, "In order to store the waveform, an "+\
                    "archive and archive_id must be set."
            approximant = lalsim.GetApproximantFromString(str(self.approximant))
            # check if we need to adjust the spin order
            if self.spin_order is not None:
                wflags = lalsim.SimInspiralCreateWaveformFlags()
                lalsim.SimInspiralSetSpinOrder(wflags, self.spin_order)
            else:
                wflags = None
            #TD waveforms
            if lalsim.SimInspiralImplementedTDApproximants(approximant):
                hplus, hcross = lalsim.SimInspiralChooseTDWaveform(
                    self.phi0, 1./sample_rate,
                    self.mass1 * lal.LAL_MSUN_SI, self.mass2*lal.LAL_MSUN_SI,
                    self.spin1x, self.spin1y, self.spin1z, self.spin2x,
                    self.spin2y, self.spin2z, self.f_min, self.f_ref,
                    self.distance * 1e6 * lal.LAL_PC_SI, self.inclination,
                    self.lambda1, self.lambda2, wflags, None,
                    self.amp_order, self.phase_order,
                    approximant)

                # taper
                lalsim.SimInspiralREAL8WaveTaper(hplus.data,
                    lalsim.GetTaperFromString(str(self.taper)))
                lalsim.SimInspiralREAL8WaveTaper(hcross.data,
                    lalsim.GetTaperFromString(str(self.taper)))

                # convert to pycbc type
                hplus = pyTypes.TimeSeries(hplus.data.data,
                    delta_t=hplus.deltaT)
                hcross = pyTypes.TimeSeries(hcross.data.data,
                    delta_t=hcross.deltaT)

                # zero pad to the desired segment length 
                if segment_length is not None:
                    N = int(sample_rate * segment_length)
                    hplus = zero_pad_h(hplus, N, position)
                    hcross = zero_pad_h(hcross, N, position)

                if store:
                    self.store_waveform(hplus, sample_rate, segment_length,
                        hcross=hcross)

            else:
                # FD waveform
                raise ValueError, "TD version of FD waveforms not " +\
                    "currently supported. Use self.get_fd_waveform and " +\
                    "convert to time series instead."

        return hplus, hcross


    def get_fd_waveform(self, sample_rate, segment_length, position=0,
            store=False):

        try:
            htilde, htilde_cross = self.waveform_from_archive(sample_rate,
                segment_length, 'FD')

        except KeyError:
            # if we're going to store, make sure that archive and archive_id
            # have been set before trying to generate the waveform
            if store and (self._archive is None or self._archive_id is None):
                raise ValueError, "In order to store the waveform, an "+\
                    "archive and archive_id must be set."
            approximant = lalsim.GetApproximantFromString(str(self.approximant))
            if lalsim.SimInspiralImplementedTDApproximants(approximant):
                hplus, hcross = self.get_td_waveform(sample_rate,
                    segment_length, position=position, archive=archive,
                    store=store)
                htilde = filter.make_frequency_series(hplus)
                htilde_cross = filter.make_frequency_series(htilde_cross)

            else:
                # FD waveform, get directly
                if self.spin_order is not None:
                    wflags = lalsim.SimInspiralCreateWaveformFlags()
                    lalsim.SimInspiralSetSpinOrder(wflags, self.spin_order)
                else:
                    wflags = None
                htilde, htilde_cross = lalsim.SimInspiralChooseFDWaveform(
                    self.phi0, 1./segment_length,
                    self.mass1 * lal.LAL_MSUN_SI, self.mass2*lal.LAL_MSUN_SI,
                    self.spin1x, self.spin1y, self.spin1z, self.spin2x,
                    self.spin2y, self.spin2z, self.f_min, self.f_max,
                    self.distance * 1e6 * lal.LAL_PC_SI, self.inclination,
                    self.lambda1, self.lambda2, wflags, None,
                    self.amp_order, self.phase_order,
                    approximant)

                # convert to pycbc type
                htilde = pyTypes.FrequencySeries(htilde.data.data,
                    delta_f=htilde.deltaF)
                htilde_cross = pyTypes.FrequencySeries(htilde_cross.data.data,
                    delta_f=htilde_cross.deltaF)
                # zero-pad out to Nyquist
                N = segment_length*sample_rate/2 + 1
                htilde.resize(N)
                htilde_cross.resize(N)

            if store:
                self.store_waveform(htilde, sample_rate, segment_length,
                    hcross=htilde_cross)

        return htilde, htilde_cross


#
#   Template Utilities
#
class Template(Waveform):

    # we add a tmplt_id for indexing
    __slots__ = Waveform.__slots__ + ['tmplt_id']

    def get_td_waveform(self, sample_rate, segment_length, store=False,
        reposition=True):
        """
        Modifies Waveform's get_td_waveform such that only hplus is used.
        Will also optionally reposition the template such that the peak is
        at the end of the segment.
        """
        try:
            h, _ = self.waveform_from_archive(sample_rate, segment_length,
                'TD')
        except KeyError:
            # if we're going to store, make sure that archive and archive_id
            # have been set before trying to generate the waveform
            if store and (self._archive is None or self._archive_id is None):
                raise ValueError, "In order to store the waveform, an "+\
                    "archive and archive_id must be set."
            h, _ = super(Template, self).get_td_waveform(sample_rate,
                store=False)
            # reposition
            if reposition:
                h = position_td_template(h, segment_length*sample_rate)
            if store:
                self.store_waveform(h, sample_rate, segment_length)
        return h

    def get_fd_waveform(self, sample_rate, segment_length, store=False):
        """
        Modifies Waveform's get_fd_waveform such that only hplus is used,
        and such that the template is placed so that the peak is at the
        end of the segment.
        """
        try:
            htilde, _ = self.waveform_from_archive(sample_rate,
                segment_length, 'FD')
        except KeyError:
            # if we're going to store, make sure that archive and archive_id
            # have been set before trying to generate the waveform
            if store and (self._archive is None or self._archive_id is None):
                raise ValueError, "In order to store the waveform, an "+\
                    "archive and archive_id must be set."
            approximant = lalsim.GetApproximantFromString(
                str(self.approximant))
            # note that if we are storing the waveform, this will also store
            # the time-domain version
            if lalsim.SimInspiralImplementedTDApproximants(approximant):
                h = self.get_td_waveform(sample_rate, segment_length,
                        store=store)
                htilde = filter.make_frequency_series(h)
            else:
                # if FD waveform, just call parent class's version
                htilde, _ = super(Template, self).get_fd_waveform(
                    sample_rate, segment_length, store=False)
            if store:
                self.store_waveform(htilde, sample_rate, segment_length)
        return htilde


class TemplateDict(dict):

    def __init__(self):
        self._sort_key = None

    def get_templates(self, connection, approximant, f_min, amp_order=None,
            phase_order=None, spin_order=None, taper=None, archive={},
            calc_f_final=True, estimate_dur=True, verbose=False,
            only_matching=False):
        if verbose:
            print >> sys.stdout, "getting templates from database"
        self.clear()
        params = ['mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z',
        'spin2x', 'spin2y', 'spin2z', 'sngl_inspiral.event_id']
        sqlquery = "SELECT %s FROM sngl_inspiral" % ', '.join(params)
        if only_matching:
            sqlquery += """
                JOIN
                    coinc_event_map AS a, coinc_event_map AS b
                ON
                    a.event_id == sngl_inspiral.event_id AND
                    a.coinc_event_id == b.coinc_event_id AND
                    b.table_name == "sim_inspiral"
                    """
        for row in connection.cursor().execute(sqlquery):
            args = dict([ [col, val] for col, val in zip(params, row)])
            # convert event_id to tmplt_id
            tmplt_id = args.pop('sngl_inspiral.event_id')
            args['tmplt_id'] = tmplt_id
            tmplt = Template(approximant=approximant, f_min=f_min,
                amp_order=amp_order, phase_order=phase_order,
                spin_order=spin_order, taper=taper, **args)
            if calc_f_final:
                tmplt.set_f_final()
            if estimate_dur:
                # we'll use fisco as the terminating frequency
                tmplt.set_duration(tmplt.estimate_duration(tmplt.f_min))
            tmplt.set_archive(archive)
            tmplt.set_archive_id()
            # add to self
            self[tmplt.tmplt_id] = tmplt

    def clear_sigmas(self):
        """
        Ensures sigma of every template is set to None.
        """
        [setattr(inj, 'sigma', None) for inj in self.values()]

    def set_sort_key(self, key):
        """
        @key: a list of parameter names to use to sort templates;
         order priority will be based on the order of values in key
        """
        self._sort_key = operator.attrgetter(*key)

    def get_sort_key(self):
        return self._sort_key

    @property
    def as_list(self):
        return sorted(self.values(), key = self.get_sort_key())


#
#   Injection Utilities
#
class Injection(Waveform):
    """
    Adds additional parameters to waveform needed to find the strain in a given
    detector. Also can apply the detector's response function to the waveform.
    """
    _injparams = ['ra', 'dec', 'polarization', 'geocent_end_time',
        'geocent_end_time_ns', 'simulation_id']
    __slots__ = Waveform.__slots__ + _injparams

    def __init__(self, **kwargs):
        # make new parameters required inputs, along with distance
        if 'distance' not in kwargs:
            raise ValueError, "please specify distance"
        # set the params using parent class's __init__
        super(Injection, self).__init__(**kwargs)
        # check that the needed injparams are also specified
        for param in self._injparams:
            if getattr(self, param) is None:
                raise ValueError, "please specify %s" % param
        # check that end times are integers
        if not isinstance(self.geocent_end_time, int):
            raise ValueError, "geocent_end_time must be an integer"
        if not isinstance(self.geocent_end_time_ns, int):
            raise ValueError, "geocent_end_time_ns must be an integer"

    @property
    def geocent_time(self):
        return lal.LIGOTimeGPS(self.geocent_end_time, self.geocent_end_time_ns)

    def detector_end_time(self, ifo):
        if ifo is None:
            # just return the geocentric time
            return self.geocent_time
        detector = lalsim.DetectorPrefixToLALDetector(ifo)
        return self.geocent_time + lal.TimeDelayFromEarthCenter(
            detector.location, self.ra, self.dec, self.geocent_time)


    def del_from_archive(self, sample_rate, segment_length, ifo, segment_start,
            dtype, del_unsegmented=True):
        '''
        Deletes an entry with the given arguments from the archive.
        '''
        archive = self._archive
        # try to delete the segmented
        if not isinstance(segment_start, lal.LIGOTimeGPS):
            segment_start = ligotimegps_from_float(segment_start)
        segment_tag = ','.join([str(ifo), str(segment_start.gpsSeconds),
            str(segment_start.gpsNanoSeconds), str(segment_length)])
        archive_key = construct_archive_key(sample_rate, segment_length,
            dtype, segment_tag, hdf5=isinstance(archive, h5py.highlevel.File))
        try:
            del archive[archive_key][self._archive_id]
        except KeyError:
            pass
        # if desired, try to delete the unsegmented
        if dtype == 'TD' and del_unsegmented:
            archive_key = construct_archive_key(sample_rate, segment_length,
                dtype, str(ifo), hdf5=isinstance(archive,
                h5py.highlevel.File))
            try:
                del archive[archive_key][self._archive_id]
            except KeyError:
                pass


    def get_td_waveform(self, sample_rate, segment_length, ifo, segment_start,
            store=False, store_unsegmented=False):
        """
        Modifies Waveform's get_td_waveform such that the detector response
        is applied to the waveform for the given detector.

        Parameters
        ----------
        sample_rate: int
            The sample rate to use to generate the waveform.
        segment_length: int
            The length of the segment, in seconds, that the injection is going
            into. The injection time-series will be 0-padded to the needed
            length.
        ifo: string or None
            The that the waveform is being injected to. If set to None, no
            detector response will be applied.
        segment_start: float or LIGOTimeGPS
            The GPS time of the start of the segment at the given ifo. The
            injection's time series will be zero-padded such that the end time
            of the injection in the desired ifo is the appropriate offset from
            the start of the injection. If ifo is None, the geocentric time
            will be used instead. If part of the injection lies outside of
            [segment_start, segment_start + segment_length), that portion will
            be clipped off.  If no part of the injection falls in the segment,
            an error will be raised.
        store: Bool
            Whether or not to store the waveform in self's archive. Default
            is False. This will store the waveform as it will be placed in the
            desired segment. This is the fastest way to call the same waveform
            in the same segment, but it requires more memory/disk space, and
            it will require regenerating the waveform if a different segment
            is used.
        store_unsegmented: Bool
            Whether or not to store the waveform prior to padding and shifting
            in the archive. This requires less space to store, but will slow
            down repeated calls to the same waveform in the same segment.

        Returns
        -------
        h: TimeSeries
            The time-series of the strain of the injection.
        """
        if not isinstance(segment_start, lal.LIGOTimeGPS):
            segment_start = ligotimegps_from_float(segment_start)
        segment_tag = ','.join([str(ifo), str(segment_start.gpsSeconds),
            str(segment_start.gpsNanoSeconds), str(segment_length)])
        # first try to get the segmented waveform
        try:
            h, _ = self.waveform_from_archive(sample_rate, segment_length,
                'TD', tag=segment_tag)
        except KeyError:
            # if segmented not present in archive, try the unsegmented one
            try:
                h, _ = self.waveform_from_archive(sample_rate, segment_length,
                    'TD', tag=str(ifo))
            except KeyError:
                # neither, generate the waveform
                # if we're going to store, make sure that archive and
                # archive_id have been set before trying to generate the 
                # waveform
                if (store or store_unsegmented) and \
                   (self._archive is None or self._archive_id is None):
                    raise ValueError, "In order to store the waveform, an "+\
                        "archive and archive_id must be set."
                hplus, hcross = super(Injection, self).get_td_waveform(
                    sample_rate, store=False)
            
                # set the epoch:  The epoch is the end time - amount of time
                # between the first sample and the end time, which for
                # time-domain waveforms we'll define as the time of the peak
                # amplitude.
                peak_idx = numpy.argmax(hplus**2 + hcross**2)
                epoch = self.geocent_time - peak_idx * hplus.delta_t
                hplus._epoch = epoch
                hcross._epoch = epoch

                # compute the strain in the detector
                if ifo is None:
                    h = hplus
                else:
                    detector = lalsim.DetectorPrefixToLALDetector(ifo)
                    h = lalsim.SimDetectorStrainREAL8TimeSeries(hplus.lal(),
                            hcross.lal(), self.ra, self.dec,
                            self.polarization, detector)

                    # convert back to pycbc type; note that h.epoch is now
                    # the start of the waveform *at the detector site* (see
                    # line 225 of LALSimulation.c)
                    h = pyTypes.TimeSeries(h.data.data, delta_t=h.deltaT,
                        epoch=h.epoch)

                if store_unsegmented:
                    self.store_waveform(h, sample_rate, segment_length,
                        tag=str(ifo))

            # zero-pad and shift h so that it sits in the appropriate
            # spot in the segment
            h = position_td_injection(h, segment_start, segment_length)

            if store:
                self.store_waveform(h, sample_rate, segment_length,
                    tag=segment_tag)

        return h


    def get_fd_waveform(self, sample_rate, segment_length, ifo, segment_start,
            store=False):
        """
        Modifies Waveform's get_fd_waveform such that the detector response
        is applied to the waveform for the given detector.

        Parameters
        ----------
        sample_rate: int
            The sample rate of the waveform if it were converted to the time
            domain. This divided by 2 gives the Nyquist frequency. The
            waveform will be zero-padded up to.
        segment_length: int
            The length of the segment, in seconds, that the injection is going
            into. The inverse of this gives the frequency step used.
        ifo: string or None
            The that the waveform is being injected to. If set to None, no
            detector response will be applied.
        segment_start: float or LIGOTimeGPS
            The GPS time of the start of the segment at the given ifo. The
            injection's time series will be zero-padded such that the end time
            of the injection in the desired ifo is the appropriate offset from
            the start of the injection. If ifo is None, the geocentric time
            will be used instead. If part of the injection lies outside of
            [segment_start, segment_start + segment_length), that portion will
            be clipped off.  If no part of the injection falls in the segment,
            an error will be raised.
        store: Bool
            Whether or not to store the waveform in the given archive. Default
            is False.

        Returns
        -------
        htilde: FrequencySeries
            The frequency series of the strain of the injection.
        """
        if not isinstance(segment_start, lal.LIGOTimeGPS):
            segment_start = ligotimegps_from_float(segment_start)
        segment_tag = ','.join([str(ifo), str(segment_start.gpsSeconds),
            str(segment_start.gpsNanoSeconds), str(segment_length)])
        try:
            htilde, _ = self.waveform_from_archive(sample_rate,
                segment_length, 'FD', tag=segment_tag)

        except KeyError:
            if not isinstance(segment_start, lal.LIGOTimeGPS):
                segment_start = ligotimegps_from_float(segment_start)
            # if we're going to store, make sure that archive and archive_id
            # have been set before trying to generate the waveform
            if store and (self._archive is None or self._archive_id is None):
                raise ValueError, "In order to store the waveform, an "+\
                    "archive and archive_id must be set."
            approximant = lalsim.GetApproximantFromString(str(self.approximant))
            if lalsim.SimInspiralImplementedTDApproximants(approximant):
                h = self.get_td_waveform(sample_rate, segment_length,
                        ifo, segment_start, store=store)
                htilde = filter.make_frequency_series(h)

            else:
                # FD approximant. We can only use this as an injection if
                # we are not apply a detector response function, and if the
                # injections lies completely within the segment
                if ifo is not None:
                    raise ValueError, """
                        Cannot apply a detector response function when using
                        an FD approximant. Either generate the waveform with
                        no ifo, (carefully) FFT to time-series, then apply,
                        or use an equivalent TD approximant instead."""
                # check that the injection would occurs completely
                # within the segment
                if self._duration is None:
                    if self._f_final is None:
                        # we'll just use schwarzschild as an estimate
                        f_final = schwarzschild_fisco(self.mass1, self.mass2)
                    else:
                        f_final = self._f_final
                    dur = estimate_duration(self.mass1, self.mass2,
                        numpy.array([self.spin1x, self.spin1y, self.spin1z]),
                        numpy.array([self.spin2x, self.spin2y, self.spin2z]),
                        self.f_min, f_final, order=4)
                else:
                    dur = self._duration
                if self.geocent_time - dur < segment_start:
                    raise ValueError, """
                        The injection starts before the start of the segment!
                        An FD approximant cannot be used if the injection
                        overlaps a segment boundary."""
                if self.geocent_time >= segment_start + segment_length:
                    raise ValueError, """
                        The injection ends after the end of the segment!
                        An FD approximant cannot be used if the injection
                        overlaps a segment boundary."""
                
                # ok, we can use the FD approximant
                htilde, _ = super(Injection, self).get_fd_waveform(
                    sample_rate, segment_length, store=False)
                # adjust the time of waveform so that its end time is correct
                tshift = float(segment_start+segment_length - \
                    self.geocent_time)
                f = numpy.arange(len(htilde))*htilde.delta_f
                htilde.data *= numpy.exp(
                    numpy.complex(0,1)*2*numpy.pi*f*tshift)

            if store:
                self.store_waveform(htilde, sample_rate, segment_length,
                    tag=segment_tag)

        return htilde


class InjectionDict(dict):

    sim_inspiral_map = {
        'mass1': 'mass1',
        'mass2': 'mass2',
        'spin1x': 'spin1x',
        'spin1y': 'spin1y',
        'spin1z': 'spin1z',
        'spin2x': 'spin2x',
        'spin2y': 'spin2y',
        'spin2z': 'spin2z',
        'geocent_end_time': 'geocent_end_time',
        'geocent_end_time_ns': 'geocent_end_time_ns',
        'distance': 'distance',
        'latitude': 'dec',
        'longitude': 'ra',
        'polarization': 'polarization',
        'phi0': 'phi0',
        'theta0': 'inclination',
        'amp_order': 'amp_order',
        'f_final': '_f_final',
        'taper': 'taper',
        'waveform': 'approximant',
        'simulation_id': 'simulation_id',
        # FIXME: fix these kludges when new tables are used 
        'eff_dist_t': 'sigma',
        'alpha': 'duration',
        'numrel_mode_min': 'phase_order',
        'numrel_mode_max': 'spin_order'
    }

    def __init__(self):
        self._sort_key = None

    def get_injections(self, connection, f_min, archive={}, calc_f_final=True,
            estimate_dur=True, verbose=False):
        if verbose:
            print >> sys.stdout, "getting injections from database"
        self.clear()
        
        params = self.sim_inspiral_map.keys()
        sqlquery = "SELECT %s FROM sim_inspiral" % ', '.join(params)
        for row in connection.cursor().execute(sqlquery):
            args = dict([ [self.sim_inspiral_map[col], val] for col, val in \
                zip(params, row)])
            inj = Injection(f_min=f_min, **args)
            inj.f_min = f_min
            if calc_f_final:
                inj.set_f_final()
            if estimate_dur:
                # we'll use fisco as the terminating frequency
                inj.set_duration(inj.estimate_duration(inj.f_min))
            inj.set_archive(archive)
            inj.set_archive_id()
            # add to self
            self[inj.simulation_id] = inj

    def clear_sigmas(self):
        """
        Ensures sigma of every injection is set to None.
        """
        [setattr(inj, 'sigma', None) for inj in self.values()]

    def set_sort_key(self, key):
        """
        @key: a list of parameter names to use to sort templates;
         order priority will be based on the order of values in key
        """
        self._sort_key = operator.attrgetter(*key)

    def get_sort_key(self):
        return self._sort_key

    @property
    def as_list(self):
        return sorted(self.values(), key=self.get_sort_key())
