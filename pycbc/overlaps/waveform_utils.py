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

from pylal import ligolw_sqlutils as sqlutils
from pycbc.overlaps import scratch
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
    # the resizing is done usin lal to allow for finer control
    newh = h.lal()
    lal.ResizeREAL8TimeSeries(newh, 0, N-position)
    lal.ResizeREAL8TimeSeries(newh, -1*position, N)
    return pyTypes.TimeSeries(newh.data.data, newh.deltaT, epoch = newh.epoch)

def get_htilde(h):#, N, df, fftplan = None):
    return filter.make_frequency_series(h)

#
#   Template Utilities
#
class Template:
    params = ['event_id',
        'mass1', 'mass2',
        'spin1x', 'spin1y', 'spin1z',
        'spin2x', 'spin2y', 'spin2z',
        'template_duration', 'f_final', 'channel']
    __slots__ = params + ['_min_sample_rate', '_approximant', '_f_min', '_waveform']

    def __init__(self, approximant, f_min, **kwargs):
        default = None
        [setattr(self, param, kwargs.pop(param, default)) for param in self.params]
        # force everything but channel and event_id to be 0 if None
        skipParams = ['channel', 'event_id'] 
        [setattr(self, param, 0.) for param in self.params if param not in skipParams and getattr(self, param) is None]
        self._min_sample_rate = self.f_final
        self._approximant = approximant
        self._f_min = f_min
        self._cache_id = None
        self.__archive = {}

    @property
    def q(self):
        return self.mass1 / self.mass2

    @property
    def mtotal(self):
        return self.mass1 + self.mass2

    @property
    def eta(self):
        return self.mass1*self.mass2 / (self.mtotal)**2.

    @property
    def mchirp(self):
        return self.eta**(3./5)*self.mtotal

    @property
    def tau0(self):
        return (5./256*numpy.pi*self._f_min*self.eta)*(numpy.pi*self.mtotal*self._f_min)**(-5./3.)

    def parse_channel(self):
        if self.channel is None:
            raise ValueError, "No cache file found in channel"
        cache_file, cache_id = self.channel.split(':')
        self._cache_id = cache_id
        return cache_file, cache_id

    def set_cache_id(self, cache_id):
        self._cache_id = cache_id

    def get_cache_id(self):
        return self._cache_id

    def get_archive_name(self):
        return self.__archive.filename

    def set_archive(self, archive):
        self.__archive = archive

    def get_archive(self):
        return self.__archive

    def get_min_sample_rate(self):
        return self._min_sample_rate

    def update_min_sample_rate(self, min_sample_rate):
        self._min_sample_rate = min_sample_rate

    def get_waveform(self, sample_rate, phi0 = 0., inc = 0., dist = 1e6, taper = None, archive = {}, store = True):

        try:
            h = pyTypes.TimeSeries(archive['%i' % sample_rate][self.event_id], delta_t = 1./sample_rate, epoch = lal.LIGOTimeGPS(0))

        except KeyError:
            #TD waveforms
            approximant = lalsim.GetApproximantFromString(self._approximant)
            if lalsim.SimInspiralImplementedTDApproximants(approximant):
                hplus, hcross = lalsim.SimInspiralChooseTDWaveform(
                    float(phi0), 1./sample_rate,
                    self.mass1 * lal.LAL_MSUN_SI, self.mass2 * lal.LAL_MSUN_SI,
                    self.spin1x, self.spin1y, self.spin1z,
                    self.spin2x, self.spin2y, self.spin2z,
                    self._f_min, 0., dist * lal.LAL_PC_SI, inc,
                    0, 0, None, None, 0, 0,
                    approximant)

                taper = get_taper_string(taper)
                lalsim.SimInspiralREAL8WaveTaper(hplus.data, lalsim.GetTaperFromString(taper))

                # CHECKME: is this always true?
                h = hplus

                # convert to pycbc type
                h = pyTypes.TimeSeries(h.data.data, delta_t = h.deltaT, epoch = h.epoch)

                if store:
                    if '%i' % sample_rate not in archive:
                        archive.create_group('%i' % sample_rate)
                    archive['%i' % sample_rate].create_dataset(self.event_id, data = h.data)

            else:
                # FD waveforms not currently supported
                raise ValueError("unrecognized approximant")
        
        return h

class TemplateCache(dict):
    def __init__(self):
        self.__scratch_files = {}

    def set_scratch_file(self, origfn, scratchfn):
        self.__scratch_files[origfn] = scratchfn
      
    def clear_scratch_file(self, scratchfn):
        for origfn, thisf in self.__scratch_files.items():
            if thisf == scratchfn and thisf != origfn:
                self.__scratch_files[origfn] = origfn

    def clear_all_scratch_files(self):
        for fn in self.__scratch_files:
            self.__scratch_files[fn] = fn

    def __setitem__(self, i, y):
        self.__scratch_files[y] = y
        dict.__setitem__(self, i, y)

    def __getitem__(self, y):
        return self.__scratch_files[dict.__getitem__(self, y)]

    def get_orig_file(self, y):
        return dict.__getitem__(self, y)

    def get_orig_files(self):
        return self.__scratch_files.keys()

    def redirection(self, y):
        return self[y] != self.get_orig_file(y)


class TemplateDict(dict):

    def __init__(self):
        self._sort_key = None

    def get_templates(self, connection, approximant, f_min, min_sample_rate = 0, err_on_no_f_final = True, err_on_no_dur = True, verbose = False, only_matching = False):
        if verbose:
            print >> sys.stdout, "getting templates from database"
        self.clear()
        # FIXME: SnglInspiral table doesn't store spins, so we use the alpha
        # parameters
        params = Template.param_map.values()
        sqlquery = "SELECT %s FROM sngl_inspiral" % ', '.join(['sngl_inspiral.'+Template.param_map[param] for param in Template.params]) 
        if only_matching:
            sqlquery += ' JOIN coinc_event_map AS a, coinc_event_map AS b, overlap_results ON a.event_id == sngl_inspiral.event_id AND a.coinc_event_id == b.coinc_event_id AND overlap_results.coinc_event_id == b.coinc_event_id'
        for row in connection.cursor().execute(sqlquery):
            args = dict([ [col, val] for col, val in zip(Template.params, row)])
            tmplt = Template(approximant, f_min, **args)
            # set min_sample_rate
            if err_on_no_f_final and not tmplt.f_final:
                raise ValueError, "no f-final found for template %s" % tmplt.event_id
            tmplt.update_min_sample_rate(max(min_sample_rate, tmplt.f_final))
            # add to self
            self[tmplt.event_id] = tmplt

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


def get_scratch_archive(tmp_path, suffix = '', start_archive = None):
    # get the temp archive name
    tmp_archive = scratch.get_temp_filename(filename = start_archive, suffix = suffix+'.hdf5', tmp_path = tmp_path, copy = start_archive is not None, replace_file = False)
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


class Injection:
    __slots__ = ['params', 'min_sample_rate', 'duration']

    def __init__(self, params, min_sample_rate = 0, duration = 0):
        """
        params: a sim_inspiral instance storing the parameters of the injection
        """
        self.params = params
        self.min_sample_rate = min_sample_rate
        self.duration = duration

    @property
    def geocent_time(self):
        return lal.LIGOTimeGPS(self.params.geocent_end_time, self.params.geocent_end_time_ns)

    def get_min_sample_rate(self):
        return self.min_sample_rate

    def update_min_sample_rate(self, min_sample_rate):
        self.min_sample_rate = min_sample_rate

    def waveform_from_archive(self, archive, sample_rate, fmin):
        if isinstance(archive, h5py.highlevel.File):
            h = pyTypes.TimeSeries(archive['%i' % sample_rate][self.params.simulation_id], delta_t = 1./sample_rate, epoch = self.geocent_time)
        elif isinstance(archive, dict):
            h = archive[sample_rate][self.params.simulation_id]
        else:
            raise ValueError, "unrecognized archive format: got %s, expected dict or h5py file" % type(archive)

        return h

    def fd_waveform_from_archive(self, archive, N, sample_rate, fmin):
        if isinstance(archive, h5py.highlevel.File):
            htilde = pyTypes.FrequencySeries(archive['%i,%i' %(N, sample_rate)][self.params.simulation_id], delta_f = sample_rate/float(N), epoch = self.geocent_time)
        elif isinstance(archive, dict):
            htilde = archive[N, sample_rate][self.params.simulation_id]

        else:
            raise ValueError, "unrecognized archive format: got %s, expected dict or h5py file" % type(archive)

        return htilde

    def store_waveform(self, h, archive, sample_rate):
        if isinstance(archive, h5py.highlevel.File):
            if '%i' % sample_rate not in archive:
                archive.create_group('%i' % sample_rate)
            archive['%i' % sample_rate].create_dataset(self.params.simulation_id, data = h.data)
        elif isinstance(archive, dict):
            archive.setdefault(sample_rate, {})
            archive[sample_rate][self.params.simulation_id] = h
        else:
            raise ValueError, "unrecognized archive format: got %s, expected dict or h5py file" % type(archive)

    def store_fd_waveform(self, htilde, archive, N, sample_rate):
        if isinstance(archive, h5py.highlevel.File):
            if '%i,%i' % (N, sample_rate) not in archive:
                archive.create_group('%i,%i' % (N, sample_rate))
            archive['%i,%i' % (N, sample_rate)].create_dataset(self.params.simulation_id, data = htilde.data)
        elif isinstance(archive, dict):
            archive.setdefault((N, sample_rate), {})
            archive[N, sample_rate][self.params.simulation_id] = htilde
        else:
            raise ValueError, "unrecognized archive format: got %s, expected dict or h5py file" % type(archive)

    def get_waveform(self, sample_rate, fmin, apply_ifo_response = True, ifo = None, optimally_oriented = False, archive = {}, store = True):

        try:
            h = self.waveform_from_archive(archive, sample_rate, fmin)

        except KeyError:

            if apply_ifo_response and ifo is None:
                raise ValueError, "must specify an ifo to calculate ifo response"
            #TD waveforms
            approximant = lalsim.GetApproximantFromString(str(self.params.waveform))
            #try:
            #    order = lalsim.GetOrderFromString(str(self.params.waveform))
            #except:
            #    order = 0
            order = 0
            if optimally_oriented:
                inc = 0.
            else:
                inc = self.params.inclination
            if apply_ifo_response:
                phi0 = self.params.phi0
            else:
                # so we can just use hplus for h
                phi0 = 0.
            # FIXME: this function doesn't seem to work at the moment
            if lalsim.SimInspiralImplementedTDApproximants(approximant):
                # generate the waveform; we will set the distance to 1Mpc, and rescale later
                hplus, hcross = lalsim.SimInspiralChooseTDWaveform(
                    phi0, 1./sample_rate,
                    self.params.mass1 * lal.LAL_MSUN_SI, self.params.mass2 * lal.LAL_MSUN_SI,
                    self.params.spin1x, self.params.spin1y, self.params.spin1z, self.params.spin2x, self.params.spin2y, self.params.spin2z,
                    fmin, 0., self.params.distance * 1e6 * lal.LAL_PC_SI, inc,
                    0, 0, None, None, self.params.amp_order, order,
                    approximant)
                hplus.epoch = hcross.epoch = self.geocent_time

                # compute the strain in the detector
                if apply_ifo_response:
                    detector = lalsim.InstrumentNameToLALDetector(ifo)
                    h = lalsim.SimDetectorStrainREAL8TimeSeries(hplus, hcross, self.params.longitude, self.params.latitude, self.params.polarization, detector)
                else:
                    h = hplus

                # taper
                lalsim.SimInspiralREAL8WaveTaper(h.data, lalsim.GetTaperFromString(str(self.params.taper)))

                # convert to pycbc type
                h = pyTypes.TimeSeries(h.data.data, delta_t = h.deltaT, epoch = h.epoch)

                if store:
                    self.store_waveform(h, archive, sample_rate)

        return h

    def get_fd_waveform(self, N, sample_rate, fmin, apply_ifo_response = True, ifo = None, optimally_oriented = False, archive = {}, store = True, fftplan = None):

        try:
            htilde = self.fd_waveform_from_archive(archive, N, sample_rate, fmin)

        except KeyError:
            h = self.get_waveform(sample_rate, fmin, apply_ifo_response = apply_ifo_response, ifo = ifo, optimally_oriented = optimally_oriented, archive = archive, store = store)
            # zero pad and transform the injection
            # we put the injection in the middle
            h = zero_pad_h(h, N, N/2)
            htilde = get_htilde(h)#, N, sample_rate / float(N), fftplan)

            if store:
                self.store_fd_waveform(htilde, archive, N, sample_rate)

        return htilde
