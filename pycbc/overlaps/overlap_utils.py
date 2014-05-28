#! /usr/bin/env python

import sqlite3
import numpy
import os, sys, shutil, socket
import tempfile
import pickle
import h5py

import lal
import lalsimulation as lalsim
from glue import segments
from glue.ligolw import lsctables
from glue.ligolw import dbtables
from pylal import ligolw_sqlutils as sqlutils

from pycbc import psd as pyPSD
from pycbc.overlaps import waveform_utils

def get_outfilename(output_directory, ifo, user_tag=None, num=None):
    tag = ''
    if user_tag is not None:
        tag = '_%s' % user_tag
    if num is not None:
        tag += '-%i' % (num)
    if ifo is None:
        ifo = 'RF'
    return '%s/%s-OVERLAPS%s.sqlite' %(output_directory, ifo.upper(), tag)


def get_exval_outfilename(output_directory, ifo, user_tag = None, num = None):
    tag = ''
    if user_tag is not None:
        tag = '_%s' % user_tag
    if num is not None:
        tag += '-%i' % (num)
    if ifo is None:
        ifo = 'RF'
    return '%s/%s-EXPECTATION_VALUES%s.sqlite' %(output_directory, ifo.upper(),
        tag)


def copy_to_output(infile, outfile, verbose = False, warn_overwrite = True, num_retries = 2):
    if verbose:
        print >> sys.stdout, "Copying %s to %s..." %(infile, outfile)
    if os.path.exists(outfile) and warn_overwrite:
        print >> sys.stderr, "Warning: File %s being overwritten." %(outfile)
    ii = 0
    copyFailure = True
    while copyFailure and ii <= abs(num_retries):
        shutil.copyfile(infile, outfile)
        ii += 1
        # check
        copyFailure = dbtables.__md5digest(infile) != dbtables.__md5digest(outfile)
    if copyFailure:
        raise ValueError, "md5 checksum failure! Checksum of %s does not match %s after %i tries" % (infile, outfile, num_tries)
    


def get_outfile_connection(infile, outfile, replace = False, verbose = False):
    if infile == outfile:
        raise ValueError, "Input file %s is same as output" % infile
    # check if outfile exists; if so, and not replace,
    # we will copy the working_file from the outfile; otherwise, we will get
    # the working_file from the infile
    #if os.path.exists(outfile) and not replace:
    #    copy_to_output(outfile, working_outfile)
    #else:
    #    copy_to_output(infile, working_outfile)

    #connection = sqlite3.connect(working_outfile)
    if not os.path.exists(outfile) or replace:
        copy_to_output(infile, outfile)

    connection = sqlite3.connect(outfile)

    return connection #working_outfile, connection


def get_template_archive_fname(output_directory, min_sample_rate, max_sample_rate, user_tag = None):
    if user_tag is not None:
        tag = '_'+user_tag
    else:
        tag = ''
    archiveFnTemplate = '%s/HL-TEMPLATE_ARCHIVE%s-%i-%i.hdf5'
    return archiveFnTemplate %(output_directory, tag, min_sample_rate, max_sample_rate)


def get_asd_file_df(asd_file):
    f = open(asd_file, 'r')
    f1, _ = map(float, f.readline().split('  '))
    f2, _ = map(float, f.readline().split('  '))
    f.close()
    return f2 - f1

def get_psd_models():
    return pyPSD.get_lalsim_psd_list()

class WorkSpace:
    def __init__(self):
        self.psds = {}
        self.qs = {}
        self.q_is = {}
        self.qtildes = {}
        self.qtilde_is = {}
        self.fftplans = {}
        self.cplxplans = {}

    def get_psd(self, df, fmin, sample_rate, psd_model):
        N = int(sample_rate/df)
        try:
            return self.psds[N, sample_rate]
        except KeyError:
            self.psds[N, sample_rate] = getattr(pyPSD, psd_model)(N/2 + 1, df, fmin)
            return self.psds[N, sample_rate]

    def get_psd_from_file(self, df, fmin, sample_rate, filename,
            is_asd_file=True):
        N = int(sample_rate/df)
        try:
            return self.psds[N, sample_rate]
        except KeyError:
            self.psds[N, sample_rate] = pyPSD.from_txt(filename, N/2 + 1, df,
                fmin, is_asd_file)
            return self.psds[N, sample_rate]

    def clear_psds(self):
        self.psds.clear()

    def get_fftplan(self, N, store = False):
        try:
            return self.fftplans[N]
        except KeyError:
            fftplan = lal.CreateForwardREAL8FFTPlan(N, 0)
            if store:
                self.fftplans[N] = fftplan 
            return fftplan


def new_snr(snr, chisq, chisq_dof):
    rchisq = float(chisq) / chisq_dof
    if rchisq < 1:
        newsnr = snr
    else:
        newsnr = snr/ ((1+rchisq**3.)/2.)**(1./6)
    return newsnr

class ParamWindow(segments.segment):
    min_jitter = None
    max_jitter = None

    def set_jitter(self, min_jitter, max_jitter):
        self.min_jitter = min_jitter
        self.max_jitter = max_jitter

    def recovery_window(self, value):
        return segments.segment(value*(1. + self.min_jitter), value*(1. + self.max_jitter))

    def in_recovered(self, value):
        return value in self.recovery_window(value)

    def get_injected_window(self):
        return self.injected_window

    @property
    def min_injected(self):
        return self[0]

    @property
    def max_injected(self):
        return self[1]

    def min_recovered(self, value):
        return self.recovery_window(value)[0]

    def max_recovered(self, value):
        return self.recovery_window(value)[1]

    @property
    def mean_injected(self):
        return self.min_injected + abs(self)/2.

class ParamWindowList(segments.segmentlist):
    param = None
    inj_apprx = None
    tmplt_apprx = None

    def set_param(self, param):
        self.param = param

    def set_approximants(self, inj_apprx, tmplt_apprx):
        self.inj_apprx = inj_apprx
        self.tmplt_apprx = tmplt_apprx

    def find(self, val):
        try:
            return segments.segmentlist.find(self, val)
        except ValueError:
            if val < self[0][0]:
                return 0
            else:
                return len(self)-1

    def write_to_database(self, connection, min_s1z = -numpy.inf, min_s2z = -numpy.inf, max_s1z = numpy.inf, max_s2z = numpy.inf):
        sqlquery = """
            CREATE TABLE IF NOT EXISTS match_windows (param, inj_apprx, tmplt_apprx, min_injected, max_injected, min_injected_spin1z, max_injected_spin1z, min_injected_spin2z, max_injected_spin2z, min_jitter, max_jitter);
            CREATE INDEX IF NOT EXISTS mw_pn_injtmpltapprx_minj_manj_idx ON match_windows (param, inj_apprx, tmplt_apprx, min_injected, max_injected);
            """
        connection.cursor().executescript(sqlquery)
        sqlquery = 'INSERT INTO match_windows (param, inj_apprx, tmplt_apprx, min_injected, max_injected, min_injected_spin1z, max_injected_spin1z, min_injected_spin2z, max_injected_spin2z, min_jitter, max_jitter) VALUES (?,?,?,?,?,?,?,?,?,?,?)'
        insert_vals = [(self.param, self.inj_apprx, self.tmplt_apprx, pw.min_injected, pw.max_injected, min_s1z, max_s1z, min_s2z, max_s2z, pw.min_jitter, pw.max_jitter) for pw in self]
        connection.cursor().executemany(sqlquery, insert_vals)

    def load_from_database(self, connection, param, inj_apprx, tmplt_apprx = 'EOBNRv2', min_s1z = None, min_s2z = None, max_s1z = None, max_s2z = None):
        self.param = param
        self.inj_apprx = inj_apprx
        self.tmplt_apprx = tmplt_apprx
        sqlquery = 'SELECT min_injected, max_injected, min_jitter, max_jitter FROM match_windows WHERE param == ? AND inj_apprx == ? AND tmplt_apprx == ?'
        select_params = [param, inj_apprx, tmplt_apprx]
        if min_s1z is not None:
            sqlquery += ' AND min_injected_spin1z == ?'
            select_params.append(min_s1z)
        if min_s2z is not None:
            sqlquery += ' AND min_injected_spin2z == ?'
            select_params.append(min_s2z)
        if max_s1z is not None:
            sqlquery += ' AND max_injected_spin1z == ?'
            select_params.append(max_s1z)
        if max_s2z is not None:
            sqlquery += ' AND max_injected_spin2z == ?'
            select_params.append(max_s2z)
        sqlquery += ' ORDER BY min_injected ASC'
        for min_injected, max_injected, min_jitter, max_jitter in connection.cursor().execute(sqlquery, tuple(select_params)):
            pw = ParamWindow(min_injected, max_injected)
            pw.set_jitter(min_jitter, max_jitter)
            self.append(pw)


class OverlapResult:
    params = ['ifo', 'effectualness', 'snr', 'chisq', 'new_snr', 'chisq_dof',
        'time_offset', 'time_offset_ns', 'snr_std', 'chisq_std', 'new_snr_std',
        'num_tries', 'num_successes', 'sample_rate', 'segment_length',
        'tmplt_approximant', 'overlap_f_min', 'waveform_f_min']
    __slots__ = params + ['template', 'injection']
    
    def __init__(self, template, injection):
        self.template = template
        self.injection = injection
        for param in self.params:
            setattr(self, param, None)

    @property
    def gps_time(self):
        return self.injection.detector_end_time(self.ifo) + \
            lal.LIGOTimeGPS(self.time_offset, self.time_offset_ns)

    def write_to_database(self, connection, coinc_event_id):
        write_params = [param for param in self.params if param in dir(self)]
        sqlquery = """
            INSERT INTO
                overlap_results (coinc_event_id, %s)""" %(
                    ', '.join(write_params)) + """
            VALUES
                (?,%s)""" %(','.join(['?' for ii in range(len(write_params))]))

        connection.cursor().execute(sqlquery, tuple([coinc_event_id] + \
            [getattr(self, col) for col in write_params]))

    def write_all_results_to_database(self, connection):
        """
        Writes the parameters to the all_results table. Does not use a
        coinc_event_id, however.
        """
        write_params = [param for param in self.params if param in dir(self)]+\
            ['simulation_id', 'tmplt_id']
        write_results = tuple([getattr(self, col) for col in self.params \
            if param in dir(self)] + [self.injection.simulation_id,
            self.template.tmplt_id])
        sqlquery = 'INSERT INTO all_results (%s) VALUES (%s)' %(
            ', '.join(write_params),
            ','.join(['?' for ii in range(len(write_params))]))
        connection.cursor().execute(sqlquery, write_results)


def get_injection_template_map(connection):
    """
    Gets injections and templates that are mapped to each other.

    Parameters
    ----------
    connection: sqlite3 connection
        A connection to a SQLite database. The database must have
        a sngl_inspiral table containing templates, a sim_inspiral
        table containing injections, and a coinc_event_map table
        linking the two together.

    Returns
    -------
    template/injection map: dict
        A dictionary keyed by the coinc_event_id of the maps.
        Elements are a tuple of the injection's simulation_id and
        the template's template_id.
    """
    sqlquery = """
        SELECT
            mapA.coinc_event_id,
            sim_inspiral.simulation_id,
            sngl_inspiral.event_id
        FROM
            coinc_event_map AS mapA
        JOIN
            sngl_inspiral, sim_inspiral, coinc_event_map AS mapB
        ON
            mapA.event_id == sim_inspiral.simulation_id AND
            mapA.coinc_event_id == mapB.coinc_event_id AND
            mapB.event_id == sngl_inspiral.event_id
        """
    return dict([ [ceid, (simid, tmpltid)] for ceid, simid, tmpltid in \
        connection.cursor().execute(sqlquery)])
        

def get_overlap_results(connection, verbose=False):
    """
    Gets the overlap_results, injections, and mapping templates.
    Returns a dectionary of overlap results keyed by the coinc_event_id.
    """
    # get the injections and templates
    # since the overlap f_min is stored in the overlap result,
    # we'll just set the injections' fmin to 0
    injections = waveform_utils.InjectionDict()
    injections.get_injections(connection, 0, calc_f_final=False,
        estimate_dur=False, verbose=verbose)
    # ditto for the templates' fmin and approximant
    templates = waveform_utils.TemplateDict()
    templates.get_templates(connection, 0, '',
            calc_f_final=False, estimate_dur=False, verbose=verbose,
            only_matching=True)
    results = {}
    sqlquery = ''.join(["""
        SELECT
            tmpltMap.event_id, injMap.event_id,
            overlap_results.coinc_event_id, """,
            ', '.join(OverlapResult.params), """
        FROM
            overlap_results
        JOIN
            coinc_event_map AS tmpltMap
        ON
            tmpltMap.coinc_event_id == overlap_results.coinc_event_id AND
            tmpltMap.table_name == "sngl_inspiral"
        JOIN
            coinc_event_map AS injMap
        ON
            injMap.coinc_event_id == overlap_results.coinc_event_id AND
            injMap.table_name == "sim_inspiral"
        """])
    for result in connection.cursor().execute(sqlquery):
        evid, simid, ceid = result[0], result[1], result[2]
        params = [result[ii] for ii in range(3, len(result))]
        thisResult = OverlapResult(templates[evid], injections[simid])
        [setattr(thisResult, param, val) for param, val in \
            zip(OverlapResult.params, params)]
        results[ceid] = thisResult

    return results


def create_all_results_table(connection):
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS all_results (""",
            ', '.join(OverlapResult.__slots__+['tmplt_id', 'simulation_id']),
            """);
        CREATE INDEX IF NOT EXISTS ar_eid_idx ON all_results (tmplt_id);
        CREATE INDEX IF NOT EXISTS ar_sid_idx ON all_results (simulation_id);
        CREATE INDEX IF NOT EXISTS ar_eff_idx ON all_results (effectualness);
        """])
    connection.cursor().executescript(sqlquery)


def create_results_table(connection):
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS
            overlap_results (coinc_event_id PRIMARY KEY, """,
            ', '.join(OverlapResult.params), """);
        CREATE INDEX IF NOT EXISTS
            or_ceid_idx ON overlap_results (coinc_event_id);
        """])
    connection.cursor().executescript(sqlquery)

############################
#
#   FIXME: The following should be moved to ligolw_sqlutils
#
def create_cem_table(connection):
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS coinc_event_map (""",
        ', '.join(lsctables.CoincMapTable.validcolumns.keys()), """);
        CREATE INDEX IF NOT EXISTS cem_tn_ei_index ON coinc_event_map (table_name, event_id);
        CREATE INDEX IF NOT EXISTS cem_cei_index ON coinc_event_map (coinc_event_id);
        """])
    connection.cursor().executescript(sqlquery)

def add_coinc_event_map_entry(connection, coinc_event_id, event_id, table_name):
    sqlquery = 'INSERT INTO coinc_event_map (coinc_event_id, table_name, event_id) VALUES (?, ?, ?)'
    connection.cursor().execute(sqlquery, (coinc_event_id, table_name, event_id))


def create_coinc_definer_table(connection):
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS coinc_definer (""",
        ', '.join(lsctables.CoincDefTable.validcolumns.keys()),
        ', %s' %(lsctables.CoincDefTable.constraints), """);
        CREATE INDEX IF NOT EXISTS cd_des_index ON coinc_definer (description);
        """])
    connection.cursor().executescript(sqlquery)

def create_coinc_event_table(connection):
    cursor = connection.cursor()
    sqlquery = ''.join(["""
        CREATE TABLE IF NOT EXISTS coinc_event (""",
        ', '.join(lsctables.CoincTable.validcolumns.keys()),
        ', %s' %(lsctables.CoincTable.constraints), ')'])
    cursor.execute(sqlquery)
    for idx_name, idx_cols in lsctables.CoincTable.how_to_index.items():
        sqlquery = 'CREATE INDEX IF NOT EXISTS %s ON coinc_event (%s)' %(idx_name, ','.join(idx_cols))
        cursor.execute(sqlquery)
    connection.commit()

sqlutils.create_cem_table = create_cem_table
sqlutils.add_coinc_event_map_entry = add_coinc_event_map_entry
sqlutils.create_coinc_definer_table = create_coinc_definer_table
sqlutils.create_coinc_event_table = create_coinc_event_table
############################

def create_log_table(connection):
    sqlquery = """
        CREATE TABLE IF NOT EXISTS
            joblog (time, simulation_id, inj_num, tmplt_id, tmplt_num,
                pickle_file, backup_archive, scratch_archive, host, username)
        """
    connection.cursor().execute(sqlquery)
    connection.commit()


def get_startup_data(connection):
    sqlquery = """
        SELECT
            simulation_id, inj_num, tmplt_id, tmplt_num, pickle_file,
            backup_archive, scratch_archive, host, username
        FROM
            joblog
        ORDER BY
            time
        DESC LIMIT 1"""
    startup_dict = {}
    for sim_id, inj_num, tmplt_id, tmplt_num, pickle_file, backup_archive, scratch_arxiv, host, username in connection.cursor().execute(sqlquery):
        startup_dict['simulation_id'] = sim_id
        startup_dict['inj_num'] = inj_num
        startup_dict['tmplt_id'] = tmplt_id
        startup_dict['tmplt_num'] = tmplt_num
        startup_dict['pickle_file'] = pickle_file
        startup_dict['backup_archive'] = backup_archive
        startup_dict['scratch_archive'] = scratch_arxiv
        startup_dict['host'] = host
        startup_dict['username'] = username

    return startup_dict


def del_old_scratch_files(local_drive_path, username, host, scratch_files = [], verbose = False):
    # construct the directory name
    scratch_dir = '%s/%s/%s' %(local_drive_path, host.split('.')[0], username)
    if verbose:
        print >> sys.stdout, "Checking for old scratch files in %s..." % scratch_dir
    for fn in scratch_files:
        if os.path.exists('%s/%s' %(scratch_dir, fn)):
            os.remove('%s/%s' %(scratch_dir, fn))
            if verbose:
                print >> sys.stdout, "Removed %s" % fn


def get_last_backups(connection):
    sqlquery = 'SELECT backup_archive FROM joblog ORDER BY time DESC LIMIT 1'
    results = connection.cursor().execute(sqlquery).fetchone()
    if results:
        last_bkup_arxv = results[0]
    else:
        last_bkup_arxv = None
    return last_bkup_arxv


def lalSeries_to_tuple(lalSeries):
    """
    @lalSeries: either a Time or Frequency Series
    """
    try:
        return ('time', lalSeries.name, lalSeries.epoch.gpsSeconds, lalSeries.epoch.gpsNanoSeconds, lalSeries.f0, lalSeries.deltaT, lalSeries.data.data)
    except AttributeError:
        return ('frequency', lalSeries.name, lalSeries.epoch.gpsSeconds, lalSeries.epoch.gpsNanoSeconds, lalSeries.f0, lalSeries.deltaF, lalSeries.data.data)


def tuple_to_lalSeries(series_tuple):
    series_type, name, sec, nanosec, f0, dx, series_data = series_tuple
    if series_type == 'time':
        series = lal.CreateREAL8TimeSeries(name, lal.LIGOTimeGPS(sec, nanosec), f0, dx, lal.lalSecondUnit, series_data.shape[0])
    elif series_type == 'frequency':
        series = lal.CreateCOMPLEX16FrequencySeries(name, lal.LIGOTimeGPS(sec, nanosec), f0, dx, lal.lalHertzUnit, series_data.shape[0])
    else:
        raise ValueError, 'unrecognized series type %s' % series_type
    series.data.data[:] = series_data

    return series

def write_backup_archive_dict(archive, fd):
    backup_archive = dict([ [key, {}] for key in archive])
    for key in archive:
        for subkey in archive[key]:
            backup_archive[key][subkey] = lalSeries_to_tuple(archive[key][subkey])
    f = os.fdopen(fd, 'w')
    pickle.dump(backup_archive, f)
    f.close()
    del backup_archive


def load_backup_archive_dict(pickle_fn):
    f = open(pickle_fn, 'r')
    backup_archive = pickle.load(f)
    f.close()
    archive = dict([ [key, {}] for key in backup_archive])
    for key in backup_archive:
        for subkey in backup_archive[key]:
            archive[key][subkey] = tuple_to_lalSeries(backup_archive[key][subkey])
    return archive
    

def write_backup_archive(backup_dir, archive):
    if isinstance(archive, h5py.highlevel.File):
        fd, backup_arxv_fn = tempfile.mkstemp(suffix='.hdf5', dir=backup_dir)
        os.close(fd)
        arxiv_fn = archive.filename
        # close the archive to ensure clean copy
        archive.close()
        shutil.copyfile(arxiv_fn, backup_arxv_fn)
    elif isinstance(archive, dict):
        arxiv_fn = None
        fd, backup_arxv_fn = tempfile.mkstemp(suffix = '.pickle',
            dir=backup_dir)
        write_backup_archive_dict(archive, fd)
    else:
        raise ValueError, "unrecognized archive format: got %s, expected dict or h5py file" % type(archive)

    return arxiv_fn, backup_arxv_fn


def checkpoint(connection, backup_dir, archive, time_now, sim_id, inj_num,
        tmplt_id, tmplt_num, username, backup_archive=False):
    # copy the archive over
    if backup_archive:
        arxiv_fn, backup_arxv_fn = write_backup_archive(backup_dir, archive)
        if arxiv_fn is not None:
            archive = h5py.File(arxiv_fn)
    # get the current host
    host = socket.gethostname()
    # get the last backup files so we can delete them
    last_bkup_arxv = get_last_backups(connection)
    if not backup_archive:
        backup_arxv_fn = last_bkup_arxv
        if isinstance(archive, h5py.highlevel.File):
            arxiv_fn = archive.filename
        else:
            arxiv_fn = None
    if not last_bkup_arxv and not backup_archive:
        backup_arxv_fn = arxiv_fn
    # update the working file
    sqlquery = '''
        INSERT INTO
            joblog (time, simulation_id, inj_num, tmplt_id, tmplt_num,
                backup_archive, scratch_archive, username, host)
        VALUES
            (?,?,?,?,?,?,?,?,?)
        '''
    connection.cursor().execute(sqlquery, (time_now, sim_id, inj_num, tmplt_id,
        tmplt_num, backup_arxv_fn, arxiv_fn, username, host))
    connection.commit()
    # delete the last backup files
    if last_bkup_arxv is not None and backup_archive:
        os.remove(last_bkup_arxv)

    return archive

def clean_backup_files(connection):#workingfile, outfile):
    # get the last backup files
    #connection = sqlite3.connect(workingfile)
    last_bkup_arxv = get_last_backups(connection)
    #connection.close()
    # copy to the output file
    #shutil.copyfile(workingfile, outfile)
    # delete the working file
    #os.remove(workingfile)
    # delete the backups
    if last_bkup_arxv is not None and os.path.exists(last_bkup_arxv):
        os.remove(last_bkup_arxv)


# FIXME: Kludges until ringdown branch is merged with master
#from glue.ligolw import ilwd
#def write_newstyle_coinc_def_entry( connection, description, search = None, search_coinc_type = None ):
#    """
#    Adds a new entry to the coinc_definer_table. The only thing used to discriminate
#    different coinc_definer entries is the description column. Search and search_coinc_type
#    can also be optionally specified.
#    """
#    sqlquery = "SELECT coinc_def_id FROM coinc_definer WHERE description == ?"
#    results = connection.cursor().execute( sqlquery, (description,) ).fetchall()
#    if results == []:
#        # none found, write new entry
#        this_id = sqlutils.get_next_id( connection, 'coinc_definer', 'coinc_def_id' )
#        sqlquery = 'INSERT INTO coinc_definer (coinc_def_id, description, search, search_coinc_type) VALUES (?, ?, ?, ?)'
#        connection.cursor().execute( sqlquery, (str(this_id), description, search, search_coinc_type) )
#        #connection.commit()
#    else:
#        this_id = ilwd.get_ilwdchar(results.pop()[0])
#
#    return this_id
#
#def add_coinc_event_entries( connection, process_id, coinc_def_id, time_slide_id, num_new_entries = 1 ):
#    """
#    Writes N new entries in the coinc_event table, where N is given by num_new_entries.
#    """
#    # get the next id
#    start_id = sqlutils.get_next_id( connection, 'coinc_event', 'coinc_event_id' )
#    # create list of new entries to add
#    new_entries = [(str(process_id), str(coinc_def_id), str(time_slide_id), str(start_id+ii)) for ii in range(num_new_entries)]
#    # add the entries to the coinc_event tabe
#    sqlquery = 'INSERT INTO coinc_event (process_id, coinc_def_id, time_slide_id, coinc_event_id) VALUES (?, ?, ?, ?)'
#    connection.cursor().executemany( sqlquery, new_entries )
#    # return the coinc_event_ids of the new entries
#    return [ilwd.get_ilwdchar(new_id[-1]) for new_id in new_entries]
#
#def get_next_id(connection, table, id_column):
#    """
#    Gets the next available id in the specified id_column in the specified table.
#    """
#    sqlquery = ' '.join(['SELECT', id_column, 'FROM', table ])
#    ids = dict([ [int(ilwd.get_ilwdchar(this_id)), ilwd.get_ilwdchar(this_id)] for (this_id,) in connection.cursor().execute(sqlquery)])
#    if ids == {}:
#        new_id = ilwd.get_ilwdchar(':'.join([table, id_column, '0']))
#    else:
#        new_id = ids[ max(ids.keys()) ] + 1
#    return new_id
#
#sqlutils.write_newstyle_coinc_def_entry = write_newstyle_coinc_def_entry
#sqlutils.add_coinc_event_entries = add_coinc_event_entries
#sqlutils.get_next_id = get_next_id
#

def write_result_to_database(connection, result, match_tag, process_id):
    # get the id associated with this match tag
    coinc_def_id = sqlutils.write_newstyle_coinc_def_entry(connection,
        match_tag)
    # add a new entry to the coinc_event_table
    ceid = sqlutils.add_coinc_event_entries(connection, process_id,
        coinc_def_id, None, 1)[0]
    # add an entry in the coinc_event_table for the injection
    sqlutils.add_coinc_event_map_entry(connection, ceid,
        result.injection.simulation_id, 'sim_inspiral')
    # add an entry for the template
    sqlutils.add_coinc_event_map_entry(connection, ceid,
        result.template.tmplt_id, 'sngl_inspiral')
    # add an entry to the overlap table
    result.write_to_database(connection, ceid)
    #connection.commit()


def delete_overlap_entries(connection, save_description, verbose = False):
    if verbose:
        print >> sys.stdout, "Removing entries not matching %s..." % save_description
    sqlquery = "DELETE FROM coinc_definer WHERE description != ?"
    connection.cursor().execute(sqlquery, (save_description,))
    sqlquery = """
        DELETE FROM coinc_event WHERE coinc_def_id NOT IN (SELECT coinc_def_id FROM coinc_definer);
        DELETE FROM coinc_event_map WHERE coinc_event_id NOT IN (SELECT coinc_event_id FROM coinc_event);
        DELETE FROM overlap_results WHERE coinc_event_id NOT IN (SELECT coinc_event_id FROM coinc_event);
        """
    connection.cursor().executescript(sqlquery)
