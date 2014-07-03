#! /usr/bin/env python

import os, sys
import math
import sqlite3
import numpy
import lal

class Result:
    def __init__(self):
        self.apprx = None
        self.unique_id = None
        self.m1 = None
        self.m2 = None
        self.s1z = None
        self.s2z = None
        self.eff_dist = None
        self.dist = None
        self.inclination = None
        self.target_snr = None
        self.effectualness = None
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
        self.threshold = None
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
        return lal.LAL_MTSUN_SI*self.mtotal

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
    def tau0(self, f0 = 40):
        return (5./(256*numpy.pi*f0*self.eta))*(numpy.pi*self.mtotal*lal.LAL_MTSUN_SI*f0)**(-5./3.)

    @property
    def v0(self, f0 = 40):
        return (2*numpy.pi*f0*self.mtotal*lal.LAL_MTSUN_SI)**(1./3)

def parse_results_cache(cache_file):
    filenames = []
    f = open(cache_file, 'r')
    for line in f:
        thisfile = line.split('\n')[0]
        if os.path.exists(thisfile):
            filenames.append(thisfile)
    f.close()
    return filenames

def get_injection_results(filenames, get_inj_map = True, ref_apprx = None, test_apprx = None, verbose=False):
    if get_inj_map and (test_apprx is None or ref_apprx is None):
        raise ValueError, "If want an inj_map, must provide a reference and test approximate"
    sqlquery = 'select sim.waveform, sim.simulation_id, sim.mass1, sim.mass2, sim.spin1z, sim.spin2z, sim.eff_dist_h, sim.distance, sim.inclination, sim.eff_dist_t, tmplt.event_id, tmplt.mass1, tmplt.mass2, res.effectualness, res.snr, res.snr_std, res.chisq, res.chisq_std, res.chisq_dof, res.new_snr, res.new_snr_std, res.num_successes, res.sample_rate, res.coinc_event_id from overlap_results as res join sim_inspiral as sim, coinc_event_map as map on sim.simulation_id == map.event_id and map.coinc_event_id == res.coinc_event_id join sngl_inspiral as tmplt, coinc_event_map as mapB on mapB.coinc_event_id == map.coinc_event_id and mapB.event_id == tmplt.event_id'
    results = {}
    reftest_map = {}
    idx = 0
    for ii,thisfile in enumerate(filenames):
        if verbose:
            print >> sys.stdout, "%i / %i\r" %(ii+1, len(filenames)),
            sys.stdout.flush()
        if not thisfile.endswith('.sqlite'):
            continue
        connection = sqlite3.connect(thisfile)
        id_map = {}
        try:
            for apprx, sim_id, m1, m2, s1z, s2z, eff_dist, dist, inc, target_snr, tmplt_evid, tmplt_m1, tmplt_m2, ff, snr, snr_std, chisq, chisq_std, chisq_dof, new_snr, new_snr_std, nsamp, sample_rate, ceid in connection.cursor().execute(sqlquery):
                results.setdefault(apprx, [])
                thisRes = Result()
                thisRes.unique_id = idx
                id_map[sim_id] = idx
                idx += 1
                thisRes.apprx = apprx
                thisRes.m1 = m1
                thisRes.m2 = m2
                thisRes.s1z = s1z
                thisRes.s2z = s2z
                thisRes.mtotal = m1+m2
                thisRes.eff_dist = eff_dist
                thisRes.dist = dist
                thisRes.inclination = inc
                thisRes.target_snr = target_snr
                thisRes.tmplt_id = tmplt_evid
                thisRes.tmplt_m1 = tmplt_m1
                thisRes.tmplt_m2 = tmplt_m2
                thisRes.effectualness = ff
                thisRes.snr = snr
                thisRes.snr_std = snr_std
                thisRes.chisq = chisq
                thisRes.chisq_sts = chisq_std
                thisRes.chisq_dof = chisq_dof
                thisRes.new_snr = new_snr
                thisRes.new_snr_std = new_snr_std
                thisRes.num_samples = nsamp
                thisRes.sample_rate = sample_rate
                thisRes.database = thisfile
                thisRes.coinc_event_id = ceid
                results[apprx].append(thisRes)
        except sqlite3.OperationalError:
            connection.close()
            continue
        except sqlite3.DatabaseError:
            connection.close()
            print "Database Error: %s" % thisfile
            continue
        # get inj map
        if get_inj_map:
            inj_map = get_reftest_map_fromdb(connection, ref_apprx, test_apprx)
            # put map into terms of unique id
            reftest_map.update(dict([ [id_map[id1], id_map[id2]] for id1,id2 in inj_map.items() if id1 in id_map and id2 in id_map]))

        connection.close()

    if verbose:
        print >> sys.stdout, ""

    return results, reftest_map

def get_reftest_map_fromdb(connection, ref_apprx, test_apprx):
    """
    Returns a map of ref. simulation_id -> test simulation_id.
    """
    sqlquery = 'SELECT ref.simulation_id, test.simulation_id FROM sim_inspiral AS test JOIN sim_inspiral AS ref, coinc_event_map AS mapA, coinc_event_map AS mapB ON test.simulation_id == mapA.event_id AND mapA.coinc_event_id == mapB.coinc_event_id AND ref.simulation_id == mapB.event_id AND ref.waveform == ? AND test.waveform == ?'
    return dict([ [refid, testid] for refid, testid in connection.cursor().execute(sqlquery, (ref_apprx, test_apprx))])

def get_testref_map(reftest_map):
    # generates inverse of reftest map; i.e. returns
    # test->ref map
    return dict([ [testid, refid] for (refid, testid) in reftest_map.items()])

def get_matching_pairs(results, reftest_map):
    idx_map = dict([ [inj.unique_id, inj] for this_list in results.values() for inj in this_list])
    matching_results = dict([ [apprx, []] for apprx in results])
    for idx1, idx2 in reftest_map.items():
        matching_results[idx_map[idx1].apprx].append(idx_map[idx1])
        matching_results[idx_map[idx2].apprx].append(idx_map[idx2])
    return matching_results


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
    try:
        return getattr(row, arg)

    except AttributeError:
        row_dict = dict([ [name, getattr(row,name)] for name in dir(row)])
        safe_dict = dict([ [name,val] for name,val in row_dict.items() + math.__dict__.items() if not name.startswith('__')])
        return eval(arg, {"__builtins__":None}, safe_dict)


def result_in_range(result, test_dict):
    cutvals = [(get_arg(result, criteria), low, high) for criteria,(low,high) in test_dict.items()] 
    return not any(x < low or x >= high for (x, low, high) in cutvals)


def result_is_match(result, test_dict):
    try:
        matchvals = [(getattr(result, criteria), targetval) for criteria,targetval in test_dict.items()]
    except:
        matchvals = [(get_arg(result, criteria), targetval) for criteria,targetval in test_dict.items()]
    return not any(x != targetval for (x, tagetval) in matchvals)


def apply_cut(result_list, test_dict):
    return [x for x in result_list if result_in_range(x, test_dict)]

def slice_results(results, test_dict):
    sliced = {}
    for apprx, full_list in results.items():
        sliced[apprx] = apply_cut(full_list, test_dict) 
    return sliced

    
def create_reftest_map(results, ref_apprx, test_apprx, match_criteria):
    """
    Creates a mapping between reference injections and test
    injections.
    """
    test_matchvals = dict([ [tuple([get_arg(inj, arg) for arg in match_criteria]), inj.unique_id] for inj in results[test_apprx]])
    ref_matchvals = dict([ [tuple([get_arg(inj, arg) for arg in match_criteria]), inj.unique_id] for inj in results[ref_apprx]])
    
    return dict([ [ref_matchvals[val], test_matchvals[val]] for val in set(test_matchvals.keys()) & set(ref_matchvals.keys()) ])


def plot2imgmap(fig, plot, links, figname, shape = 'rect', view_width = 1000):
    """
    @param plot RegularPolyCollection from matplotlib. The sort of thing
    returned by scatter()
    @links: a dictionary of links. If shape is 'rect', the keys are the (x,y) coordinates
     of the top left and bottom right corners of the link, in the form
     ((x1, y1), (x2, y2)). If shape is 'circle', the keys are the (x,y) coordinates.
     The values are the links.
    """

    dpi = fig.get_dpi()
    img_height = fig.get_figheight() * dpi
    img_width = fig.get_figwidth() * dpi
    scalefac = float(view_width)/img_width

    trans = plot.axes.transData
    if shape == 'rect':
        img_coords = dict([[(tuple(trans.transform(coord1)), tuple(trans.transform(coord2))), link] for (coord1, coord2),link in links.items()])
    elif shape == 'circle':
        img_coords = dict([[tuple(trans.transform(coords)), link] for coords,link in links.items()])
    else:
        raise ValueError, 'Unrecognized shape %s' % shape

    tmplt = """
<img src="%s" alt="Click on a tile to go to info" usemap="#points" border="0" width="%i">
<map name="points">
%s
</map>
"""
    if shape == 'rect':
        maptmplt = '<area shape="rect" coords="%i,%i,%i,%i", href="%s">'
        maps = [maptmplt  %(scalefac*ix1, scalefac*(img_height - iy1), scalefac*ix2, scalefac*(img_height - iy2), link) for ((ix1,iy1), (ix2, iy2)),link in img_coords.items()]
    else:
        maptmplt = '<area shape="circle" coords="%i,%i,5", href="%s">'
        maps = [maptmplt  %(scalefac*ix, scalefac*(img_height - iy), link) for (ix,iy),link in img_coords.items()]

    return tmplt %(figname, view_width, '\n'.join(maps))

