#! /usr/bin/env
import os
from glue import lal as glal
from glue import segments

def get_outcache_entry(output_directory, ifo, user_tag = None, fnum = 0, gz = True):
    tag = 'TMPLTBANK'
    if user_tag is not None:
        tag += '_%s' % user_tag
    extension = 'xml'
    if gz:
        extension += '.gz'
        
    outfilename = '%s/%s-%s-%i.%s' %(output_directory, ifo, tag, fnum, extension)

    return glal.CacheEntry(ifo, tag, segments.segment(fnum, 0), os.path.abspath(outfilename)) 

def get_outcache_name(output_directory, ifo, user_tag = None):
    tag = 'TMPLTBANK'
    if user_tag is not None:
        tag += '_%s' % user_tag
    return '%s/%s-%s.cache' %(output_directory, ifo, tag)
