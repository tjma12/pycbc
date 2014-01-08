# Copyright (C) 2007  Kipp Cannon, Collin Capano
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


import os
import shutil
import signal
import sys
import tempfile
import threading
import warnings


__author__ = "Kipp Cannon <kipp.cannon@ligo.org>"


#
# Module-level variable used to hold references to
# tempfile.NamedTemporaryFiles objects to prevent them from being deleted
# while in use.  NOT MEANT FOR USE BY CODE OUTSIDE OF THIS MODULE!
#


temporary_files = {}
temporary_files_lock = threading.Lock()


#
# Module-level variable to hold the signal handlers that have been
# overridden as part of the clean-up-scratch-files-on-signal feature.  NOT
# MEANT FOR USE BY CODE OUTSIDE OF THIS MODULE!
#


origactions = {}


def install_signal_trap(signums = (signal.SIGTERM, signal.SIGTSTP), retval = 1):
    """
    Installs a signal handler to erase temporary scratch files when a
    signal is received.  This can be used to help ensure scratch files
    are erased when jobs are evicted by Condor.  signums is a squence
    of the signals to trap, the default value is a list of the signals
    used by Condor to kill and/or evict jobs.

    The logic is as follows.  If the current signal handler is
    signal.SIG_IGN, i.e. the signal is being ignored, then the signal
    handler is not modified since the reception of that signal would
    not normally cause a scratch file to be leaked.  Otherwise a signal
    handler is installed that erases the scratch files.  If the
    original signal handler was a Python callable, then after the
    scratch files are erased the original signal handler will be
    invoked.  If program control returns from that handler, i.e.  that
    handler does not cause the interpreter to exit, then sys.exit() is
    invoked and retval is returned to the shell as the exit code.

    Note:  by invoking sys.exit(), the signal handler causes the Python
    interpreter to do a normal shutdown.  That means it invokes
    atexit() handlers, and does other garbage collection tasks that it
    normally would not do when killed by a signal.

    Note:  this function will not replace a signal handler more than
    once, that is if it has already been used to set a handler
    on a signal then it will be a no-op when called again for that
    signal until uninstall_signal_trap() is used to remove the handler
    from that signal.

    Note:  this function is called by get_connection_filename()
    whenever it creates a scratch file.
    """
    temporary_files_lock.acquire()
    try:
        # ignore signums we've already replaced
        signums = set(signums) - set(origactions)

        def temporary_file_cleanup_on_signal(signum, frame):
            temporary_files_lock.acquire()
            temporary_files.clear()
            temporary_files_lock.release()
            if callable(origactions[signum]):
                # original action is callable, chain to it
                return origactions[signum](signum, frame)
            # original action was not callable or the callable
            # returned.  invoke sys.exit() with retval as exit code
            sys.exit(retval)

        for signum in signums:
            origactions[signum] = signal.getsignal(signum)
            if origactions[signum] != signal.SIG_IGN:
                # signal is not being ignored, so install our
                # handler
                signal.signal(signum, temporary_file_cleanup_on_signal)
    finally:
        temporary_files_lock.release()


def uninstall_signal_trap(signums = None):
    """
    Undo the effects of install_signal_trap().  Restores the original
    signal handlers.  If signums is a sequence of signal numbers the
    only the signal handlers for thos signals will be restored.  If
    signums is None (the default) then all signals that have been
    modified by previous calls to install_cleanup_handler() are
    restored.

    Note:  this function is called by put_connection_filename() and
    discard_connection_filename() whenever they remove a scratch file
    and there are then no more scrach files in use.
    """
    temporary_files_lock.acquire()
    try:
        if signums is None:
            signums = origactions.keys()
        for signum in signums:
            signal.signal(signum, origactions.pop(signum))
    finally:
        temporary_files_lock.release()


#
# Functions to work with temp files in scratch space
#

def get_temp_filename(filename = None, suffix = '', tmp_path = None, copy = True, replace_file = False, verbose = False):
    """
    Utility code for moving files to a (presumably local)
    working location for improved performance and reduced fileserver
    load.
    """
    def mktmp(path, suffix, verbose = False):
        # make sure the clean-up signal traps are installed
        install_signal_trap()
        # create the temporary file and replace it's unlink()
        # function
        temporary_file = tempfile.NamedTemporaryFile(suffix = suffix, dir = path)
        # NamedTemporaryFile opens a file object; close it to avoid
        # danling open files
        temporary_file.file.close()
        filename = temporary_file.name
        temporary_files_lock.acquire()
        try:
            temporary_files[filename] = temporary_file
        finally:
            temporary_files_lock.release()
        if verbose:
            print >>sys.stderr, "using '%s' as workspace" % filename
        # mkstemp() ignores umask, creates all files accessible
        # only by owner;  we should respect umask.  note that
        # os.umask() sets it, too, so we have to set it back after
        # we know what it is
        #umsk = os.umask(0777)
        #os.umask(umsk)
        #os.chmod(filename, 0666 & ~umsk)
        return filename

    def truncate(filename, verbose = False):
        if verbose:
            print >>sys.stderr, "'%s' exists, truncating ..." % filename,
        try:
            fd = os.open(filename, os.O_WRONLY | os.O_TRUNC)
        except Exception, e:
            if verbose:
                print >>sys.stderr, "cannot truncate '%s': %s" % (filename, str(e))
            return
        os.close(fd)
        if verbose:
            print >>sys.stderr, "done."

    def cpy(srcname, dstname, verbose = False):
        if verbose:
            print >>sys.stderr, "copying '%s' to '%s' ..." % (srcname, dstname),
        shutil.copy2(srcname, dstname)
        if verbose:
            print >>sys.stderr, "done."
        try:
            # try to preserve permission bits.  according to
            # the documentation, copy() and copy2() are
            # supposed preserve them but don't.  maybe they
            # don't preserve them if the destination file
            # already exists?
            shutil.copystat(srcname, dstname)
        except Exception, e:
            if verbose:
                print >>sys.stderr, "warning: ignoring failure to copy permission bits from '%s' to '%s': %s" % (filename, target, str(e))

    if filename is not None:
        database_exists = os.access(filename, os.F_OK)
    else:
        database_exists = False

    if tmp_path is not None:
        target = mktmp(tmp_path, suffix, verbose = verbose)
        if copy:
            if database_exists:
                if replace_file:
                    # truncate database so that if this job
                    # fails the user won't think the database
                    # file is valid
                    truncate(filename, verbose = verbose)
                else:
                    # need to copy existing database to work
                    # space for modifications
                    i = 1
                    while True:
                        try:
                            cpy(filename, target, verbose = verbose)
                        except IOError, e:
                            import errno
                            import time
                            if e.errno not in (errno.EPERM, errno.ENOSPC):
                                # anything other
                                # than out-of-space
                                # is a real error
                                raise
                            if i < 5:
                                if verbose:
                                    print >>sys.stderr, "warning: attempt %d: %s, sleeping and trying again ..." % (i, errno.errorcode[e.errno])
                                time.sleep(10)
                                i += 1
                                continue
                            if verbose:
                                print >>sys.stderr, "warning: attempt %d: %s: working with original file '%s'" % (i, errno.errorcode[e.errno], filename)
                            temporary_files_lock.acquire()
                            del temporary_files[target]
                            temporary_files_lock.release()
                            target = filename
                        break
    else:
        temporary_files_lock.acquire()
        try:
            if filename in temporary_files:
                raise ValueError, "file '%s' appears to be in use already as a temporary database file and is to be deleted" % filename
        finally:
            temporary_files_lock.release()
        target = filename
        if database_exists and replace_file:
            truncate(target, verbose = verbose)

    del mktmp
    del truncate
    del cpy

    return target


#
# FIXME:  this is only here temporarily while the file corruption issue on
# the clusters is diagnosed.  remove when no longer needed
#

try:
    # >= 2.5.0
    from hashlib import md5 as __md5
except ImportError:
    # < 2.5.0
    from md5 import new as __md5
def __md5digest(filename):
    """
    For internal use only.
    """
    m = __md5()
    f = open(filename)
    while True:
        d = f.read(4096)
        if not d:
            break
        m.update(d)
    return m.hexdigest()


def put_temp_filename(filename, working_filename, verbose = False):
    """
    This function reverses the effect of a previous call to
    get_connection_filename(), restoring the working copy to its
    original location if the two are different.  This function should
    always be called after calling get_connection_filename() when the
    file is no longer in use.

    During the move operation, this function traps the signals used by
    Condor to evict jobs.  This reduces the risk of corrupting a
    document by the job terminating part-way through the restoration of
    the file to its original location.  When the move operation is
    concluded, the original signal handlers are restored and if any
    signals were trapped they are resent to the current process in
    order.  Typically this will result in the signal handlers installed
    by the install_signal_trap() function being invoked, meaning any
    other scratch files that might be in use get deleted and the
    current process is terminated.
    """
    if working_filename != filename:
        # initialize SIGTERM and SIGTSTP trap
        deferred_signals = []
        def newsigterm(signum, frame):
            deferred_signals.append(signum)
        oldhandlers = {}
        for sig in (signal.SIGTERM, signal.SIGTSTP):
            oldhandlers[sig] = signal.getsignal(sig)
            signal.signal(sig, newsigterm)

        # replace document
        if verbose:
            print >>sys.stderr, "moving '%s' to '%s' ..." % (working_filename, filename),
        digest_before = __md5digest(working_filename)
        shutil.move(working_filename, filename)
        digest_after = __md5digest(filename)
        if verbose:
            print >>sys.stderr, "done."
        if digest_before != digest_after:
            print >>sys.stderr, "md5 checksum failure!  checksum on scratch disk was %s, checksum in final location is %s" % (digest_before, digest_after)
            sys.exit(1)

        # remove reference to tempfile.TemporaryFile object.
        # because we've just deleted the file above, this would
        # produce an annoying but harmless message about an ignored
        # OSError, so we create a dummy file for the TemporaryFile
        # to delete.  ignore any errors that occur when trying to
        # make the dummy file.  FIXME: this is stupid, find a
        # better way to shut TemporaryFile up
        try:
            file(working_filename, "w").close()
        except:
            pass
        temporary_files_lock.acquire()
        try:
            del temporary_files[working_filename]
        finally:
            temporary_files_lock.release()

        # restore original handlers, and send outselves any trapped signals
        # in order
        for sig, oldhandler in oldhandlers.iteritems():
            signal.signal(sig, oldhandler)
        while deferred_signals:
            os.kill(os.getpid(), deferred_signals.pop(0))

        # if there are no more temporary files in place, remove the
        # temporary-file signal traps
        temporary_files_lock.acquire()
        no_more_files = not temporary_files
        temporary_files_lock.release()
        if no_more_files:
            uninstall_signal_trap()


def discard_temp_filename(working_filename, verbose = False):
    """
    Like put_connection_filename(), but the working copy is simply
    deleted instead of being copied back to its original location.
    This is a useful performance boost if it is known that no
    modifications were made to the file, for example if queries were
    performed but no updates.

    Note that the file is not deleted if the working copy and original
    file are the same, so it is always safe to call this function after
    a call to get_connection_filename() even if a separate working copy
    is not created.
    """
    if verbose:
        print >>sys.stderr, "removing '%s' ..." % working_filename,
    # remove reference to tempfile.TemporaryFile object
    temporary_files_lock.acquire()
    try:
        del temporary_files[working_filename]
    finally:
        temporary_files_lock.release()
    if verbose:
        print >>sys.stderr, "done."

    # if there are no more temporary files in place, remove the
    # temporary-file signal traps
    temporary_files_lock.acquire()
    no_more_files = not temporary_files
    temporary_files_lock.release()
    if no_more_files:
        uninstall_signal_trap()
