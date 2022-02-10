#!/usr/bin/env python
from __future__ import print_function
import operator, os, sys, subprocess, time
import pandas as pd


############################################################
# exec_par
#
# Execute the commands in the list 'cmds' in parallel, but
# only running 'max_proc' at a time.
############################################################
def make_directory(path):
    """Short summary.

    Parameters
    ----------
    path : Full path to the directory

    """

    if not os.path.isdir(path):
        os.mkdir(path)
        print("Making directory: " + path)
    else:
        print("Directory already exists!")

def shift_unmap(shift_size, output_unmap, unmap_path='/home/shush/genomes/GRCh38_unmap.bed'):
    unmap = pd.read_csv(unmap_path,  sep='\t', header=None)
    first_shift = pd.DataFrame(['chr8', '0', '0']).T
    complete_unmap = pd.concat([first_shift, unmap])
    complete_unmap.reset_index(inplace=True)
    new_ends = pd.to_numeric(complete_unmap[2])+shift_size
    shift_unmap = pd.DataFrame([complete_unmap[0], complete_unmap[1], new_ends]).T
    interm_bed = 'nonmerged_shifted.bed'
    shift_unmap.to_csv(interm_bed, index=None, sep='\t', header=None)
    cmd = 'bedtools merge -i {} > {}; rm {}'.format(interm_bed, output_unmap, interm_bed)

    process = subprocess.Popen(cmd, shell=True)
    output, error = process.communicate()
    print(error)

def exec_par(cmds, max_proc=None, verbose=False):
    total = len(cmds)
    finished = 0
    running = 0
    p = []

    if max_proc == None:
        max_proc = len(cmds)

    if max_proc == 1:
        while finished < total:
            if verbose:
                print(cmds[finished], file=sys.stderr)
            op = subprocess.Popen(cmds[finished], shell=True)
            os.waitpid(op.pid, 0)
            finished += 1

    else:
        while finished + running < total:
            # launch jobs up to max
            while running < max_proc and finished+running < total:
                if verbose:
                    print(cmds[finished+running], file=sys.stderr)
                p.append(subprocess.Popen(cmds[finished+running], shell=True))
                #print 'Running %d' % p[running].pid
                running += 1

            # are any jobs finished
            new_p = []
            for i in range(len(p)):
                if p[i].poll() != None:
                    running -= 1
                    finished += 1
                else:
                    new_p.append(p[i])

            # if none finished, sleep
            if len(new_p) == len(p):
                time.sleep(1)
            p = new_p

        # wait for all to finish
        for i in range(len(p)):
            p[i].wait()
