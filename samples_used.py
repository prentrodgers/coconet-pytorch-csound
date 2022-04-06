#!/usr/bin/env python
# coding: utf-8

# ## Find out what samples are used in the csound instance
# 
# The purpose of this function is to determine which of the Bosendorfer samples files are being used in the Csound file. Sometimes the velocity is out of whack, and very few of the range of velocities of piano sounds are called for. This function reads in the log file and csd file from the instance of Csound recently run. The log file contains lines like this: <code>instr 1:  iFtable = 777.000</code> which shows when a sample file was actually called for by the csound orchestra. The csd file contains lines like this: <code>f777 0 0 1 "/home/prent/Dropbox/csound/samples/Bosendor/47 emp G4-.wav" 0 0 0 ;</code> which shows a sample file being read into main memory. That sample file might not be ever used. We don't know until the notes are processed if these samples are actually needed. The Bosendorfer file names of the samples start with a number, in this case 47, which indicates the relative velocity the key was struck when making the sample file. These numbers vary from 31 to 85, and there are a few others that were left out for storage reasons. If some samples are never used, we can eliminate them from the csd file, or preferably expand the velocity on the input to csound to ensure that all the sample sets are used.


import numpy as np


def load_samples_used(logfile='goldberg5.log'):

    with open(logfile, 'r') as l:
        log_data = l.readlines()

    samples_used = np.empty(0)
    for row in log_data:
        #split row in each space, so each column will become an element item and attribute it to data
        data = row.split()
        if data != []:
            if data[0] == 'instr':
                np.asarray(data[4])
                text_array = np.asarray(data)
                sample_num = text_array[4].astype(float).astype(int)
                samples_used = np.hstack((samples_used, sample_num))

    samples_used.sort()
    samples_used = np.unique(samples_used)
    return (samples_used)


def load_samples_in_csd(csd_file='goldberg_aria1.csd',sample_array_size=700):
    
    sample_tag = 'samples/Bosendor/'
    velocity_tag = '\"/home/prent/Dropbox/csound/samples/Bosendor/'
    wav_tag = '.wav\"'
    
    array_size = sample_array_size # one for every sample file that could be loaded for this csound csd input file
    with open(csd_file, 'r') as c:
        csd_data = c.readlines()
    
    sample_num = np.zeros((array_size),dtype=int)
    velocity = np.zeros((array_size),dtype=int)
    note = np.zeros((array_size),dtype='U2')
    i = 0
    for row in csd_data:
        data = row.split()
        if data != []:     #print only if there are items in the input stream
            text_array = np.asarray(data)
            found_array = np.char.rfind(text_array,sample_tag)
            if any(found_array != -1): 
                sample_num[i] = int((np.chararray.replace(text_array[0], 'f', '')))
                velocity[i] = int((np.chararray.replace(text_array[4], velocity_tag, '')))
                note[i] = np.chararray.replace(text_array[6], wav_tag,'')
                i += 1

    sample_num = sample_num[:i]
    velocity = velocity[:i]
    note = note[:i]
    return(sample_num, velocity, note)


def report_samples_used():
    samples_used_in_csound = load_samples_used()
    sample_number, velocity, notes = load_samples_in_csd()

    print('Here\'s a list of all the samples numbers, velocities, and notes that were actually used in the call to csound:')
    how_many = 0
    print(f'sample\t\tvelocity\tMIDI note')
    for samples in samples_used_in_csound:
        found = np.where(samples == sample_number)
        print(f'{sample_number[found]}\t\t{velocity[found]}\t\t{notes[found]}')        
        how_many += 1
    return (how_many)

# to use these functions call the functions as so:s
# samples_in_use = report_samples_used()
# print(f'Total samples used in this csound execution was {samples_in_use}')



