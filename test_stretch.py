import numpy as np
import copy
import mido
import time
from midi2audio import FluidSynth
from IPython.display import Audio, display
import os
import muspy
import piano 
import selective_stretching_codes
import samples_used
import subprocess
from numpy.random import default_rng
# rng = np.random.default_rng(seed=42)
rng = np.random.default_rng()

CSD_FILE = 'goldberg_aria1.csd'
LOGNAME = 'goldberg5.log'
root = 5
mode = 'major'


def main():
      # pull in an interesting chorale from the collection of numpy segments
      filename = os.path.join('segmented_chorales','chorale_90.npy')
      voice_low = 9
      voice_high = 16
      print(filename)
      play_chorale = selective_stretching_codes.transpose_up_segment(selective_stretching_codes.expand_and_concatenate(np.load(filename)),30) 
      play_chorale = play_chorale[voice_low:voice_high,:] # this is where you need to pick the voices
      print(f'play_chorale.shape: {play_chorale.shape}, play_chorale.dtype: {play_chorale.dtype}')
      # now double up the voices to become twice as many voices.
      play_chorale = np.concatenate((play_chorale,play_chorale),axis = 0)
      print(f'play_chorale.shape: {play_chorale.shape}, play_chorale.dtype: {play_chorale.dtype}')
      # Before you do the discover, stretch the chorale out by making it 4 times as long.
      start_decompress = 0
      end_decompress = play_chorale.shape[1]
      factor = 1
      print(f'play_chorale.shape: {play_chorale.shape}')
      play_chorale = selective_stretching_codes.decompress_segment(play_chorale,start_decompress,end_decompress,factor=factor) 
      print(f'play_chorale.shape: {play_chorale.shape}, play_chorale.dtype: {play_chorale.dtype}')

      # print(f'Decompression by a factor of {factor} results in change by a factor of: {play_chorale.shape[1]/end_decompress}')

      # Now we have to find the the in tune time_steps, and challenging time_steos: range_in_tune and the range_of_steps
      scores = selective_stretching_codes.assign_scores_to_time_steps(play_chorale)
      range_of_steps = selective_stretching_codes.find_challenging_time_steps(scores)
      range_in_tune = selective_stretching_codes.find_in_tune_time_steps(play_chorale, range_of_steps)

      # give me a set of random numbers from 2 up to but not including 8
      factors = rng.integers(low=2,high=8,size=range_of_steps.shape[0])
      print(f'decompression factors for each of the sections to be expanded: \n{factors}')

      decom_chorale, challenging_steps, in_tune_steps = selective_stretching_codes.expand_challenging_time_steps(play_chorale, range_of_steps, range_in_tune)

      skip_time_step_in_arp = 3 # Arpeggiate one time, then skip the next three, then arpegiate the next, and so on.
      mask = rng.integers(low=0, high=2, size=(16,16), dtype=np.uint8)
      arp_chorale = selective_stretching_codes.arpeggiate_and_stretch(decom_chorale,mask,skip_time_step_in_arp)
      tpq = 1 # time per quarter note.
      upsamp = 6 # the higher the more mellow the sound
      velocity_base = 67 # this cents the middle velocity sample from the Bosendorfer set
      volume = 14 # avoid distortion. Slowly increse this until the overall amps gets close to but not over 32k.
      if selective_stretching_codes.valid_midi(arp_chorale): 
            selective_stretching_codes.piano_roll_to_csound(arp_chorale,velocity_base,volume,tpq,upsamp)
      number = samples_used.report_samples_used()
      print(f'total samples used: {number}')


main()

