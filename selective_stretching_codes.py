#!/usr/bin/env python
# coding: utf-8

# Selectively stretch and decompress segments based on metrics
# This is a library of helper functions to transform a 264 note chorale into a longer piece. 
# The goal is to enable selectively staying on one place for a longer time. This can be accomplished by selecting a segment of an array to double, triple, any arbitrary amount in length.
# The assumption is that we will take the segmented chorale numpy arrays in directory segmented_chorales

from re import T
import numpy as np
import copy
import mido
import time
from midi2audio import FluidSynth
from IPython.display import Audio, display
import os
import muspy
import piano 
import subprocess
from numpy.random import default_rng
rng = np.random.default_rng()
import csv

CSD_FILE = 'goldberg_aria1.csd'
LOGNAME = 'goldberg5.log'

file_name = '/home/prent/Downloads/chorales_018007b_(c)greentree.mid'

def load_csound_piano():
    piano.start_logger(fname=LOGNAME)
    piano.logging.info(f'Logging messages to: {LOGNAME}')
    csd_content, lines = piano.load_csd(CSD_FILE)
    piano.logging.info(f'Loaded the csd file {CSD_FILE}. There are {lines} lines read containg {len(csd_content)} bytes')
    cs, pt = piano.load_csound(csd_content)
    return (pt,cs)

# function for converting arrays of shape (T, N) into midi files
# the input array has entries that are np.nan (representing a rest)
# or an integer between 0 and 127 inclusive
#
# Altered to accept pieces of arbitrary number of voices. 
# Mine start life as 264 notes by 16 voices per chorale

def piano_roll_to_midi(piece, bpm = 50):
    """
    piece is a an array of shape (T, 4) for some T.
    The (i,j)th entry of the array is the midi pitch of the jth voice at time i. It's an integer in range(128).
    outputs a mido object mid that you can convert to a midi file by called its .save() method
    """
    # piece = np.concatenate([piece, [[np.nan, np.nan, np.nan, np.nan]]], axis=0) # this assumes a four part chorale, which mine are not.

    microseconds_per_beat = 60 * 1000000 / bpm

    mid = mido.MidiFile()
    
    # modified to make the number of voices dependent on what is passed into the function
    v = 0
    tracks = {}
    past_pitches = {}
    delta_time = {}
    for voice in piece:
        tracks['piano' + str(v)] = mido.MidiTrack()
        past_pitches['piano' + str(v)] = np.nan
        delta_time['piano' + str(v)] = 0
        v += 1
    
    # create a track containing tempo data
    metatrack = mido.MidiTrack()
    metatrack.append(mido.MetaMessage('set_tempo',
                                      tempo=int(microseconds_per_beat), time=0))
    mid.tracks.append(metatrack)

    # create the N voice tracks (was 4)
    for voice in tracks:
        mid.tracks.append(tracks[voice])
        tracks[voice].append(mido.Message(
            'program_change', program=0, time=0)) # choir aahs=52, piano = 0

    # add notes to the N voice tracks
    # this function expects an array in this form: chorale type: <class 'numpy.ndarray'>
    # piece.shape: (33, 4) 
    # mine are (16,264)
    
    pitches = {}
    for i in range(piece[1].shape[0]): # 0 - 263 in my case
        v = 0
        for voice in piece: # 0-15 in my case
            pitches['piano'+str(v)] = piece[v,i] # i is from 0 to 263, v is 0 to 15
            v += 1
        for voice in tracks:
            if np.isnan(past_pitches[voice]):
                past_pitches[voice] = None
            if np.isnan(pitches[voice]):
                pitches[voice] = None
            if pitches[voice] != past_pitches[voice]:
                if past_pitches[voice]:
                    tracks[voice].append(mido.Message('note_off', note=int(past_pitches[voice]),
                                                      velocity=64, time=delta_time[voice]))
                    delta_time[voice] = 0
                if pitches[voice]:
                    tracks[voice].append(mido.Message('note_on', note=int(pitches[voice]),
                                                      velocity=64, time=delta_time[voice]))
                    delta_time[voice] = 0
            past_pitches[voice] = pitches[voice]
            # 480 ticks per beat and each line of the array is a 16th note
            delta_time[voice] += 120

    return mid


# ## Decompression of segments of the model output back to the 2 1/2 measure segments.
# ## Generalized to enable stretching any part of the segment by doubling the number of notes
# This section turns the a segment the model into a longer slot segment. We compressed the segment going into the model, so we decompress it coming out of the model. Decompress does several things. This is relevant for the Schmucke chorale that has some 2 1/2 measure segments, and the model expects 2 measure segments. But it can also be used to double the size of an array, or increase one particular part by an arbitrary amount. 
# For example, it could turn this: a b c d into a b b b c c c d by repeated calls to the decompress_segment function. 
# Back to it's use in Schmucke:
#         
# - expands the end of segment for 0,1,2,3 and convert it to 4x40 array
# - fixes the end of the 4,5,6th by adding padding to convert to a 4x40 array
#     
# This will be called just after emerging from the model and before the midi file is written    
# 3/28/22 added the ability to decompress_segment by a larger amount. The parameter *factor*, when 1 does what it always did, repeat every note once in the range passed to the function. If factor is 2, then it repeats the notes twice. To turn a 1/16 note into a quarter note, use factor of 3, three extra notes for every existing note.

# this function expects a segment of shape (voice,notes) for any value of voice,notes. It can be used for any number of voices and notes per voice.
# it can only work on the end of a matrix. 
def decompress_segment(this_segment, start, end, factor=1):
      # this function will turn any arbitrary segment of an array of voices and notes into a segment that is twice as long as the original segment
      new_start = start - (end - start) # 32 - (48 - 32) = 16 or 32 - (40 - 32) = 24
      # print(f'new_start: {new_start}, start: {start}, end: {end}')
      expanded = np.zeros((this_segment.shape[0],(this_segment.shape[1] + (end - start) * factor)), dtype=int) 
      # print(f'input shape of this_segment: {this_segment.shape} shape of expanded: {expanded.shape}, factor: {factor}')
      v = 0 # index to voices
      for voices in this_segment:
            s = 0 # index to source
            t = 0 # index to target
            # if v == 0: print(f'voices.shape: {voices.shape}')
            for notes in voices:
                  expanded[v,t] = this_segment[v,s]
                  if new_start <= s < end: # if s is both greater or equal to start and less than end
                        for x in range(factor):
                              t += 1
                              expanded[v,t] = this_segment[v,s]
                  s += 1
                  t += 1
            v += 1
      return(expanded)


# ## Transpose from the key of C used in the model to the original key: root
# This is done to restore what the input midi file key was. I found that model inputs in the key of C are harmonized much better than those that are in other keys. I thought they took care of this in the model by transposing to different keys, but my experience suggests otherwize.
# Add the value of root (F is 5) to each note in the array, with the exception of the 0's, which have to remain the same 0. 


def transpose_up_segment(my_segment,root):
    new_segment = copy.deepcopy(my_segment) # just make a copy, you will change the non zero elements 
    v = 0
    for voice in new_segment:
        n = 0
        for note in voice:
            if note > 0:
                new_segment[v,n] = note + root
            n += 1
        v += 1
        
    return(new_segment)

def transpose_up(segments,root): # read in (Segment, Voices, Notes) then call transpose_up_segment.
    s = 0
    new_segment = copy.deepcopy(segments)
    for seg in segments:
        new_segment[s] = transpose_up_segment(seg,root)
        s += 1
    return(new_segment)  

def piano_roll_to_csound(piece,velocity,volume,tpq,upsample,challenging_steps,min_delay=20,zfactor=10):
    if os.path.exists('goldberg5.log'):
        os.remove("goldberg5.log") # make sure the log starts over with a fresh log file. Next line starts the logger.
    pt,cs = load_csound_piano() # load the csd file and return a performance thread and a Csound instance, start the logger.
    piano.logging.info('ins star dur vel ton oc voi stero env glis upsa rEnv 2nd 3rd vol chan')
    tp16th = tpq / 4 # time per 1/16th note
    hold = 0.2 # how long to hold to make more of a legato
    pfields = []
    new_volume = volume
    volume_increases = 0
    volume_decreases = 0
    velocity_increases = 0
    velocity_decreases = 0
    challenging_step_count = 0
    volume_reset = 0
    velocity_reset = 0
    not_c_step = 0
    v = 0
    for voice in piece: # once for each voice
        new_velocity = velocity
        prev_note = 0
        duration = 0
        start_time = 0
        first = True
        time_step = 0
        for note in voice: # one note for each time step in this voice [69 67 67 67 65 67 67 67 67]
            if first:
                prev_note = note
                first = False
            if note == prev_note:
                duration += tp16th # add another 1/16th note duration to the current duration, but don't play it
            else: # send the note to csound
                octave = prev_note // 12 # csound needs octave and tone as sepatate fields
                if octave > 0: # not a zero note value
                    tone = prev_note - 12 * octave
                    if (octave > 8) or (octave < 2):
                        print(f'Warning. voice:{v} note: {note} Out of range: prev_note: {prev_note}, octave: {octave}, tone: {tone}')
                    if in_challenging_step(time_step, challenging_steps):
                        challenging_step_count += 1
                        if rng.integers(low=0,high=2) == 0: 
                              new_velocity += 1 # increase velocity 50% of the time
                              velocity_increases += 1
                        if rng.integers(low=0,high=2) == 0: 
                              new_volume += 0.25 # maybe increase volume
                              volume_increases += 1
                    else: # not in_in_challenging steps
                        not_c_step += 1
                        if rng.integers(low=0,high=3) == 0: 
                              new_velocity -= 1 # maybe decrease velocity
                              velocity_decreases += 1
                        if rng.integers(low=0,high=3) == 0: 
                              new_volume -= 0.25 # maybe decrease volume
                              volume_decreases += 1
                    if (new_volume < 10) or (new_volume > 18): # out of range
                          volume_reset += 1
                          new_volume = volume # if you are out of line, reset
                    if (new_velocity < 58) or (new_velocity > 71): 
                          velocity_reset += 1
                          new_velocity = velocity # minumum velocity is passed in the parameters
                    random_start = start_time + rng.standard_normal()/(zfactor * 5 * tpq) # higher zfactor, less perturbation of time.
                    if random_start < 0: random_start = 0 # perterb the start time by a bit, but not below 0
                    stereo = rng.choice([1,3,5,7,8,9,11,13,15]) # locate in stereo field randomly
                    new_upsample = upsample + rng.choice([-1, 0, 1]) # give it some variability
                    #        Inst      Start         Dur              Vel           Ton   Oct   Voice  Stere  Envlp Gli Upsamp      R-Env 2nd-gl 3rd Mult      Line # ; Channel
                    pfields.append([1, random_start, duration + hold, new_velocity, tone, octave, 1.0, stereo, 0.0, 0.0, new_upsample, 0.0, 0.0, 0.0, new_volume, 1])
                start_time += duration
                duration = tp16th
                prev_note = note 
            time_step += 1
        v += 1
    print(f'challenging_step_count: {challenging_step_count}',end='\t')
    print(f'not_c_step: {not_c_step}')
    print(f'velocity_increases: {velocity_increases}',end='\t')
    print(f'velocity_decreases: {velocity_decreases}')
    print(f'volume_increases: {volume_increases}',end='\t')
    print(f'volume_decreases: {volume_decreases}')
    print(f'velocity_reset: {velocity_reset}',end='\t')
    print(f'volume_reset: {volume_reset}')
    pfields.sort() # This is done automatically by csound when called from the command line, but not when note events are sent

    
    
    print(f'list of notes is {len(pfields)} long')
    for i in range(len(pfields)):
        # print(f'index: {i}\t{pfields[i]}')
        pt.scoreEvent(0, 'i', pfields[i]) # here is where the notes are sent to ctcsound
        piano.logging.info(pfields[i]) 

    piano.printMessages(cs)

    delay_time =  max(min_delay,len(pfields) // 20) # need enough time to prevent csound being told to stop 
#     delay_time = min_delay

    print(f'about to delay to allow ctcsound to process the notes. delay_time: {delay_time}')
    print(f'last start time was at {round(start_time,1)}. Set f0 to {round(start_time+1,1)}')
    pfields.append(([0,"start","dur","velocity","tone","octave","voice","stereo","envelop","glis","upsamp","rEnvel","2nd G","3rd G","volume","chan"]))
    pfields.sort() # again after adding the header line
    save_file = open('pfields.csv','w+',newline='')
    with save_file:
          write = csv.writer(save_file)
          write.writerows(pfields) 
    save_file.close()
    time.sleep(delay_time) # once you hit the next line csound stops
    pt.stop() # this is important I think. It closes the output file.
    pt.join()   
    piano.printMessages(cs)    
    cs.reset()
    
    subprocess.run(['grep', 'invalid\|range\|error\|replacing\|overall\|rtevent', 'goldberg5.log']) # look in the log for important messages
    audio = Audio('/home/prent/Music/sflib/goldberg_aria1.wav')
    display(audio)  



# pass in any arbitrary array dimension, and return one with fewer positions in the second dimension.
# This function will take a voice,notes array and return a voice,notes array with fewer notes. 
# It compresses the notes from {start} to {end} to 1/2 their original size.
def compress_segment(input_array, start, end, skip=2):
    v = 0
    for voice in input_array:
        n = start
        for i in range(start, end, skip): # for example, start at 24, increment until just before 40 by 2 each time
            input_array[v,n] = voice[i]
            n += 1
        v += 1
    return(input_array[:,:(start - end) // 2]) # return all four voices all notes in each voice. Return only the first 32 slots


# ## Play the passed chorale to a wave file and perform it. 


def quick_play(chorale, gain, outfile = 'test.wav', outmidi='test.mid', bpm=60, soundfont='../font.sf2'): # save the chorale to midi and put up a display to play it. Fast, simple, and out of tune.
    pad8=np.zeros((chorale.shape[0],8),dtype=int)
    chorale = np.concatenate((chorale,pad8),axis=1) # midi likes and ending to avoid the edge case bugs.
    midi_output = piano_roll_to_midi(chorale, bpm=bpm) # convert to mido object at 60 time_steps per minute, one per second.
    music = muspy.from_mido(midi_output) # convert mido to muspy music
    muspy.write_midi(outmidi, music) # write the midi file to disk
    muspy.write_audio(outfile, music, 'wav', soundfont, 44100, gain) # generate a wav file using fluidsynth
    audio = Audio(outfile) # get ready to show the audio widget
    display(audio) # display the audio widget
    return (muspy.scale_consistency(music))

# ## Use the loaded model to create new chorales by masking out voices sequentially
# This process starts with the seven sements of the chorale after they have been compressed down to digestable chunks of (4,32) MIDI notes between zero and 57. That's what the model was designed to work with. We take each segment one at a time, and repeatedly masks one voice and predicts what that voice might have been. It's usually pretty close to the original, maybe 80% of what the original notes were. We start with voices S A T B. The first time through it creates a chorale with S' A T B. Then it makes S' A' T B, followed by S' A' T' B, finally making S' A' T' B'. It saves that generated chorale (4,32) in a slot in a (4,4,32) array. Then it does it again, gradually shifting from the original chorale to one that includes some odd notes. Each is stored in the (4,4,32) array. At the end, it reshapes that into a (16,32) array and returns it to the calling program.

# create the mask that will allow replacement of a voice
def mask_voice(chop_voice):
    mask = np.ones((4,32),dtype= int)
    mask[chop_voice,] = 0
    return mask


# ## Expand the compressed segments and concatenate the segments into a full chorale
#   
# - Start with the chorale that has been split into 7 (4,32) segments
# - Then decompress the segments to restore their original sizes
# - Then concatenate the segments together in a (16,264) array

def expand_and_concatenate(chorale, expand_codes):
      # chorale has a shape of (7,16,32) where 7 is the number of segments, 16 voices, 32 1/16th notes
      # expand_codes is 32,40, but decompress segment expects 24,32, because we want to take those 8 and make them 16.
     
      expanded_size = max(expand_codes[:,1]) # what is the maximum size of the expand_codes axis=1: 40 or 48
      segment_count = expand_codes.shape[0] # number of segments to expand
      voices = chorale.shape[1] # number of voice in the input chorale structure
      expanded_segment = np.zeros((segment_count,voices,expanded_size),dtype=int) # (7,16,40) where 7 is segments, 16 is voices, 40 is max length
      concat_chorale = np.empty((voices,0),dtype=int) # what the shape will be after expansion of segments
      
      for s in range(segment_count):
            if expand_codes[s,0] != expand_codes[s,1]: # you need to expand this segment
                  temp_expanded = decompress_segment(chorale[s], expand_codes[s,0], expand_codes[s,1])
                  if temp_expanded.shape[1] == expanded_size:
                        expanded_segment[s] = temp_expanded
                  else:
                        pad_size = expanded_size - temp_expanded.shape[1]
                        padding = np.zeros((chorale.shape[1],pad_size),dtype=int)
                        expanded_segment[s] = np.concatenate((temp_expanded,padding),axis=1)
            else:
                  pad_size = expanded_size - expand_codes[s,1] 
                  padding = np.zeros((chorale.shape[1],pad_size),dtype=int)
                  expanded_segment[s] = np.concatenate((chorale[s,:,:expand_codes[s,1]],padding),axis=1)
            concat_chorale = np.concatenate((concat_chorale, expanded_segment[s,:,:expand_codes[s,1]]),axis=1)
      return(concat_chorale)

# arpeggiate the chorale to make it more pianistic
# Also spread out the octaves a bit, adding 12 to an occasional soprano note, and subtracting 12 from a random bass note.
# Randomize the arpeggiation from 1 to 5 half measures at a time


# This imposes a mask on a chorale and causes arpeggiation. It also lowers some bass notes by an octave. Expects a (V,N) array
def arpeggiate(local_chorale, mask, skip):
    chorale = copy.deepcopy(local_chorale)
    chorale_voices = chorale.shape[0]
    # print(f'chorale_voices: {chorale_voices}')
    # print(f'mask.shape: {mask.shape}') # (16,8)
    mask = mask[:chorale_voices,:] # make the mask smaller to match the number of voices in the input chorale
    # print(f'mask.shape: {mask.shape}') # (16,8)
    for i in range(0, chorale.shape[1] // mask.shape[1], skip): # 0,notes // mask slots by skip
        start = i * mask.shape[1] # 8
        end = (i+1) * mask.shape[1] # 8
        chorale[:,start:end] = mask * chorale[:,start:end] # so here is where the mask is broadcasting all voices in the mask
    v = 0
    for voice in chorale: # 0 through 15 more or less 
        n = 0
        for note in voice:
            # if v == 0 : # soprano - need to randomly add an octave or two to the soprano voices
            #     if note != 0:
            #         chorale[v,n] = note + rng.choice([0,0,0,12,0,0,0,0]) # I can't get this to be rare enough.
            if v == 3: # bass reduce the octave
                if note != 0:
                    chorale[v,n] = note + rng.choice([0,0,-12,0,0,0])
            n += 1
        v += 1
    return(chorale)

def octave_shift(local_chorale, percent, clump):
      chorale = copy.deepcopy(local_chorale)
      inverse_percent = 1 / percent
      clump_factor = inverse_percent * clump # 5 * 24 = 120 
      v = 0
      while v < local_chorale.shape[0]:
            n = 0
            while n < local_chorale.shape[1]:
                  if rng.integers(clump_factor) == 0: # 0 out of 120 chances, but then do all the next notes in the voice for clump
                        c = 0
                        while (c < clump) and (c + n < local_chorale.shape[1]):
                              if chorale[v, n + c] != 0: 
                                    chorale[v, n + c] += 12 
                              c += 1
                        n += c
                  n += 1
            v += 1
      return(chorale)


# ## Generalize voice selection
# The goal is to simplify the selection of voices to perform. The arrays saved in segmented_chorales are(7,16,32) in shape. They were created by the coconet model, after masking one voice and having the model synthesize the missing voice. This process is repeated 16 times, creating an array 16 voices, playing seven segments from the chorale, each with 32 notes per segment. This is passed through function that expands that to a 16x264 array of voices and notes by taking the first four 32 note segments (0,1,2,3) and expanding them to 40 by doubling the length of the last 8 notes to 16. Those arrays are then concatenated with the  next two segments (4,5), which remain 32 notes, and the final whole note (6) which has 40 notes. Doing the math, 4 * 40 + 2 * 32 + 40 = 264 time steps of 16 voices. 
# This cell takes that 16x264 array, and selects which voices to include in the final chorale. Keeping all 16 voices results in too many "wrong" notes, since the model was unaware of the need to harmonize with any more than 4 voices at a time. I need a systematic way to select voices based on some criteria, perhaps consonance, or entropy, or some other metric.
# I prefer to just listen to different combinations and select the most interesting one. But I need a way to do this systematically, and in a way that preserves my impressions.
# The machine learning framework OpenCV, used for image classification and recognition, has a function called "Select Region of Interest". It displays an image, and the user at the terminal is able to draw a box using a mouse to provide the focus on a region of interest in the image. I used it to localize on something in the image that I want to crop the image to before passing it into a deep learning image recognition network. In this way, the network would have just what it needed and little other than what was needed to recognize the object at a later time. It's kind of human in the middle method to find what you are looking for. I need an analog to that to find the best combination of the 16 possible voices to select. And since I have about a hundred 16 voice chorales to choose from, I need a way to find the interesting voices easily and systematically.

# pick voices based on a mask, and how many voices you want to return
def voice_chooser(chorale,voice_mask,voices):
    chorale_mix = np.zeros((voices,264),dtype=int)
    chorale_mix = np.concatenate((chorale[voice_mask[0]:voice_mask[0]+voices,0:40],
                                  chorale[voice_mask[1]:voice_mask[1]+voices,40:40+40],
                                  chorale[voice_mask[2]:voice_mask[2]+voices,80:80+40],
                                  chorale[voice_mask[3]:voice_mask[3]+voices,120:120+40],
                                  chorale[voice_mask[4]:voice_mask[4]+voices,160:160+32],
                                  chorale[voice_mask[5]:voice_mask[5]+voices,192:192+32],
                                  chorale[voice_mask[6]:voice_mask[6]+voices,224:224+40]),axis=1)
    return(chorale_mix[:voices,:]) # return just the voices you want and all the notes in the array

def in_challenging_step(time_step, challenging_steps):
      for steps in challenging_steps:
            if (time_step >= steps[2]) and (time_step < steps[3]): 
                  return(True)
      return(False)

def valid_midi(chorale):
    valid = True
    for voice in chorale:
        for note in voice:
            valid = (0 <= note <= 127)
            if not valid: 
                print(f'midi not valid: {valid}. note: {note}')
                return (False)
    return(valid)


# note, you can't just multiply by other than 1. If you multiply by 2, you take a midi number like 65 and make it 130, 
# that's out of range for MIDI.
def set_mask(randomized=False):
      if randomized:
            mask = rng.integers(low=0, high=2, size=(16,18), dtype=np.uint8)
      else:
            mask = np.zeros((16,8))

            # 1st part - each of these is just a half measure - 8 slots - but it could be any dimension that is a multiple of 8. I've had success with 16x16
            mask[0,] =  [0,0,0,1,0,0,0,1]
            mask[1,] =  [0,0,1,1,0,0,1,1]
            mask[2,] =  [0,1,1,1,0,1,1,1]
            mask[3,] =  [1,1,0,1,1,0,1,0]
            # 2nd part
            mask[4,] =  [1,0,1,1,0,1,1,1]
            mask[5,] =  [1,0,1,1,0,0,1,1]
            mask[6,] =  [1,0,1,1,0,0,0,1]
            mask[7,] =  [1,1,0,1,1,0,0,1]
            # 3rd part
            mask[8,] =  [0,0,1,1,0,1,1,1]
            mask[9,] =  [0,1,1,1,0,0,0,1]
            mask[10,] = [0,0,0,1,0,0,1,1]
            mask[11,] = [1,1,1,1,1,0,1,0]
            # 4th part
            mask[12,] = [0,0,0,1,1,0,1,1]
            mask[13,] = [0,0,1,1,0,1,1,1]
            mask[14,] = [0,1,1,0,1,1,1,0]
            mask[15,] = [1,1,1,1,1,1,0,1]
      return(mask)

def shaded_mask(mask_shape, zero_ratio, one_ratio):
      total_size =  mask_shape[0] * mask_shape[1] # 320
      total_ratio = (zero_ratio + one_ratio) # 5
      divisor = total_size / total_ratio # 64
      print(f'{total_size}, {total_ratio * divisor}') # are they equal? If not, then try again.
      if total_size == total_ratio * divisor:
            zero_vector = np.zeros(int(divisor * zero_ratio), dtype=int)
            one_vector = np.ones(int(divisor * one_ratio), dtype=int)
            m = np.concatenate((zero_vector,one_vector),axis=0)
            rng.shuffle((m),axis=0) # in place shuffle
            m = m.reshape(mask_shape)
      else: 
            m = None
            print(f'The sum of the ratios {total_ratio} must be divisible into the product of the mask shapes: {mask_shape[0] * mask_shape[1]}')
            print(f'{mask_shape[0] * mask_shape[1] / total_ratio} is not an integer')
      return(m)      

# this code is stolen from the source code for the python music library muspy. Great code there.
def _get_scale(root: int, mode: str):
    """Return the scale mask for a specific root."""
    if mode == "major":
        c_scale = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], bool)
    elif mode == "minor":
        c_scale = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], bool)
    else:
        raise ValueError("`mode` must be either 'major' or 'minor'.")
    return np.roll(c_scale, root) # shifts every element of the array {root} places over with rollovers

# this function checks the consonance of a single time step of N notes. 
# I need to find out which notes are not in the root and mode
def quick_score(time_step, root, mode='major'):
    scale = _get_scale(root, mode.lower())
    note_count = 0
    in_scale_count = 0
    bad_note = 0
    for note in time_step:
        if scale[note % 12]:
            in_scale_count += 1
        else: 
            bad_note = note_count
        note_count += 1
    if note_count < 1:
        return np.nan
    return (in_scale_count / note_count), bad_note # this returns the last bad note found in the time_step

def sort_key(e):
    return (-e[8]) # sort by the value in the slot #8, class entropy, highest on top

def print_chorale_metric_report(dirname, from_voice, to_voice, expand_codes):

      metrics = []
      
      print(f'\tfile\t\tpitch\tpitch\tpitch\tpitch\tpoly-\tpoly\tscale\tclass\tfrom\tto')
      print(f'\tname\t\trange\tuse\tclas\tentr\tphony\trate\tconsist\tentropy\tvoice\tvoice')

      for filename in os.listdir(dirname):
            path = os.path.join(dirname,filename)
            # print(f'from_voice: {from_voice}, to_voice: {to_voice}')
            chorale = np.load(path) # bring in a premade chorale stored as a numpy array (16,264)
            # print(f'Just loaded. chorale.shape: {chorale.shape}')

            chorale = expand_and_concatenate(chorale, expand_codes) # decompress 
            # print(f'Expanded. chorale.shape: {chorale.shape:}')
            
            midi_output = piano_roll_to_midi(chorale[from_voice:to_voice,:]) # convert to mido object
            music = muspy.from_mido(midi_output) # convert mido to muspy music
            metric = [os.path.basename(filename), # make a list of metrics
                  muspy.pitch_range(music),
                  muspy.n_pitches_used(music),
                  muspy.n_pitch_classes_used(music), 
                  round(muspy.pitch_entropy(music),3),
                  round(muspy.polyphony(music),3), 
                  round(muspy.polyphony_rate(music),3),
                  round(muspy.scale_consistency(music),3), 
                  round(muspy.pitch_class_entropy(music),3),from_voice,to_voice]
            metrics.append(metric)
      metrics.sort(key=sort_key)
      for metric in metrics:
            print('{:12}'.format(metric[0]),'\t',metric[1],'\t',metric[2],          '\t',metric[3],'\t',metric[4],'\t',metric[5],'\t',metric[6],          '\t',metric[7],'\t',metric[8],'\t',metric[9],'\t',metric[10])
      return(metrics)

def report_expansion(scores, range_of_steps, challenging_steps, chorale, concat_challenging_steps, lines=10):
          # Get a report of all the in_tune steps and the lengths, and the challenging steps and their lengths
    print('--------------------------------------------------------------------------')
    print(f'scores.shape: {scores.shape}')
    print(f'first {lines} lines of original scores array. score, voice out of key, time_step')
    
    for t in range(0, scores.shape[0], 4):
            if t + 4 < scores.shape[0]:  print(f't: {t}\t{scores[t+0]}\t{t+1}\t{scores[t+1]}\t{t+2}\t{scores[t+2]}\t{t+3}\t{scores[t+3]}')
            if t > lines: break

    if t + 3 < scores.shape[0]:  print(f't: {t}\t{scores[t+0]}\t{t+1}\t{scores[t+1]}\t{t+2}\t{scores[t+2]}\t{t+3}\t{scores[t+3]}')
    elif t + 2 < scores.shape[0]:  print(f't: {t}\t{scores[t+0]}\t{t+1}\t{scores[t+1]}\t{t+2}\t{scores[t+2]}')
    elif t + 1 < scores.shape[0]:  print(f't: {t}\t{scores[t+0]}\t{t+1}\t{scores[t+1]}')
    else: print(f't: {t}\t{scores[t+0]}')
        
    print(f'\nchallenging_steps.shape: {challenging_steps.shape}, one for each set of steps in the source and target chorale')
   
    t = 0
    print('--------------------------------------------------------------------------')
    print(f'Contents of range_of_steps.shape: {range_of_steps.shape} before expansion. First {lines} lines.')
    print(f'\t\t\tInitial')
    print(f'\tstart\tend\tsize')
    for steps in range_of_steps:
        initial_size = steps[1] - steps[0]
        print(f't: {t},\t{steps[0]},\t{steps[1]},\t{initial_size}')
        t += 1 
        if t > lines: break
    
    print('--------------------------------------------------------------------------')
    print(f'Contents of challenging_steps after expansion. shape: {challenging_steps.shape}. First {lines} lines.')
    print(f'\t\t\tInitial\t\tExpanded')
    print(f'\tstart\tend\tsize\tstart\tend\tsize\tfactor')
    total_challenging_steps = 0
    t=0
    for steps in challenging_steps:
        initial_size = steps[1] - steps[0]
        if initial_size > 0: # why would it not be greater than 0
            final_size = steps[3] - steps[2]
            total_challenging_steps += final_size
            increase = final_size // initial_size
            print(f't: {t},\t{steps[0]},\t{steps[1]},\t{initial_size}\t{steps[2]},\t{steps[3]}\t{final_size}\t{increase}')
        else: print(f't: {t},\t{steps[0]},\t{steps[1]},\t{initial_size}\t{steps[2]},\t{steps[3]}')
        t += 1
        if t > lines: break
    print(f'total_challenging_steps: {total_challenging_steps}')
    print(f'total steps in chorale: {chorale.shape[1]}')
    
    print('--------------------------------------------------------------------------')
    print(f'Contents of concat_challenging_steps. Shape: {concat_challenging_steps.shape} after expansion. First {lines} lines.')
    print(f'\tInitial\t\t\tExpanded')
    print(f'\tstart\tend\tsize\tstart\tend\tsize\tincrease factor')
    total_challenging_steps = 0
    
    t=0
    for steps in concat_challenging_steps:
        initial_size = steps[1] - steps[0]
        if initial_size > 0: # why would it not be greater than 0
            final_size = steps[3] - steps[2]
            total_challenging_steps += final_size
            increase = final_size // initial_size
            print(f't: {t},\t{steps[0]}\t{steps[1]}\t{initial_size}\t{steps[2]}\t{steps[3]}\t{final_size}\t{increase}')
            #       t: 48,      678,        689,        11               1055,        1110           55           5
        else: print(f't: {t}\t{steps[0]}\t{steps[1]}\t{initial_size}\t{steps[2]}\t{steps[3]}')
        t += 1
        if t > lines: break
    print(f'total_challenging_steps: {total_challenging_steps}')
    print(f'total steps in chorale: {chorale.shape[1]}')
    print('--------------------------------------------------------------------------')


def midi_name(notes):
      note_names = np.empty(0,dtype='S2') # unicode string of length 2

      names = np.array(['C♮','C♯','D♮','D♯','E♮','F♮','F♯','G♮','G♯','A♮','A♯','B♮'])
      for note in notes:
            midi_octave = note // 12 
            midi_note = (note % 12)
            note_names = np.append(note_names,[names[midi_note]],0)
      return(note_names)


def assign_scores_to_time_steps(play_chorale, root, mode):
      # determine which time steps have notes that are not in the root key
      time_first_chorale = play_chorale.transpose() 
      # find those time steps that have a score less than 1 in this segment
      scores = np.zeros((len(time_first_chorale),3)) # defaults to floating point data type
      step = 0 # index to play_chorale
      t = 0 # index to scores
      print(f'in assign_scores_to_time_steps')
      for time_step in time_first_chorale:
            # print(f'time_step.shape: {time_step.shape}, time_step: {time_step}, root: {root}',end='\t')
            # print(f'type(time_step): {type(time_step)}, type(root): {type(root)}')
            in_key, bad_note = quick_score(time_step, root, mode=mode)
            if in_key < 1: # if it's not perfectly in tune with the root key
                  # if t < 10: print(f'scores index: {t} {time_step}, names: {midi_name(time_step)}, {time_step[bad_note]}, in_key: {in_key}, chorale step: {step}')
                  scores[t,0] = in_key
                  scores[t,1] = bad_note
                  scores[t,2] = step
                  t += 1
            step += 1
      
      # print(f'initial shape of scores: {scores.shape}')
      scores = scores[:t,]
      # print(f'scores.shape (number of time steps, measures): {scores.shape}')
      return(scores)

# you now have an array {scores} that has the {in_key} score, the voice with the wrong note, and the time_step in which it was found,
# only for those time_steps that had score that was less than perfect
# find a consecutive set of time steps with wrong notes in them that you can play with.

def find_challenging_time_steps(scores, chorale):
    time_steps = chorale.shape[1]
    r = 0 # index to range_of_steps, which you are building in this function
    range_of_steps = np.zeros((time_steps, 4), dtype=int) # second dimension set to 4. The represent original_start, original_end, new_start, new_end
    score_index = 0 # index to scores
    # last_found = int(scores[score_index, 2]) # the first in the array # starts with N'th time step in the original chorale before expansion
    for t in range(time_steps): # examine every time step to see if it scored low. 
        # print(f't: {t}, score_index: {score_index}, {int(scores[score_index,2])}',end='\n')
        if t == int(scores[score_index, 2]): # we have a match between the scores at score index and the time_step t
            # print(f'found {t} = int(scores[{score_index},2]): {int(scores[score_index,2])}')
            # print(f'int(scores[{score_index + 1},2]): {int(scores[score_index + 1,2])}')
            range_of_steps[r, 0] = t  # 0 is the location in the range_of_steps for the start of the challenging section in the chorale
            while (int(scores[score_index, 2] + 1) == int(scores[score_index + 1, 2])): # consecutive scores < 1
                # print(f'found two in a row. int(scores[{score_index}, 2] + 1): {int(scores[score_index, 2] + 1)}\t') # 
                # print(f'int(scores[{score_index + 1},2]): {int(scores[score_index + 1,2])}')
                score_index += 1 # advance through the score steps to see if they are continuous
                if score_index > scores.shape[0] - 2 : break
            range_of_steps[r, 1] = int(scores[score_index,2]) # last one not equal to the next one, so store in range_of_steps[r,1]
            # print(f'int(scores[{score_index},2]): {int(scores[score_index,2])}')
            r += 1 # advance in range_of_steps and prepare to look through the time steps for another match between score and time_steps
            score_index += 1
            if score_index > scores.shape[0] - 2 : break
    
    r -= 1
    # print(f'before truncation. range_of_steps.shape: {range_of_steps.shape}')
    range_of_steps = range_of_steps[:r+1,] # trim it down to the valid steps
    # print(f'after truncation. range_of_steps.shape: {range_of_steps.shape}')
    return(range_of_steps)         

# each time you expand a section, the previous ones need to move higher 
def find_in_tune_time_steps(play_chorale,range_of_steps):
    range_in_tune = np.zeros((play_chorale.shape[1],4), dtype=int)
#     print(f'starting find_in_tune_steps with play_chorale.shape: {play_chorale.shape}')
    in_tune = 0
    start_in_tune = 0
    for steps in (range_of_steps):
        if steps[0] > 0:
            range_in_tune[in_tune,0] = start_in_tune # 0
            range_in_tune[in_tune,1] = steps[0] # 80
            # print(f'loading range_in_tune[{in_tune}]: {range_in_tune[in_tune]}',end=' ')
            # print(f'range out of tune: {steps}')
            in_tune += 1
            start_in_tune = steps[1]
    range_in_tune[in_tune,0] = start_in_tune        
    range_in_tune[in_tune,1] = len(range_in_tune)
    range_in_tune = range_in_tune[:in_tune+1,]
    return (range_in_tune)

# Create the decompressed chorale with the interesting sections extended
def expand_challenging_time_steps(play_chorale, range_of_steps, range_in_tune, low=2, high=8):
    # 
    factors = rng.integers(low=low,high=high,size=range_of_steps.shape[0]) # give me a set of random numbers from 2 to but not including 8
    rs = 0 # index to range_of_steps - the challenging steps
    challenging_steps = copy.deepcopy(range_of_steps) # you are going to make changes, so you need to start with a copy of the array
#     inserted_steps = 0
    in_tune_steps = copy.deepcopy(range_in_tune) # you are going to make changes, so you need to start with a copy of the array
#     print(f'incoming play_chorale.shape: {play_chorale.shape}')
    decom_chorale = np.empty(shape=(play_chorale.shape[0],0),dtype=int) # create an empty array that will be concatenated to create the expanded chorale
    total_notes = 0
    new_in_tune_start = 0
    for steps in in_tune_steps:
        if rs < range_of_steps.shape[0]: # less than 8
            in_tune_steps[rs,2] = new_in_tune_start # new start of the in_tune steps in the expanded array
            in_tune_steps[rs,3] = new_in_tune_start + steps[1] - steps[0] # new ending location of the in_tune steps in the expanded array
            challenging_steps[rs,2] = steps[3] # new location of the challenging steps in the expanded array
            # inserted_steps = (range_of_steps[rs,1] - range_of_steps[rs,0]) * (factors[rs] - 1) # 
            second = play_chorale[:,steps[0]:steps[1]] # the current in_tune steps
            #                              +-- all voices
            #                              | +-- array containing the location in the chorale of the challenging steps
            #                              | |              +-- index to challenging steps
            #                              | |              |  +-- start of location of challenging steps in the chorale
            #                              | |              |  |  +-- array containing the location in the chorale of the challenging steps
            #                              | |              |  |  |                 +-- end of location of challenging steps in the chorale
            #                              | |              |  |  |                 |   +-- how many times to repeat this step
            #                              | |              |  |  |                 |   |       +-- index to challenging steps
            #                              | |              |  |  |                 |   |       |        +-- add notes not voices
            third = np.repeat(play_chorale[:,range_of_steps[rs,0]:range_of_steps[rs,1]],factors[rs],axis=1) # the lengthened challenging steps
            challenging_steps[rs,3] = challenging_steps[rs,2] + third.shape[1] # update the new ending point for this section in the expanded chorale
            new_in_tune_start = challenging_steps[rs,3] # start next time through
            total_notes += second.shape[1] + third.shape[1]
            decom_chorale = np.concatenate((decom_chorale, second, third),axis=1) 
        rs += 1
    second = play_chorale[:,steps[0]:steps[1]]
    total_notes += second.shape[1] 
    decom_chorale = np.concatenate((decom_chorale, second),axis=1) 
#     print(f'decom_chorale.shape: {decom_chorale.shape}')
#     print(f'total_notes: {total_notes}')
    in_tune_steps[rs-1,2] = new_in_tune_start
    in_tune_steps[rs-1,3] = new_in_tune_start + in_tune_steps[rs-1,1] - in_tune_steps[rs-1,0]
    return (decom_chorale, challenging_steps, in_tune_steps)    

def dropout(my_chorale, percent_to_zero=0.1):
      chorale = copy.deepcopy(my_chorale)
      indices = np.random.choice(np.arange(chorale.size), replace=False, size=int(chorale.size * percent_to_zero))
      chorale[np.unravel_index(indices, chorale.shape)] = 0
      return(chorale)

def midi_to_input(midi_file):
      music = muspy.read(midi_file)
      if music.key_signatures != []: # check if the midi file includes a key signature - some don't
            root = music.key_signatures[0].root 
            mode = music.key_signatures[0].mode # major or minor
      else: 
            print('Warning: no key signature found. Assuming C major')
            mode = "major"
            root = 0    
      if music.time_signatures != []: # check if the midi file includes a time signature - some don't
            numerator = music.time_signatures[0].numerator
            denominator = music.time_signatures[0].denominator 
      else: 
            print('Warning: no time signature found. Assuming 4/4')
            numerator = 4
            denominator = 4
      # turn it into a piano roll
      piano_roll = muspy.to_pianoroll_representation(music,encode_velocity=False) # boolean piano roll if False, default True
      # print(piano_roll.shape) # should be one time step for every click in the midi file
      q = music.resolution # quarter note value in this midi file. 
      q16 = q // 4 # my desired resolution is by 1/16th notes
      print(f'time signatures: {numerator}/{denominator}')
      time_steps = piano_roll.shape[0] // q16
      print(f'music.resolution is q: {q}. q16: {q16} time_steps: {time_steps} 1/16th notes')
      sample= np.zeros(shape=(time_steps,4)).astype(int) # default is float unless .astype(int)
      # This loop is able to load an array of shape N,4 with the notes that are being played in each time step
      for click in range(0,piano_roll.shape[0],q16): # q16 is skip 240 steps for 1/16th note resolution
            voice = 3 # start with the low voices and decrement for the higher voices as notes get higher
            for i in range(piano_roll.shape[1]): # check if any notes are non-zero
                  time_interval = (click) // q16 
                  if (piano_roll[click][i]): # if velocity anything but zero - unless you set encode_velocity = False
                  # if time_interval % 16 == 0:
                  #     print(f'time step: {click} at index {i}, time_interval: {time_interval}, voice: {voice}')
                  # i is the midi note number. I want to transpose it into C
                        sample[time_interval][voice] = i - root # index to the piano roll with a note - transposed by the key if not C which is 0
                        voice -= 1 # next instrument will get the higher note
      return (sample,root,mode)                  