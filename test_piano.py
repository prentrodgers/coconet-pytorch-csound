import pandas as pd
import time
import piano 
import sys

CSD_FILE = 'goldberg5.csd' # 214 tones to the octave tonality diamond to the 31 limit
NOTES_FILE = "goldberg5.mac.csv" # "
CSD_FILE = 'goldberg_aria1.csd' # VRWT version
NOTES_FILE = "goldberg_aria1.mac.csv" # "
LOGNAME = 'goldberg5.log'

def main():
      piano.start_logger(fname=LOGNAME)
      piano.logging.info(f'Logging messages to: {LOGNAME}')
      csd_content, lines = piano.load_csd(CSD_FILE)
      piano.logging.info(f'Loaded the csd file {CSD_FILE}. There are {lines} lines read containg {len(csd_content)} bytes')
      cs, pt = piano.load_csound(csd_content)
      df = piano.load_notes(NOTES_FILE)
      piano.logging.info(f'Read in {len(df)} notes from {NOTES_FILE}')
      # delay_time = 5
      # piano.logging.info(f'Taking a {delay_time} second delay to enable the orchestra to load the samples') 
      # I don't this is necessary. It will start when it's good and ready
      # time.sleep(delay_time)
      piano.logging.info('Inst Start  Dur   Vel   Ton  Oct  Voic Ste Envl Glis Upsm Renv 2-gl 3r-gl  Mult chan')
      for index, _ in df.iterrows():
            pfields = df.iloc[index].values.tolist()
            pfields[0] = pfields[0] / 32 # reduce the start time
            pfields[1] = pfields[1] / 32 # reduce the duration to accomodate the lack of t0 1776 support in ctcsound.
            pfields.insert(0,1) # insert the instrument number before all the other elements of the list.
            if index < 10: piano.logging.info (pfields) # piano.logging.info only the first few
                  #  +-- method scoreEvent on pd.CsoundPerformanceThread(cs.csound())
                  #  |    +-- if non-zero start time from performance start instead of the default of relative to the current time.
                  #  |    |   +-- opcode is i for note event
                  #  |    |   |   +-- pfields are the tuple, list, or ndarray of MYFLTs with event p values
            pt.scoreEvent(0, 'i', pfields) 
            if index > 440: break
      piano.printMessages(cs)      
      delay_time = df.iloc[len(df)-1]['Start'] / 32 
      piano.logging.info(f'Delay for {round(delay_time,0) + 10} seconds to allow the csound instance to play all the notes')
      time.sleep(delay_time + 5) # this delay has to be at least as long as you want the instrument to play.
      # Once you get to the next line, it's all over.
      pt.stop() # this is important I think. It closes the output file.
      pt.join()              # Join will wait for the other thread to complete. If we did not call join(),
                        # after t.play() returns we would immediate move to the next line, c.stop(). 
                        # That would stop Csound without really giving it time to run. 
      piano.printMessages(cs)    
      cs.reset()
main()