import ctcsound
import pandas as pd
import time
import logging

def start_logger(fname = 'piano.log'):
      logger = logging.getLogger()
      fhandler = logging.FileHandler(filename=fname, mode='w')
      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
      fhandler.setFormatter(formatter)
      logger.addHandler(fhandler)
      logger.setLevel(logging.DEBUG)

def load_csd(csd_file):
      # csd_file = 'goldberg5.csd' 
      csd_content = ""
      lines = 0
      empty = False
      with open(csd_file,'r') as csd:
            while not empty:
                  read_str = csd.readline()
                  empty = not read_str
                  lines += 1
                  if read_str.startswith('i') or read_str.startswith('t'):
                        pass
                  else:
                        csd_content += read_str
      csd.close()        
      return(csd_content, lines)

# added 1/24/22 to supress messages. modified 1/29/22 to send messages to log file
def flushMessages(cs, delay=0):
      s = ""
      if delay > 0:
            time.sleep(delay)
      for i in range(cs.messageCnt()):
            s += cs.firstMessage()
            cs.popFirstMessage()
      return s

def printMessages(cs, delay=0):
      s = flushMessages(cs, delay)
      this_many = 0
      if len(s)>0:
            logging.info(s)
                  

def load_csound(csd_content):
      cs = ctcsound.Csound()    # create an instance of Csound
      cs.createMessageBuffer(0)    
      cs.setOption('-odac')
      cs.setOption("-G")  # Postscript output
      cs.setOption("-W")  # create a WAV format output soundfile
      printMessages(cs)
      cs.compileCsdText(csd_content)       # Compile Orchestra from String - already read in from a file 
      printMessages(cs)
      cs.start()             # When compiling from strings, this call is necessary before doing any performing
      flushMessages(cs)
      
      pt = ctcsound.CsoundPerformanceThread(cs.csound()) # Create a new CsoundPerformanceThread, passing in the Csound object
      pt.play()              # starts the thread, which is now running separately from the main thread. This 
                        # call is asynchronous and will immediately return back here to continue code
                        # execution.
      return (cs, pt)

def load_notes(note_file):
      # read in from a file called goldberg_aria1.mac.csv into a padas dataframe, and remove 
      df = pd.read_csv(note_file) # this is the complete aria.
      df = df[df.chan != 1] # remove any of the channel 1 rows
      df.reset_index(drop=True, inplace=True) # drop the previous index, and do it in place.
      return (df)




