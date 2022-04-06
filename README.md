# coconet-pytorch-csound

This repo is derived from the pytorch implementation of the original coconet paper (https://arxiv.org/pdf/1903.07227.pdf) created by Kevin Donoghue: (https://github.com/kevindonoghue/coconet-pytorch). I was able to get his code to run with a few modifications. This notebook uses some of his code to train, save, and later load the model trained over 30,000 iterations.

Before running the notebooks and python programs, you will need to install an environment. My preference is for Ananconda, or Miniconda. Those can be obtained from the Ananconda.org site here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

Once that is installed, then issue the following command on the conda command line:

<code>conda env create -f gym_conda_environment.yml</code>

<code>conda activate gym</code>
      
You will also need to install a few other programs not available through Anaconda:

- fluidsynth, available from most linux distributions
- A sound font to use with fluidsynth. These are too large to include in the github repo, and are available here: https://musical-artifacts.com/?apps=fluidsynth&formats=sf2&license=free&order=top_rated&tags=soundfont I recommend the Roland soundfont, since it has most of what you would want. 
- csound-devel - if you want to use csound to generate wave files - that's optional, since fluidsynth is capable of generating serviceable wave files from midi. I use Csound because it does a better job and supports different sample files, tuning systems, and convolution. But it can be a challenge to get working. 
- midi2audio, available with <code>pip install midi2audio</code>

Here is what the current repo contains:

## Notebook to train the model: coconet_train_the model.ipynb

This is taken almost directly from https://github.com/kevindonoghue/coconet-pytorch with a few changes to allow the training to take place without a GPU and other minor fixes. This takes about 30 hours to train. You could save 30 hours by obtaining from the author. Let me know. prent at ripnread.com.
Or you could reduce the training cycles from 30,000 to something smaller than that. 

## Load the trained model and generate some chorales: coconet_generate_many_chorales.ipynb

This notebook loads the model and generates many 16 voice chorales based on the bass part of "Schmucke dich, o liebe Seele" by JS Bach BWV 180. It's an interesting chorale, because the first four phrases are 2 1/2 measures long, comprising 40 1/16th notes. The model only accepts 32 1/16th note input. So I have to compress the last 16 1/16 notes into 8, then expand them on the back side back out to 40. That took a while to figure out. Each is saved to a numpy array of dimension (16,264). The models create 4 voice chorales by preserving only the bass line of the original chorale, and synthesizing the other three parts. It does this four times and stacks those four chorales on top of each other to make 16 voice chorales. The drawback of this method is that each of the four stacked chorales was generated with absolutely no idea what the others were up to. It produces some challenging results. These can then be read in an further modified as you like. I found them too strange to get much value out of them. The directory "numpy_chorales" contains some examples. This notebook relies on midi realizations, plus some that use csound. Csound can be a challenge to set up. 

## Load the trained model and generate chorales: coconet_incremental_synthesis_hp800.ipynb

This notebook takes a different approach. I start with the Schmucke dich chorale, and erase on of the four parts, and allow the model to synthesize that one part. Then I take that four part chorale, mask the next voice and synthesis its replacement. I do this 16 times, so that in the end I have replaced all four voices four times. It still sounds somewhat like the original, but different enough to be interesting. Each step retains about 80% of the notes. The model was trained to get as close as possible to the notes that were in the original chorale, and it does a pretty good job of that. After 16 iterations, it's not at all the same as the original. I did this to create 100 chorales, and saved them all as numpy arrays of dimentions (7,16,32).  I included some examples in the directory "segmented_chorales". It takes about 20 minutes to make one 16 voice chorale with 32 1/16th notes in seven segments. Those need to be decompressed and concatenated to make a full chorale. This notebook takes most of the functions in previous notebooks and puts them in python libraries that are called from the notebook.

## Load the numpy arrays saved above and generate some chorales with arpegiation and other modifications: coconet_selective_stretch_library.ipynb 

This takes a numpy array from the directory "segmented_chorales" and makes some expansions of it to create some interesting music. The specific alterations include arpeggiation, stretching of interesting sections, and other changes. 

I also include code to use Csound to realize the chorale using some very large Bosendorfer samples I purchased. It is set to use the Victorian Rational Well Temperament, a tuning that sounds better than twelve tone equal temperament for Bach.
