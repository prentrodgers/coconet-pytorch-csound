# coconet-pytorch-csound

This repo is uses code from the pytorch implementation of the original coconet paper (https://arxiv.org/pdf/1903.07227.pdf) created by Kevin Donoghue: (https://github.com/kevindonoghue/coconet-pytorch). I was able to get his code to run with a few modifications. This notebook uses some of his code to train, save, and later load the model trained over 30,000 iterations, and to generate some variations on the chorales. 

If you are interested in hearing some examples, you can visit my web page for some recent work: http://ripnread.com/sample-page/code/fantasia-on-artificial-chorales/

Before running the notebooks and python programs, you will need to install an environment. My preference is for Ananconda, or Miniconda. Those can be obtained from the Ananconda.org site here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

Once that is installed, then issue the following command on the conda command line:

<code>conda env create -f gym_conda_environment.yml</code>

<code>conda activate gym</code>
      
You will also need to install a few other programs not available through Anaconda:

- fluidsynth, available from most linux distributions
- A sound font to use with fluidsynth. These are too large to include in the github repo, and are available here: https://musical-artifacts.com/?apps=fluidsynth&formats=sf2&license=free&order=top_rated&tags=soundfont I recommend the Roland soundfont, since it has most of what you would want. 
- csound-devel - if you want to use csound to generate wave files - that's optional, since fluidsynth is capable of generating serviceable wave files from midi. I use Csound because it does a better job and supports different sample files, tuning systems, and convolution. But it can be a challenge to get working. You will also need some samples. I use a set of Bosendorfer piano samples that I paid a good deal of money for, and which are not avaiable online to my knowledge. Csound can be used to make wonderful music, but it has a steep learning curve.
- 
- midi2audio, available with <code>pip install midi2audio</code>

Here is what the current repo contains:

## Notebook to train the model: coconet_with_initial_fixes.ipynb

This is taken almost directly from https://github.com/kevindonoghue/coconet-pytorch with a few changes to allow the training to take place without a GPU and other minor fixes. This takes about 30 hours to train. You could save 30 hours by obtaining from the author. Let me know. prent at ripnread.com.
Or you could reduce the training cycles from 30,000 to something smaller than that. Those who know Dropbox can obtain a copy here: https://www.dropbox.com/s/tsa95byleku5gwj/model_zipped.zip?dl=0

## Load the trained model and generate chorales: coconet_incremental_synthesis_hp800.ipynb

This notebook takes a different approach. I start with a real Bach chorale, and erase on of the four parts, and allow the model to synthesize that one part. Then I take that four part chorale, mask the next voice and synthesis its replacement. I do this 16 times, so that in the end I have replaced all four voices four times. It still sounds somewhat like the original, but different enough to be interesting. Each step retains about 80% of the notes. The model was trained to get as close as possible to the notes that were in the original chorale, and it does a pretty good job of that. After 16 iterations, it's not at all the same as the original. I did this to create about 100 numpy arrays of artificial chorales, and saved them as numpy arrays of dimentions (7,16,32).  I included some examples in the directories: dearest_jesus_chorales, schmucke_chorales, and schmucke_chorales. It takes about 20-40 minutes to make one 16 voice chorale with 32 1/16th notes in several segments. Those need to be decompressed and concatenated to make a full chorale. This notebook takes most of the functions in previous notebooks and puts them in python libraries that are called from the notebook. 

## Read the generated chorale numpy arrays in and make some music out of them: coconet_dearest_jesus.ipynb, coconet_schmucke.ipynb, and coconet_wake_up.ipynb

These notebooks load the trained model and generates music through a variety of algorithmic techniques. These include stretching and compressing, arpeggiating, changing the dynamics, and other alterations to make the music sound more human. Many of the modifications were made to customize the csound audio generation, and won't work if you don't have csound. This notebook creats some midi realizations, and in most cases csound audio files as well. Csound can be a challenge to set up. There are three separate notebooks, one each for Schmücke dich, o liebe Seele (Adorn yourself, O dear soul), Wachet doch, erwacht, ihr Schläfer (Wake up, wake up Sleepers) and Liebster Jesu, wir sind hier (Dearest Jesus).

I also include code to use Csound to realize the chorale using some very large Bosendorfer samples I purchased. It is set to use the Victorian Rational Well Temperament, a tuning that sounds better than twelve tone equal temperament for Bach.
