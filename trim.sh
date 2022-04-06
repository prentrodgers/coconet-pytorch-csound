#! /bin/bash
set -v
csound goldberg_aria1c.csd   
sox /home/prent/Music/sflib/goldberg_aria1a-c.wav save1.wav reverse
sox save1.wav save2.wav silence 1 0.01 0.01
sox save2.wav save1.wav reverse
sox save1.wav /home/prent/Music/sflib/goldberg_aria1-t14.wav silence 1 0.01 0.01
rm save1.wav
rm save2.wav
ls -lth /home/prent/Music/sflib/goldberg_aria1-t14.wav
ffmpeg -y -i /home/prent/Music/sflib/goldberg_aria1-t14.wav -b:a 320k /home/prent/Music/sflib/goldberg_aria1-t14.mp3