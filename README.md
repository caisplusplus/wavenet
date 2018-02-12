# Wavenet

This code is largely based off of https://github.com/tomlepaine/fast-wavenet.  All credit goes to that repo. 

This version is hopefully easier to read and learn about what Wavenet does.

For the training code obviously look at `train.py`. This code should be pretty
straightforward if you understand the basic idea of Wavenet. The audio
generation code in `generate.py` is a little bit more complex because it uses a
dynamic programming technique to speed up generation. 
