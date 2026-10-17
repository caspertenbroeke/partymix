import sys, numpy as np, json
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
f=sys.argv[1]; out=sys.argv[2]
act=RNNDownBeatProcessor()(f)
np.save(out+".act.npy",act)
r=DBNDownBeatTrackingProcessor(beats_per_bar=[4],fps=100,min_bpm=100,max_bpm=160)(act)
np.save(out+".npy",r)
print(f, len(r))
