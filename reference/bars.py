import numpy as np, json, librosa
G=json.load(open('grid2.json'))
def beats_for(k):
    g=G[k]; T=g['T']
    if k=='09':
        r=np.load('mm/09.npy'); b=r[:,0]; pos=r[:,1]
        # smooth intervals with median filter to remove jitter
        from scipy.signal import medfilt
        d=medfilt(np.diff(b),9); b2=np.concatenate([[b[0]],b[0]+np.cumsum(d)])
        # re-anchor drift: blend toward original every beat
        b2=b+ (np.convolve(b-b2,np.ones(9)/9,'same')*0) ; b2=0.5*b+0.5*np.convolve(b,np.ones(5)/5,'same'); b2[:2]=b[:2]; b2[-2:]=b[-2:]
        first=np.where(pos==1)[0][0]
        return b2, first
    per=60/g['bpm']; b=np.arange(g['first_bar'],T-0.01,per); return b,0
out={}
for k in G:
    b,first=beats_for(k)
    y,sr=librosa.load(f'wav/{k}.wav',sr=22050,mono=True)
    S=np.abs(librosa.stft(y,n_fft=2048,hop_length=512)); f=librosa.fft_frequencies(sr=sr,n_fft=2048); fps=sr/512
    def bandE(lo,hi):
        e=(S[(f>=lo)&(f<hi)]**2).sum(0); return e
    lo,mi,hi=bandE(20,150),bandE(150,3000),bandE(3000,11000)
    bars=b[first::4]
    feat=[]
    for i in range(len(bars)-1):
        a,z=int(bars[i]*fps),int(bars[i+1]*fps)
        feat.append([10*np.log10(x[a:z].mean()+1e-9) for x in (lo,mi,hi)])
    feat=np.array(feat)
    out[k]=dict(beats=b.tolist(),first=int(first),feat=feat.tolist())
json.dump(out,open('bars.json','w'))
for k in out:
    F=np.array(out[k]['feat']); L=F[:,0]; top=np.percentile(L,90)
    s=''.join(str(int(np.clip((v-top+18)/2,0,9))) for v in L)   # 9 = full bass, each step 2 dB
    print(k,G[k]['name'],'%.2f'%G[k]['bpm'])
    for i in range(0,len(s),32): print('   %3d '%i+' '.join(s[i+j:i+j+8] for j in range(0,32,8)))
