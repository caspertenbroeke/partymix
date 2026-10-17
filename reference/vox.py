import numpy as np, json, soundfile as sf, sys, os
B=json.load(open('bars.json')); G=json.load(open('grid2.json'))
V={}
for k in B:
    if not os.path.exists(f'stems/{k}.done'): continue
    v,sr=sf.read(f'stems/{k}.wav',dtype='float32'); v=v.mean(1) if v.ndim>1 else v
    b=np.array(B[k]['beats']); bars=b[B[k]['first']::4]
    e=np.array([10*np.log10((v[int(bars[i]*sr):int(bars[i+1]*sr)]**2).mean()+1e-10) for i in range(len(bars)-1)])
    e=np.nan_to_num(e,nan=-100); top=np.percentile(e,90)
    vs=''.join(str(int(np.clip((x-top+20)/2.2,0,9))) for x in e)
    F=np.array(B[k]['feat']); L=F[:,0]; t2=np.percentile(L,90)
    bs=''.join(str(int(np.clip((x-t2+18)/2,0,9))) for x in L)
    V[k]=vs
    print(k,G[k]['name'],'%.2f'%G[k]['bpm'],'  (B=bass V=vocals, 8 bars per group)')
    for i in range(0,len(bs),32):
        print('  %3d B '%i+' '.join(bs[i+j:i+j+8] for j in range(0,32,8)))
        print('      V '+' '.join(vs[i+j:i+j+8] for j in range(0,32,8)))
json.dump(V,open('vox.json','w'))
