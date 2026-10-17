import numpy as np, json
G=json.load(open('grid.json')); G2={}
for k,g in G.items():
    r=np.load(f'mm/{k}.npy'); b=r[:,0]; pos=r[:,1]
    per0=np.median(np.diff(b))
    idx=np.concatenate([[0],np.cumsum(np.round(np.diff(b)/per0))])
    # iterative robust fit
    m=np.ones(len(b),bool)
    for _ in range(5):
        A=np.polyfit(idx[m],b[m],1); res=b-np.polyval(A,idx)
        m=np.abs(res)<0.03
    per,b0=A
    # downbeat phase: beat index mod 4 of madmom downbeats
    di=idx[pos==1].astype(int)%4
    cnt=np.bincount(di,minlength=4); ph=int(np.argmax(cnt))
    first_bar=b0+ph*per
    while first_bar-4*per>-0.05: first_bar-=4*per
    G2[k]=dict(name=g['name'],bpm=60/per,first_bar=float(first_bar),T=g['T'])
    print(k,g['name'][:14].ljust(14),'bpm %.3f'%(60/per),'res std %.1f ms'%(res[m].std()*1000),'inliers %d/%d'%(m.sum(),len(m)),'downbeat votes',cnt,'first_bar %.3f'%first_bar)
json.dump(G2,open('grid2.json','w'),indent=1)
