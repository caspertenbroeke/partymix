import numpy as np, sys
sys.argv=['x',sys.argv[1]]; import engine as E
pre=sys.argv[0]
R=[]
for i in range(7):
    try: r=np.load(f'chunks/v2_{i}.wav.mm.npy')
    except: continue
    r=r.copy(); r[:,0]+=i*280
    lo=i*280+(5 if i else 0); hi=i*280+285
    R.append(r[(r[:,0]>=lo)&(r[:,0]<hi)])
r=np.concatenate(R); bt=r[:,0]; db=bt[r[:,1]==1]
exp_beats=E.T(np.arange(0,E.TOTAL_BARS*4)/4); exp_db=E.T(np.arange(E.TOTAL_BARS))
def near(a,b):
    i=np.searchsorted(b,a).clip(1,len(b)-1); return np.minimum(np.abs(a-b[i]),np.abs(a-b[i-1]))
cues=[(float(E.T(E.O[i])),E.SET[i][0]) for i in range(E.n)]+[(E.TOTAL_SEC,'end')]
lag=np.median([ (bt[j]-exp_beats[np.argmin(np.abs(exp_beats-bt[j]))]) for j in range(0,len(bt),7)])
print('median lag detected vs grid %.0f ms'%(lag*1000)); bt=bt-lag; db=db-lag
for (t0,k),(t1,_) in zip(cues[:-1],cues[1:]):
    m=(bt>=t0)&(bt<t1); mb=bt[m]; per=np.median(np.diff(mb))
    d=near(mb,exp_beats); mdb=db[(db>=t0)&(db<t1)]; dd=near(mdb,exp_db)
    print(f'{k:4s} {int(t0//60)}:{int(t0%60):02d}  beats on grid {100*(d<0.035).mean():5.1f}%  half-beat off {100*(d>per*0.3).mean():4.1f}%  downbeats on bar-1 {100*(dd<0.05).mean():5.1f}%')
