# Daubechies (min phase) and symlet filters by spectral factorisation, in mpmath for precision.
import itertools, mpmath as mp
mp.mp.dps = 50
from math import comb
def roots_y(p):
    # P(y) = sum_{k<p} C(p-1+k,k) y^k ; roots in y
    coeffs = [comb(p-1+k,k) for k in range(p)][::-1]
    return mp.polyroots(coeffs, maxsteps=200, extraprec=200) if p>1 else []
def z_roots(p):
    # y = (2 - z - 1/z)/4 -> z^2 - (2-4y) z + 1 = 0 ; pick |z|<1
    out=[]
    for y in roots_y(p):
        b = 2-4*y
        disc = mp.sqrt(b*b-4)
        z1=(b+disc)/2; z2=(b-disc)/2
        out.append((z1,z2))
    return out
def build(p, choice):
    zs=[]
    for (pair,c) in zip(z_roots(p),choice):
        z1,z2=pair
        small = z1 if abs(z1)<1 else z2
        big = z2 if abs(z1)<1 else z1
        zs.append(small if c==0 else big)
    # polynomial: (1+z)^p * prod (z - r)
    poly=[mp.mpf(1)]
    def mul(poly, r):
        res=[mp.mpc(0)]*(len(poly)+1)
        for i,c in enumerate(poly):
            res[i]+=c; res[i+1]-=c*r
        return res
    for _ in range(p): poly=mul(poly,-1)
    for r in zs: poly=mul(poly,r)
    h=[mp.re(c) for c in poly]
    s=sum(h); h=[c*mp.sqrt(2)/s for c in h]
    return h
def phase_nonlin(h):
    # deviation of the phase from linear
    import cmath, math
    hf=[float(c) for c in h]
    ws=[math.pi*k/64 for k in range(1,48)]
    ph=[]
    for w in ws:
        H=sum(c*cmath.exp(-1j*w*n) for n,c in enumerate(hf))
        ph.append(cmath.phase(H))
    # unwrap
    import numpy as np
    ph=np.unwrap(ph); A=np.vstack([ws,np.ones(len(ws))]).T
    coef,res,_,_=np.linalg.lstsq(A,ph,rcond=None)
    return float(np.sum((A@coef-ph)**2))
out={}
for p in [1,2,3,4,8]:
    h=build(p,[0]*(p-1))
    # Daubechies' order: largest-first convention h0 ... ; min-phase root choice gives db
    out['db%d'%p]=h
# symlet 4: try all root choices (complex pairs must be chosen consistently)
p=4
best=None
for ch in itertools.product([0,1],repeat=p-1):
    h=build(p,list(ch))
    if any(abs(mp.im(c))>1e-30 for c in h): pass
    e=phase_nonlin(h)
    if best is None or e<best[0]: best=(e,ch,h)
out['sym4']=best[2]
print(best[1])
for k,h in out.items():
    print(k, '[' + ', '.join(mp.nstr(c,17) for c in h) + ']')
    print('  sum', mp.nstr(sum(h),20), 'sumsq', mp.nstr(sum(c*c for c in h),20))
print('--- sym4 candidates with sumsq=1')
for ch in itertools.product([0,1],repeat=3):
    h=build(4,list(ch))
    ss=sum(c*c for c in h)
    if abs(ss-1)<1e-20:
        print(ch, phase_nonlin(h), '[' + ', '.join(mp.nstr(c,17) for c in h) + ']')
