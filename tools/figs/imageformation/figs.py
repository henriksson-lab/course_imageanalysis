import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j1, j0
from math import pi
R='/data/henriksson/github/course/imageanalysis'
plt.rcParams.update({"font.size": 15, "axes.titlesize": 16})
rng=np.random.default_rng(7)
def otf_airy(s):
    s=np.minimum(np.abs(s),1.0); return 2/pi*(np.arccos(s)-s*np.sqrt(1-s*s))
def airy(v):
    v=np.where(np.abs(v)<1e-9,1e-9,v); return (2*j1(v)/v)**2
cell=plt.imread(R+'/demos/src/fourier2d/cell256.png')
if cell.ndim==3: cell=cell[...,0]
cell=cell.astype(float)
# ---------------------------------------------------------------- 1. imaging chain
n=256; px=40.0; NA=0.8; lam=520.0
f=np.fft.fftfreq(n,d=px)  # cycles/nm
FX,FY=np.meshgrid(f,f); fr=np.hypot(FX,FY); fc=2*NA/lam
H=otf_airy(fr/fc)
obj=cell/cell.max()
blur=np.real(np.fft.ifft2(np.fft.fft2(obj)*H))
k=4  # 160 nm pixels
samp=blur.reshape(n//k,k,n//k,k).mean(axis=(1,3))
photons=20
noisy=rng.poisson(np.clip(samp,0,None)*photons)/photons
q=np.round(noisy/noisy.max()*7)/7
fig,ax=plt.subplots(1,5,figsize=(16,3.9))
for a,img,t in zip(ax,[obj,blur,samp,noisy,q],["object","optics: ∗ PSF","sampled: 160 nm pixels","noise: ~20 photons","quantised: 3 bits"]):
    a.imshow(img,cmap="gray",interpolation="nearest"); a.set_title(t); a.axis("off")
for i in range(4):
    fig.text(0.2*(i+1)-0.004,0.47,"→",fontsize=28,ha="center",va="center")
plt.tight_layout(w_pad=2.2); plt.savefig(R+'/psf/fig-chain.png',dpi=90); plt.close()
# ---------------------------------------------------------------- 2. Airy PSF, profile, OTF
NA=1.4; lam=520.0
xs=np.linspace(-700,700,281); X,Y=np.meshgrid(xs,xs)
v=2*pi*NA*np.hypot(X,Y)/lam; A=airy(v)
fig,ax=plt.subplots(1,3,figsize=(16,4.6))
ax[0].imshow(np.log10(np.maximum(A,1e-4)),cmap="inferno",extent=[-0.7,0.7,-0.7,0.7]); ax[0].set_title("Airy pattern (log scale)"); ax[0].set_xlabel("x (µm)"); ax[0].set_ylabel("y (µm)")
r=np.linspace(0,700,701); sig=0.21*lam/NA
ax[1].plot(r,airy(2*pi*NA*r/lam),lw=3,label="Airy (ideal lens)")
ax[1].plot(r,np.exp(-r**2/(2*sig**2)),"--",lw=2.5,label="Gaussian, σ = 0.21λ/NA")
ax[1].axvline(0.61*lam/NA,color="#d62728",ls=":",lw=2); ax[1].text(0.61*lam/NA+10,0.8,"0.61λ/NA\n= 227 nm",color="#d62728")
ax[1].set_xlabel("distance from the centre (nm)"); ax[1].set_ylabel("intensity"); ax[1].legend(fontsize=12); ax[1].set_title("PSF profile, NA 1.4, λ 520 nm")
fcu=2*NA/lam*1000; ff=np.linspace(0,8,400)
ax[2].plot(ff,otf_airy(ff/fcu),lw=3,label="Airy OTF")
ax[2].plot(ff,np.exp(-2*pi**2*(sig/1000)**2*ff**2),"--",lw=2.5,label="Gaussian OTF")
ax[2].axvline(fcu,color="#d62728",ls=":",lw=2); ax[2].text(fcu+0.1,0.8,"2NA/λ\n= 5.4 /µm",color="#d62728")
ax[2].set_xlabel("spatial frequency (cycles/µm)"); ax[2].set_ylabel("OTF (contrast kept)"); ax[2].legend(fontsize=12); ax[2].set_title("OTF: a low-pass with a hard edge")
plt.tight_layout(); plt.savefig(R+'/psf/fig-airy.png',dpi=90); plt.close()
# ---------------------------------------------------------------- 3. two points
fig,ax=plt.subplots(1,4,figsize=(16,3.8),sharey=True)
x=np.linspace(-500,500,1001); u=lam/NA
for a,d,t in zip(ax,[1.0,0.61,0.47,0.35],["d = 1.0 λ/NA","Rayleigh: 0.61 λ/NA","Sparrow: 0.47 λ/NA","0.35 λ/NA"]):
    D=d*u; p1=airy(2*pi*NA*np.abs(x-D/2)/lam); p2=airy(2*pi*NA*np.abs(x+D/2)/lam)
    a.plot(x,p1,color="#999",lw=1.5); a.plot(x,p2,color="#999",lw=1.5); a.plot(x,p1+p2,lw=3)
    m=(p1+p2).max(); mid=(p1+p2)[500]; a.set_title(t+f"\ndip {100*max(0,1-mid/m):.0f} %"); a.set_xlabel("x (nm)")
ax[0].set_ylabel("intensity")
plt.tight_layout(); plt.savefig(R+'/resolution/fig-twopoints.png',dpi=90); plt.close()
# ---------------------------------------------------------------- 4. axial PSF and missing cone
def psf_rz(rs,zs,NA,lam,n,M=400):
    kmax=NA/lam; kn=n/lam; rho=(np.arange(M)+0.5)*kmax/M; w=rho*kmax/M
    kz=np.sqrt(kn**2-rho**2)-kn
    J=j0(2*pi*np.outer(rs,rho))
    E=np.exp(2j*pi*np.outer(zs,kz))*w
    U=E@J.T
    I=np.abs(U)**2; return I/I.max()
NA=1.4; n=1.518
xs=np.linspace(-1500,1500,121); zs=np.linspace(-3000,3000,241)
I=psf_rz(np.abs(xs),zs,NA,lam,n)
fig=plt.figure(figsize=(16,5.6))
ax1=fig.add_subplot(1,3,1); ax2=fig.add_subplot(1,3,2); ax3=fig.add_subplot(1,3,3)
ext=[-1.5,1.5,-3,3]
ax1.imshow(np.log10(np.maximum(I,1e-4)),cmap="inferno",extent=ext,origin="lower",aspect="equal"); ax1.set_title("widefield PSF, xz (log)"); ax1.set_xlabel("x (µm)"); ax1.set_ylabel("z (µm)")
ax2.imshow(np.log10(np.maximum(I**2,1e-4)),cmap="inferno",extent=ext,origin="lower",aspect="equal"); ax2.set_title("confocal PSF, xz (log)"); ax2.set_xlabel("x (µm)")
# 3D OTF by FFT of a sampled 3D PSF (widefield), kx-kz slice
N3=128; d=50.0
c=np.arange(N3)-N3//2
XX,YY=np.meshgrid(c*d,c*d)
rr=np.hypot(XX,YY)
ru=np.unique(np.round(rr.ravel(),3))
z3=c*d
Irz=psf_rz(ru,z3,NA,lam,n,M=300)  # (nz, nr)
idx=np.searchsorted(ru,np.round(rr,3))
V=Irz[:,idx]  # nz x N3 x N3
O=np.abs(np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(V))))
O/=O.max()
sl=O[:,:,N3//2]  # kz x ky? axes: (z, y, x) -> take x=0 slice -> (kz, ky)
k=np.fft.fftshift(np.fft.fftfreq(N3,d=d))*1000
ax3.imshow(np.log10(np.maximum(sl,1e-4)),cmap="viridis",extent=[k[0],k[-1],k[0],k[-1]],origin="lower",aspect="equal")
ax3.set_xlim(-7,7); ax3.set_ylim(-7,7); ax3.set_xlabel("lateral frequency (cycles/µm)"); ax3.set_ylabel("axial frequency (cycles/µm)")
ax3.set_title("widefield OTF (log): the missing cone")
ax3.annotate("missing\ncone",xy=(0,2.2),xytext=(2.5,5),color="white",fontsize=14,arrowprops=dict(color="white",arrowstyle="->"))
plt.tight_layout(); plt.savefig(R+'/resolution/fig-axial.png',dpi=90); plt.close()
# ---------------------------------------------------------------- 5. invisible grating
n=256; px=40.0; NA=1.4
f=np.fft.fftfreq(n,d=px); FX,FY=np.meshgrid(f,f); H=otf_airy(np.hypot(FX,FY)/(2*NA/lam))
A=cell[::1,::1]/cell.max()
gx=np.arange(n); per=160.0
G=0.25*np.cos(2*pi*gx*px/per)[None,:]*np.ones((n,1))
B=A+G
iA=np.real(np.fft.ifft2(np.fft.fft2(A)*H)); iB=np.real(np.fft.ifft2(np.fft.fft2(B)*H))
fig,ax=plt.subplots(1,5,figsize=(16,3.9))
crop=slice(64,192)
for a,img,t in zip(ax,[A,B,iA,iB,iB-iA],["object A","object B = A + grating\n(period 160 nm)","image of A","image of B","image B − image A"]):
    if t.startswith("image B −"): a.imshow(img[crop,crop],cmap="RdBu_r",vmin=-0.25,vmax=0.25)
    else: a.imshow(img[crop,crop],cmap="gray",vmin=-0.25 if "B =" in t else 0,vmax=1.25 if "B =" in t else 1)
    a.set_title(t); a.axis("off")
print("max |iB-iA|", np.abs(iB-iA).max())
plt.tight_layout(); plt.savefig(R+'/illposed/fig-invisible.png',dpi=90); plt.close()
# ---------------------------------------------------------------- 6. bridge: truth, observed, naive inverse, with prior knowledge
n=128; img=cell.reshape(128,2,128,2).mean(axis=(1,3)); img=img/img.max()
f=np.fft.fftfreq(n); FX,FY=np.meshgrid(f,f); s=1.5
H=np.exp(-2*pi**2*s**2*(FX**2+FY**2))
g=np.real(np.fft.ifft2(np.fft.fft2(img)*H))+0.001*rng.standard_normal((n,n))
G=np.fft.fft2(g)
inv=np.real(np.fft.ifft2(G/H))
snr=1e4
W=H/(H**2+1/snr); wien=np.real(np.fft.ifft2(G*W))
print("bridge rmse obs", np.sqrt(np.mean((g-img)**2)), "inv", np.sqrt(np.mean((inv-img)**2)), "wiener", np.sqrt(np.mean((wien-img)**2)))
fig,ax=plt.subplots(1,4,figsize=(14,4))
for a,im,t in zip(ax,[img,g,inv,wien],["truth","observed: blur + 0.1 % noise","naive inverse 1/H","with prior knowledge\n(next day)"]):
    a.imshow(im,cmap="gray",vmin=0,vmax=1); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig(R+'/illposed/fig-bridge.png',dpi=90); plt.close()
print("done")
