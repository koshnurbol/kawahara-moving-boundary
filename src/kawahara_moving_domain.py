"""
Kawahara (moving boundary): восстановленное ФИЗИЧЕСКОЕ решение v_m(y,t) на
подвижной области, безфорсинга (f=0), исправленной схемой
(квартичные B-сплайны, краевой базис через нуль-пространство, Кранк--Николсон
с коэффициентами в середине шага).

Строит две 3D-поверхности:
  Пример 1 — монотонно расширяющаяся область, u0 = x^3 (1-x)^3;
  Пример 2 — осциллирующая область,        u0 = A * sin^3(pi x)  (A=0.01).

Запуск:  python3 kawahara_moving_domain.py
Результат:  ex1_moving.png, ex2_moving.png
"""
import numpy as np, math
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# ---------- эталонный квартичный B-сплайн на [0,5] и производные ----------
def B4(z):
    z=np.asarray(z,float)
    c=[(z>=0)&(z<1),(z>=1)&(z<2),(z>=2)&(z<3),(z>=3)&(z<4),(z>=4)&(z<=5)]
    f=[z**4/24,(z**4-5*(z-1)**4)/24,(z**4-5*(z-1)**4+10*(z-2)**4)/24,
       ((5-z)**4-5*(4-z)**4)/24,(5-z)**4/24]; return np.select(c,f,0.0)
def dB4(z):
    z=np.asarray(z,float)
    c=[(z>=0)&(z<1),(z>=1)&(z<2),(z>=2)&(z<3),(z>=3)&(z<4),(z>=4)&(z<=5)]
    f=[4*z**3/24,(4*z**3-20*(z-1)**3)/24,(4*z**3-20*(z-1)**3+40*(z-2)**3)/24,
       (-4*(5-z)**3+20*(4-z)**3)/24,-4*(5-z)**3/24]; return np.select(c,f,0.0)
def ddB4(z):
    z=np.asarray(z,float)
    c=[(z>=0)&(z<1),(z>=1)&(z<2),(z>=2)&(z<3),(z>=3)&(z<4),(z>=4)&(z<=5)]
    f=[12*z**2/24,(12*z**2-60*(z-1)**2)/24,(12*z**2-60*(z-1)**2+120*(z-2)**2)/24,
       (12*(5-z)**2-60*(4-z)**2)/24,12*(5-z)**2/24]; return np.select(c,f,0.0)
def dddB4(z):
    z=np.asarray(z,float)
    c=[(z>=0)&(z<1),(z>=1)&(z<2),(z>=2)&(z<3),(z>=3)&(z<4),(z>=4)&(z<=5)]
    f=[24*z/24,(24*z-120*(z-1))/24,(24*z-120*(z-1)+240*(z-2))/24,
       (-24*(5-z)+120*(4-z))/24,-24*(5-z)/24]; return np.select(c,f,0.0)

# ---------- безфорсинговый решатель (нуль-пространственный краевой базис) ----------
def solve_unforced(M,T,alpha,gamma,alphap,gammap,u0f,Tfinal=1.0,nx_out=161):
    h=1.0/M; P=M+4
    val=lambda k,x,d=0:[B4,dB4,ddB4,dddB4][d]((np.asarray(x,float)-(k-4)*h)/h)/h**d
    ng=14; gx,gw=np.polynomial.legendre.leggauss(ng); XN=[];WN=[]
    for l in range(M):
        a,b=l*h,(l+1)*h; XN.append(0.5*(b-a)*gx+0.5*(a+b)); WN.append(0.5*(b-a)*gw)
    XN=np.concatenate(XN); W=np.concatenate(WN)
    V0=np.vstack([val(k,XN,0) for k in range(P)]);V1=np.vstack([val(k,XN,1) for k in range(P)])
    V2=np.vstack([val(k,XN,2) for k in range(P)]);V3=np.vstack([val(k,XN,3) for k in range(P)])
    A=(V0*W)@V0.T;S3=(V1*W)@V2.T;S5=(V2*W)@V3.T;Bc1=(V0*W)@V1.T;Bc2=(V0*W*XN)@V1.T
    Ct=np.einsum('an,bn,cn->abc',V0*W,V0,V1,optimize=True)
    Gm=np.array([[val(k,0.,0),val(k,0.,1),val(k,0.,2),val(k,1.,0),val(k,1.,1)] for k in range(P)]).T
    Z=sla.null_space(Gm); r=Z.shape[1]
    Ar,S3r,S5r,Bc1r,Bc2r=(Z.T@Xm@Z for Xm in (A,S3,S5,Bc1,Bc2))
    dt=Tfinal/T; tt=np.arange(T+1)*dt
    c=np.zeros((T+1,r)); c[0]=np.linalg.solve(Ar,Z.T@(V0@(W*u0f(XN))))
    for k in range(1,T+1):
        tm=0.5*(tt[k-1]+tt[k]); cf=Z@c[k-1]
        Cr=Z.T@np.tensordot(cf,Ct,axes=([0],[1]))@Z
        Bconv=(alphap(tm)/gamma(tm))*Bc1r+(gammap(tm)/gamma(tm))*Bc2r
        b1,b2,b3=1/gamma(tm),1/gamma(tm)**3,1/gamma(tm)**5
        Mr=-Bconv-b2*S3r+b3*S5r+b1*Cr
        c[k]=np.linalg.solve(Ar+dt/2*Mr,(Ar-dt/2*Mr)@c[k-1])
    xf=np.linspace(0,1,nx_out)
    BZ=np.vstack([val(k,xf) for k in range(P)]).T@Z
    U=np.array([BZ@c[k] for k in range(T+1)])           # (nt,nx) = u_m(x,t)
    return tt,xf,U

# ---------- аккуратная 3D-поверхность в физических координатах ----------
def surface(tt,xf,U,alpha,gamma,fname,title,stride_t=2):
    plt.rcParams.update({"font.size":11,"mathtext.fontset":"cm","axes.linewidth":0.6})
    ts=tt[::stride_t]; Us=U[::stride_t]
    Tg=np.repeat(ts[:,None],len(xf),axis=1)
    Yg=np.array([alpha(ts[k])+xf*gamma(ts[k]) for k in range(len(ts))])
    ls=LightSource(azdeg=315,altdeg=45)
    cmap=plt.cm.viridis
    norm=plt.Normalize(Us.min(),Us.max())
    rgb=ls.shade(Us,cmap=cmap,norm=norm,vert_exag=0.12,blend_mode="soft")
    fig=plt.figure(figsize=(7.6,5.6),dpi=220)
    ax=fig.add_subplot(111,projection="3d")
    surf=ax.plot_surface(Yg,Tg,Us,facecolors=rgb,rstride=1,cstride=1,
                         linewidth=0,antialiased=True,shade=False,rasterized=True)
    ax.set_xlabel(r"$y$",labelpad=8); ax.set_ylabel(r"$t$",labelpad=8)
    ax.set_zlabel(r"$v_m(y,t)$",labelpad=10)
    ax.view_init(elev=26,azim=-58)
    ax.xaxis.pane.set_alpha(0.04); ax.yaxis.pane.set_alpha(0.04); ax.zaxis.pane.set_alpha(0.04)
    ax.grid(True,alpha=0.25)
    m=plt.cm.ScalarMappable(cmap=cmap,norm=norm); m.set_array(Us)
    cb=fig.colorbar(m,ax=ax,shrink=0.6,pad=0.11,aspect=16)
    cb.ax.tick_params(labelsize=9)
    ax.set_title(title,fontsize=12,pad=4)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    plt.savefig(fname,dpi=220,bbox_inches="tight"); plt.close()

if __name__=="__main__":
    # ----- Пример 1: монотонное расширение -----
    a1 =lambda t:-t/(t+1);        g1 =lambda t:(3*t+1)/(t+1)
    a1p=lambda t:-1/(t+1)**2;     g1p=lambda t:2/(t+1)**2
    u01=lambda x:x**3*(1-x)**3
    tt,xf,U1=solve_unforced(48,600,a1,g1,a1p,g1p,u01)
    surface(tt,xf,U1,a1,g1,"ex1_moving.png",
            "Example 1: monotonically expanding domain")
    print(f"Ex1 saved. v_m range [{U1.min():.3e}, {U1.max():.3e}]")

    # ----- Пример 2: осциллирующая область (малая амплитуда, квазилинейный режим) -----
    A=0.01
    a2 =lambda t:0.5*np.cos(2*np.pi*t)-0.5;  g2=lambda t:2-np.cos(2*np.pi*t)
    a2p=lambda t:-np.pi*np.sin(2*np.pi*t);   g2p=lambda t:2*np.pi*np.sin(2*np.pi*t)
    u02=lambda x:A*np.sin(np.pi*x)**3
    tt,xf,U2=solve_unforced(48,600,a2,g2,a2p,g2p,u02)
    surface(tt,xf,U2,a2,g2,"ex2_moving.png",
            r"Example 2: oscillating domain")
    print(f"Ex2 saved. v_m range [{U2.min():.3e}, {U2.max():.3e}]")
