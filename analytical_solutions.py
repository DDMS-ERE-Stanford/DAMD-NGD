def analyticalx(x0,t,phi): 
    # computes x(t) given deteministic constant reaction rate a 
    phi0 = phi[0]
    phi1 = phi[1]
    a = phi0 
    
    x = x0*np.exp(-a*t)

    return x

def Bayesianseq(X,tm,dm,phicurrent,sdnoise):  # Bayesian update with Gaussian likelihood, not yet normalized
    
    x_pred = np.zeros((1,1)) 
    phi0_pred = np.zeros((1,1)) 
    phi1_pred = np.zeros((1,1)) 
    t_pred = np.zeros((1,1)) 
    
    x_pred[0] = X
    phi0_pred[0] = phicurrent[0]
    phi1_pred[0] = phicurrent[1]
    t_pred[0] = tm

    prior = model0predictPDF(x_pred,phi0_pred,phi1_pred,t_pred)

    likelihood = sp.stats.norm.pdf(dm,loc=X,scale=sdnoise)

    return prior*likelihood 

def normconstBseq(tm,dm,phiprior,sdnoise,Xmin,Xmax):
    normconst = sp.integrate.quad(Bayesianseq,Xmin,Xmax,args=(tm,dm,phiprior,sdnoise,))[0]
    return normconst    


def pu(U,t,phi0,phi1):
    theta2 = np.log(phi1)
    theta1 = phi0/phi1
     
    return np.exp(-(np.exp(theta2)*(t*theta1 + W0) + np.log(U))**2/(2.*np.exp(2.*theta2)*t) - theta2)/(np.sqrt(2.*np.pi)*np.sqrt(t)*np.abs(U))
    
def Fu(U,t,phi0,phi1): 
    theta2 = np.log(phi1)
    theta1 = phi0/phi1
    return 1.-(1. - sp.special.erf((t*theta1 + W0 + np.log(U)/np.exp(theta2))/(np.sqrt(2)*np.sqrt(t))))/2.

def invFu(y,phi,t):
    
    theta2 = np.log(phi[1])
    theta1 = phi[0]/phi[1]
    
    arg = sp.special.erfinv(2.*y-1)*np.sqrt(2*t) - theta1*t - W0
    
    return np.exp(np.exp(theta2)*arg) 

def dFudtheta1(U,t,phi0,phi1): # this is actually dFudphi0 here

    return (np.sqrt(t)/(np.exp(((phi0*t)/phi1 + W0 + np.log(U)/phi1)**2/(2.*t))*phi1*np.sqrt(2*np.pi)))


def dFudtheta2(U,t,phi0,phi1): # this is actually dFudphi1 here

    return ((-((phi0*t)/phi1**2) - np.log(U)/phi1**2)/(np.exp(((phi0*t)/phi1 + W0 + np.log(U)/phi1)**2/(2.*t))*np.sqrt(2*np.pi)*np.sqrt(t)))

def dpudtheta1(U,t,phi0,phi1): # this is actually dpudphi0 here

    return -((phi0*t + phi1*W0 + np.log(U))/(np.exp((phi0*t + phi1*W0 + np.log(U))**2/(2.*phi1**2*t))*phi1**3*np.sqrt(2*np.pi)*np.sqrt(t)*np.abs(U)))
    

def dpudtheta2(U,t,phi0,phi1): # this is actually dpudphi1 here

    return (t*(-phi1**2 + phi0**2*t + phi0*phi1*W0) + (2*phi0*t + phi1*W0)*np.log(U) + np.log(U)**2)/(np.exp((phi0*t + phi1*W0 + np.log(U))**2/(2.*phi1**2*t))*phi1**4*np.sqrt(2*np.pi)*t**1.5*np.abs(U))

             
             
def model0predictCDF(x,phi0,phi1,t):  
    phi1 = np.abs(phi1)
    return Fu(x,t,phi0,phi1)
    
def model0predictPDF(x,phi0,phi1,t):  
    phi1 = np.abs(phi1)
    return pu(x,t,phi0,phi1)

def model0predictderW2(x,phi0,phi1,t):
    
    r0 = pu(x,t,phi0,phi1) # dFdx
    
    r1 = dFudtheta1(x,t,phi0,phi1) # dFdphi0
    
    r2 = dFudtheta2(x,t,phi0,phi1) # dFdphi1
    
    return r0,r1,r2
    
def model0predictderKL(x,phi0,phi1,t):
    
    r0 = pu(x,t,phi0,phi1) # p
    
    r1 = dpudtheta1(x,t,phi0,phi1) # dpdphi0
    
    r2 = dpudtheta2(x,t,phi0,phi1) # dpdphi1
    
    return r0,r1,r2    

def model0predictderL2(x,phi0,phi1,t):
    
    r0 = pu(x,t,phi0,phi1) # p
    
    r1 = dFudtheta1(x,t,phi0,phi1) # dFdphi0
    
    r2 = dFudtheta2(x,t,phi0,phi1) # dFdphi1
    
    return r0,r1,r2  


def pobs(X,tm,dm,phiprior,sdnoise,normconst):
    return np.maximum(Bayesianseq(X,tm,dm,phiprior,sdnoise)/normconst,10.**-6)

def Fobs(X,tm,dm,phiprior,sdnoise,normconst):
    F = sp.integrate.quad(pobs,Xmin,X,args=(tm,dm,phiprior,sdnoise,normconst,))[0]
    return F

def Fobsmonotonic(x,phi,tm,normconst,Xcmin,Xcmax):
    
    if x<Xcmin:
        F_pred = 0.
    elif x>Xcmax:
        F_pred = 1.
    else:
        F_pred = sp.integrate.quad(pobs,Xcmin,x,args=(tm,dm,phi,sdnoise,normconst,))[0]
            
    return F_pred 


def Fobsinv(y,tm,dm,phiprior,sdnoise,normconst):
    
    Finv = inversefunc(Fobs, y_values=y, args = (tm,dm,phiprior,sdnoise,normconst,) )
    
    return Finv

def gradpredNN(x,tm,phicurrent): 
    
    Nxx = np.shape(x)[0]
    
    x_pred = np.zeros((Nxx,1)) 
    phi0_pred = np.zeros((Nxx,1)) 
    phi1_pred = np.zeros((Nxx,1)) 
    t_pred = np.zeros((Nxx,1)) 
    
    x_pred[:,0] = x
    phi0_pred[:,0] = phicurrent[0]
    phi1_pred[:,0] = phicurrent[1]
    t_pred[:,0] = tm
    
    Fx_pred, Fxp0_pred,Fxp1_pred = model0predictderKL(x_pred,phi0_pred,phi1_pred,t_pred)
    Fx_pred[0,0] = 0.
    Fxp0_pred[0,0] = 0.
    Fxp1_pred[0,0] = 0.
    
    return Fx_pred[:,0], Fxp0_pred[:,0],Fxp1_pred[:,0] 
 

def gradFpredNN(x,tm,phicurrent):
    
    Nxx = np.shape(x)[0]
    
    x_pred = np.zeros((Nxx,1)) 
    phi0_pred = np.zeros((Nxx,1)) 
    phi1_pred = np.zeros((Nxx,1)) 
    t_pred = np.zeros((Nxx,1)) 
    
    x_pred[:,0] = x
    phi0_pred[:,0] = phicurrent[0]
    phi1_pred[:,0] = phicurrent[1]
    t_pred[:,0] = tm

    Fx_pred, d0,d1 = model0predictderW2(x_pred,phi0_pred,phi1_pred,t_pred)

    return Fx_pred[:,0],d0[:,0],d1[:,0] 

def gradFpredNN_L2(x,tm,phicurrent):
    
    Nxx = np.shape(x)[0]
    
    x_pred = np.zeros((Nxx,1)) 
    phi0_pred = np.zeros((Nxx,1)) 
    phi1_pred = np.zeros((Nxx,1)) 
    t_pred = np.zeros((Nxx,1)) 
    
    x_pred[:,0] = x
    phi0_pred[:,0] = phicurrent[0]
    phi1_pred[:,0] = phicurrent[1]
    t_pred[:,0] = tm
    
    d00, d0,d1 = model0predictderL2(x_pred,phi0_pred,phi1_pred,t_pred)
 
    return d00[:,0],d0[:,0],d1[:,0] 


def ppredNN(x,phi,tm):
        
        Nxx = np.shape(x)[0]
       
        x_pred = np.zeros((Nxx,1)) 
        phi0_pred = np.zeros((Nxx,1)) 
        phi1_pred = np.zeros((Nxx,1)) 
        t_pred = np.zeros((Nxx,1)) 

        x_pred[:,0] = x
        phi0_pred[:,0] = phi[0]
        phi1_pred[:,0] = phi[1]
        t_pred[:,0] = tm

        F_pred = model0predictPDF(x_pred,phi0_pred,phi1_pred,t_pred)
            
        return F_pred      
    
def FpredNN(x,phi,tm):
    
        Nxx = np.shape(x)[0]
        
        x_pred = np.zeros((Nxx,1)) 
        phi0_pred = np.zeros((Nxx,1)) 
        phi1_pred = np.zeros((Nxx,1)) 
        t_pred = np.zeros((Nxx,1)) 

        x_pred[:,0] = x
        phi0_pred[:,0] = phi[0]
        phi1_pred[:,0] = phi[1]
        t_pred[:,0] = tm

        F_pred = model0predictCDF(x_pred,phi0_pred,phi1_pred,t_pred)

        return F_pred     
    
def FpredNNmonotonic(x,phi,tm,Xcmin,Xcmax):
    
    if x<Xcmin:
        F_pred = 0.
    elif x>Xcmax:
        F_pred = 1.
    else:
        phi = np.abs(phi)

        x0_pred = np.zeros((1,1)) 
        phi0_pred = np.zeros((1,1)) 
        phi1_pred = np.zeros((1,1)) 
        t_pred = np.zeros((1,1)) 

        x0_pred[0,0] = x
        phi0_pred[0,0] = phicurrent[0]
        phi1_pred[0,0] = phicurrent[1]
        t_pred[0,0] = tm

        F_pred = model0predictCDF(x0_pred,phi0_pred,phi1_pred,t_pred)

    return F_pred 

def FpredNNinv(y,phi,tm,Xcmin,Xcmax):
    
    Finv = invFu(y,phi,tm) 
    
    return Finv


def GF(idiv,phi,t): 
    
    phi = np.reshape(phi,Nphi)

    G = np.zeros((Nphi,Nphi))
    
    ppredNNa = np.zeros(NXOBS+1)
    gradpredNNa = np.zeros((Nphi,NXOBS+1))   
        
    if idiv==0:
        ppredNNa, gradpredNNa[0,:], gradpredNNa[1,:] = gradpredNN(xx,t,phi) 
    if idiv ==1:
        ppredNNa, gradpredNNa[0,:], gradpredNNa[1,:] = gradFpredNN(xx,t,phi) 
    
    if idiv==0:
        
        integrand = np.zeros((Nphi,Nphi,NXOBS+1))
        for ix in range(0,NXOBS+1):
            if ppredNNa[ix]>=0.001:
                integrand[0,0,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradpredNNa[0,ix]*gradpredNNa[0,ix])
                integrand[0,1,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradpredNNa[0,ix]*gradpredNNa[1,ix])
                integrand[1,1,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradpredNNa[1,ix]*gradpredNNa[1,ix])
            else:
                integrand[:,:,ix] = 0.
        
        G[0,0] = np.trapz(integrand[0,0,:],x=xx)
        G[0,1] = np.trapz(integrand[0,1,:],x=xx)
        G[1,0] = G[0,1]
        G[1,1] = np.trapz(integrand[1,1,:],x=xx)
        
    elif idiv==1: 

        integrand = np.zeros((Nphi,Nphi,NXOBS+1))
        for ix in range(0,NXOBS+1):
            if ppredNNa[ix]>=0.001:
                integrand[0,0,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradpredNNa[0,ix]*gradpredNNa[0,ix])
                integrand[0,1,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradpredNNa[0,ix]*gradpredNNa[1,ix])
                integrand[1,1,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradpredNNa[1,ix]*gradpredNNa[1,ix])
            else:
                integrand[:,:,ix] = 0.
        G[0,0] = np.trapz(integrand[0,0,:],x=xx)
        G[0,1] = np.trapz(integrand[0,1,:],x=xx)
        G[1,0] = G[0,1]
        G[1,1] = np.trapz(integrand[1,1,:],x=xx)   
        
    return G 


def gradloss2(phi,idiv,tm,dm,phiprior,sdnoise):
    
    phi = np.reshape(phi,Nphi)
    phi = np.abs(phi)
    
    FpredNNa = np.zeros(NXOBS+1)
    FpredNNmon = np.zeros(NXOBS+1)
    
    ppredNNa = np.zeros(NXOBS+1)
    FpredNNinva = np.zeros(NXOBS+1)
    gradpredNNa = np.zeros((Nphi,NXOBS+1))
    gradFpredNNa = np.zeros((Nphi,NXOBS+1))
    
    G = np.zeros((Nphi,Nphi))
    
    FpredNNa = FpredNN(xx,phi,tm) 
    ppredNNa = ppredNN(xx,phi,tm) 
    
    
    if idiv==0:
        pp, gradpredNNa[0,:], gradpredNNa[1,:] = gradpredNN(xx,tm,phi) 
    if idiv ==1 or idiv == 2:
        pp, gradFpredNNa[0,:], gradFpredNNa[1,:] = gradFpredNN(xx,tm,phi) 
 
    
    ic = np.argmin(np.abs(xx-analyticalx(X0,tm,phi))) 
    
    arrL1 = np.array(np.where(ppredNNa[0:ic,0] < 0.))
    arrL2 = np.array(np.where(FpredNNa[0:ic,0] < 0.))
    arrR1 = np.array(np.where(ppredNNa[ic:NXOBS+1,0] < 0.)+ic)
    arrR2 = np.array(np.where(FpredNNa[ic:NXOBS+1,0] > 1.)+ic)
    idL1 = 0
    idL2 = 0
    idR1 = NXOBS
    idR2 = NXOBS
    if arrL1.size > 0:
        idL1 = np.max(np.where(ppredNNa[0:ic,0] < 0.001))
    if arrL2.size > 0:
        idL2 = np.max(np.where(FpredNNa[0:ic,0] < 0.))
    if arrR1.size > 0:
        idR1 = np.min(np.where(ppredNNa[ic:NXOBS+1,0] < 0.)+ic)
    if arrR2.size > 0:
        idR2 = np.min(np.where(FpredNNa[ic:NXOBS+1,0] > 1.)+ic)
    idL = np.max([idL1,idL2,0])
    idR = np.min([idR1,idR2,NXOBS])
    
    FpredNNa[0:idL,0] = 0.
    FpredNNa[idR:NXOBS+1,0] = 1.
    ppredNNa[0:idL,0] = 0.
    ppredNNa[idR:NXOBS+1,0] = 0.

    FpredNNmon = FpredNNa
    
    ppredNNa[0,0] = 0.
    
    
    
    if idiv==1:
        Ny = 1000
        ymin = .000001 
        ymax = .99999 
        yy = np.linspace(ymin,ymax,Ny)
        
        Finv = np.zeros(Ny)
        Fhatinv = np.zeros(Ny)
        gradlossW2 = np.zeros((Nphi+1,Ny))
        
        Fhinvf = sp.interpolate.interp1d(Fobsmon,xx)
        for iy in range(0,Ny):
            Finv[iy] = invFu(yy[iy],phi,tm) 
            Fhatinv[iy] = Fhinvf(yy[iy]) 
            
            gradlossW2[0,iy] = -1./(ppredNN([Finv[iy]],phi,tm)+10.**-6)*gradFpredNN([Finv[iy]],tm,phi)[1] 
            gradlossW2[1,iy] = -1./(ppredNN([Finv[iy]],phi,tm)+10.**-6)*gradFpredNN([Finv[iy]],tm,phi)[2]
            gradlossW2[2,iy] = ppredNN([Finv[iy]],phi,tm) 
    
    gl = np.zeros(Nphi)
    
    if idiv ==0:
        
        integrand = np.zeros((Nphi,NXOBS+1))

        
        for ix in range(0,NXOBS+1):
            integrand[0,ix] = gradpredNNa[0,ix]*(1.+np.log((np.maximum(ppredNNa[ix,0],10.**-6)/(pobsa[ix]+10.**-6))))
            integrand[1,ix] = gradpredNNa[1,ix]*(1.+np.log((np.maximum(ppredNNa[ix,0],10.**-6)/(pobsa[ix]+10.**-6))))

        gl[0] = np.trapz(integrand[0,:],x=xx)
        gl[1] = np.trapz(integrand[1,:],x=xx)


    elif idiv==1:
        integrand = np.zeros((Nphi,Ny))

        for iy in range(0,Ny):
            integrand[0,iy] = gradlossW2[0,iy]*(Finv[iy]-Fhatinv[iy]) 
            integrand[1,iy] = gradlossW2[1,iy]*(Finv[iy]-Fhatinv[iy]) 
                
        gl[0] = np.trapz(integrand[0,1:Ny-2],x=yy[1:Ny-2]) 
        gl[1] = np.trapz(integrand[1,1:Ny-2],x=yy[1:Ny-2]) 
    
    elif idiv == 2:
        integrand = np.zeros((Nphi,NXOBS+1))
        for ix in range(0,NXOBS+1):
            integrand[0,ix] = 2*(FpredNNa[ix,0]-Fobsmon[ix])*gradFpredNNa[0,ix]  
            integrand[1,ix] = 2*(FpredNNa[ix,0]-Fobsmon[ix])*gradFpredNNa[1,ix] 
        integrand[:,0] = 0
        gl[0] = np.trapz(integrand[0,:],x=xx)
        gl[1] = np.trapz(integrand[1,:],x=xx) 
        
    gl = np.reshape(gl,Nphi)
    

    return gl


def loss2(phi,idiv,tm,dm,phiprior,sdnoise):

    phi = np.reshape(phi,Nphi)
    
    phi = np.abs(phi) 
    
    FpredNNa = np.zeros(NXOBS+1)
    FpredNNmon = np.zeros(NXOBS+1)
    
    ppredNNa = np.zeros(NXOBS+1)
    FpredNNinva = np.zeros(NXOBS+1)

    FpredNNa = FpredNN(xx,phi,tm) 
    ppredNNa = ppredNN(xx,phi,tm) 
    ic = np.argmin(np.abs(xx-analyticalx(X0,tm,phi))) 
    
    arrL1 = np.array(np.where(ppredNNa[0:ic,0] < 0.))
    arrL2 = np.array(np.where(FpredNNa[0:ic,0] < 0.))
    arrR1 = np.array(np.where(ppredNNa[ic:NXOBS+1,0] < 0.)+ic)
    arrR2 = np.array(np.where(FpredNNa[ic:NXOBS+1,0] > 1.)+ic)
    idL1 = 0
    idL2 = 0
    idR1 = NXOBS
    idR2 = NXOBS
    if arrL1.size > 0:
        idL1 = np.max(np.where(ppredNNa[0:ic,0] < 0.001))
    if arrL2.size > 0:
        idL2 = np.max(np.where(FpredNNa[0:ic,0] < 0.))
    if arrR1.size > 0:
        idR1 = np.min(np.where(ppredNNa[ic:NXOBS+1,0] < 0.)+ic)
    if arrR2.size > 0:
        idR2 = np.min(np.where(FpredNNa[ic:NXOBS+1,0] > 1.)+ic)
    idL = np.max([idL1,idL2,0])
    idR = np.min([idR1,idR2,NXOBS])
    
    FpredNNa[0:idL,0] = 0.
    FpredNNa[idR:NXOBS+1,0] = 1.
    ppredNNa[0:idL,0] = 0.
    ppredNNa[idR:NXOBS+1,0] = 0.

    FpredNNmon = FpredNNa
    
    ppredNNa[0,0] = 0.
    
    Ny = 1000
    
    if idiv==1:
                
        ymin = .000001 
        ymax = .99999 
        yy = np.linspace(ymin,ymax,Ny)
        
        Finv = np.zeros(Ny)
        Fhatinv = np.zeros(Ny)
        
           
        Fhinvf = sp.interpolate.interp1d(Fobsmon,xx,fill_value=(0.,1.))
        for iy in range(0,Ny):
            Finv[iy] = invFu(yy[iy],phi,tm) 
            Fhatinv[iy] = Fhinvf(yy[iy]) 
    
    if idiv==0:
            
        integrand = np.maximum(ppredNNa[:,0],np.zeros(NXOBS+1))*np.log(np.divide(np.maximum(ppredNNa[:,0],np.ones(NXOBS+1)*10.**-6),(pobsa+10.**-5)))
        loss = np.trapz(integrand,x=xx)
        
    elif idiv == 1:

        dw2 = np.sqrt(np.trapz((Finv-Fhatinv)**2,x=yy))
        loss = 1./2*dw2**2
    
    elif idiv == 2:
        integrand = (FpredNNa[:,0]-Fobsmon[:])**2 
        loss = np.trapz(integrand,x=xx)
 
    return loss



def gradloss3(phi,idiv,tm,dm,phiprior,sdnoise):
    
    
    phi = np.reshape(phi,Nphi)
    phi = np.abs(phi) 
    
    FpredNNa = np.zeros(NXOBS+1)
    FpredNNmon = np.zeros(NXOBS+1)
    
    ppredNNa = np.zeros(NXOBS+1)
    FpredNNinva = np.zeros(NXOBS+1)
    gradpredNNa = np.zeros((Nphi,NXOBS+1))
    gradFpredNNa = np.zeros((Nphi,NXOBS+1))
    
    G = np.zeros((Nphi,Nphi))
    
    FpredNNa = FpredNN(xx,phi,tm) 
    ppredNNa = ppredNN(xx,phi,tm) 
    
    
    if idiv==0:
        pp, gradpredNNa[0,:], gradpredNNa[1,:] = gradpredNN(xx,tm,phi) 
    if idiv ==1:
        pp, gradFpredNNa[0,:], gradFpredNNa[1,:] = gradFpredNN(xx,tm,phi) 

    ic = np.argmin(np.abs(xx-analyticalx(X0,tm,phi))) 
    
    arrL1 = np.array(np.where(ppredNNa[0:ic,0] < 0.))
    arrL2 = np.array(np.where(FpredNNa[0:ic,0] < 0.))
    arrR1 = np.array(np.where(ppredNNa[ic:NXOBS+1,0] < 0.)+ic)
    arrR2 = np.array(np.where(FpredNNa[ic:NXOBS+1,0] > 1.)+ic)
    idL1 = 0
    idL2 = 0
    idR1 = NXOBS
    idR2 = NXOBS
    if arrL1.size > 0:
        idL1 = np.max(np.where(ppredNNa[0:ic,0] < 0.001))
    if arrL2.size > 0:
        idL2 = np.max(np.where(FpredNNa[0:ic,0] < 0.))
    if arrR1.size > 0:
        idR1 = np.min(np.where(ppredNNa[ic:NXOBS+1,0] < 0.)+ic)
    if arrR2.size > 0:
        idR2 = np.min(np.where(FpredNNa[ic:NXOBS+1,0] > 1.)+ic)
    idL = np.max([idL1,idL2,0])
    idR = np.min([idR1,idR2,NXOBS])
    
    
    FpredNNa[0:idL,0] = 0.
    FpredNNa[idR:NXOBS+1,0] = 1.
    ppredNNa[0:idL,0] = 0.
    ppredNNa[idR:NXOBS+1,0] = 0.

    FpredNNmon = FpredNNa
    
    ppredNNa[0,0] = 0.
    
    
    
    if idiv==1:
        Ny = 1000
        ymin = .000001 
        ymax = .99999 
        yy = np.linspace(ymin,ymax,Ny)
        
        Finv = np.zeros(Ny)
        Fhatinv = np.zeros(Ny)
        gradlossW2 = np.zeros((Nphi+1,Ny))
        
        Fhinvf = sp.interpolate.interp1d(Fobsmon,xx)
        for iy in range(0,Ny):
            Finv[iy] = invFu(yy[iy],phi,tm) 
            Fhatinv[iy] = Fhinvf(yy[iy]) 
            
            gradlossW2[0,iy] = -1./(ppredNN([Finv[iy]],phi,tm)+10.**-6)*gradFpredNN([Finv[iy]],tm,phi)[1] 
            gradlossW2[1,iy] = -1./(ppredNN([Finv[iy]],phi,tm)+10.**-6)*gradFpredNN([Finv[iy]],tm,phi)[2]
            gradlossW2[2,iy] = ppredNN([Finv[iy]],phi,tm) 
    
    gl = np.zeros(Nphi)
    
    if idiv ==0:
        
        integrand = np.zeros((Nphi,NXOBS+1))
        integrandG = np.zeros((Nphi,Nphi,NXOBS+1))
        
        for ix in range(0,NXOBS+1):
            integrand[0,ix] = gradpredNNa[0,ix]*(1.+np.log((np.maximum(ppredNNa[ix,0],10.**-6)/(pobsa[ix]+10.**-6))))
            integrand[1,ix] = gradpredNNa[1,ix]*(1.+np.log((np.maximum(ppredNNa[ix,0],10.**-6)/(pobsa[ix]+10.**-6))))

            if ppredNNa[ix]>=0.001:
                integrandG[0,0,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradpredNNa[0,ix]*gradpredNNa[0,ix])
                integrandG[0,1,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradpredNNa[0,ix]*gradpredNNa[1,ix])
                integrandG[1,1,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradpredNNa[1,ix]*gradpredNNa[1,ix])
            else:
                integrandG[:,:,ix] = 0.

        gl[0] = np.trapz(integrand[0,:],x=xx)
        gl[1] = np.trapz(integrand[1,:],x=xx)

    elif idiv==1:
        integrand = np.zeros((Nphi,Ny))
        integrandG = np.zeros((Nphi,Nphi,NXOBS+1))

        for iy in range(0,Ny):
            integrand[0,iy] = gradlossW2[0,iy]*(Finv[iy]-Fhatinv[iy]) 
            integrand[1,iy] = gradlossW2[1,iy]*(Finv[iy]-Fhatinv[iy]) 
        
        for ix in range(0,NXOBS+1):
             if ppredNNa[ix]>=0.001:
                integrandG[0,0,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradFpredNNa[0,ix]*gradFpredNNa[0,ix])
                integrandG[0,1,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradFpredNNa[0,ix]*gradFpredNNa[1,ix])
                integrandG[1,1,ix] = 1./(np.maximum(ppredNNa[ix],10.**-6))*(gradFpredNNa[1,ix]*gradFpredNNa[1,ix])
             else:
                integrandG[:,:,ix] = 0.


        gl[0] = np.trapz(integrand[0,1:Ny-2],x=yy[1:Ny-2]) 
        gl[1] = np.trapz(integrand[1,1:Ny-2],x=yy[1:Ny-2]) 
        
    
    
    G[0,0] = np.trapz(integrandG[0,0,:],x=xx)
    G[0,1] = np.trapz(integrandG[0,1,:],x=xx)
    G[1,0] = G[0,1]
    G[1,1] = np.trapz(integrandG[1,1,:],x=xx) 
    
    determinant = np.linalg.det(G)
    
    if determinant<10.**-8:
        Ginv = np.linalg.inv(np.identity(Nphi)) 
    else:
        Ginv = np.linalg.inv(G)
        
    gl = np.reshape(gl,Nphi)
    
    gl = np.matmul(Ginv,gl)
    
    return gl         
             
             
             
