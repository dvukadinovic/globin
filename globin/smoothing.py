import astropy
from astropy.io import fits
from scipy import ndimage, misc 
import numpy as np

# To read fits
def read_fits(path):
    f = fits.open(path)
    imgs = np.array(f[0].data)
    hdr = f[0].header
    f.close()

    return imgs, hdr

# to save the fits
def write_data(output_filename, cube,hdr):
    #Write output parameters
    hdu = fits.PrimaryHDU(cube,hdr)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(output_filename, overwrite=True)

def sqr(x):
    return x*x

def mygsmooth(a, num, sd):
    d=int(num/2)
    dim=a.shape
    ny=dim[0]
    nx=dim[1]

    # x, y = np.meshgrid(np.arange(num))

    rv=np.zeros((ny,nx))
    f =np.zeros((ny,nx))
    for x in range(num):
        for y in range(num):
            r=((x-d)**2+(y-d)**2)/(2.0*sd)**2
            f[y,x]=np.exp(-r)
    
    for x in range(nx):
        xl=np.max([x-d,0])
        xh=np.min([x+d,nx-1])
        for y in range(ny):
            yl=np.max([y-d,0])
            yh=np.min([y+d,ny-1])
            nn=np.sum(f[d+(yl-y):d+(yh-y),d+(xl-x):d+(xh-x)])
            rv[y,x]=np.sum(f[d+(yl-y):d+(yh-y),d+(xl-x):d+(xh-x)]*a[yl:yh,xl:xh])/nn
            
    return rv

def mysmooth(a,num):
    d=int(num/2)
    dim=a.shape
    ny=dim[0]
    nx=dim[1]
    rv=np.zeros((ny,nx))
    sd=np.zeros((ny,nx))
    for x in range(nx):
        xl=np.max([x-d,0])
        xh=np.min([x+d,nx-1])
        mm=xh-xl+1
        for y in range(ny):
            yl=np.max([y-d,0])
            yh=np.min([y+d,ny-1])
            nn=mm*(yh-yl+1)
            rv[y,x]=np.sum(a[yl:yh,xl:xh])/nn
            sd[y,x]=np.sum(sqr(a[yl:yh,xl:xh]-rv[y,x]))/nn
    return rv,sd
    
def azismooth(tmp,num):
    eps=0.01
    ma=360.0
    a1=((tmp+ma) % 180.0)             #   0 .. 180
    a2=((tmp+ma+90.0) % 180.0)-90.0   # -90 ..  90
  
    sa1,d1=mysmooth(a1,num)
    sa2,d2=mysmooth(a2,num)
  
    sa2b=((sa2+180.0) % 180.0)           #   0 .. 180

    
    idx1=np.where((np.abs(sa1-sa2b) >  eps) & (d2 > d1))  # problem points
    idx2=np.where((np.abs(sa1-sa2b) >  eps) & (d1 >= d2)) # problem points

    az=(sa1+sa2b)/2.0
    
    nidx1=len(idx1[0])
    nidx2=len(idx2[0])
    

    if nidx1 > 0: 
        for x in range(nidx1):
            az[idx1[0][x],idx1[1][x]]=sa1[idx1[0][x],idx1[1][x]]
    if nidx2 > 0:         
        for x in range(nidx2):
            az[idx2[0][x],idx2[1][x]]=sa2[idx2[0][x],idx2[1][x]]

    return ((az+90.0) % 180.0) - 90.0
    
def tempinit(a, index, thresh, gtlt):
    """
    Preparing the cube. Filter out the pixels with given 'treshold' and up/below it
    based on 'gtlt' keyword.
    """
    b  = np.copy(a)      
    dim = a.shape
    x = np.arange((dim[1]))/np.float(dim[1]-1)
    y = np.arange((dim[2]))/np.float(dim[2]-1)
    ax = np.zeros((dim[1],dim[2]))

    # xx = np.zeros((dim[1],dim[2]))
    # for i in range(dim[2]): 
    #     xx[:,i]=x
    
    # yy = np.zeros((dim[1],dim[2]))
    # for i in range(dim[2]): 
    #     yy[:,i]=y
    
    if gtlt>0:
        idx = np.where(a[index,::] > thresh)
    else:
        idx = np.where(a[index,::] < thresh)

    nidx=len(idx[0])
    if nidx >0:
        for x in range(nidx):
            ax[idx[0][x],idx[1][x]] = 1.0
    
    for i in range(3):
        aaa = np.copy(a[i,::])
        for ix in range(dim[2]):
            for iy in range(dim[1]):
                ol = np.max([dim[1], dim[2]])
                for offs in range(ol):
                    if(ix-offs >      0): 
                        xl=ix-offs
                    else:
                        xl=0 
                    if(ix+offs < dim[2]): 
                        xh=ix+offs 
                    else: 
                        xh=dim[2]-1
                    if(iy-offs >      0): 
                        yl=iy-offs 
                    else: 
                        yl=0 
                    if(iy+offs < dim[1]): 
                        yh=iy+offs 
                    else:
                        yh=dim[1]-1
                    if (np.sum(ax[yl:yh,xl:xh]) > 0):
                        b[i,iy,ix] = np.sum(ax[yl:yh,xl:xh]*aaa[yl:yh,xl:xh])/np.sum(ax[yl:yh,xl:xh])
                        break 
    return b

def smooth_parameters(atmos, num=5, std=2.5):
    for parameter in atmos.nodes:
        for idn in range(len(atmos.nodes)):
            if parameter=="chi":
                aux = azismooth(atmos.values[parameter][idn], num)
            else:
                aux = mygsmooth(atmos.values[parameter][idn], num, std)

            atmos.values[parameter][idn] = ndimage.median_filter(aux, size=4)

    return atmos

def mkinput_init(inname,outname):

    a, hdr = read_fits(inname)
    bla = tempinit(a, 5, 3500.0, 1)
    num = 5
    sd = 2.50

    lab  = ['TEMPE','BFIEL','GAMMA','AZIMU','VELOS','VMICI']
    model= [3,      3,    3,      3,      3       ,1]
    for zz in range(len(lab)):
        t=hdr[lab[zz]]
        for i in range(t-1,t+model[zz]-1):
            if zz == 0:
                print(lab[zz],i)
                bla[i,:,:]=mygsmooth(bla[i,:,:],num,sd)     # TEMPE
            if zz == 1:
                print(lab[zz],i)
                bla[i,:,:]=mygsmooth(bla[i,:,:],num,sd)     # BFIEL
            if zz == 2:

                print(lab[zz],i)
                bla[i,:,:]=mygsmooth(bla[i,:,:],num,sd)     # GAMMA
            if zz == 3:
                print(lab[zz],i)
                bla[i,:,:]=azismooth(bla[i,:,:],num)       # AZIMU
            if zz == 4:
                print(lab[zz],i)
                bla[i,:,:]=mygsmooth(bla[i,:,:],num,sd)     # VELOS
            if zz == 5:
                print(lab[zz],i)
                bla[i,:,:]=mygsmooth(bla[i,:,:],num,sd)     # VMICI
    
    #Save the smooth cube 
    for i in range(0,bla.shape[0]):
         print(i,bla.shape[0])
         bla[i,::]=ndimage.median_filter(bla[i,::], size=4)
         
      # to find check if non-realistic vels appear in the umbra 
    zz = 0 #TEMPE
    i=hdr[lab[0]]-1
    iv=hdr[lab[4]]-1
    idx=np.where(bla[i,:,:] < 3900)
    nidx = len(idx[0]) 
    
    print('UMBRA',lab[0],i,nidx)
    if nidx >0:
        
        
        dbla=np.copy(bla)
        for ii in range(0,model[4]):            
            vidx = np.where((bla[i,:,:] < 3900))
            nvidx= len(vidx[0])
            print(lab[4],i,iv+ii,nvidx)
            if nvidx > 0:
               for jj in range(0,model[4]):
                   print(jj,ii,iv+jj,dbla.shape)
                   for x in range(nvidx):
                      #print(iv+jj,vidx[0][x],vidx[1][x],dbla[iv+jj,vidx[0][x],vidx[1][x]])
                      dbla[iv+jj,vidx[0][x],vidx[1][x]]*=0.1
                      #print('--->',dbla[iv+jj,vidx[0][x],vidx[1][x]])
                      if jj==0:
                        dbla[hdr[lab[5]]-1,vidx[0][x],vidx[1][x]]=0.1  
                   
                   if jj==model[4]-1:
                      print('VMICI',hdr[lab[5]]-1,nvidx)
                
                
        itmp = ndimage.median_filter(dbla[i,::], size=7)
        itmp = ndimage.median_filter(dbla[i,::], size=11)
        for i in range(0,bla.shape[0]):
            for x in range(nidx):
                bla[i,idx[0][x],idx[1][x]]= dbla[i,idx[0][x],idx[1][x]]
    
         
    print('Saving cube...')
    write_data(outname,bla,hdr)
    
    return bla

import globin
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import time

# atmos = globin.Atmosphere("./globin_inversion/hinode/runs/m1_x70-109_y270-319/inverted_atmos.fits")

# out = mkinput_init('inverted_atmos.fits','input_atmos.fits')

inp,_ = read_fits("inverted_atmos.fits")
out,_ = read_fits("input_atmos.fits")

_, nx, ny = inp.shape

atmos = globin.Atmosphere()
atmos.nodes["temp"] = np.array([-2.0, -0.8, 0])
atmos.values["temp"] = inp[14:17]

atmos.nodes["chi"] = np.array([-2.0, -0.8, 0])
atmos.values["chi"] = inp[14:17]

# start = time.time()
# atmos = smooth_parameters(atmos)
# print(time.time() - start)

aux = gaussian_filter(inp[5], 1.25, truncate=2.0)
aux = ndimage.median_filter(aux, size=4)

plt.figure()
plt.imshow(out[5].T, origin="lower")
plt.colorbar()

plt.figure()
plt.imshow(aux.T, origin="lower")
plt.colorbar()

# plt.figure()
# plt.imshow(atmos.values["chi"][0].T, origin="lower")
# plt.colorbar()

plt.show()

# temp = atmos.values["mag"].T

# # bla = tempinit(temp, 2, 3500.0, 1)
# # bla[i,:,:]=mygsmooth(bla[i,:,:],num,sd)     # VELOS
# bla = mygsmooth(temp[0], 5, 2.5)

# # plt.figure()
# # plt.imshow(temp[0])

# plt.figure()
# plt.imshow(bla)
# plt.colorbar()

# plt.figure()
# plt.imshow(gaussian_filter(temp[0], 1))
# plt.colorbar()

# plt.show()