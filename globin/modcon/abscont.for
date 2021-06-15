*******************************************************************************
*                               CONTABS                                       *
*******************************************************************************

C   These routines calculate the electron pressure (in subrout. PEMAKE), the 
C   ionisation equilibrium (in subrout.  JON etc.) and the continuum 
C   absorption coefficient (in subrout. ABSKO etc.) upon input of the 
C   temperature, gas pressure, a wavelength set and data for the different ions
C   and their absorption mechanisms.
C   The calling program has first to call INITAB (just once) and then PEMAKE
C   and ABSKO for each (T,P) point. ABSKO calls all the rest of the necessary
C   routines. As an illustration a little test program is given below. The
C   input variables are described there.

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C      common /coutr/nto,ntpo(10)
C      data pe/1./,lread,lwrite,lscr1,lscr2,iouts/9,10,12,13,0/
Cc
Cc initiate absko code, set flag for diagnostic io
Cc
C      call initab(lread,lwrite,lscr1,lscr2,iouts)
C      nto=1
C      ntpo(1)=1
Cc
Cc test loop
Cc
C      do 100 i=1,4
C        t=4000.+i*1000.
C        do 100 j=1,4
C          pg=10.**(j+2)
C          call pemake(t,pe,pg,pe)
Cccccccc        pemake(t,pe,pg,pex)
C          call absko(   2, 1,t,pe,    1,1,RHO,absc,scatc)
Cccccccc        absko(newt,nt,t,pe,iseta,j,RHO,absc,scatc)
C          write(lwrite,101)t,pg,pe,absc,scatc
C101       format('0t,pg,pe,absc,scatc=',5e12.5)
C100   continue
Cc
C      end

C   Variables for INITAB:
C   LREAD and LWRITE are the input and output units. LSCR1 and LSCR2 are the
C   units of the two scratch files used internally by the routines; they can
C   be deleted at the end of a run. IOUTS decides on the amount of diagnostics
C   to be done, in particular it controls the amount of output.

C   NTO: Tells the routines the number of T-PE points per call at which they
C   should make a print out. NTPO(I): contains the numbers of the points at
C   which the print out is wanted. E.g. NTO=1 and NTPO(1)=1 means that only
C   one printout per call is required and that this should be of the first
C   T-PE point of the call. NTO=0 results in no printout being made.

C   Variables for PEMAKE:
C   T: temperature, PE: initial estimate of the electron pressure.
C   PG: gas pressure, PEX: Output; the electron pressure value found to
C   be consistent with T and PG.

C   Variables for ABSKO:
C   NEWT: Tells the routine if new or old values of T and PE are to be used.
C   NT: Contains the dimensions of T and PE. T: temperature, PE: electron
C   pressure, ISETA: Wavelength set number. J: Number of the wavelenght point
C   in the wavelength set ISETA for which the absorption coefficient is to be
C   calculated, RHO: Gas density, ABSC: Absorption coefficient per gram solar 
C   matter, SCATC: Scattering Coefficient per gram solar matter.
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%% END OF TEST CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine initab(lread,lwrite,lscr1,lscr2,iouts)
      parameter (mkomp=30)
c
c 'initab' initiates the absko-code. parameters:
c
c   lread : input unit for the absko tables
c   lwrite: output unit for diagnostics
c   lscr1 : first scratch unit
c   lscr2 : second scratch unit
c   iouts : switch for amount of diagnostics (0=none, 1=some)
c
      common /ca1/duma(mkomp,12),nextt,nutzt
      common /cfil/ireset(10),islask,ireat
      common /cxlset/nset,nl(10),xl(20,10)
      common /cros/wros(20)
      common /utput/iread,iwrit
      common /coutr/nto,ntpo(10)
      common /cvaagl/nlb,xla(500),w(500)
c
c logical units
      iread=lread
      iwrit=lwrite
      call injon(iouts)
      islask=lscr1
      ireat=lread
      isave=iwrit
      do 1 i=1,10
1     ireset(i)=lscr2
c
c zeroset weights
      do 4 i=1,500
4     w(i)=0.
      do 5 i=1,20
5     wros(i)=0.
c
c wavelength sets
c rosseland
      call vaagl(nlbro,xl,wros)
      nl(1)=nlbro
c wavelengths for transport equation
      call vaagl(nlb,xla,w)
      lmax=15
      ifirst=2
      ilast=10
      call setdis(nlb,xla,lmax,ifirst,ilast)
c 5000. $ngstr@m standard
      nl(1)=nl(1)+1
      nl1=nl(1)
      xl(nl1,1)=5000.
      if(iouts.lt.0) iwrit=4
      call inabs(iouts)
      if(iouts.ge.-1.) iwrit=isave
      nl(1)=nl(1)-1
c
c control parameters
      nto=0
      nextt=0
      nutzt=0
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine pemake(t,pe,pg,pex)
c
c 'pemake' calculates, from the temperature t and an initial
c estimate pe, the electron pressure pex that is consistent with
c the gas pressure pg. a modified regula falsi procedure
c is used on the log-log pg-pe relation.
c 76.03.08  *nord*
c
      data it,n,eps/0,20,1.e-3/
c
c start
c subroutine jon(t,pe,iepro,pg,ro,e,ioutr)
      a=alog(pe)
      pex=pe
      call jon(t,pe,1,fa,ro,e,0)
c******print 40,t,pg,pe,fa
40    format(' t,pg,pe,pgp=',4e10.4)
      it=it+1
      fa=alog(fa/pg)
      if(abs(fa).lt.eps) goto 101
      b=a-0.69*fa
      pex=exp(b)
c one pemake iteration, cf. pemake
      call jon(t,pex,1,fb,ro,e,0)
c******print 40,t,pg,pe,fb
      it=it+1
      fb=alog(fb/pg)
      if(abs(fb).lt.eps) goto 101
      x=b
c
c loop
      do 100 i=1,n
      xold=x
c
c interpolate to find new x
      x=a-(b-a)/(fb-fa)*fa
      pex=exp(x)
C      print*, 'new pex: ', pex
      if(abs(x-xold).lt.eps) goto 101
      call jon(t,pex,1,fx,ro,e,0)
C      print 40,t,pg,pex,fx
      it=it+1
      fx=alog(fx/pg)
c
c check if a or b closest to x
      if(abs(a-x).lt.abs(b-x)) goto 102
      a=x
      fa=fx
      goto 100
102   b=x
      fb=fx
c
c end of loop
100   continue
      print 51,n,t,pe,pg,a,b,fa,fb,eps
51    format('0***pemake, max iter.: n,t,pe,pg,a,b,fa,fb,eps=',
     * /,1x,i2,8e11.4)
      return
c DV: a, fa i fb su NAN, zasto?
c
c normal end
101   continue
      return
c
c count entry
      entry pecnt
      print 52,it
52    format('0total number of calls to jon from pemake-r =',i5)
      return
c---- debug subchk
      end

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine vaagl(nlb,xl,w)
c
c        denna rutin beraeknar vaaglaengdspunkter och motsvarande
c        vikter.  glamd(i),i=1,jlbds  aer diskontinuiteter eller del-
c        ningspunkter i vaaglaengdsled, inklusive vaaglaengdsskalans
c        aendpunkter.  mld(i), i=1,jlbds-1   aer det oenskade antalet
c        vaaglaengdspunkter i resp. intervall.   dessa storheter
c        inlaeses i rutinen.
c        vi anvaender subr.  g a u s i .
c        ****   observera. jlbds faar hoegst vara 100 med haer brukade dimension
c        och formatsatsen 100 ******
c
      dimension xl(500),w(500),glamd(100),mld(100),xlp(10),wp(10)
      common/utput/iread,iwrit
      common/cline3/glamd,jlbds
      read(iread,100)jlbds
      if(jlbds.gt.1) goto 31
c allow one point standard opacity
      read(iread,102)xl(1)
      w(1)=1.
      nlb=1
      return
31    continue
      jp=jlbds-1
      do3 k=1,jp
    3 read(iread,102)glamd(k),mld(k)
      read(iread,102)glamd(jlbds)
      i=0
      do2 k=1,jp
      if(mld(k).gt.0)go to 21
      jip=2
      xlp(1)=glamd(k)
      wp(1)=(glamd(k+1)-glamd(k))*0.5
      xlp(2)=glamd(k+1)
      wp(2)=wp(1)
      mld(k)=1
      if(k.eq.jp)go to 22
      if(mld(k+1).eq.0)go to 23
   22 mld(k)=2
      go to 23
   21 continue
      call gausi(mld(k),glamd(k),glamd(k+1),wp,xlp)
      jip=mld(k)
   23 continue
      do1 j=1,jip
      ip=i+j
      xl(ip)=xlp(j)
    1 w(ip)=wp(j)+w(ip)
    2 i=i+mld(k)
      nlb=i
  100 format(i5,5x,2f10.0)
  102 format(f10.5,i5)
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine setdis(nlb,xlb,lmax,ifirst,ilast)
c
c        this subrout. distributes the nlb wavelengths given in xlb in
c        wavelength sets, with max. lmax wavenlengths in each. the first
c        set number is ifirst, the last ilast. if more sets are necessary the
c        execution is stopped with a print-out.
c
      dimension xlb(nlb)
      common/cxlset/nset,nl(10),xl(20,10)
      common/utput/iread,iwrit
c
      nlp=1
      nset=ifirst
      do2 j=1,nlb
      xl(nlp,nset)=xlb(j)
      if(j.eq.nlb)go to 3
      nlp=nlp+1
      if(nlp.le.lmax)go to 2
      nl(nset)=nlp-1
      nlp=1
      nset=nset+1
      if(nset.gt.ilast)go to 4
    2 continue
c
    4 write(iwrit,200)
      stop
c
    3 nl(nset)=nlp
      return
c
  200 format(1h ,'too few sets allowed or too many wavelength points wan
     *ted in setdis')
c---- debug subchk
      end

C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine absko(newt,nt,tskal,peskal,iseta,j,RHO,absk,sprid)
      parameter (mt=60,mkomp=30,mkompr=15)
      parameter (ifadim=(mkomp-mkompr)*mt)
      parameter (kfadim=mkompr*mt+(mkomp-mkompr)*mt*3)
c
c        the routine administers the computation of absorption
c        coefficients. it calls the routines, giving the proper thermo-
c        dynamic information ( j o n ) , the details of the absorption
c        mechanisms ( d e t a b s ) and the factors for the interpolation
c        in t  ( t a b s ) . it chooses (if necessary reads) the right set
c        of absorption-coefficient data (iseta), statement no. 5 and makes
c        the interpolation in t, statements nos. 10-18, and the summation
c        of a rosseland mean, if indicated by j = 0, statements nos. 25-28.
c
c        newt should be gt 1 the first time this routine is used,
c                       eq 1 when a new set of t-pe is used,
c                       eq 0 otherwise.
c
c        nt is the number of t-pe points. the temperatures t should be ex-
c        pressed in degrees kelvin, the electron pressures pe in dynes per cm2.
c        iseta is the wavelength-set number, j the wavelength number in that
c        set. j equal to zero indicates that a rosseland mean is wanted.
c        this mean is computed using the wavelength points of the actual
c        set (iseta) and the quadrature weights given in rosw.
c        in absk and sprid the absorption and scattering coefficients per gram
c        of stellar matter are stored.
c
c        dimensions necessary
c        ab(nkomp),absk(1),fakt(nkomp),faktp(500),ntpo(nto),pe(nt),peskal(1),
c        rosw(max(nl)),sprid(1),sumw(nt),t(nt),tskal(1),xla(max(nl)),
c        xla3(max(nl))
c        the dimensions are lower limits.
c        dimensions of arrays in commons /ca2/,/ca3/ and /cfil/ are commented
c        on in subrout. inabs, those of arrays in common /ca4/ in subrout.
c        tabs.
c        nkomp is the number of 'components'
c        nl(i) is the number of wavelengths in wavelength set i
c        nt is the number of t-pe points supplied in tskal and peskal
c        nto is the number of points in those scales for which a detailed
c              print-out is wanted.
c
c      integer newt, nt, iseta
c      real RHO

      dimension tskal(nt),peskal(nt),absk(nt),sprid(nt)
      dimension faktp(500)
      dimension sumw(mt)
      common/utput/iread,iwrit
      common/ca2/abkof(4000),kompla(600),kompr,komps,nkomp
      common/ca3/ilogta(30),null
      common/ca4/afak(kfadim),nofak(ifadim),nplats(ifadim)
      common/ca5/ab(30),fakt(30),pe(mt),t(mt),xla(20),xla3(20),ro,
     *sumabs,sumsca,viktr,iset,nlb
      common/cfil/ireset(10),islask,ireat
      common/coutr/nto,ntpo(10)
      common/cros/rosw(20)
      COMMON/SCR2/ISETT(10),NLBB(10),XLAA(20,10),XLA33(20,10),
     *NABKOFF(10),ABKOFF(4000,10),NKOMPLL(10),KOMPLAA(600,10)
c
c      nto=0
c      data ntpo/1,1,1,1,1,1,1,1,1,1/
      iset=iseta
      if(newt.gt.1)isetp=-1
      if(newt.eq.0)go to 5
c
c        factors only dependent on t-pe
c
      call tabs(nt,tskal)
      ifak=1
      kfak=1
      jp=0
      kp=1
c
c        loop over the t-pe points ('the first ntp-loop')
      do4 ntp=1,nt
      t(ntp)=tskal(ntp)
      pe(ntp)=peskal(ntp)
c        is print-out wanted for t-pe point no. ntp
      ioutr=0
      if(kp.gt.nto)go to 3
      if(ntp.eq.ntpo(kp))go to 1
      go to 3
    1 ioutr=1
      kp=kp+1
    3 continue
c
      call jon(t(ntp),pe(ntp),1,pg,ro,dum,ioutr)
      RHO=ro

      call detabs(j,0,ntp,ioutr)
c
c        we store the fakt array, made in jon-detabs in longer arrays namely
c                  in afak for temperature-independent components
c                  in faktp for temperature-dependent ones
      do2 komp=1,kompr
      afak(kfak)=fakt(komp)
    2 kfak=kfak+1
      do4 komp=komps,nkomp
      faktp(ifak)=fakt(komp)
      kfak=kfak+nofak(ifak)
    4 ifak=ifak+1
c        end of 'the first ntp-loop'
c
c        reading  of a new wavelength set if indicated by iset
    5 if(iset.eq.isetp)go to 6
      ireadp=ireset(iset)
CSCR2
      ISETP=ISETT(ISET)
      NLB=NLBB(ISET)
      DO 610 ISCR2=1,20
        XLA(ISCR2)=XLAA(ISCR2,ISET)
        XLA3(ISCR2)=XLA33(ISCR2,ISET)
 610  CONTINUE
      NABKOF=NABKOFF(ISET)
      DO 611 ISCR2=1,4000
        ABKOF(ISCR2)=ABKOFF(ISCR2,ISET)
 611  CONTINUE
      NKOMPL=NKOMPLL(ISET)
      DO 612 ISCR2=1,600
        KOMPLA(ISCR2)=KOMPLAA(ISCR2,ISET)
 612  CONTINUE
CSCR2   51 read(ireadp,end=52)isetp,nlb,xla,xla3,nabkof,abkof,nkompl,kompla
CSCR2      go to 5
CSCR2   52 rewind ireadp
CSCR2      go to 51
c        rosseland mean or not
    6 if(j.gt.0)go to 9
    7 j1=1
      j2=nlb
      do8 ntp=1,nt
      sumw(ntp)=0.
    8 absk(ntp)=0.
      go to 10
    9 j1=j
      j2=j
c
c        interpolation in t
c        loop over all the wavelengths in a possible rosseland mean. this
c        loop ends in statement no. 26
   10 do26 jp=j1,j2
      kfak=1
      ifak=1
      kp=1
c
c        loop over the t-pe points ('the second ntp-loop')
      do26 ntp=1,nt
c
c        is print-out wanted for t-pe point no. ntp
      ioutr=0
      if(kp.gt.nto)go to 93
      if(ntp.eq.ntpo(kp))go to 92
      go to 93
   92 ioutr=1
      kp=kp+1
      if(kp.eq.2)ioutr=2
   93 continue
      iu=jp
c
c        components with absorption coefficients independent of the
c        temperature
c
      do14 komp=1,kompr
      if(kompla(iu).le.0)go to 12
c        the vector kompla is determined in subrout. inabs.
c        kompla greater than zero gives the index in abkof, where the table for
c        this component and wavelength begins.
c        kompla less than or equal to zero indicates that the actual absorption
c        coefficient for this component and wavelength is zero, as found in sub-
c        routine inabs.
c
   11 index=kompla(iu)
      ab(komp)=afak(kfak)*abkof(index)
      go to 13
   12 ab(komp)=0.
   13 kfak=kfak+1
   14 iu=iu+nlb
c
c        components with t-dependent absorption coefficients
      do19 komp=komps,nkomp
      nop=nofak(ifak)
      if(nop.eq.0)go to 17
      if(kompla(iu).le.0)go to 17
   15 index=nplats(ifak)-1+kompla(iu)
c        the vector nplats is determined by subrout. tabs. it gives the array
c        index of the temperature at which the interpolation in abkof
c        begins. nofak, giving information on the t-interpolation and
c        possibly indicating that ab=0 (nofak=0) is also determined by tabs.
c
c        interpolation
      delsum=0.
      do16 np=1,nop
      delsum=delsum+afak(kfak)*abkof(index)
      kfak=kfak+1
   16 index=index+1
c
c        has the interpolation been made on the logarithm
      if(ilogta(komp).gt.0)delsum=exp(delsum)
c        multiplication by factor from jon-detabs
      delsum=delsum*faktp(ifak)
      if(delsum.ge.0)go to 162
c
c        a negative interpolation result
  161 if(null.gt.0)write(iwrit,200)komp,delsum,jp,iset,t(ntp)
  200 format(4h ab(,i4,11h) negative=,e12.4,5x,17hfor wavelength no,i5,
     *5x,6hset no,i5,5x,2ht=,f10.4,'  and therefore put =0 ***absko***')
      ab(komp)=0.
      go to 18
  162 ab(komp)=delsum
      go to 18
   17 ab(komp)=0.
      kfak=kfak+nop
   18 iu=iu+nlb
   19 ifak=ifak+1
c
c        we multiply by wavelength-dependent  factors and add up. this is
c        done in detabs.
      call detabs(j,jp,ntp,ioutr)
c
      if(j.le.0)go to 25
   24 absk(ntp)=sumabs
c      print*, 'sumabs @ ntp', sumabs, ntp
      sprid(ntp)=sumsca
      go to 26
c
c        summation to get a rosseland mean
   25 if(j.eq.0) absk(ntp)=absk(ntp)+rosw(jp)*viktr/(sumabs+sumsca)
      if(j.lt.0) absk(ntp)=absk(ntp)+rosw(jp)*viktr/sumabs
      sumw(ntp)=sumw(ntp)+rosw(jp)*viktr
   26 continue
c
c        end of 'the second ntp-loop'
c
      if(j.gt.0)go to 29
   27 do28 ntp=1,nt
      sprid(ntp)=0.
 28   absk(ntp)=sumw(ntp)/absk(ntp)
c
   29 continue
      return
c---- debug subchk
      end

         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine ainv2(a,m)
      dimension a(6,6),c(6),ind(6)
c***** this subroutine evaluates the inverse of a                       
c***** square m*m matrix a                                              
      double precision a,c,amax,sto,w1,w
  100 amax=0.0                                                          
      do 2 i=1,m                                                        
      ind(i)=i                                                          
      if(dabs(a(i,1))-amax)2,2,3
    3 amax=dabs(a(i,1))
      i4=i                                                              
    2 continue                                                          
      mm=m-1                                                            
      do 111 j=1,mm                                                     
      if(i4-j)6,6,4                                                     
    4 isto=ind(j)                                                       
      ind(j)=ind(i4)                                                    
      ind(i4)=isto                                                      
      do 5 k=1,m                                                        
      sto=a(i4,k)                                                       
      a(i4,k)=a(j,k)                                                    
      a(j,k)=sto                                                        
    5 continue                                                          
    6 amax=0.0                                                          
      j1=j+1                                                            
      do 11 i=j1,m                                                      
      a(i,j)=a(i,j)/a(j,j)                                              
      do 10 k=j1,m                                                      
      a(i,k)=a(i,k)-a(i,j)*a(j,k)                                       
      if (k-j1)14,14,10                                                 
   14 if(dabs(a(i,k))-amax)10,10,17
   17 amax=dabs(a(i,k))
      i4=i                                                              
   10 continue                                                          
   11 continue                                                          
  111 continue                                                          
      do 140 i1=1,mm                                                    
      i=m+1-i1                                                          
      i2=i-1                                                            
      do 41 j1=1,i2                                                     
      j=i2+1-j1                                                         
      j2=j+1                                                            
      w1=-a(i,j)                                                        
      if(i2-j2)141,43,43                                                
   43 do 42 k=j2,i2                                                     
      w1=w1-a(k,j)*c(k)                                                 
   42 continue                                                          
  141 c(j)=w1                                                           
   41 continue                                                          
      do 40 k=1,i2                                                      
      a(i,k)=c(k)                                                       
   40 continue                                                          
  140 continue                                                          
      do 150 i1=1,m                                                     
      i=m+1-i1                                                          
      i2=i+1                                                            
      w=a(i,i)                                                          
      do 56 j=1,m                                                       
      if (i-j)52,53,54                                                  
   52 w1=0.0                                                            
      go to 55                                                          
   53 w1=1.0                                                            
      go to 55                                                          
   54 w1=a(i,j)                                                         
   55 if(i1-1)156,156,57                                                
   57 do 58 k=i2,m                                                      
      w1=w1-a(i,k)*a(k,j)                                               
   58 continue                                                          
  156 c(j)=w1                                                           
   56 continue                                                          
      do 50 j=1,m                                                       
      a(i,j)=c(j)/w                                                     
   50 continue                                                          
  150 continue                                                          
      do 60 i=1,m                                                       
   63 if(ind(i)-i)61,60,61                                              
   61 j=ind(i)                                                          
      do 62 k=1,m                                                       
      sto=a(k,i)                                                        
      a(k,i)=a(k,j)                                                     
      a(k,j)=sto                                                        
   62 continue                                                          
      isto=ind(j)                                                       
      ind(j)=j                                                          
      ind(i)=isto                                                       
      go to 63                                                          
   60 continue                                                          
      return                                                            
c---- debug subchk
      end                                                               
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function bpl(t,x)
c
c bpl evaluates the planck function at temperature t and wavelength x
c
      common /bplc/ex,x5
      data cp/1.191e27/,c2/1.438e8/
      x5=((x**2)**2)*(x/cp)
      ex=exp(-c2/(t*x))
      bpl=ex/((1.-ex)*x5)
      return
c
c      entry divbp(t,x)
c
c divbp returns the temperature derivative of the planck function.
c
      x6=x5*x
      tex=t*(1.-ex)
      divbp=c2*(ex/tex)/(tex*x6)
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine bplv(n,t,x,b)
c
c bplv is vectorizable planck function subroutine
c
      dimension t(n),b(n)
      data cp/1.191e27/,c2/1.438e8/
      do 100 k=1,n
      x5=((x**2)**2)*(x/cp)
      ex=exp(-c2/(t(k)*x))
      b(k)=ex/((1.-ex)*x5)
100   continue
      return
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine dbplv(n,t,x,b,d)
c
c dbplv is a vectorizable planck function temperature derivative subroutine
c
      dimension t(n),b(n),d(n)
      data cp/1.191e27/,c2/1.438e8/
      do 100 k=1,n
      x5=((x**2)**2)*(x/cp)
      x6=x5*x
      ex=exp(-c2/(t(k)*x))
      tex=t(k)*(1.-ex)
      b(k)=ex/((1.-ex)*x5)
      d(k)=c2*(ex/tex)/(tex*x6)
100   continue
      return
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine detabs(j,jp,ntp,ioutr)
      parameter (mt=60)
c
c        this routine gives the details of the absorption mechanisms.
c        changes in the absorption-coefficient program are expected to
c        be confined to the tables and to this routine.
c        detabs has two purposes
c        1. jp=0   determination of wavelength-independent factors (dep.on
c                  t, pe and the component) stored in fakt.
c        2. jp= the actual wavelength number.
c                  multiplication of ab, computed in subroutine absko,
c                  by wavelength-dependent factors. summation of the total
c                  absorption and scattering coefficients ( sumabs and
c                  sumsca ).
c
c        n o t e .  before a call on detabs for purpose 1, subrout.
c        jon must have been called.
c
c        if j is less than or equal to zero, the weight for a rosseland mean
c        will be computed and stored in viktr (the weight being 1/viktr).
c        ntp is the array index of the t-pe point.
c        if ioutr is greater than zero at a call with jp greater than zero
c        (part two of the routine), details of the absorption coefficients
c        are printed. if ioutr is greater than one, a table heading is also
c        printed.
c
c
c        contents of common/ci5/, communicating physical information from
c        subrout. jon.
c             abund  abundances
c             anjon  fractions of ionization
c             h      quantum number of the highest existing hydrogenic level
c             part   partition functions
c             dxi    decrease of ionization energy of hydrogen in electron volt
c             f1     n(hi)/n(h)
c             f2     n(hii)/n(h)
c             f3     n(h-)/n(h)
c             f4     n(h2+)/n(h)
c             f5     n(h2)/n(h)
c             xkhm   'dissociation constant' of h-
c             xmh    mass of the hydrogen atom in grams
c             xmy    grams of stellar matter/grams of hydrogen
c
c        dimensions necessary
c        abund(nel),anjon(nel,max(nj)),els(nt),h(5),hrest(nt),part(nel,max(nj))
c        prov(nkomp)
c        the dimensions are lower limits. dimensions in common /ca5/ are
c        commented on in subrout. absko.
c        nel is the number of chemical elements initiated in subrout. injon
c        nj(i) is the number of stages of ionization, including the neutral
c             stage, for element i
c        nkomp is the number of components, not including those added by
c             analytical expressions after statement no. 13.
c        nt   is the number of temperatures-electron pressures given at the
c             call of subrout. absko.
c
c
c      dimension els(mt), hrest(mt)
      dimension els(mt)
      dimension fakray(mt)
      dimension h2ray(mt)
      common/ci5/abund(16),anjon(16,5),h(5),part(16,5),dxi,f1,f2,f3,f4,
     *f5,xkhm,xmh,xmy
      common/ca2/ca2dum(4602),nkomp
      common/ca5/ab(30),fakt(30),pe(mt),t(mt),xla(20),xla3(20),ro,
     *sumabs,sumsca,viktr,iset,nlb
      common/utput/iread,iwrit
      common/carc4/prov(30),source(30),abname(30),nprova,nprovs,nprov
      common/dusan/hrest(mt)

      logical first
C The following double precision variables are to be declared REAL on the CRAY
C Comment out the following lines on the CRAY
      double precision source,abname
      double precision nhmin,nels,nhray,nh2ray
C      character*8 source,abname
C      character*8 nhmin,nels,nhray,nh2ray
C End of lines to be changed on the CRAY or other 64bit machines

C      data nels/'el-sc'/,nhray/'h-sc'/,nh2ray/'h2-sc'/,nhmin/'h-'/
      data first/.true./
c
c        save absorption component names the first time detabs is called
c
c      print*, '----- new call -----'
c      print*, 'Pars ', j, jp, ntp, ioutr, hrest(1)

      if (.not.first) go to 1
      do 2 komp=19,nkomp
      abname(1)=nhmin
    2 abname(komp-16)=abname(komp)
      nprova=nkomp-16
      nprovs=3
      nprov=nprova+nprovs
      abname(nprov-2)=nels
      abname(nprov-1)=nhray
      abname(nprov)=nh2ray
      first=.false.
    1 continue
c
      teta=5040./t(ntp)
      if(jp.gt.0)go to 7
c
c        1. computation of wavelength-independent quantities
c
      hn=1./(xmh*xmy)
      hnh=f1*hn
c        h-
      fakt(1)=pe(ntp)*hnh*1.e-17/xkhm
      fakt(18)=pe(ntp)*hnh*2.e-26/part(1,1)
c        hi
      teta31=31.30364*teta
      xfakh=2.0898e-26/part(1,1)*hnh
      nniv=15
      xniv=15.
      if(h(1).lt.xniv)nniv=int(h(1))
      do3 m=1,nniv
      xm2=m*m
      xm3=xm2*float(m)
    3 fakt(m+1)=xfakh*exp(-teta31*(1.-1./xm2))/xm3
      fakt(nniv+1)=fakt(nniv+1)*amin1(h(1)-nniv,1.)
      if(nniv.ge.15)go to 6
    4 n1=nniv+1
      do5 m=n1,15
    5 fakt(m+1)=0.
c
c        free-free hi absorption
    6 umc=2.3026*dxi*teta
      expj=xfakh*exp(-teta31+umc)/(2.*teta31)
      addf=exp(teta31/((float(nniv)+0.5)**2)-umc)-1.
      if(h(1).lt.xniv+0.5)addf=0.
      fakt(17)=expj
      hrest(ntp)=expj*addf
c      print*, 'expj | addf', expj, addf
c
c        h+h
      fakt(19)=(hnh*1.e-25)*(hnh*1.e-25)*ro
c        h2+
      fakt(20)=(hnh*1.e-20)**2*ro*anjon(1,2)/anjon(1,1)
c        h2-
      fakt(21)=pe(ntp)*f5*hn
c        c i
      fakt(22)=anjon(3,1)*abund(3)*hn*9./part(3,1)
c        mg i
      fakt(23)=anjon(8,1)*abund(8)*hn/part(8,1)
c        al i
      fakt(24)=anjon(9,1)*abund(9)*hn*6./part(9,1)
c        si i
      fakt(25)=anjon(10,1)*abund(10)*hn*9./part(10,1)
c        he i
      fakt(26)=anjon(2,1)*abund(2)*hn/part(2,1)
c        electron scattering
      els(ntp)=4.8206e-9*pe(ntp)/(t(ntp)*ro)
c        rayleigh scattering
      fakray(ntp)=hnh*2./part(1,1)
      h2ray(ntp)=f5*hn
c      print*, 'expj | addf | hrest', expj, addf, hrest(ntp)
      return
c        n o t e . apart from vectors hrest and els, none of the
c        temperature- or pressure-dependent variables defined above can
c        generally be used at the next visit below.
c        any set of factors which is wanted should be stored in an array with
c        dimension = nt, like hrest and els, or in fakt, where the data for
c        further use are stored in subr. absko.
c
c        2. wavelength-dependent factors. summation.
c        correction for stimulated emission
    7 expa=exp(-28556.*teta/xla(jp))
   11 stim=1.-expa
c
c        absorption
      sumabs=0.
c        h i
      do12 komp=2,17
      sumabs=sumabs+ab(komp)
   12 continue
c DV: until here it is good
c      print*, 'hrest(ntp) | xla3(jp)', hrest(ntp), xla3(jp)
      sumabs=(sumabs+hrest(ntp))*xla3(jp)
      prov(2)=sumabs
c      print*, 'detabs:sumabs', sumabs
c      print*, ''
c        h-
      hmin=ab(1)+ab(18)/stim
      sumabs=sumabs+hmin
      prov(1)=hmin
c        h+h, h2+, he i, c i, mg i, al i, si i
      do13 komp=19,nkomp
      sumabs=sumabs+ab(komp)
      prov(komp-16)=ab(komp)
   13 continue
c
c        here further absorption mechanisms, given in tables or by
c        analytical expressions, may be added.
      sumabs=sumabs*stim
c
c        scattering
      xray=amax1(xla(jp),1026.)
      xray2=1./(xray*xray)
      rayh=xray2*xray2*(5.799e-13+xray2*(1.422e-6+xray2*2.784))*
     *fakray(ntp)
      rayh2=xray2*xray2*(8.14e-13+xray2*(1.28e-6+xray2*1.61))*h2ray(ntp)
      sumsca=els(ntp)+rayh+rayh2
      prov(nprov-2)=els(ntp)
      prov(nprov-1)=rayh
      prov(nprov)=rayh2
c
      if(j.gt.0)go to 15
c
c        weight for a rosseland mean
   14 viktr=expa/(stim*stim*(xla3(jp)*1e-3)**2)
   15 continue
c
      if(ioutr-1)23,21,20
c
c        **** print-out ****
   20 write(iwrit,200)xla(jp),(abname(kp),kp=1,nprov)
  200 format('0wavel.=',f7.0,'    abs       scat  ',16a6)
   21 do22 kp=1,nprova
   22 prov(kp)=prov(kp)/sumabs*stim
      do24 kp=1,nprovs
   24 prov(nprova+kp)=prov(nprova+kp)/sumsca
      write(iwrit,201)t(ntp),sumabs,sumsca,(prov(kp),kp=1,nprov)
  201 format(3h t=,f7.1,5x,2e10.4,16f6.4)
   23 continue
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine dubint(nxskal,xskal,nyskal,yskal,iybeg,iyend,n,x,y,
     *                                 factor,ix,iy1,iy2)
      parameter (mt=60)
c
c        this routine computes interpolation factors for interpolation in an,
c        not necessarily equidistant, two dimensional table. scales
c           xskal (nxskal points)
c           yskal(nyskal points)
c        the table is only defined for y-values yskal(l), where l is
c        within the interval iybeg(nx) to iyend(nx) for a given xskal(nx).
c        arguments are given in x and y (n points).
c        resulting factor for point k is put in factor(k,1-2,1-2)
c        starting points at interpolation in ix(k), referring to the xscale, and
c        in iy1(k) and iy2(k), referring to the restricted y scales defined for
c        xscale(ix(k)) and xscale(ix(k)+1), respectively.
c        interpolations and extrapolations are  l i n e a r .
c
      dimension xskal(nxskal),yskal(nxskal),x(n),y(n),factor(mt,2,2)
     & ,ix(n)
      dimension iybeg(nxskal),iyend(nxskal),iy1(nxskal),iy2(nxskal)
      do5 k=1,n
      do1 j=2,nxskal
      jmem1=j
      if(x(k).lt.xskal(j))go to 2
    1 continue
    2 ix(k)=jmem1-1
      jm1=jmem1-1
      do3 j=2,nyskal
      jmem2=j
      if(y(k).lt.yskal(j))go to 4
    3 continue
    4 iy=jmem2-1
      iy1(k)=iy+1-iybeg(jm1)
      iy1(k)=min0(iy1(k),iyend(jm1)-iybeg(jm1))
      iy1(k)=max0(iy1(k),1)
      iy2(k)=iy+1-iybeg(jmem1)
      iy2(k)=min0(iy2(k),iyend(jmem1)-iybeg(jmem1))
      iy2(k)=max0(iy2(k),1)
      i1=iy1(k)+iybeg(jm1)-1
      i2=iy2(k)+iybeg(jmem1)-1
      dx=(x(k)-xskal(jmem1-1))/(xskal(jmem1)-xskal(jmem1-1))
      dy1=(y(k)-yskal(i1))/(yskal(i1+1)-yskal(i1))
      dy2=(y(k)-yskal(i2))/(yskal(i2+1)-yskal(i2))
      factor(k,1,1)=(1.-dx-dy1+dx*dy1)
      factor(k,2,1)=(1.-dy2)*dx
      factor(k,1,2)=(1.-dx)*dy1
    5 factor(k,2,2)=dx*dy2
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine gausi(k,a,b,ai,xmyi)
c
c        rutinen ger vikter och integrationspunkter foer gaussintegration
c        mellan a och b - b aer oevre graens , a nedre. kaella foer data/
c        lowan, davids, levenson,  bull amer math soc  48 sid 739  (1942) /
c        ai=vikter, xmyi=integrationspunkter.
c        integrationsordning k.  k vaeljes mellan 2 och 10.
c
      dimension ai(k),xmyi(k),ap(29),xmyp(29),indov(9)
      double precision ap,xmyp
c               10 datakort foer ap, 9 foer xmyp och 1 foer indov
      data ap/1.0d0,0.55555555555555d0,.88888888888888d0,.347854845137d0
     *,0.65214515486254d0,0.23692688505618d0,0.47862867049936d0,        
     * 0.56888888888888d0,0.17132449237917d0,0.36076157304813d0,        
     * 0.46791393457269d0,0.12948496616887d0,0.27970539148927d0,        
     * 0.38183005050511d0,0.41795918367346d0,0.10122853629037d0,        
     * 0.22238103445337d0,0.31370664587788d0,0.36268378337836d0,        
     * 0.08127438836157d0,0.18064816069485d0,0.26061069640293d0,        
     * 0.31234707704000d0,0.33023935500126d0,0.06667134430868d0,        
     * 0.14945134915058d0,0.21908636251598d0,0.26926671930999d0,        
     * 0.29552422471475d0/                                              
      data xmyp/
     *0.57735026918962d0,.77459666924148d0,.0d0,0.86113631159405d0,
     *0.33998104358485d0,.90617984593866d0,.53846931010568d0,.0d0,      
     *0.93246951420315d0,.66120938646626d0,.23861918608319d0,           
     *0.94910791234275d0,.74153118559939d0,.40584515137739d0,.0d0,      
     *0.96028985649753d0,.79666647741362d0,.52553240991632d0,           
     *0.18343464249565d0,.96816023950762d0,.83603110732663d0,           
     *0.61337143270059d0,.32425342340380d0,.0d0,0.97390652851717d0,     
     *0.86506336668898d0,.67940956829902d0,.43339539412924d0,           
     *0.14887433898163d0/
      data indov/1,3,5,8,11,15,19,24,29/
      if(k.eq.1)go to 7
      kud=0
      flk=float(k)/2.
      k2=k/2
      fk=float(k2)
      if(abs(flk-fk)-1.e-7)2,1,1
    1 k2=k2+1
      kud=1
    2 ioev=indov(k-1)
      ined=ioev-k2
      do3 i=1,k2
      ip=ined+i
      xmyi(i)=-xmyp(ip)*(b-a)*0.5+(b+a)*0.5
    3 ai(i)=(b-a)*0.5*ap(ip)
      k2=k2+1
      do4 i=k2,k
      ip=ioev+k2-i
      if(kud)6,6,5
    5 ip=ip-1
    6 continue
      xmyi(i)= xmyp(ip)*(b-a)*0.5+(b+a)*0.5
    4 ai(i)=(b-a)*0.5*ap(ip)
      return
    7 xmyi(1)=(b+a)*0.5
      ai(1)=b-a
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine inabs(iouts)
      parameter (mt=60)
c
c        this routine  reads absorption coefficient tables and inter/extra-
c        polates them to our wavelengths given in xl. the interpolation is
c        performed separately for each wavelength set.
c
c        nkomp is the number of components in the full table.
c        nextl should be greater than zero if a print-out is wanted on extra-
c              polation in wavelength,
c        nutzl if print-out is wanted when we put the coefficient =0 outside the
c              wavelength region of the tables.
c        nextt and nutzl are the corresponding quantities on interpolation in
c              t, made in subrout. tabs.
c        null  should be greater than zero if a print-out is wanted (from sub-
c              routine absko) when a coefficient is found to be less than zero
c              on interpolation in t and therefore put equal to zero.
c
c        for each component the following parameters must be specified
c        abname is the name of, or a symbol for, the absorption mechanism.
c        source indicates the source or reference of the data
c        1. parameters for the wavelength interpolation.
c          ilogl should be greater than zero if interpolation in wavelength is
c                to be performed on the logarithmic absorption coefficients
c                (with subsequent exponentiation of the results - here if ilogt
c                is equal to zero or in subrout. absko if ilogt is greater
c                than zero). otherwise interpolation in wavelength is made
c                directly on the absorption coefficients themselves.
c          kvadl should be greater than zero if quadratic interpolation in
c                wavelength is wanted. otherwise interpolation will be linear.
c          minex should be gt 0 if linear extrapolation (instead of putting the
c                coefficient = 0) is wanted towards shorter wavelengths.
c          maxex, corresponding towards longer wavelengths.
c          nlatb is the number of wavelength points of the absorption coeffi-
c                cient table to be read.
c          xlatb are those wavelengths. they should be given in increasing order
c        2. parameters for the temperature interpolation.
c          ilogt, kvadt, minet, maxet and ntetb are the t-interpolation
c                analogues to ilogl-nlatb.
c          iteta is put greater than zero when teta values (teta=5040./t) are
c                given in xtet instead of temperatures.
c          xtet are the temperature (teta) values of the absorption
c                coefficient table to be read. the xtet values should be given
c                in increasing order and equidistantly, however (ielmax-1)
c                changes of the interval are allowed. the program checks that
c                this number is not exceeded.
c        xkap is the absorption coefficient table for the actual component. the
c                wavelengths increases more rapidly than t (teta).
c
c        the tables for  t e m p e r a t u r e - i n d e p e n d e n t
c        c o m p o n e n t s  s h o u l d  b e  p u t  f i r s t .
c           the resulting table is put in abkof. here t (teta) increases more
c        rapidly than xla, which increases more rapidly than komp. if the result
c        of the interpolation is zero for a certain xla(j) and komp, this is not
c        put in abkof. instead a note is made in kompla (kompla(nlb*(komp-
c        1)+j) is put equal to zero). otherwise the kompla value tells where in
c        abkof the table for the component komp and the wavelength j begins.
c
c        a detailed print-out is given if iouts is greater than zero.
c
c
c        dimensions necessary
c        abkof(nabdim),abname(nkomp),delt(nkomp,ielmax),idel(nkomp),
c        idiskv(max(nlatb)),ilogta(nkomp),ireset(nset),isvit(nkomp),iteta(nkomp)
c        kompla(max(nl)*nkomp),kvadt(nkomp),maxet(nkomp),minet(nkomp),
c        nl(nset),ntaet(nkomp),ntm(nkomp,ielmax),source(nkomp),
c        tbolt(nkomp,ielmax),xkap(max(nlatb),max(ntetb)),xl(max(nl),nset),
c        xla(max(nl)),xla3(max(nl)),xlatb(max(nlatb)),xtet(max(ntetb)),
c        xtetp(max(ntetb))
c
c        the dimensions are lower limits
c        ielmax is the maximum number of different t intervals (given below) in
c               any absorption coefficient table.
c        nabdim is the dimension of the abkof array (given below).
c        nkomp is the number of 'components', i.e. equal to the number of
c               different absorption coefficient tables to be read.
c        nl(i)  is the number of wavelengths in the wavelength set i.
c        nlatb(komp) is the number of wavelength points in the table to be read
c               for the component komp.
c        nset   is the number of wavelength sets.
c        ntetb  is the number of temperature points in the table for the com-
c               ponent being considered.
c
      integer kvadl(30),minex(30),maxex(30),nlatb(30),ilogl(30),
     *idisk(30),ntetb(30),ilogt(30)
      dimension idiskv(mt,30),xlatb(mt,30),xtet(30,30),ntaet(30),
     *xkap(mt,30,30),xla3(20),xla(20),xtetp(30)
      common/carc4/prov(30),source(30),abname(30),nprova,nprovs,nprov
      common/utput/iread,iwrit
      common/ca1/delt(30,2),tbot(30,2),idel(30),isvit(30),iteta(30),
     *kvadt(30),maxet(30),minet(30),ntm(30,2),nextt,nutzt
      common/ca2/abkof(4000),kompla(600),kompr,komps,nkomp
      common/ca3/ilogta(30),null
      common/cfil/ireset(10),islask,ireat
      common/cxlset/nset,nl(10),xl(20,10)
      COMMON/SCR2/ISETT(10),NLBB(10),XLAA(20,10),XLA33(20,10),
     *NABKOFF(10),ABKOFF(4000,10),NKOMPLL(10),KOMPLAA(600,10)
C The following double precision variables are to be declared REAL on the CRAY
C Comment out following line on the CRAY
      double precision abname,source
C End of lines to be changed on the CRAY or other 64bit machines
c
c        ielmax is the maximum number of different t intervals in the xkap-
c        table. the dimensions of tbot, delt and ntm are affected by this number
      ielmax=2
c        the dimension of the abkof array
      nabdim=4000
      do705 l=1,30
  705 xtetp(l)=0.
      if(iouts.gt.0)write(iwrit,229)
c
      read(ireat,101)nkomp,nextl,nutzl,nextt,nutzt,null
c
      kompr=0
CSCR1      rewind islask
c
c        loop over components starts (the 'first komp-loop')
      do720 komp=1,nkomp
      read(ireat,105)abname(komp),source(komp)
      read(ireat,102)ilogl(komp),kvadl(komp),minex(komp),maxex(komp),
     *nlatb(komp)
      read(ireat,103)(xlatb(j,komp),j=1,nlatb(komp))
c
c        we find the discontinuities in wavelength
c        a discontinuity in a table is defined by two wavelength points
c        within less than two angstroems.
      idisk(komp)=0
      idiskv(1,komp)=0
      do700 j=2,nlatb(komp)
      idiskv(j,komp)=0
      if((xlatb(j,komp)-xlatb(j-1,komp)).ge.2.)go to 700
      idiskv(j-1,komp)=1
      idiskv(j,komp)=1
      idisk(komp)=1
  700 continue
c
c        continue reading
      read(ireat,102)ilogt(komp),kvadt(komp),minet(komp),maxet(komp),
     *ntetb(komp),iteta(komp)
      ilogta(komp)=ilogt(komp)
      if(ntetb(komp).gt.1)go to 702
  701 kompr=kompr+1
      go to 703
  702 read(ireat,103)(xtet(l,komp),l=1,ntetb(komp))
c
c        finally the absorption coefficient table is read
  703 do 704 k=1,ntetb(komp)
  704 read(ireat,104)(xkap(jj,k,komp),jj=1,nlatb(komp))
c
c        we take the logarithms before the wavelength interpolation
c        if ilogl(komp) is greater than zero.
      if(ilogl(komp).lt.1)go to 712
  710 do 711 k=1,ntetb(komp)
      do 711 jj=1,nlatb(komp)
      if(xkap(jj,k,komp).gt.0.)go to 711
c
c        a coefficient for which the logarithm should be taken is zero
      write(iwrit,207)jj,k,xkap(jj,k,komp),komp
      xkap(jj,k,komp)=1.e-37
  711 xkap(jj,k,komp)=alog(xkap(jj,k,komp))
  712 continue
c
c        preparation of the t-interpolation in subrout. tabs
c
c        we find out whether isvit(komp) can be chosen greater than zero. this
c        is the case if the t scale and maxet, minet and kvadt are identical
c        with those of the previous component. if isvit is greater than zero,
c        the time spent in subr. tabs will be decreased.
      isvit(komp)=0
      if(ntetb(komp).le.1)go to 719
      do 721 l=1,ntetb(komp)
      if(xtet(l,komp).ne.xtetp(l))go to 722
  721 continue
      if(ntetb(komp).ne.ntetbp)go to 722
      if(maxet(komp).ne.maxetp) go to 722
      if(minet(komp).ne.minetp) go to 722
      if(kvadt(komp).ne.kvadtp)go to 722
      isvit(komp)=1
  722 continue
c
c        we remember temperatures etc. for next component
      do723 l=1,ntetb(komp)
  723 xtetp(l)=xtet(l,komp)
      ntetbp=ntetb(komp)
      maxetp=maxet(komp)
      minetp=minet(komp)
      kvadtp=kvadt(komp)
c
c        we find the intervals in the t (teta) scale
      tbot(komp,1)=xtet(1,komp)
      delt(komp,1)=xtet(2,komp)-xtet(1,komp)
      ntm(komp,1)=1
      idel(komp)=1
      if(ntetb(komp).eq.2)go to 719
c
      j=1
      lf=1
      do714 l=3,ntetb(komp)
      diff=xtet(l,komp)-xtet(l-1,komp)
      if(abs(1.-diff/delt(komp,j)).lt.1.e-4)go to 714
      j=j+1
      if(j.gt.ielmax)go to 715
      tbot(komp,j)=xtet(l-1,komp)
      delt(komp,j)=diff
      ntm(komp,j-1)=lf
      lf=0
  714 lf=lf+1
      ntm(komp,j)=lf
      idel(komp)=j
      go to 719
c        too many different intervals in the t-table for this component
  715 write(iwrit,203)komp,ielmax
      write(iwrit,206)(xtet(l,komp),l=1,ntetb(komp))
      stop
c
  719 ntaet(komp)=ntetb(komp)
c        all data necessary below for this component are stored on unit
c        islask
CSCR1      write(islask)kvadl(komp),minex(komp),maxex(komp),nlatb(komp),
CSCR1     *ilogl(komp),idisk(komp),(idiskv(j,komp),j=1,
CSCR1     *nlatb(komp)),(xlatb(j,komp),j=1,nlatb(komp)),ntetb(komp),
CSCR1     *ilogt(komp),(xtet(l,komp),l=1,ntetb(komp))
CSCR1     *,((xkap(jj,k,komp),jj=1,nlatb(komp)),k=1,ntetb(komp))
c
c        **** print-out ****
      if(iouts.le.0)go to 7
      write(iwrit,211)komp,abname(komp),source(komp),abname(komp)
      write(iwrit,212)
      write(iwrit,213)xlatb(1,komp),xlatb(nlatb(komp),komp)
      if(minex(komp).eq.0)write(iwrit,214)
      if(minex(komp).gt.0)write(iwrit,215)
      if(kvadl(komp).eq.0)write(iwrit,216)
      if(kvadl(komp).gt.0)write(iwrit,217)
      if(ilogl(komp).eq.0)write(iwrit,218)
      if(ilogl(komp).gt.0)write(iwrit,219)
      if(maxex(komp).eq.0)write(iwrit,220)
      if(maxex(komp).gt.0)write(iwrit,221)
      if(idisk(komp).gt.0)write(iwrit,222)
      if(ntetb(komp)-1)8,8,9
    8 write(iwrit,230)
      go to 7
    9 continue
      write(iwrit,223)
      write(iwrit,213)xtet(1,komp),xtet(ntetb(komp),komp)
      if(minet(komp).eq.0)write(iwrit,214)
      if(minet(komp).gt.0)write(iwrit,215)
      if(kvadt(komp).eq.0)write(iwrit,216)
      if(kvadt(komp).gt.0)write(iwrit,217)
      if(ilogta(komp).eq.0)write(iwrit,218)
      if(ilogta(komp).gt.0)write(iwrit,219)
      if(maxet(komp).eq.0)write(iwrit,220)
      if(maxet(komp).gt.0)write(iwrit,221)
      if(isvit(komp).gt.0)write(iwrit,224)
      write(iwrit,231)
    7 continue
  720 continue
c        end of 'the first komp-loop'
c
      komps=kompr+1
c
c
c        we build the abkof array. interpolation in wavelength.
c
c        loop over wavelength sets ('the iset-loop')
      do 70 iset=1,nset
CSCR1      rewind islask
      nlb=nl(iset)
      do1 j=1,nlb
      xla(j)=xl(j,iset)
    1 xla3(j)=xla(j)**3
      index=1
c
c        loop over components starts ('the second komp-loop')
      do60 komp=1,nkomp
CSCR1      read(islask)kvadl(komp),minex(komp),maxex(komp),nlatb(komp),
CSCR1     *ilogl(komp),idisk(komp),(idiskv(j,komp),j=1,
CSCR1     *nlatb(komp)),(xlatb(j,komp),j=1,nlatb(komp)),ntetb(komp),
CSCR1     *ilogt(komp),(xtet(l,komp),l=1,ntetb(komp))
CSCR1     *,((xkap(jj,k,komp),jj=1,nlatb(komp)),k=1,ntetb(komp))
      ji=1
      lambi=1
c
c        loop over wavelengths ('the j-loop') starts
      do60 j=1,nlb
c        searching in wavelength
      iu=nlb*(komp-1)+j
      kompla(iu)=index
      do24 jj=1,nlatb(komp)
      ihelp=jj
      if(xla(j)-xlatb(jj,komp))25,24,24
   24 lambi=jj
   25 continue
      if(ihelp-1)45,45,26
   26 if(kvadl(komp))33,33,27
   33 if(nlatb(komp)-lambi-1)41,31,31
   27 if(nlatb(komp)-lambi-1)41,28,29
c
c        quadratic interpolation
   28 lambi=lambi-1
   29 continue
c        are discontinuities present
      if(idisk(komp).le.0)go to 299
      if(idiskv(lambi+1,komp).le.0)go to 299
      if(xla(j).gt.xlatb(lambi+1,komp))go to 292
  291 if(idiskv(lambi,komp).gt.0)go to 31
      if(lambi.eq.1)go to 31
      lambi=lambi-1
      go to 299
  292 lambi=lambi+1
      if(idiskv(lambi+1,komp).gt.0)go to 31
      if(lambi+1.eq.nlatb(komp))go to 31
  299 continue
      dxx1=xla(j)-xlatb(lambi,komp)
      dxx2=xla(j)-xlatb(lambi+1,komp)
      dxx3=xla(j)-xlatb(lambi+2,komp)
      dx21=xlatb(lambi+1,komp)-xlatb(lambi,komp)
      dx32=xlatb(lambi+2,komp)-xlatb(lambi+1,komp)
      dx31=xlatb(lambi+2,komp)-xlatb(lambi,komp)
      a1=dxx2*dxx3/(dx21*dx31)
      a2=dxx1*dxx3/(dx21*dx32)
      a3=dxx1*dxx2/(dx31*dx32)
c
      do30 k=1,ntetb(komp)
      abkof(index)=a1*xkap(lambi,k,komp)-a2*xkap(lambi+1,k,komp)+
     *a3*xkap(lambi+2,k,komp)
   30 index=index+1
      go to 59
c
c        linear inter- and extrapolation
   31 a2=(xla(j)-xlatb(lambi,komp))/(xlatb(lambi+1,komp)-
     *xlatb(lambi,komp))
      a1=1.-a2
      do32 k=1,ntetb(komp)
      abkof(index)=a1*xkap(lambi,k,komp)+a2*xkap(lambi+1,k,komp)
   32 index=index+1
      go to 59
c
c        too great a wavelength - outside the table
   41 if(maxex(komp))50,50,42
   42 lambi=lambi-1
      if(nextl.gt.0)write(iwrit,201) komp,xla(j)
      go to 31
c
c        too small a wavelength - outside the table
   45 if(minex(komp))50,50,46
   46 if(nextl.gt.0)write(iwrit,201)komp,xla(j)
      go to 31
c
c        abs. coeff. is put = zero
   50 kompla(iu)=0
      if(nutzl.gt.0)write(iwrit,202)komp,xla(j)
      go to 60
c
   59 if(ilogl(komp).lt.1)go to 592
      if(ilogt(komp).gt.0)go to 60
c
c        logarithmic interpolation only in wavelength
      lip=index-ntetb(komp)
      lap=index-1
      do 591 ll=lip,lap
  591 abkof(ll)=exp(abkof(ll))
c
  592 continue
c
      if(ilogt(komp).le.0)go to 60
c        we take the logarithm before the t interpolation if ilogt(komp) gt 0
      lip=index-ntetb(komp)
      lap=index-1
      do593 ll=lip,lap
      if(abkof(ll).gt.0.)go to 593
c
c        impossible to take the logarithm of a negative coefficient
      lus=ll-lip+1
      write(iwrit,208)ll,abkof(ll),komp,j,iset,lus
      abkof(ll)=1.e-37
  593 abkof(ll)=alog(abkof(ll))
   60 continue
c        end of 'the j-loop'
c        end of 'the second komp-loop'
c
c        write the data of the set iset on unit ireset(iset)
      nabkof=index-1
      nkompl=iu
      ireadp=ireset(iset)
CSCR2
      ISETT(ISET)=ISET
      NLBB(ISET)=NLB
      DO 610 ISCR2=1,20
        XLAA(ISCR2,ISET)=XLA(ISCR2)
        XLA33(ISCR2,ISET)=XLA3(ISCR2)
 610  CONTINUE
      NABKOFF(ISET)=NABKOF
      DO 611 ISCR2=1,4000
        ABKOFF(ISCR2,ISET)=ABKOF(ISCR2)
 611  CONTINUE
      NKOMPLL(ISET)=NKOMPL
      DO 612 ISCR2=1,600
        KOMPLAA(ISCR2,ISET)=KOMPLA(ISCR2)
 612  CONTINUE
CSCR2      write(ireadp)iset,nlb,xla,xla3,nabkof,abkof,nkompl,kompla
c
CSCR2      end file ireadp
CSCR2      backspace ireadp
c
c        check dimension of abkof
      if(iouts.gt.0) write(iwrit,204)nabkof,iset
      if(nabkof.le.nabdim)go to 70
c        too small dimension for abkof
      write(iwrit,205)nabdim
      stop
   70 continue
c
c        end of 'the iset-loop'
c
c
      do 71 iset=1,nset
      ireadp=ireset(iset)
CSCR2      rewind ireadp
71    continue
      if(iouts.le.0) goto 74
c
c        **** print-out ****
c        on wavelength sets and files
      write(iwrit,225)ireat,islask
      write(iwrit,226)
      do73 m=1,nset
      np=nl(m)
      write(iwrit,227)m,ireset(m)
      write(iwrit,228)(xl(j,m),j=1,np)
   73 continue
      write(iwrit,232)
   74 continue
  101 format(8x,i2,5(9x,i1))
  102 format(4(9x,i1),8x,i2,9x,i1)
  103 format(6f10.0)
  104 format(6e10.3)
  105 format(2a8)
  201 format(' extrapolation for component',i5,5x,'and wavelength=',f10.
     *3,5x,'***inabs***')
  202 format(' abs.coeff. put =0 at wavelength-inter/extrapolation for
     *component',i5,5x,'and wavelength=',f10.3,5x,'***inabs***')
  203 format(' too many different intervals in the t-(teta-)table for co
     *mponent ',i5,5x,'max is',i5,5x,'***inabs***')
  204 format(' necessary dimension for abkof=',i5,5x,'in set',i5,5x,'***
     *inabs***')
  205 format(' dimension allowed =',i5,5x,'too small     ***inabs***')
  206 format(6h xtet=,10e12.4)
  207 format(' xkap(',i2,',',i2,')=',e12.5,' put = 1.e-77 before log has
     *been taken.   ***inabs***')
  208 format(' abkof(',i4,')=',e12.5,' for component ',i2,' wavelength',
     *i3,' set ',i2,' xtet nr ',i2,' abkof put=1.e-77   ***inabs***')
  211 format('0************  component no',i3,', ',a8,' source  ',a8,'
     *****************',5x,a8)
  212 format('0     i n t e r p o l a t i o n  i n  w a v e l e n g t h'
     *)
  213 format(1h ,15x,f10.2,25x,f10.2)
  214 format(1h+,'   kappa=0   - ')
  215 format(1h+,' lin. extrap.- ')
  216 format(1h+,25x,' -lin. int. ')
  217 format(1h+,25x,' -quad. int.')
  218 format(1h+,37x,'  kappa    - ')
  219 format(1h+,37x,' log(kappa)- ')
  220 format(1h+,60x,' -   kappa=0')
  221 format(1h+,60x,' -lin. extrap.')
  222 format(1h0,' discontinuities present.')
  223 format('0     i n t e r p o l a t i o n  i n  t  ( t e t a )')
  224 format('0 t scale etc. identical with preceeding component')
  225 format(1h0,'f i l e s  u s e d  b y  t h e  a b s - b l o c k'//'
     * initial file ',i3,', preliminary file',i3)
  226 format(1h0,'set           wavelengths',81x,'file')
  227 format (1h ,i2,105x,i2)
  228 format(1h ,5x,10f10.2)
  229 format(1h1,'d a t a  f r o m  s u b r o u t i n e  i n a b s')
  230 format(1h0,'     n o  t-  ( t e t a - ) d e p e n d e n c e')
  231 format(1h0)
  232 format(1h1)
      return
c---- debug subchk
      end
 
 
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine injon(iouts)
c
c
c        this routine reads data necessary for the computation of ionization
c        equilibria etc. (in subr. jon).
c        1. nel= the number of chemical elements considered.
c           a=   the ratio of the number of hydrogen nuclei to the number of
c                nuclei of metallic elements.
c           nmet=the number of metallic elements in the list of chemical
c                elements considered. the last nmet elements of the list are
c                considered to be metallic, for the calculation of the
c                quantity a (defined above).
c        2. iel  is the array which will contain the symbols for the chemical
c                elements considered.
c           abund is the array which will contain the prevailing abundances of
c                the chemical elements considered at input. these abundances
c                are expressed on a logarithmic scale (base 10) and need not be
c                normalized. the abundances are modified in this subroutine
c                so that the right value of a (defined above) is obtained.
c        3. ai   is the array which will contain the atomic weights of the
c                elements considered.
c        4. data for the computation of the partition functions is read next.
c           nj(i)= the number of stages of ionization considered for element i.
c         for each stage of ionization ja the following quantities are read
c           g0(ja)=the statistical weight of the ground level,
c           nk(ja)=the number of electron configurations considered.
c         for each electron configuration jb the following quantities are read
c           xion(jb)=the ionization energy in electron volts,
c           g2(jb)=the statistical weight (2l+1)*(2j+1)
c           xl(jb)=the lowest quantum number of the asymptotic (hydrogenic) part
c                of the partition function,
c           nl(jb)=the number of terms in the (approximate) expression for the
c                'middle part' of the partition function ('qprime').
c           alfa is an array which will contain the 'statistical weights' of
c                the (approximate) expressions for the 'middle parts' of the
c                partition functions.
c           gamma is an array containing the corresponding 'excitation
c                potentials' (expressed in electron volts).
c         for the method used see traving et al., abh. hamb. viii, i (1966).
c        5. elements and stages of ionization that should be disregarded are
c           indicated by ielem(i)=0 for element i and by ion(i,j)=0 for
c           ionization stage j. otherwise indicators should be = 1.
c        6. nqfix is the number of partition functions that should be constant.
c                the values are read into the vector parco and an indication is
c                made in iqfix.  iqfix(i,j)=0 means that the partition function
c                for element i, stage of ionization j, is considered to be
c                constant.
c           nqtemp is the number of partition functions  that should be
c                pressure-independent and interpolated in t. values of four
c                temperatures (tparf, the same for all elements) and
c                corresponding partition functions (parf) are read. iqfix(i,j)=1
c                means that a pressure-independent partition function for inter-
c                polation in t is given.
c        7. ifish is a parameter for the choice of the asymptotic partition
c                function. ifish=0 means that the asymptotic part will be evalu-
c                ated following baschek et al., abh. hamb. viii,26 (1966). ifish
c                =1 means that it will be evaluated following fischel and sparks
c                astrophys. j. 164, 356 (1971).
c        8. tmolim is the higher temperature limit beyond which molecules will
c                not be considered
c
c        moreover some initiating work is done for subr. jon. unlogarithmic
c        abundances are normalized on hydrogen, xmy and sumh (defined below)
c        are computed and some further quantities are evaluated at the end.
c        a detailed printout is given if iouts is equal to one. after injon
c        has been called once, a new detailed printout is obtained if
c        injon is called with iouts greater than one.
c
c        dimensions necessary
c        abund(nel),ai(nel),alfa(lmax),anjon(nel,max(nj)),fl2(5),f1q(3),f2q(2),
c        gamma(lmax),g0(jmax),g2(kmax),h(5),iel(nel),ielem(nel),
c        ion(nel,max(nj)),iqfix(nel,max(nj)),jamem(nel,max(nj)),jbbeg(jmax),
c        jcbeg(jmax),nj(nel),nk(jmax),nl(kmax),parco(jmax),parf(4*jmax),
c        parpp(4),parpt(4),parq(4*jmax),part(nel,max(nj)),shxij(5),tparf(4),
c        xion(kmax),xiong(nel,max(nj)),xl(kmax)
c        the dimensions are lower limits
c        jmax is the total number of stages of ionization, including neutral
c             atoms.
c        kmax is the total number of electron configurations.
c        lmax is the total number of terms in the (approximate) expressions
c             for the middle part of the partition functions ('qprime'),
c             according to traving et al., cited above.
c        nel  is the number of chemical elements.
c        nj(i) is the number of stages of ionization, including the neutral
c             stage, for element i.
c
      dimension ai(16),f1q(3),f2q(2),parf(180),parpp(4),parpt(4)
      dimension jamem(16,5)
      common/ci1/fl2(5),parco(45),parq(180),shxij(5),tparf(4),
     *xiong(16,5),eev,enamn,sumh,xkbol,nj(16),iel(16),nel,summ
      common/ci9/ai
      common/ci3/alfa(300),gamma(300),g0(45),g2(80),xion(80),xl(80),
     *jbbeg(45),jcbeg(45),nk(45),nl(80),ifish
      common/ci4/ ielem(16),ion(16,5),tmolim,molh
      common/ci5/abund(16),anjon(16,5),h(5),part(16,5),dxi,f1,f2,f3,f4,
     *f5,xkhm,xmh,xmy
      common/ci6/iqfix(16,5),nqtemp,tp
      common/utput/iread,iwrit
c
        if(iouts.gt.1) go to 25
c
c        reading of the abundances and their associated quantities
c        **** 1 ****
c
      read(iread,100)nel,a,nmet
c
c        **** 2 ****
c
c
      read(iread,110)(iel(i),i=1,nel)
      read(iread,101)(abund(i),i=1,nel)
c
c        **** 3 ****
c
      read(iread,102)(ai(i),i=1,nel)
      nu=nel
      sum=0.
      summ=0.
      fakt=1.
c
c        the abundances are converted from a logarithmic scale to a direct
c        scale, and are then normalized on hydrogen. xmy=grams of stellar matter
c        /grams of hydrogen. sumh=number of other nuclei/number of hydrogen
c        nuclei.
c        summ=number of nuclei other than h, c, n, o / number of hydrogen nuclei
c
      if(nmet.le.0)go to 22
      nu=nel-nmet+1
      do1 i=nu,nel
      abund(i)=10.**abund(i)
    1 sum=abund(i)+sum
c
      fakt=sum*a/10.**abund(1)
      nu=nu-1
   22 do2 i=1,nu
      abund(i)=10.**abund(i)*fakt
    2 sum=sum+abund(i)
      xmy=0.
      aha=abund(1)
      do3 i=1,nel
      abund(i)=abund(i)/aha
      summ=summ+abund(i)
    3 xmy=xmy+abund(i)*ai(i)
      xmy=xmy/ai(1)
      sumh=sum/aha-1.
      summ=summ-abund(1)-abund(3)-abund(4)-abund(5)
c
c        **** 4 ****
c
c        reading of data for the partition functions.
c        for the symbols, see above.
c
      read(iread,103)(nj(i),i=1,nel)
      ja=1
      jb=1
      jc1=1
      do11 i=1,nel
      njp=nj(i)
      do11 j=1,njp
      jamem(i,j)=ja
      jbbeg(ja)=jb
      jcbeg(ja)=jc1
c        jbbeg and jcbeg are indicators used by function qtrav
c
      read(iread,104)g0(ja),nk(ja)
      nkp=nk(ja)
      iqfix(i,j)=2
c        iqfix(i,j)=2 means that a 'full' partition function should be
c        computed. this may be changed under **** 7 ****.
c
      ja=ja+1
      do11 k=1,nkp
      read(iread,105)xion(jb),g2(jb),xl(jb),nl(jb)
      if(k.gt.1)go to 9
      xiong(i,j)=xion(jb)
c        xiong is the ionization energy in electron volts for the ground state,
c        used in the computation of ionization equilibria in subrout.  jon.
c
    9 continue
      jc2=nl(jb)+jc1-1
      jbm=jb
      jb=jb+1
      if(nl(jbm).le.0)go to 10
      read(iread,106)(gamma(l),alfa(l),l=jc1,jc2)
   10 jc1=jc2+1
   11 continue
c
c        **** 5 ****
c
c        reading of the indicators of the elements and the stages of ionization
c        to be disregarded.
      do12 i=1,nel
      njp=nj(i)
      read(iread,107)ielem(i),(ion(i,j),j=1,njp)
   12 continue
c
c        **** 6 ****
c
c        specification of those partition functions given as constants.
c        indication in iqfix, if not marked by iqf.
      read(iread,103)nqfix
      if(nqfix.le.0)go to 15
   13 do14 i=1,nqfix
      read(iread,111)i1,j1,parcop,iqf
      ja=jamem(i1,j1)
      parco(ja)=parcop
      iqfix(i1,j1)=0
      if(iqf.ne.0)iqfix(i1,j1)=2
   14 continue
   15 continue
c
c        specification of those partition functions to be interpolated in t.
c        indication in iqfix.
      read(iread,103)nqtemp
      if(nqtemp.eq.0)go to 20
      read(iread,101)tparf
      do17 i=1,nqtemp
      read(iread,109)i1,j1,(parpp(k),k=1,4)
      iqfix(i1,j1)=1
c
c        preparation for interpolation of partition functions in t (concluded
c        in subrout. jon).
      do16 k=1,3
   16 f1q(k)=(parpp(k+1)-parpp(k))/(tparf(k+1)-tparf(k))
      do161 k=1,2
  161 f2q(k)=(f1q(k+1)-f1q(k))/(tparf(k+2)-tparf(k))
      f3q=(f2q(2)-f2q(1))/(tparf(4)-tparf(1))
      parpt(1)=parpp(1)
      parpt(2)=f1q(1)
      parpt(3)=f2q(1)
      parpt(4)=f3q
      ja=jamem(i1,j1)
      do17 k=1,4
      jk=(ja-1)*4+k
      parq(jk)=parpt(k)
   17 parf(jk)=parpp(k)
c        parq is in common/ci1/ and is used in subrout. jon. parf is just
c        used below.
c
   20 continue
c
c        **** 7, 8 ****
c
c        the parameters ifish and tmolim. initiating work for subrout. jon.
c        when molh is greater than zero the molecular formation will be computed
c        in subr. moleq (only h2 and h2+), else more complete molecular
c        formation will be evaluated in subr. mol.
c
      read(iread,100)ifish
      read(iread,4528) tmolim,molh
 4528 format(f10.0,i5)
      do21 j=1,5
      flj=j
      fl2(j)=31.321*flj*flj
   21 shxij(j)=sqrt(13.595*flj)
c
c        eev=the electron volt (expressed in terms of ergs)
c        xmh=the mass of the hydrogen atom (expressed in grams)
c        xkbol=boltzmann's constant (expressed in ergs per degree kelvin)
      eev=1.602095e-12
      xmh=1.67339e-24
      xkbol=1.38053e-16
      enamn=eev/(xmh*xmy)
      tp=0.
c        tp is the temperature at the 'preceding' call of jon.
c
c        **** print-out ****
c
      if(iouts.le.0)go to 40
   25 continue
      write(iwrit,201)
      write(iwrit,202)
      write(iwrit,203)
      do33 i=1,nel
      njp=nj(i)
      write(iwrit,204)iel(i),abund(i),ielem(i),(ion(i,j),iqfix(i,j),j=1,
     *                                                          njp)
   33 continue
      write(iwrit,207)
      write(iwrit,208)
      if(ifish.eq.1)write(iwrit,211)
      if(ifish.eq.0)write(iwrit,210)
      if(nqtemp.gt.0.or.nqfix.gt.0)write(iwrit,214)
      if(nqtemp.gt.0)write(iwrit,209)tparf
      ja=1
      do32 i=1,nel
      njp=nj(i)
      do32 j=1,njp
      jp=j-1
      if(iqfix(i,j).eq.0)write(iwrit,205)iel(i),jp,parco(ja)
      jk1=(ja-1)*4+1
      jk2=(ja-1)*4+4
      if(iqfix(i,j).eq.1)write(iwrit,206)iel(i),jp,(parf(jk),jk=jk1,jk2)
   32 ja=ja+1
      if(nqtemp.gt.0)write(iwrit,215)
      write(iwrit,212)tmolim
      if(molh.le.0) write(iwrit,216)
      if(molh.gt.0) write(iwrit,217)
      write(iwrit,213)xmy,sumh
   40 continue
      return
c
  100 format(i10,f10.4,i10)
  101 format(6f10.4)
  102 format(6f10.4)
  103 format(12i5)
  104 format(f5.0,i5)
  105 format(f6.3,f4.0,f5.1,i5)
  106 format(4(f10.3,f10.4))
  107 format(i10,5i5)
  108 format(2f10.4)
  109 format(2i5,4f10.4)
  110 format(16a3)
  111 format(2i5,f10.4,i5)
  201 format(1h1,'d a t a  f r o m  s u b r o u t i n e  i n j o n')
  202 format(1h0,30x,1hi,14x,2hii,13x,3hiii,12x,2hiv)
  203 format(1h ,'    abundance   ielem      ion   pf       ion   pf
     *   ion   pf       ion   pf')
  204 format(1h ,a2,e12.4,i5,5x,5(i5,i5,5x))
  205 format(1h ,a2,', stage of ionization=',i2,' partition function (co
     *nstant)=',f10.3)
  206 format(1h ,a2,', stage of ionization=',i2,' part.func. (t-dep.) ='
     *,4f10.3)
  207 format(1h0,'ielem and ion = 1 or 0 means element and ionization
     *stage should be considered or disregarded resp.')
  208 format(1h0,'pf=2 full part. func., =1 part. func. to be interpolat
     *ed in t, =0 constant part. func.')
  209 format(1h ,'t-dependent partition functions given for t =  ',4f10.
     *0)
  210 format(1h ,'asymptotic parts of part. func. following baschek et a
     *l. 1966')
  211 format(1h ,'asymptotic parts of part. func. following fischel and
     *sparks 1971')
  212 format(1h0,'molecules considered below t=',f7.0,' degrees kelvin')
  213 format(1h0,'xmy=grams stellar matter/grams of hydrogen=',f7.4,5x,
     *'sumh=number of other atoms/number of h=',f8.5)
  214 format(1h0,'partition functions supplied by the user')
  215 format(1h ,'if t outside range for interpolations detailed part. f
     *unctions are computed')
  216 format(1h0,'molecules considered:  h2, h2+, h2o, oh, ch, co, cn, c
     *2, n2, o2, no, nh')
  217 format(1h0,'molecules considered:  h2, h2+')
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine jon(t,pe,iepro,pg,ro,e,ioutr)
      parameter (mt=60)
c
c
c        this routine computes ionization equilibria for a given temperature (t,
c        expressed in degrees kelvin) and a given electron pressure (pe, in
c        dynes per cm2). the fractions of ionization are put in the anjon vector
c        and the partition functions are put in part. if iepro is greater than
c        zero, the gas pressure (pg,in dynes per cm2), density (ro, in grams
c        per cm3) and inner energy (e, in ergs per gram) are also evaluated.
c        n o t e . radiation pressure is not included in e.
c
c        the energies of ionization are reduced by dxi, following baschek et
c        al., abh. hamb. viii, 26 eq. (10). these reductions are also made in
c        the computation of e.
c        the energy of dissociation for h- has been reduced by 2*dxi, following
c        tarafdar and vardya, third harv. smiths. conf., page 143. the formation
c        of molecules is considered for t less than tmolim.
c
c        if ioutr is greater than zero, a detailed print-out will be given.
c
c
c        the function  qtrav and subrout. moleq are called.
c        they call qas and molfys respectively.
c
c        dimensions necessary
c        a(5),dq(4),f(max(nj)),pfak(max(nj)),rfak(jmax)
c        dimensions of arrays in commons /ci1/,/ci4/,/ci5/ and /ci6/ are
c        commented on in subrout. injon.
c        jmax is the total number of stages of ionization, including neutral
c             atoms.
c        nj(i) is the number of stages of ionization, including the neutral
c             stage, for element i.
c
c
      dimension dq(4),f(5),pfak(5),rfak(45)
      dimension presmo(30)
      common/ci1/fl2(5),parco(45),parq(180),shxij(5),tparf(4),
     *xiong(16,5),eev,enamn,sumh,xkbol,nj(16),iel(16),nel,summ
      common/ci4/ ielem(16),ion(16,5),tmolim,molh
      common/ci5/abund(16),anjon(16,5),h(5),part(16,5),dxi,f1,f2,f3,f4,
     *f5,xkhm,xmh,xmy
      common/ci6/iqfix(16,5),nqtemp,tp
      common/ci7/a(5),pfish,itp
      common/utput/iread,iwrit
      common/rabell/xxrho(mt),xyrho
      common/ci8/yypg,yyrho,yye
      common/cmol1/eh,fe,fh,fhe,fc,fce,fn,fne,fo,foe,fk,fke,fs,fse
      common/cmol2/nmol,pk(30)
      common/carc3/f1p,f3p,f4p,f5p,hnic,presmo
c        statement function for 10.**
      exp10(x)=exp(2.302585*x)
c
      itp=1
c
c        is t=the temperature of the preceding call
C DVukadinovic      if(abs((t-tp)/t).lt.1.e-8)go to 53
   51 itp=0
c
c        some quantities, only dependent on t
      teta=5040./t
      teta25=1.202e9/(teta*teta*sqrt(teta))
      do52 j=1,5
   52 a(j)=fl2(j)*teta
c        a=alfa(baschek et al., cited above)
c
      if(nqtemp.eq.0)go to 53
c
c        preparation for interpolation of partition functions in t
      dq(1)=1.
      dq(2)=t-tparf(1)
      dq(3)=dq(2)*(t-tparf(2))
      dq(4)=      dq(3)*(t-tparf(3))
c
c        some quantities also dependent on pe
c        the pfak factors are used in the saha equation. h(j) is the
c        quantum number of the cut of the partition functions (according
c        to baschek et al., cited above) for j-1 times ionized atoms. h is
c        used in qas.
c
c        xnel= the electron (number) density (per cm3)
c        pfish= p(fischel and sparks, astrophys. j. 164, 359 (1971)) is used in
c        function qas.
c
   53 dxi=4.98e-4*teta*sqrt(pe)
      dum=teta25/pe
      dim=exp10(dxi*teta)
      pfak(1)=dim*dum
      sqdxi=1./sqrt(dxi)
      h(1)=shxij(1)*sqdxi
      do54 j=2,5
      pfak(j)=pfak(j-1)*dim
   54 h(j)=shxij(j)*sqdxi
      xnel=pe/(xkbol*t)
      pfish=4.2e3/xnel**0.166666667
c
c        partition functions and ionization equilibria
c
      xnecno=0.
      xnenh=0.
      ejon=0.
      ja=1
c
c        beginning of loop over elements ('the i-loop').
      do24 i=1,nel
      njp=nj(i)
c
c        should element no. i be considered
      if(ielem(i).gt.0)go to 9
      do 55 j=1,njp
      anjon(i,j)=0.
      part(i,j)=0.
   55 continue
      go to 23
c
c        beginning of loop over stages of ionization ('the j-loop')
    9 do19 j=1,njp
      jm1=j-1
c
c        should stage of ionization no. j be considered
      if(ion(i,j).gt.0)go to 10
      anjon(i,j)=0.
      part(i,j)=0.
      go to 18
c
c        which kind of partition function should be computed
c
   10 if(iqfix(i,j)-1)14,11,13
   11 if(t.lt.tparf(1).or.t.gt.tparf(4))go to 13
      partp=part(i,j)
      if(itp.gt.0)go to 15
c
c        partition functions to be interpolated in t
      jparf=(ja-1)*4+1
      partp=0.
      do12 ip=1,4
      partp=partp+parq(jparf)*dq(ip)
   12 jparf=jparf+1
      go to 15
c
c        partition functions following traving et al., abh. hamb. viii, 1 (1966)
   13 partp=qtrav(teta,h(j),j,ja)
      go to 15
c
c        the partition function is constant
   14 partp=parco(ja)
   15 part(i,j)=partp
c
c        ionization equilibria and total number of electrons
c
      if(j.le.1)go to 19
      if(itp.gt.0)go to 17
      rfak(ja)=exp10(-xiong(i,jm1)*teta)
   17 f(jm1)=pfak(jm1)*rfak(ja)*partp/part(i,j-1)
      go to 19
   18 if(j.gt.1)f(jm1)=0.
   19 ja=ja+1
c        end of 'the j-loop'
c
      fil=1.
      do20 j=2,njp
      ll=njp-j+1
   20 fil=1.+f(ll)*fil
      anjon(i,1)=1./fil
      xnen=0.
      do21 j=2,njp
      jm1=j-1
      anjon(i,j)=anjon(i,jm1)*f(jm1)
      if(i.le.1)go to 24
      fljm1=jm1
   21 xnen=anjon(i,j)*fljm1+xnen
      if(i.gt.2.and.i.lt.6) xnecno=xnecno+xnen*abund(i)
      xnenh=xnen*abund(i)+xnenh
c        xnenh=number of electrons from elements other than hydrogen (q in
c        mihalas, meth. comp. phys. 7, 1 (1967), eq. (35))
c        xnecno=number of electrons from elements other than h, c, n, o
c
c
c        computation of the energy of ionization (ejon). hydrogen is not
c        included.
c
      xerg=0.
c        xerg= the energy of ionization per atom (in electron volts)
c
      do22 j=2,njp
      jm1=j-1
      fljm1=jm1
   22 xerg=anjon(i,j)*(xiong(i,jm1)-dxi*fljm1)+xerg
      ejon=xerg*abund(i)+ejon
      go to 24
   23 ja=ja+njp
   24 continue
c        end of 'the i-loop'
c
c
      xnecno=xnenh-xnecno
      tp=t
      if(iepro.le.0)go to 71
c
c        comp. of pressure, density and inner energy
c
      xih=xiong(1,1)-dxi
      xihm=0.747-2.*dxi
c        xih and xihm are the energies of ionization for h and h- respectively
c        (in electron volts).
c
      xkhm=teta25*2.*exp10(-teta*xihm)
c        xkhm = the 'dissociation constant' for h-.
c
      hjonh=anjon(1,2)/anjon(1,1)
      if(t.gt.tmolim)go to 42
      if(molh.le.0) goto 45
c
c        formation of molecules. only h2 and h2+
   41 call moleq(t,pe,hjonh,xih,xkhm,xihm,xnenh,f1,f2,f3,f4,f5,fe,fsum,
     *   eh)
      fepe=pe/fe
      f1p=f1*fepe
      f3p=f3*fepe
      f4p=f4*fepe
      f5p=f5*fepe
      go to 43
c        formation of molecules composed of h,c,n,o
   45 if(anjon(3,1).le.0..or.anjon(4,1).le.0..or.anjon(5,1).le.0.)
     * goto 41
      hjonc=anjon(3,2)/anjon(3,1)
      hjonn=anjon(4,2)/anjon(4,1)
      hjono=anjon(5,2)/anjon(5,1)
      abuc=abund(3)/abund(1)
      abun=abund(4)/abund(1)
      abuo=abund(5)/abund(1)
      call mol(t,pe,hjonh,hjonc,hjonn,hjono,abuc,abuo,abun,xih,xkhm,xihm
     *,xnecno,f1,f2,f3,f4,f5)
      sumpmo=0.
      presmo(1)=fhe*pk(1)
      presmo(2)=fhe*fhe*pk(2)
      presmo(3)=fhe*fhe*hjonh*pk(3)
      presmo(4)=fhe*fhe*foe*pk(4)
      presmo(5)=fhe*foe*pk(5)
      presmo(6)=fhe*fce*pk(6)
      presmo(7)=fce*foe*pk(7)
      presmo(8)=fce*fne*pk(8)
      presmo(9)=fce*fce*pk(9)
      presmo(10)=fne*fne*pk(10)
      presmo(11)=foe*foe*pk(11)
      presmo(12)=fne*foe*pk(12)
      presmo(13)=fne*fhe*pk(13)
      presmo(14)=fce*fce*fhe*fhe*pk(14)
      presmo(15)=fhe*fce*fne*pk(15)
      presmo(16)=fce*fce*fhe*pk(16)
      presmo(18)=fhe*fse*pk(18)
      presmo(19)=fke*fhe*pk(19)
      presmo(20)=fce*fce*fce*fhe*pk(20)
      presmo(21)=fce*fce*fce*pk(21)
      presmo(22)=fce*fse*pk(22)
      presmo(23)=fke*fce*pk(23)
      presmo(24)=fke*fce*fce*pk(24)
      presmo(25)=fne*fse*pk(25)
      presmo(26)=fke*fne*pk(26)
      presmo(27)=fke*foe*pk(27)
      presmo(28)=fse*foe*pk(28)
      presmo(29)=fse*fse*pk(29)
      presmo(30)=fke*fse*pk(30)
      do 30 i=1,nmol
      presmo(i)=presmo(i)*pe
   30 sumpmo=sumpmo+presmo(i)
      sumpa=pe*(fhe+fce+fne+foe)
      sumpi=pe*(fhe*hjonh+fce*hjonc+fne*hjonn+foe*hjono)
      hnic=pe*fhe
      hpnic=hnic*hjonh
      pg=pe+sumpmo+sumpa+sumpi+pe*summ/fe
      goto 46
c
c        no molecules
   42 f2=anjon(1,2)
      fe=xnenh+f2
      f1=anjon(1,1)
      f3=0.
      f4=0.
      f5=0.
      fsum=1.
      eh=-xih*f1
C problem sa fsum fe
   43 pg=pe*(1.+(fsum+sumh)/fe)
   46 ro=pe*xmy*(xmh/xkbol)/(fe*t)
       xyrho=ro
      e=1.5*pg/ro+(eh+ejon)*enamn
      yypg=pg
      yyrho=ro
      yye=e
c
      if(ioutr.le.0)go to 71
c
c        **** print-out ****
c
      write(iwrit,204)t,pe,pg,ro,e
      write(iwrit,201)
      write(iwrit,202)
      do93 i=1,nel
      njp=nj(i)
      write(iwrit,203)iel(i),abund(i),(anjon(i,j),j=1,njp)
      write(iwrit,207)(part(i,j),j=1,njp)
   93 continue
      if(t.gt.tmolim)go to 44
      if(molh.le.0)goto 47
      write(iwrit,205)f1p,f3p,f5p,f4p
      go to 71
   47 continue
      write(iwrit,208)hnic,(presmo(i),i=1,13)
      goto 71
   44 write(iwrit,206)
c
   71 continue
      return
  201 format(1h0,'element  abundance  ionization fractions',17x,
     *'partition functions')
  202 format(1h ,23x,1hi,7x,2hii,6x,3hiii,5x,2hiv,12x,1hi,9x,2hii,8x,
     *3hiii,7x,2hiv)
  203 format(6h      ,a2,e12.4,4f8.4)
  204 format(3h0t=,f7.1,5x,3hpe=,e12.4,5x,3hpg=,e12.4,5x,3hro=,e12.4,
     *5x,2he=,e12.4)
  205 format(1h0,'partial pressures'/4x,'h',8x,'h-',7x,'h2',7x,'h2+'/1x,
     *4(1pe9.2))
  206 format(1h0,'no molecules considered')
  207 format(1h+,56x,4e10.3)
  208 format(1h0,'partial pressures'/4x,'h',8x,'h-',7x,'h2',7x,'h2+',6x,
     *'h2o',6x,'oh',7x,'ch',7x,'co',7x,'cn',7x,'c2',7x,'n2',7x,'o2',7x,
     *'no',7x,'nh'/1x,14(1pe9.2))
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine mol(t,pe,g2,gc,gn,go,abuc,abuo,abun,xih,xkhm,xihm,xnen,
     *f1,f2,f3,f4,f5)
c
c        this routine computes dissociation equilibria for the molecules h2,h2+,
c        h2o,oh,ch,co,cn,c2,o2,n2,nh and no with h,h-,h+,c,c+,o,o+,n,n+ con-
c        sidered, using a newton-raphson scheme. some features come from the
c        monster and from mihalas, meth. comp. phys. 7,1.
c
c        g2=n(hii)/n(hi), gc=n(cii)/n(ci) etc.
c        abuc= the number of carbon nuclei per hydrogen nucleus, abuo and abun
c        are the corresponding values for oxygen and nitrogen.
c        xih = the ionization energy of hydrogen
c        xihm= the 'dissociation energy' of h-
c        xkhm= the 'dissociation constant' of h-
c        xnen= the number of electrons per unit volume from elements other than
c        hydrogen, carbon, oxygen and nitrogen.
c
c        the subscript in akd(i) etc. has the following meaning
c        i=1 h-, 2 h2, 3 h2+, 4 h2o, 5 oh, 6 ch, 7 co, 8 cn, 9 c2, 10 n2,
c         11 o2, 12 no, 13 nh
c
c        this routine calls molmat and ainv2
c        the data for computing the dissociation constants are from tsuji
c        (astron. astrophys.  23,411 (1973))
c
c        the routine gives fh, fc, fo, fn, fe. fh=p(hi)/ph, fc=p(ci)/ph etc.,
c        where ph=nh*kt (nh is the number of hydrogen nuclei per cm3).
c
      double precision aka(12),ak1(12),ak2(12),ak3(12),ak4(12),akd(13),
     *pk(13),th,wkh,wkc,wkn,wko,fh,fc,
     *fn,fo,fe,cam,root,qn,qnn,a(6,6),f(6),d(6),fhp,fcp,fop,fnp,fep
     *,s,r,b,bk
      dimension b1(5),b2(5),dis(10)
      common /cmol1/eh,ffe,ffh,fhe,ffc,fce,ffn,fne,ffo,foe,ffk,fke
     & ,ffs,fse
      common /cmol2/nmol,ppk(30)
      data b1/2.6757,1.4772,0.60602,0.12427,0.00975/,
     *b2/2.9216,2.0036,1.7231,0.82685,0.15253/,
     *dis/9.50,4.38,3.47,11.11,7.90,6.12,9.76,5.12,6.51,3.21/
      data aka/12.739d0,11.206998d0,25.420d0,12.371d0,12.135d0,13.820d0,
     *12.805d0,12.804d0,13.590d0,13.228d0,12.831d0,12.033d0/
      data ak1/5.1172d0,2.7942767d0,10.522d0,5.0578d0,4.0760d0,11.795d0,
     *8.2793d0,6.5178d0,10.585d0,5.5181d0,7.1964d0,3.8435d0/
      data ak2/1.2572d-1,-7.9196803d-2,1.6939d-1,1.3822d-1,1.2768d-1,
     *1.7217d-1,6.4162d-2,9.7719d-2,2.2067d-1,6.9935d-2,1.7349d-1,
     &1.3629d-1/
      data ak3/1.4149d-2,-2.4790744d-2,1.8368d-2,1.6547d-2,1.5473d-2,
     *2.2888d-2,7.3627d-3,1.2739d-2,2.9997d-2,8.1511d-3,2.3065d-2,
     *1.6643d-2/
      data ak4/6.3021d-4,0.d0,8.1730d-4,7.7224d-4,7.2661d-4,1.1349d-3,
     *3.4666d-4,6.2603d-4,1.4993d-3,3.7970d-4,1.1380d-3,7.8691d-4/
c
c        computation of dissociation constants (akd) and pe/k(ab) (pk).
      th=5040.d0/t
      akd(1)=xkhm
      pk(1)=pe/xkhm
      do 4 j=2,13
      m=j-1
      akd(j)=10.d0**(aka(m)-(ak1(m)-(ak2(m)-(ak3(m)-ak4(m)*th)*th)*th)*
     *th)
    4 pk(j)=pe/akd(j)
      pk(4)=pe*pk(4)
c
c        computation of starting values for fh,fc etc.
c
      wkh=1.d0+g2+pk(1)
      wkc=1.d0+gc
      wko=1.d0+go
      wkn=1.d0+gn
      fe=xnen
      cam=fe*(1.d0+g2+pk(1))/(2.d0*pk(2))
      root=dsqrt(dabs(cam*cam+fe/pk(2)))
      fh=-cam+root
      fe=fh*(g2-pk(1))+xnen
      if(fe.gt.0.d0) goto 60
      r=pk(2)/pk(1)/pk(1)
      cam=(wkh*xnen/pk(1)-1.d0-2.d0*xnen*r)
      s=r-wkh/pk(1)
      cam=cam/2.d0/s
      root=dsqrt(cam*cam-r*xnen*xnen/s)
      fe=-cam-root
      if(fe.lt.0.) fe=-cam+root
      fh=(xnen-fe)/pk(1)
   60 continue
      r=1.d0+go+pk(4)*fh*fh/fe/fe+pk(5)*fh/fe
      s=1.d0+gc+pk(6)*fh/fe
      b=pk(7)/fe
      cam=(b*(abuc-abuo)+r*s)/(2.d0*r*b)
      bk=abuo/r*s/b
      fo=-cam+dsqrt(cam*cam+bk)
      fc=abuc/(s+b*fo)
   61 qn=1.d0+(fh*pk(13)+fc*pk(8))/fe
      qnn=fe/4.d0/pk(10)
      cam=qn*qnn
      root=dsqrt(dabs(cam*cam+2.d0*abun*qnn))
      fn=-cam+root
      if(fn.lt.0.d0) fn=0.d0
c
c        computation of fh,fc, etc. using all relevant molecules, atoms and ions
c        diff gives the approx. accuracy to which fh etc. have to converge
c        before the iterations are stopped.
c
      diff=0.001
      m=5
      do 2 j=1,100
      fhp=fh
      fcp=fc
      fop=fo
      fnp=fn
      fep=fe
      nico=j
      call molmat(pk,g2,gc,gn,go,abuc,abun,abuo,fh,fc,fn,fo,fe,xnen,f,a)
      call ainv2(a,m)
      do 30 l=1,m
      qn=0.d0
      do 3 ll=1,m
    3 qn=qn+a(l,ll)*f(ll)
      d(l)=-qn
   30 continue
      fh=fh+d(1)
      fc=fc+d(2)
      fo=fo+d(3)
      fn=fn+d(4)
      fe=fe+d(5)
      chech=dabs(1.d0-fhp/fh)
      checc=dabs(1.d0-fcp/fc)
      checo=dabs(1.d0-fop/fo)
      checn=dabs(1.d0-fnp/fn)
      chece=dabs(1.d0-fep/fe)
      if(nico.le.2) goto 2
      if(chech.lt.diff.and.checc.lt.diff.and.checo.lt.diff.and.checn.lt.
     *diff.and.chece.lt.diff) goto 23
    2 continue
      print 305,fh,fc,fo,fn,fe,chech,checc,checo,checn,chece
  305 format(' ***mol*** the desired accuracy was not achieved after 100
     * iterations.  latest values and differences were'/2x,5(1pe11.4),10
     *x,5e10.3)
   23 continue
c        computation of the inner energy. deh2 and deh2p are the sum of
c        dissociation, rotation and vibration energies (in ev per molecule) for
c        h2 and h2+. dis(i) is the dissociation energy for the molecule (i+3)
c        in the list of molecules (values are from tsuji). for these molecules
c        the rotation and vibration energies are neglected.
   98 teta=th
      deh2=(b1(1)-(b1(2)-(b1(3)-(b1(4)-b1(5)*teta)*teta)*teta)*teta)*
     *8.617e-5*t-4.476
      deh2p=(b2(1)-(b2(2)-(b2(3)-(b2(4)-b2(5)*teta)*teta)*teta)*teta)*
     *8.617e-5*t-2.648
      fhe=fh/fe
      fce=fc/fe
      foe=fo/fe
      fne=fn/fe
      eh2=(-2.*xih+deh2)*fhe*fh*pk(2)
      eh2p=(deh2p-xih)*fhe*fh*g2*pk(3)
      ehm=-(xihm+xih)*fh*pk(1)
      ehj=-xih*fh
      eh2o=-(2.*xih+dis(1))*fhe*fh*fo*pk(4)
      eoh=-(xih+dis(2))*foe*fh*pk(5)
      ech=-(xih+dis(3))*fhe*fc*pk(6)
      eco=-dis(4)*fce*fo*pk(7)
      ecn=-dis(5)*fce*fn*pk(8)
      ec2=-dis(6)*fce*fc*pk(9)
      en2=-dis(7)*fne*fn*pk(10)
      eo2=-dis(8)*foe*fo*pk(11)
      eno=-dis(9)*fne*fo*pk(12)
      enh=-(dis(10)+xih)*fne*fh*pk(13)
      eh=eh2+eh2p+ehm+ehj+eh2o+eoh+ech+eco+ecn+ec2+en2+eo2+eno+enh
      ffh=fh
      ffc=fc
      ffn=fn
      ffo=fo
      ffe=fe
      nmol=13
      do 72 i=1,nmol
   72 ppk(i)=pk(i)
      f1=fh
      f2=g2*fh
      f3=fh*pk(1)
      f4=fh*fhe*g2*pk(3)
      f5=fh*fhe*pk(2)
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine moleq(t,pe,g2,xih,xkhm,xihm,xnenh,f1,f2,f3,f4,f5,fe,
     *                 fsum,eh)
c
c        this routine computes dissociation equilibria for the molecules
c        h2 and h2+ with h+, h and h- considered. it mainly follows mihalas,
c        meth. comp. phys. 7, 1 (1967).
c
c        the inner energy of the hydrogen gas, eh, is also evaluated.
c
c        xih=the ionization energy of hydrogen
c        xkhm=the 'dissociation constant' of h-
c        xihm=the 'dissociation energy' of h-.
c        xnenh=the number of electrons per unit volume from elements other than
c        hydrogen (q in mihalas's article)
c        g2,f1,f2 etc. see ref.
c        double precision necessary for relatively low pressures.
c
c
      common/utput/iread,iwrit
      double precision g3,g4,g5,a,e,b,c,d,c1,c2,c3,cam,f1d,f2d,f3d,f4d,
     *f5d,fed,fsumd,root
c
c        call molfys for physical data
      call molfys(t,xkh2,xkh2p,deh2,deh2p)
c
c        calculation of the equilibrium
      g3=pe/xkhm
      g4=pe/xkh2p
      g5=pe/xkh2
      a=1.+g2+g3
      e=g2*g4/g5
      b=2.*(1.+e)
      c=g5
      d=g2-g3
      c1=c*b*b+a*d*b-e*a*a
      c2=2.*a*e-d*b+a*b*xnenh
      c3=-(e+b*xnenh)
      cam=c2/(2.*c1)
      root=dsqrt(cam*cam-c3/c1)
      f1d=-cam+root
      if(f1d.gt.1.d0)f1d=-cam-root
      f5d=(1.d0-a*f1d)/b
      f4d=e*f5d
      f3d=g3*f1d
      f2d=g2*f1d
      fed=f2d-f3d+f4d+xnenh
      fsumd=f1d+f2d+f3d+f4d+f5d
      f1=f1d
      f2=f2d
      f3=f3d
      f4=f4d
      f5=f5d
      fe=fed
      fsum=fsumd
c
c        calculation of the energies
      eh2=(-2.*xih+deh2)*f5
      eh2p=(-xih+deh2p)*f4
      ehm=-(xihm+xih)*f3
      ehj=-xih*f1
      eh=ehj+ehm+eh2+eh2p
    1 continue
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine molfys(t,xkh2,xkh2p,deh2,deh2p)
c
c        this routine gives dissociation constants xkh2 (=n(h i)*n(h i)/n(h2) )
c        and xkh2p (=n(h i)*n(h ii)/n(h2+)), expressed in number per cm3, and
c        the sum of dissociation, rotation and vibration energies, deh2 and
c        deh2p for h2 and h2+, respectively (expressed in ergs per molecule).
c        the data are from vardya, m.n.r.a.s. 129, 205 (1965) and earlier
c        references. the dissociation constant for h2 is from tsuji,
c        astron. astrophys. 1973.
c
      dimension a1(5),a2(4),b1(5),b2(5),te(5)
      data a1/12.739,-5.1172,1.2572e-1,-1.4149e-2,6.3021e-4/,
     *a2/11.20699 ,-2.794276 ,-0.079196   ,0.024790   /,
     *b1/2.6757,-1.4772,0.60602,-0.12427,0.009750/,
     *b2/2.9216,-2.0036,1.7231,-0.82685,0.15253/
      tex=5040./t
      te(1)=1.
      do1 k=1,4
    1 te(k+1)=te(k)*tex
      xkh2=0.
      xkh2p=0.
      deh2=0.
      deh2p=0.
      do2 k=1,4
      xkh2=a1(k)*te(k)+xkh2
      xkh2p=a2(k)*te(k)+xkh2p
      deh2=b1(k)*te(k)+deh2
    2 deh2p=b2(k)*te(k)+deh2p
      xkh2=a1(5)*te(5)+xkh2
      deh2=(b1(5)*te(5)+deh2)*8.617e-5*t-4.476
      deh2p=(b2(5)*te(5)+deh2p)*8.617e-5*t-2.648
      xkh2=10.**xkh2
      xkh2p=10.**xkh2p
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine molmat(pk,gh,gc,gn,go,ac,an,ao,fh,fc,fn,fo,fe,xnen,f,a)
c
c        this routine computes the elements of matrix a and vector f in the
c        newton-raphson procedure for determining the molecular equilibrium.
c        it is called by subr. mol.
c
      double precision pk(13),f(6),fh,fc,fn,fo,fe,
     *h,hh,c,o,xn,a(6,6),fhe,fce,fne,foe
      fhe=fh/fe
      fce=fc/fe
      fne=fn/fe
      foe=fo/fe
      h=1.d0+gh+pk(1)+pk(5)*foe+pk(6)*fce+pk(13)*fne
      hh=2.d0*fhe*(pk(2)+gh*pk(3)+pk(4)*foe)
      c=1.d0+gc+pk(7)*foe+pk(6)*fhe+pk(8)*fne
      o=1.d0+go+pk(4)*fhe*fhe+pk(5)*fhe+pk(7)*fce+pk(12)*fne
      xn=1.d0+gn+pk(13)*fhe+pk(8)*fce+pk(12)*foe
      f(1)=fh*(h+hh)-1.d0
      f(2)=fc*(c+2.d0*pk(9)*fce)-ac
      f(3)=fo*(o+2.d0*pk(11)*foe)-ao
      f(4)=fn*(xn+2.d0*pk(10)*fne)-an
      f(5)=fh*(gh+pk(3)*gh*fhe-pk(1))+gc*fc+gn*fn+go*fo-fe+xnen
      a(1,1)=h+2.d0*hh
      a(1,2)=fhe*pk(6)
      a(1,3)=fhe*(2.d0*pk(4)*fhe+pk(5))
      a(1,4)=fhe*pk(13)
      a(1,5)=-fhe*(pk(5)*foe+pk(6)*fce+pk(13)*fne+2.d0*fhe*(pk(2)+gh*pk(
     *3)+2.d0*pk(4)*foe))
      a(2,1)=fce*pk(6)
      a(2,2)=c+4.d0*pk(9)*fce
      a(2,3)=fce*pk(7)
      a(2,4)=fce*pk(8)
      a(2,5)=-fce*(foe*pk(7)+fhe*pk(6)+fne*pk(8)+2.d0*pk(9)*fce)
      a(3,1)=foe*(2.d0*pk(4)*fhe+pk(5))
      a(3,2)=foe*pk(7)
      a(3,3)=o+4.d0*pk(11)*foe
      a(3,4)=foe*pk(12)
      a(3,5)=-foe*(fhe*pk(5)+fce*pk(7)+fne*pk(12)+2.d0*(foe*pk(11)+pk(4)
     **fhe*fhe))
      a(4,1)=fne*pk(13)
      a(4,2)=fne*pk(8)
      a(4,3)=fne*pk(12)
      a(4,4)=xn+4.d0*fne*pk(10)
      a(4,5)=-fne*(fhe*pk(13)+fce*pk(8)+foe*pk(12)+2.d0*pk(10)*fne)
      a(5,1)=gh*(1.d0+2.d0*pk(3)*fhe)-pk(1)
      a(5,2)=gc
      a(5,3)=go
      a(5,4)=gn
      a(5,5)=-gh*pk(3)*fhe*fhe-1.d0
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function qas(h,xl,a,z,pfish,ifish)
c
c        this routine computes the asymptotic parts of the partition
c        functions following
c           baschek et al., abh. hamb. viii, 26 (1966) if ifish = 0
c           fischel and sparks, ap. j. 164, 359 (1971) if ifish = 1
c           (approximating the zeta functions by integrals).
c
c        xl=quantum number for the first level of the asymptotic part
c        h=quantum number of the cut (for ifish=0)
c        a=dz(fischel and sparks)=alfa(baschek et al.)
c        pfish=p(fischel and sparks), only necessary if ifish = 1
c
c
      common/utput/iread,iwrit
c
c        which type
      qas=0.0
      if (ifish.lt.0) return
      if(ifish.gt.0)go to 1
c
c        baschek et al.
      qas=0.333333*(h*(h+1.)*(h+0.5)-xl*(xl+1.)*(xl+0.5)) +
     *                       a*(h-xl)+0.5*a*a*(h-xl)/(h*xl)
      return
c
c        fischel and sparks
    1 p=pfish*z
c
c        fischel and sparks, eq. (26)
      p2=p*p
      p3=p2*p
      if(p.le.xl)go to 2
      xlm1=xl-1.
      r2=xlm1*xlm1
      r3=r2*xlm1
      qas=1.3333333*p3+0.5*p2+0.16666667*p+1.33333333*a*p-0.4*a*a/p-
     *0.33333333*r3-0.5*r2-0.16666667*xlm1-a*xlm1+0.5*a*a/xl
      return
c
c        fischel and sparks, eq. (27)
    2 axl2=a/(xl*xl)
      qas=p3*p/xl*(1.+axl2*(0.33333333+0.1*axl2))
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function qtrav(teta,hp,j,ja)
c
c        here the partition functions according to traving et al., abh. hamb.
c        sternw. viii, 1 (1966) are computed. the symbols are given
c        in the comments at the beginning of subrout. injon.
c        function qas is called.
c
c        dimensions necessary
c        a(5),asde(kmax),h(5),qprim(kmax)
c        kmax is the total number of electron configurations.
c        dimensions of arrays in common /ci3/ are commented on in subroutine
c        injon.
c
      dimension asde(80),h(5),qprim(80)
      common/ci3/alfa(300),gamma(300),g0(45),g2(80),xion(80),xl(80),
     *jbbeg(45),jcbeg(45),nk(45),nl(80),ifish
      common/ci7/a(5),pfish,itp
c
c
c        statement function for 10.**
      exp10(x)=exp(2.302585*x)
c
      flj=j
      jb=jbbeg(ja)
      jc1=jcbeg(ja)
      nkp=nk(ja)
      qsum=0.
c
c        we start the loop over different electron configurations ('the k-loop'
      do5 k=1,nkp
      jc2=nl(jb)+jc1-1
c
c        is teta=preceding teta
      if(itp.gt.0)go to 4
      pra=xion(jb)*teta
      if(pra.lt.12.)go to 1
      asde(jb)=0.
      go to 2
    1 asde(jb)=g2(jb)*exp10(-pra)
c
    2 qprim(jb)=0.
      if(nl(jb).le.0)go to 4
      do3 l=jc1,jc2
      pre=gamma(l)*teta
      if(pre.gt.12.)go to 3
      qprim(jb)=qprim(jb)+alfa(l)*exp10(-pre)
    3 continue
    4 jc1=jc2+1
      qsum=qprim(jb)+asde(jb)*qas(hp,xl(jb),a(j),flj,pfish,ifish)
     *             +qsum
    5 jb=jb+1
c        end of 'the k-loop'
      qtrav=g0(ja)+qsum
c
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine tabs(nt,t)
      parameter (mt=60,mkomp=30,mkompr=15)
      parameter (ifadim=(mkomp-mkompr)*mt)
      parameter (kfadim=mkompr*mt+(mkomp-mkompr)*mt*3)
c
c        this routine computes  factors for interpolation in t (teta if
c        iteta(komp) is greater than zero) in the abkof table, initiated by
c        subrout. inabs. concerning the other control integers, see inabs.
c        the resulting factors are put in afak. the number of factors for
c        the component komp at temperature t(ntp) is given in
c        nofak((nkomp-kompr)*(ntp-1)+komp-kompr). here kompr is the number
c        of components with t-indep. coefficients. nofak=0 means that the
c        absorption coefficient should be =0. nplats (index as for nofak)
c        gives the array index of the temperature point at which the
c        interpolation in abkof should start.
c
c        nt=number of temperatures
c        t= array of temperatures
c
c        dimensions necessary
c        afak(kfadim),nofak(ifadim),nplats(ifadim),t(1)
c        the dimensions are lower limits. dimensions of arrays in commons /ca1/
c        and /ca2/ are commented on in subrout. inabs.
c        ifadim should be at least =(nkomp-kompr)*nt, where nkomp is the number
c               of components, kompr the number of temperature-independent
c               components and nt the number of temperature points (in the para
c               meter list).
c        kfadim should be at least =kompr*nt+(nkomp-kompr)*nt*num, where num is
c               between 2 and 3 and dependent on the type of temperature
c               interpolation used.
c
c
      dimension t(nt)
      common/utput/iread,iwrit
      common/ca1/delt(30,2),tbot(30,2),idel(30),isvit(30),iteta(30),
     *kvadt(30),maxet(30),minet(30),ntm(30,2),nextt,nutzt
      common/ca2/abkof(4000),kompla(600),kompr,komps,nkomp
      common/ca4/afak(kfadim),nofak(ifadim),nplats(ifadim)
c
      ifak=1
      kfak=1
      nsvit=1
c        this is just a dummy statement to give nsvit a formal value
c
      do81 ntp=1,nt
      tp=t(ntp)
      kfak=kfak+kompr
      do81 komp=komps,nkomp
      if(isvit(komp).gt.0)go to (51,61,70),nsvit
      if(iteta(komp).le.0)go to 2
    1 ts=5040./t(ntp)
      go to 3
    2 ts=t(ntp)
c
c        searching
    3 if((ts-tbot(komp,1)).ge.0.)go to 10
      if(minet(komp).le.0)go to 70
c
c        extrapolation downwards
      if(nextt.gt.0)write(iwrit,200)ts,komp
      inta=1
      ap=(ts-tbot(komp,1))/delt(komp,1)
      ip=0
      go to 60
c
c        searching continues
   10 intap=1
      idp=idel(komp)
      do11 i=1,idp
      ap=(ts-tbot(komp,i))/delt(komp,i)
      ip=int(ap)
      inta=ip+intap
      inap=ntm(komp,i)-1+intap
      if(inta.le.inap) go to 20
   11 intap=inap+1
      if(maxet(komp).le.0)go to 70
c
c        extrapolation downwards
      if(nextt.gt.0)write(iwrit,200)ts,komp
      inta=inap
      ip=ntm(komp,idp)-1
      go to 60
c
   20 if(kvadt(komp).le.0)go to 60
c
c        quadratic interpolation
   21 if(inta.lt.inap)go to 50
      inta=inta-1
      ip=ip-1
c
   50 dxx1=ap-float(ip)
      dxx2=dxx1-1.
      dxx3=dxx1-2.
      a1=dxx2*dxx3*0.5
      a2=-dxx1*dxx3
      a3=dxx1*dxx2*0.5
   51 afak(kfak)=a1
      afak(kfak+1)=a2
      afak(kfak+2)=a3
      nplats(ifak)=inta
      nofak(ifak)=3
      ifak=ifak+1
      kfak=kfak+3
      nsvit=1
      go to 80
c
c        linear inter/extrapolation
   60 a2=ap-float(ip)
      a1=1.-a2
   61 afak(kfak)=a1
      afak(kfak+1)=a2
      nplats(ifak)=inta
      nofak(ifak)=2
      ifak=ifak+1
      kfak=kfak+2
      nsvit=2
      go to 80
c
c        outside table. abs.coeff. should be = 0
   70 if(nutzt.gt.0)write(iwrit,201)ts,komp
      nofak(ifak)=0
      ifak=ifak+1
      nsvit=3
c
   80 continue
      if(kfak.gt.kfadim)go to 90
      if(ifak.gt.ifadim)go to 91
   81 continue
c
      go to 92
   90 write(iwrit,202)kfak,kfadim,nt
      stop
   91 write(iwrit,203)ifak,ifadim,nt
      stop
   92 continue
c
  200 format(33h extrapolation in tabs, t (teta)=,e12.5,5x,12hcomponent
     *no,i5)
  201 format(24h zero in tabs, t (teta)=,e12.5,5x,12hcomponent no,i5)
  202 format(6h kfak=,i5,5x,11h gt kfadim=,i5,5x,12hin tabs, nt=,i5)
  203 format(6h ifak=,i5,5x,11h gt ifadim=,i5,5x,12hin tabs, nt=,i5)
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine tb04a1(n,f,d,w)
      dimension f(n),d(n),w(3,n)
c
c tb04a1 calculates the coefficients of a cubic spline.
c given the function values f(), it returns derivatives d().
c this version of tb04a assumes constant dx=1.
c this subroutine is not used in the absko-code, but may be useful
c in setting up tables for interpolation.
c
c first point
      w(1,1)=1.
      w(2,1)=0.
      w(3,1)=-1.
      d(1)=2.*(2.*f(2)-f(1)-f(3))
c
c last point
      w(1,n)=1.
      w(2,n)=0.
      w(3,n)=-1.
      d(n)=2.*(2.*f(n-1)-f(n-2)-f(n))
c
c interior points
      n1=n-1
      do 100 k=2,n1
      w(1,k)=1.
      w(2,k)=4.
      w(3,k)=1
      d(k)=3.*(f(k+1)-f(k-1))
100   continue
c
c eliminate at first point
      c=-w(3,1)/w(3,2)
      w(1,1)=w(1,1)+c*w(1,2)
      w(2,1)=w(2,1)+c*w(2,2)
      d(1)=d(1)+c*d(2)
      w(3,1)=w(2,1)
      w(2,1)=w(1,1)
c
c eliminate at last point
      c=-w(1,n)/w(1,n-1)
      w(2,n)=w(2,n)+c*w(2,n-1)
      w(3,n)=w(3,n)+c*w(3,n-1)
      d(n)=d(n)+c*d(n-1)
      w(1,n)=w(2,n)
      w(2,n)=w(3,n)
c
c eliminate subdiagonal
      do 110 k=2,n
      c=-w(1,k)/w(2,k-1)
      w(2,k)=w(2,k)+c*w(3,k-1)
      d(k)=d(k)+c*d(k-1)
110   continue
c
c backsubstitute
      d(n)=d(n)/w(2,n)
      do 120 kk=2,n
      k=(n+1)-kk
      d(k)=(d(k)-w(3,k)*d(k+1))/w(2,k)
120   continue
c
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine termon(t,pe,prad,pg,dpgt,dpgp,ro,drot,drop,cp,tgrad,q)
c
c        rutinen beraeknar  ovanstaaende storheter utgaaende
c        fraan t, pe och prad (straalningstrycket). ingen av stor-
c        heterna ges eller erhaalles logaritmerad. metoden med dif-
c        ferensformler aer vardayas
c        vi anvaender rutinen taet.
c        *** observera. ocksaa taet maaste arbeta med och ge
c        i c k e  l o g a r i t m e r a d e  s t o r h e t e r . *******
c
c version of 73.02.05. double precision added. *nord*
c
      double precision eh,hh,ph,prah
c
      dimension pgh(4),roh(4),eh(4),hh(4),ph(4),ep(4),prah(4)
c
      derep=0.01
      deret=0.001
      delpe=derep*pe
      delt=deret*t
      pinv=1./(2.*delpe)
      tinv=1./(2.*delt)
c
      call jon(t,pe,1,pg,ro,epp,0)
      p=prad+pg
c
      pep=pe-delpe
      call jon(t,pep,1,pgh(1),roh(1),ep(1),0)
      prah(1)=prad
      pep=pe+delpe
      call jon(t,pep,1,pgh(2),roh(2),ep(2),0)
      prah(2)=prad
      tp=t-delt
      call jon(tp,pe,1,pgh(3),roh(3),ep(3),0)
      prah(3)=prad*(1.-4.*deret)
      tp=t+delt
      call jon(tp,pe,1,pgh(4),roh(4),ep(4),0)
      prah(4)=prad*(1.+4.*deret)
c
      do1 i=1,4
      eh(i)=3.*prah(i)/roh(i)+ep(i)
      ph(i)=prah(i)+pgh(i)
    1 hh(i)=eh(i)+ph(i)/roh(i)
c
      det=(eh(4)-eh(3))*tinv
      dep=(eh(2)-eh(1))*pinv
      drot=(roh(4)-roh(3))*tinv
      drop=(roh(2)-roh(1))*pinv
      dht=(hh(4)-hh(3))*tinv
      dhp=(hh(2)-hh(1))*pinv
      dpt=(ph(4)-ph(3))*tinv
      dpp=(ph(2)-ph(1))*pinv
      dpgt=(pgh(4)-pgh(3))*tinv
      dpgp=(pgh(2)-pgh(1))*pinv
c
      cv=det-dep*drot/drop
      cp=dht-dhp*dpt/dpp
      hjalp=drot-drop*dpt/dpp
      tgrad=-p*hjalp/(cp*ro*ro)
      q=-t/ro*(drot-drop*dpgt/dpgp)
      u2=cp*dpp/(cv*drop)
c
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      function tg01b1(n,f,d,xx)
      dimension f(n),d(n)
c
c tg01b1 calculates the value of a spline constructed by tb04a1.
c xx is the coordinate, in "integer" units.
c
      kk=xx
      kk=min0(n-1,max0(1,kk))
      p=xx-kk
      q=1.-p
      df=f(kk+1)-f(kk)
      tg01b1=q*f(kk)+p*f(kk+1)+p*q*
     & (q*(d(kk)-df)-p*(d(kk+1)-df))
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine traneq
      parameter (mt=60)
c
c traneq solves the transfer equation including continuum scattering.
c features:
c
c 1. cannons perturbation technique is used on the angular quadrature.
c    the basic idea in this technique is to replace the inversion of
c    a complicated (mmu order) operator with the inversion of a simple
c    operator (one point=eddington approximation), plus iteration on
c    the error.
c 2. aitken extrapolation accellerates the convergence.
c 3. a trick due to robert stein (priv. comm., 1979) is used to
c    eliminate the need for double precision storage of the matrix
c    elements. the idea is to store the (small) sum of the three
c    matrix elements on a row, instead of the (large) diagonal ele-
c    ment.
c 4. the solution is 4th order accurate, using the hermitean
c    method of auer. this is accomplished with the correction
c    terms ad and bd in subroutine tranfr.
c 5. the scattering is treated as dipole scattering instead of the normally
c    used isotropic approximation. this can be done very simply in the
c    iterating cannon scheme.
c 6. a boundary condition which includes an estimated infalling
c    radiation makes the solution good also for values of x+s
c    large compared with 1./tau(1). a logarithmic tau-scale
c    should be used.
c
c this version of traneq is compatible with previous traneq's.
c 79.06.21 *nord*
c
      common /ctran/nt,tau(mt),x(mt),s(mt),bplan(mt),xj(mt),xh(mt)
     & ,xk(mt)
      common /space2/source(mt),error(mt),qp1(mt),qp2(mt),qp3(mt),p(mt)
     & ,sp1(mt,6),sp2(mt,6),sp3(mt,6),an(mt),ad(mt),bd(mt)
     & ,fact(mt),dso(mt),c(6),t(6),ex(6)
      dimension a(7)
c
c initiate
      do 100 k=1,nt
      fact(k)=1.
      dso(k)=0.
      xj(k)=0.
      xk(k)=0.
      error(k)=bplan(k)*x(k)/(x(k)+s(k))
100   source(k)=0.
c
c calculate the matrix elements
      call tranfr
      call transc
c
c iteration loop
      itmax=7
      do 110 it=1,itmax
110   a(it)=0.
      do 140 it=1,itmax
      itm=it
c
c solve the continuum scattering problem in the eddington approximation
      call scattr
      do 120 k=1,nt
      xj(k)=xj(k)+p(k)
      xk(k)=xk(k)+.333333*p(k)
c
c aitken extrapolation used for convergence accelleration
      ds=error(k)+p(k)*s(k)/(x(k)+s(k))
      if(dso(k).ne.0.) fact(k)=amin1(1.25,amax1(.8,fact(k)-ds/dso(k)))
      ds=ds/fact(k)
      if(it.ge.2) dso(k)=ds
120   source(k)=source(k)+ds
c
c solve the transfer equation with given source function
      call formal
c
c check error in source function
      do 130 k=1,nt
130   a(it)=amax1(a(it),abs(error(k)/source(k)))
c
c end of iteration loop
      if(a(it).lt.0.0001) go to 141
140   continue
      print 50,(a(it),it=1,itm)
50    format(' maxfel =',12f10.7)
141   continue
c
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine tranfr
      parameter (mt=60)
c
c formal solves the transfer equation with given source function 'source'.
c 'error' is the resulting error in the definition of the continuum
c scattering source function. transfr calculates the matrix elements
c of the problem. flux and intensities at tau=0 are returned in /csurf/.
c 79.06.21 *nord*
c
      common /cangle/mmu,xmu(6),xmu2(6),h(6)
      common /ctran/nt,tau(mt),x(mt),s(mt),bplan(mt),xj(mt),xh(mt)
     & ,xk(mt)
      common /space2/source(mt),error(mt),qp1(mt),qp2(mt),qp3(mt),p(mt)
     & ,sp1(mt,6),sp2(mt,6),sp3(mt,6),an(mt),ad(mt),bd(mt)
     & ,fact(mt),dso(mt),c(6),t(6),ex(6)
      common /csurf/hsurf,y1(6)
      data c6,c12/0.1666666666666667,0.08333333333333333/
c
c mu loop
      nt1=nt-1
      nt2=nt-2
      do 110 i=1,mmu
c
c k=1
      dtaub=.5*(x(1)+s(1)+x(2)+s(2))*(tau(2)-tau(1))/xmu(i)
      a=1./dtaub
      b=a**2
      sp2(1,i)=1.+2.*a
      sp3(1,i)=-2.*b
c let p be the even part of the intensity, then
c
c         p(2)= p(1) + d*p'(1) + .5*d2*p''(1)
c or      p(2)= p(1) + d*(p(1)-i(1,-mu)) + .5*d2*(p(1)-s(1)) .
c where   i(1,-mu) = s(1)*(1.-exp(-t))
c
c the difference as compared to the usual second order boundary condition
c is the additional term   i(1,-mu)=s(1)*(1.-exp(-t)). thus the coefficient
c for s(1) in the first equation should be changed as follows
c         s(1)=s(1)*(1.+c*(1.-exp(-t))
c where   c=2./d
c *nord* 751009
      c(i)=2.*a
      t(i)=tau(1)*(x(1)+s(1))/xmu(i)
      ex(i)=t(i)*(1.-.5*t(i)*(1.-.3333*t(i)))
      if(t(i).gt.0.1) ex(i)=1.-exp(-t(i))
c
c k=2,nt-1
      do 100 k=2,nt1
      dtaua=dtaub
      dtaub=.5*(x(k)+s(k)+x(k+1)+s(k+1))*(tau(k+1)-tau(k))/xmu(i)
      dtauc=.5*(dtaua+dtaub)
      a=1./(dtaua*dtauc)
      b=1./(dtaub*dtauc)
      ad(k)=c6-c12*a*dtaub**2
      bd(k)=c6-c12*b*dtaua**2
      sp1(k,i)=-a+ad(k)
      sp2(k,i)=1.
      sp3(k,i)=-b+bd(k)
100   continue
c
c k=nt
      sp2(nt,i)=1.
c
c end of mu loop
110   continue
c
c eliminate subdiagonal, save factors in sp1
      do 121 i=1,mmu
      do 120 k=1,nt2
      sp1(k,i)=-sp1(k+1,i)/(sp2(k,i)-sp3(k,i))
      sp2(k+1,i)=sp2(k+1,i)+sp1(k,i)*sp2(k,i)
120   sp2(k,i)=sp2(k,i)-sp3(k,i)
121   sp2(nt-1,i)=sp2(nt-1,i)-sp3(nt-1,i)
c
      return
c
      entry formal
c
c zeroset
      do 130 k=1,nt
      an(k)=(3.*xk(k)-xj(k))/8.*s(k)/(x(k)+s(k))
      xk(k)=0.
130   xj(k)=0.
c
c mu loop
      xh(1)=0.
      hsurf=0.
      do 170 i=1,mmu
c
c initiate approximative source function
      p(1)=source(1)+an(1)*(3.*xmu2(i)-1.)
c note the anisotropic scattering correction
      s0=p(1)
      p(1)=p(1)*(1.+c(i)*ex(i))
      do 140 k=2,nt1
140   p(k)=(1.-ad(k)-bd(k))*(source(k)+an(k)*(3.*xmu2(i)-1.))
     & +ad(k)*(source(k-1)+an(k-1)*(3.*xmu2(i)-1.))
     & +bd(k)*(source(k+1)+an(k+1)*(3.*xmu2(i)-1.))
      p(nt)=source(nt)
c
c accumulate right hand side
      do 150 k=1,nt2
150   p(k+1)=p(k+1)+sp1(k,i)*p(k)
c
c backsubstitute
      do 160 k=1,nt1
      p(nt-k)=(p(nt-k)-sp3(nt-k,i)*p(nt-k+1))/sp2(nt-k,i)
      xk(nt-k)=xk(nt-k)+h(i)*p(nt-k)*xmu2(i)
160   xj(nt-k)=xj(nt-k)+h(i)*p(nt-k)
c
c end of mu loop
      xk(nt)=xk(nt)+h(i)*p(nt)*xmu2(i)
      r1=p(1)-s0*ex(i)
      xh(1)=xh(1)+h(i)*xmu(i)*r1
      p0=p(1)*(1.-ex(i))+.5*s0*ex(i)**2
      hsurf=hsurf+h(i)*xmu(i)*p0
      y1(i)=2.*p0
c hsurf and y1(6) are the flux and intensities at the surface
170   continue
      xj(nt)=p(nt)
c
c 'xj' is the new mean intensity
      do 180 k=1,nt
180   error(k)=(x(k)*bplan(k)+s(k)*xj(k))/(x(k)+s(k))-source(k)
c
c flux and second moment
      do 190 k=2,nt
190   xh(k)=2.*(xk(k)-xk(k-1))/(x(k)+s(k)+x(k-1)+s(k-1))/
     /(tau(k)-tau(k-1))
c
      return
c---- debug subchk
      end
         
C%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      subroutine transc
      parameter (mt=60)
c
c scattr solves the transfer equation including continuum scattering
c in the eddington approximation, i.e., using only one mu point.
c 'error' is the inhomogeneous term of the equation, and 'p' contains
c the estimated mean intensity on exit. transc calculates the matrix
c elements for scattr.
c 79.06.21 *nord*
c
      common /ctran/nt,tau(mt),x(mt),s(mt),bplan(mt),xj(mt),xh(mt)
     & ,xk(mt)
      common /space2/source(mt),error(mt),qp1(mt),qp2(mt),qp3(mt),p(mt)
     & ,sp1(mt,6),sp2(mt,6),sp3(mt,6),an(mt),ad(mt),bd(mt)
     & ,fact(mt),dso(mt),c(6),t(6),ex(6)
      data xmu/0.5773503/
c
c k=1
      dtaub=.5*(x(1)+s(1)+x(2)+s(2))*(tau(2)-tau(1))/xmu
      a=1./dtaub
      b=a**2
      qp2(1)=2.*a+x(1)/(s(1)+x(1))
      qp3(1)=-2.*b
      cc=2.*a
      tt=tau(1)*(x(1)+s(1))/xmu
      eex=tt*(1.-.5*tt*(1.-.33333*tt))
      if(tt.gt.0.1) eex=1.-exp(-tt)
      qp2(1)=qp2(1)-cc*s(1)/(x(1)+s(1))*eex
c
c k=2,nt-1
      nt1=nt-1
      do 100 k=2,nt1
      dtaua=dtaub
      dtaub=.5*(x(k)+s(k)+x(k+1)+s(k+1))*(tau(k+1)-tau(k))/xmu
      dtauc=.5*(dtaua+dtaub)
      a=1./(dtaua*dtauc)
      b=1./(dtaub*dtauc)
      qp1(k)=-a
      qp2(k)=x(k)/(s(k)+x(k))
      qp3(k)=-b
100   continue
c
c k=nt
      qp2(nt)=x(nt)/(x(nt)+s(nt))
c
c eliminate subdiagonal
      nt2=nt-2
      do 110k=1,nt2
      qp1(k)=-qp1(k+1)/(qp2(k)-qp3(k))
      qp2(k+1)=qp2(k+1)+qp1(k)*qp2(k)
110   qp2(k)=qp2(k)-qp3(k)
      qp2(nt-1)=qp2(nt-1)-qp3(nt-1)
      return
c
      entry scattr
c
c initiate inhomogenous terms
      p(1)=error(1)
      p(nt)=error(nt)
      do 120 k=2,nt-1
120   p(k)=error(k)
      dsdt=0.
c prelim
      p(1)=p(1)*(1.+cc*eex)-dsdt*cc*(eex-tt*(1.-eex))
c
c accumulate inhomogenous terms
      do 130 k=1,nt2
130   p(k+1)=p(k+1)+qp1(k)*p(k)
c
c backsubstitute
      p(nt)=p(nt)/qp2(nt)
      do 140 k=1,nt1
140   p(nt-k)=(p(nt-k)-qp3(nt-k)*p(nt-k+1))/qp2(nt-k)
c
      return
c---- debug subchk
      end