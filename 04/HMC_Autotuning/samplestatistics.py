import matplotlib.pyplot as plt
import numpy as np


class stats:
    
    def __init__(self,dim1,dim2,N):
        """
        Class to accumulate and plot 2D sample statistics.

        :param dim1: Index of the 1st dimension of interest.
        :param dim2: Index of the 2nd dimension of interest.
        :param N: Total number of samples.
        """

        self.N=N
        self.dim1=dim1
        self.dim2=dim2
        self.m1=np.zeros(N)
        self.m2=np.zeros(N)
        self.logdet=np.zeros(N)


    def get(self,m,logdet,it):
        """
        Accumulate sample values and log of determinant of the mass matrix.
        """

        self.m1[it]=m[self.dim1]
        self.m2[it]=m[self.dim2]
        
        if it==0: self.mean=m
        else: self.mean+=m
            
        self.logdet[it]=logdet

    def save(self):
        """
        Save sample values for later analysis.
        """

        np.save('OUTPUT/m1.npy',self.m1)
        np.save('OUTPUT/m2.npy',self.m2)

       

    def display(self):
        """
        Compute and plot selected quantities.
        """

        # mean
        plt.subplots(1, figsize=(20,10))
        plt.plot(self.mean/self.N,'k',linewidth=4)
        plt.title('mean as function of parameter index',pad=20)
        plt.xlabel('parameter index')
        plt.ylabel('mean value')
        ylim=1.1*np.max(np.abs(self.mean/self.N))
        plt.xlim((0.0,len(self.mean)))
        plt.ylim((-ylim,ylim))
        plt.savefig('OUTPUT/means.png', bbox_inches='tight', format='png')
        plt.grid()
        plt.show()
        

        # traces 
        plt.subplots(1, figsize=(20,10))
        plt.plot(self.m1,'k',linewidth=4)
        plt.plot(self.m2,'r',linewidth=4,alpha=0.7)
        plt.title('trace plot (black=parameter 1, red=parameter 2)',pad=20)
        plt.ylabel(r'$m_1$, $m_2$')
        plt.xlabel('sample index')
        ylim=np.maximum(np.max(np.abs(self.m1)),np.max(np.abs(self.m2)))
        plt.xlim((0.0,self.N))
        plt.ylim((-ylim,ylim))
        plt.grid()
        plt.savefig('OUTPUT/traces.png', bbox_inches='tight', format='png')
        plt.show()
            

        if 1==1:

            # intra-parameter correlations
            mu1=self.mean[self.dim1]/self.N
            mu2=self.mean[self.dim2]/self.N
            cc1=np.correlate((self.m1-mu1),(self.m1-mu1),'full')[self.N-1:2*self.N]
            cc2=np.correlate((self.m2-mu2),(self.m2-mu2),'full')[self.N-1:2*self.N]

            cc1=cc1/np.max(cc1)
            cc2=cc2/np.max(cc2)
                
            # regular
            plt.subplots(1, figsize=(20,10))
            plt.plot(cc1,'k',linewidth=4)
            plt.plot(cc2,'r',linewidth=4)
            plt.title('intra-parameter correlations (black=parameter 1, red=parameter 2)',pad=20)
            plt.xlabel('sample index')
            plt.ylabel(r'$cc(m_1)$, $cc(m_2)$')
            plt.grid()
            plt.xlim((-10.0,self.N))
            plt.savefig('OUTPUT/correlations.png', bbox_inches='tight', format='png')
            plt.show()

            # zoomed and effective sample size
            # Find index where successive values are below zero.
            N1=1
            while (N1<self.N and (cc1[N1-1]>0 or cc1[N1]>0)): N1+=1
            N2=1
            while (N2<self.N and (cc2[N2-1]>0 or cc2[N2]>0)): N2+=1
            Nmax=np.max((N1,N2))
        
            plt.subplots(1, figsize=(20,10))
            plt.plot(cc1[0:np.min((2*Nmax,self.N))],'k',linewidth=4)
            plt.plot(cc2[0:np.min((2*Nmax,self.N))],'r',linewidth=4)
            plt.title('intra-parameter correlations (black=parameter 1, red=parameter 2)',pad=20)
            plt.xlabel('sample index')
            plt.ylabel(r'$cc(m_1)$, $cc(m_2)$')
            plt.grid()
            plt.xlim((-2.0,2*Nmax))
            plt.savefig('OUTPUT/correlations_zoom.png', bbox_inches='tight', format='png')
            plt.show()

            n_eff1=1.0/(1.0+2.0*np.sum(cc1[0:N1]))
            n_eff2=1.0/(1.0+2.0*np.sum(cc2[0:N2]))
            print('effective sample fractions: (%f,%f)' % (n_eff1,n_eff2))
            
        
        # log determinant
        plt.subplots(1, figsize=(20,10))
        plt.plot(self.logdet,'k',linewidth=4)
        plt.xlabel('sample index')
        plt.ylabel('log(det(M))')
        plt.title('log determinant')
        plt.xlim((0.0,len(self.logdet)))
        plt.grid()
        plt.savefig('OUTPUT/logdet.png', bbox_inches='tight', format='png')
        plt.show()

        
        # 1D and 2D posterior histograms.
        burnin=0

        plt.subplots(1, figsize=(12,12))
        plt.hist(self.m1[burnin:],bins=30,density=True,color='k')
        plt.grid()
        plt.xlabel(r'$m_1$')
        plt.title('posterior 1D histogram parameter 1',pad=20)
        plt.savefig('OUTPUT/marginal_1D_1.png', bbox_inches='tight', format='png')
        plt.show()

        plt.subplots(1, figsize=(12,12))
        plt.hist(self.m2[burnin:],bins=30,density=True,color='k')
        plt.grid()
        plt.xlabel(r'$m_2$')
        plt.title('posterior 1D histogram parameter 2',pad=20)
        plt.savefig('OUTPUT/marginal_1D_2.png', bbox_inches='tight', format='png')
        plt.show()

        m=np.max( (np.max(np.abs(self.m1)), np.max(np.abs(self.m2)) ))
        plt.subplots(1, figsize=(12,12))
        plt.hist2d(self.m1[burnin:],self.m2[burnin:],bins=30,cmap='binary',density=True)
        plt.xlabel(r'$m_1$')
        plt.ylabel(r'$m_2$')
        plt.colorbar()
        plt.title('posterior 2D histogram',pad=20)
        plt.savefig('OUTPUT/marginal_2D.png', bbox_inches='tight', format='png')
        plt.show()
