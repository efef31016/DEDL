import numpy as np
from scipy.integrate import simpson
import time


class IntegralofSingular:
    def __init__(self, g01, vw, low, upp, pts):   # g01 = np.array([g0,g1]); vw = np.array([v,w]); low = g0; upp = g1

        self.g01 = g01
        self.vw = vw
        self.low = low
        self.upp = upp
        self.pts = pts
        
        # 積分切點
        self.n1 = np.linspace(self.low,1,self.pts)
        self.n2 = np.linspace(0,self.upp,self.pts)
        
        self.g0v = lambda x : g01[0](x) * self.vw[0].predict(np.array(x).reshape(-1,1)).reshape(-1)
        self.g0w = lambda x : g01[0](x) * self.vw[1].predict(np.array(x).reshape(-1,1)).reshape(-1)
        self.g1v = lambda x : g01[1](x) * self.vw[0].predict(np.array(x).reshape(-1,1)).reshape(-1)
        self.g1w = lambda x : g01[1](x) * self.vw[1].predict(np.array(x).reshape(-1,1)).reshape(-1)

    
    def AMatrixEle(self, fcn, which):

        if which==0:
            n = self.n1
        else:
            n = self.n2

        return simpson(fcn(n), n)
    
    def twod_inv(self, A):
        return 1/(A[0,0]*A[1,1]-A[0,1]*A[1,0]) * np.array([[A[1,1],-A[0,1]],[-A[1,0],A[0,0]]])
    
    def get_A(self, eps, mu):
        st = time.time()
        a11 = 1 - self.AMatrixEle(self.g0v,0) / eps
        a12 = -self.AMatrixEle(self.g0w,0) / eps
        a21 = -self.AMatrixEle(self.g1v,1) / eps
        a22 = 1 - self.AMatrixEle(self.g1w,1) / eps
        en = time.time()
        print("the integral time spend %d seconds.\n" % round(en-st,0))
        A = np.array([[a11,a12],[a21,a22]])
        A_inv = self.twod_inv(A)
        ub = np.dot(A_inv,mu)
        
        return A, ub
    
    def solve_u(self, test_points, u, vw):
        return u[0]*vw[0] + u[1]*vw[1]