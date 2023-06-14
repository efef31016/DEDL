import numpy as np

def u0u1(A11,A12,A21,A22,mu):

    A = np.array([[A11,A12],[A21,A22]])
    svd_A = np.linalg.svd(A)
    cond_A = np.max(svd_A[1])/np.min(svd_A[1])
    print("==============================")
    print("κ(A) = %.5f" % cond_A)
    print("==============================")
    A_inv = np.linalg.inv(A)
    ub = np.dot(A_inv,mu)

    return A, ub, cond_A


# special case
def two_norm(x,y):
    return sum((x-y)**2) / len(x)

def u_exact_direct(x, eps):
    A = -1+np.sqrt(5)
    B = -1-np.sqrt(5)
    alpha = 1 - 2/A * (np.exp(A/(2*eps)) - np.exp(A/(4*eps)))
    beta = 1 - 2/B * (np.exp(B/(2*eps)) - np.exp(B/(4*eps)))
    gamma = np.exp(A/(2*eps)) - 2/A * (np.exp(A/(4*eps)) - 1)
    eta = np.exp(B/(2*eps)) - 2/B * (np.exp(B/(4*eps)) - 1)
    # C1 = (eta - beta) / (alpha - gamma) / (beta + alpha * ((eta - beta) / (alpha - gamma)))
    C1 = (eta - beta) / (alpha*eta - beta*gamma)
    # C2 = 1 / (beta + alpha * (eta - beta) / (alpha - gamma))
    C2 = (alpha - gamma) / (alpha*eta - beta*gamma)
    return C1 * np.exp(A/(2*eps)*x) + C2 * np.exp(B/(2*eps)*x)

def V(x, c2, eps):
    A = -1+np.sqrt(5)
    B = -1-np.sqrt(5)
    c1 = 1-c2
    return c1 * np.exp(A/(2*eps)*x) + c2 * np.exp(B/(2*eps)*x)

def W(x, c2, eps):
    A = -1+np.sqrt(5)
    B = -1-np.sqrt(5)
    c1 = -c2
    return c1 * np.exp(A/(2*eps)*x) + c2 * np.exp(B/(2*eps)*x)