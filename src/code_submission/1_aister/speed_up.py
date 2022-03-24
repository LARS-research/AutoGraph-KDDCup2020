#import numpy as np
#
#try:
#    from scipy.sparse import csc_matrix
#except:
#    os.system('pip scipy.sparse')
#    from scipy.sparse import csc_matrix
    
from util import timeit
import torch

@timeit
def sparse_dot(matrix_a,matrix_b):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.mm(torch.Tensor(matrix_a).to(device),torch.Tensor(matrix_b).to(device)).cpu().numpy()

#    a = np.array(matrix_a)
#    b = np.array(matrix_b)
#    
#    a_rate = (a!=0).sum()/(a.shape[0]*a.shape[1])
#    b_rate = (b!=0).sum()/(b.shape[0]*b.shape[1])
#    print(a_rate,b_rate)
#    
#    if a_rate<0.02:
#        a_csc = csc_matrix(a)
#        if b_rate<0.01:
#            b_csc = csc_matrix(b)
#            return a_csc.dot(b_csc).toarray()
#        else:
#            return a_csc.dot(b)
#    else:
#        if b_rate<0.01:
#            b_csc = csc_matrix(b)
#            return csc_matrix.dot(a,b_csc)
#        else:
#            return np.dot(a,b)
    
#import time
#import os
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#
#a = np.random.rand(10000,10000)
#a[a>0.01] = 0
#b = np.random.rand(10000,9999)
#
#s = time.time()
#print(sparse_dot(a,b))
#print(f'tensor:{time.time()-s}')

#s = time.time()
#k = np.array(([i for i in range(100)]))
#row,col = np.nonzero(a)
#result = np.zeros((10000,9999))
#for i in np.unique(row):
#    target = k#col[row==i]
##    result[i,:] = np.dot(a[i,target],b[target,:])
#print(result)
#print(f'for:{time.time()-s}')



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#a_t = torch.Tensor(a)
#b_t = torch.Tensor(b)
#
#
#print(device)
#a_t.to(device)
#b_t.to(device)
#s = time.time()
#print(torch.mm(a_t,b_t).numpy())
#print(f'cuda tensor:{time.time()-s}')
#
#s = time.time()
#print(np.dot(a,b))
#print(f'np:{time.time()-s}')
