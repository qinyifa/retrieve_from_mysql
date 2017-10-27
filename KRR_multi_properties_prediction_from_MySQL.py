
# coding: utf-8

# The code below predicts multiple molecular properties by Kernel ridge regression(KRR). The number of molecules is 109, up to 60 molecules are used as samples to "train" the KRR model, and the rest are "out-of-sample" molecules. 

# In[2]:

import numpy as np;
import scipy.io;
import random;
import os;
import sys;
import scipy.spatial.distance as scipy_distance;

# import some functions from krr.py 


# In[17]:

directory = "D:/tensorflow_work/retrieve_from_mysql";
os.chdir(directory)


# In[18]:

mol_data = scipy.io.loadmat('molecular.mat');
type(mol_data)

mol_data.keys()


# In[19]:

p = ['homo', 'lumo', 'r2', 'cv', 'u0'];
p_unit =['Hatree', 'Hatree', 'Bohr^2', 'cal/(mol*k)', 'Hatree'];
p_limit = [0.0016, 0.0016, 0.1, 0.01, 0.0016];


# In[20]:

# Split the total molecules set into training and test sets.
# return properties (containing in list p) and colomb matrixs, spectrums of colomb matrixs
def split_training_test(mol_data,p,
                       N_total    = len(mol_data['id']),
                       N_training = round(0.67*len(mol_data['id'])-1),
                       N_test     = len(mol_data['id'])-round(0.67*len(mol_data['id'])-1)):

    index_entire = list(range(N_total))
    random.shuffle(index_entire)
    
    index_training  = index_entire[0:N_training]
    index_test      = index_entire[N_training:N_test+N_training]
    
    num_p = len(p)
    properties_training = np.zeros([N_training,num_p])
    properties_test     = np.zeros([N_test,num_p])

    for i in list(range(num_p)):
        properties_training[:,i]  = np.reshape(mol_data[p[i]][index_training],[N_training,])
        properties_test[:,i]      = np.reshape(mol_data[p[i]][index_test],[N_test,])

    cm_eigenvalue_training     = mol_data['colomb_matrix_eigenvalues'][index_training]
    cm_eigenvalue_test         = mol_data['colomb_matrix_eigenvalues'][index_test]
    
    return properties_training,properties_test,cm_eigenvalue_training,cm_eigenvalue_test 


# In[21]:

# Calculate the distance matrix of vectors in two sets
def distance_matrix_of_cm_eigenvalue(cm_eigenvalue_set1,cm_eigenvalue_set2,lp=2):

        distance_matrix = scipy_distance.cdist(cm_eigenvalue_set1,
                                               cm_eigenvalue_set2,
                                               'minkowski',lp)
       
        return distance_matrix
# d_ij = lp-norm distance between the i-th vector in set1 and the j-th vector in set2. 
# when Euclidean distance, choose lp=2.0 .


# Kernelized distance matrix by Gaussian function
def gaussian_k(distance_matrix,sigma):
    k_matrix = np.exp(-1.0*distance_matrix/sigma)
    return k_matrix
# k_ij = exp(-1*d_ij/sigma)



# Calculate the inverse of regularized k_matrix
def inverse_k_matrix(k_matrix,lambda0):
    k_matrix_dimension = np.shape(k_matrix) 
    k_matrix = k_matrix + lambda0*np.eye(k_matrix_dimension[1])
    k_matrix = np.mat(k_matrix)      # Convert the numpy array K_matrix into a numpy matrix    
  
    inverse_k = k_matrix.I
    inverse_k = np.array(inverse_k)   
    return inverse_k
# inverse_k = (k_matrix+lambda0*I_matrix)^-1


# In[22]:

def training_krr(properties_training,properties_test,cm_eigenvalue_training,cm_eigenvalue_test):
    
    distance_matrix_training = distance_matrix_of_cm_eigenvalue(cm_eigenvalue_training,cm_eigenvalue_training,2.0)
    sigma                    = np.max(distance_matrix_training )/np.log(2.0)

    inverse_k                = inverse_k_matrix(distance_matrix_training,0.0)
    coeffiecients_alpha      = np.mat(inverse_k) * np.mat(properties_training)
    
    return coeffiecients_alpha


# In[23]:

def test_krr(coeffiecients_alpha,cm_eigenvalue_test):
    k_test_training          = distance_matrix_of_cm_eigenvalue(cm_eigenvalue_test,cm_eigenvalue_training)
    properties_test_predict  = np.mat(k_test_training)*coeffiecients_alpha
    return properties_test_predict


# In[25]:

properties_training, properties_test, cm_eigenvalue_training, cm_eigenvalue_test  =    split_training_test(mol_data,p)


# In[26]:

coeffiecients_alpha     = training_krr(properties_training, 
                                    properties_test, 
                                    cm_eigenvalue_training, 
                                    cm_eigenvalue_test)
properties_test_predict = test_krr(coeffiecients_alpha,
                                   cm_eigenvalue_test)


# In[27]:

delta = np.array(properties_test-properties_test_predict)
mae   = np.sum(np.abs(delta),axis=0)/np.shape(delta)[0]
rmse  = np.sqrt(np.sum(np.square(delta),axis=0)/np.shape(delta)[0])


# In[28]:

print(p)
print(p_unit)
print(mae)
print(rmse)
print(p_limit)


# In[ ]:



