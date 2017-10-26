#usr/bin/python
#from array import *
import decimal as de
import numpy as np
import scipy.io as sio
import pymysql as pm
import scipy.spatial.distance as distance


def mysql_cursor(mysql_query):
    conn =  pm.connect(host='10.1.20.140',user='root',passwd='hzw2014',
                       database='school',port=3306,cursorclass=pm.cursors.DictCursor)
    cursor = conn.cursor()
    cursor.execute(mysql_query)
    answer = cursor.fetchall()
    m_id = []
    for j in answer:
        for i in j.keys():
          if isinstance(j[i],de.Decimal):
            m_id.append(float(j[i]))
          else:
            m_id.append(j[i])
    a_id = np.array(m_id)
    a_id = np.reshape(a_id,[-1,1])
    return a_id
mysql_query="SELECT id FROM basicproperties where homo < '2.5' ;"
mo_id=mysql_cursor(mysql_query)
molnum = np.size(mo_id)





#-------------------Retrieve the xyzs0 and elementid-------------------------------#
mysql_query = "SELECT atomnum FROM basicproperties where homo < '2.5' ;"
atomnum = mysql_cursor(mysql_query)
atomnum_max = np.max(atomnum)
atomnum_cumsum = np.cumsum(atomnum)
atomnum_cumsum = atomnum_cumsum.tolist()
atomnum_cumsum.pop()


mysql_query="SELECT x,y,z,elementid,id_b FROM xyzs0 WHERE id_b IN (SELECT id FROM basicproperties WHERE homo < '2.5');"
xyzs0 = mysql_cursor(mysql_query)
xyzs0 = np.reshape(xyzs0,[-1,5])
elementid =  xyzs0[:,3] 
xyzs0 = xyzs0[:,0:3]

xyzs0 = np.split(xyzs0,atomnum_cumsum)
elementid = np.split(elementid,atomnum_cumsum)


mysql_query = "SELECT homo FROM basicproperties where homo < '2.5' ;"
homo = mysql_cursor(mysql_query)
mysql_query = "SELECT lumo FROM basicproperties where homo < '2.5' ;"
lumo = mysql_cursor(mysql_query)
mysql_query = "SELECT u0 FROM basicproperties where homo < '2.5' ;"
u0 =  mysql_cursor(mysql_query)
mysql_query = "SELECT r2 FROM basicproperties where homo < '2.5' ;"
r2 =  mysql_cursor(mysql_query)
mysql_query = "SELECT cv FROM basicproperties where homo < '2.5' ;"
cv =  mysql_cursor(mysql_query)

#--------------------Calculate the Colomb Matrix---------------------------------#
def xyz_to_colombmatrix(xyz,elementid,lp=2.0,dimension=atomnum_max):

    colombmatrixs_unified = np.zeros([np.size(xyz),dimension,dimension])
    colombmatrixs_eigenvalue_unified = np.zeros([np.size(xyz),dimension])

    for i in range(np.size(xyz)):
        size_i    = np.shape(xyz[i])[0]
        # the total number of atoms in the i-th molecule
        colombmatrix = distance.cdist(xyz[i],xyz[i],'minkowski',lp)
        # calculate the distance lp-norm distance metrix of atoms in the i-th molecule
        colombmatrix = 1.0/(colombmatrix + np.eye(size_i))
        
        
        chargeouterproduct = np.outer(elementid[i],elementid[i])
        # the l,m element in the chargeouterproduct matrix is Z_l * Z_m 
        colombmatrix = np.multiply(chargeouterproduct,colombmatrix)
        # element-wise multipy
        # the l,m element in colombmatrix = Z_l * Z_m / distance(R_l, R_m), l not equal to m
        

        for j in range(size_i):
            colombmatrix[j,j] = 0.5*np.power(colombmatrix[j,j],2.4/2.0)    
        # now in the colomb matrix, l,m element equals 0.5*Z_l**2.4, when l = m

        for j in range(size_i):
            norm_row = np.sum(colombmatrix**2.0,axis=1)
            sort_index     = np.argsort(-1*norm_row)
        
            colombmatrix       = colombmatrix[sort_index]
            colombmatrix       = colombmatrix[:,sort_index]
        # sort the colomb matrix according to the l2-norm of each row
        # and then resort the colomns by the same order as rows

        eigenvalues = np.linalg.eigvals(np.mat(colombmatrix))
        eigenvalues_sort_index = np.argsort(np.abs(eigenvalues))
        # eigenvalues are sored in descending order of the absolute values
        eigenvalues = eigenvalues[eigenvalues_sort_index]
        eigenvalues = eigenvalues[::-1]
        # get the i-th molecule's colomb matrix's eigenvalue array

        colombmatrixs_unified[i][0:size_i,0:size_i] = colombmatrix
        # ensure all the colomb matrixs have the same shape by filling empty sites with 0.
        colombmatrixs_eigenvalue_unified[i][0:size_i] = eigenvalues

    return colombmatrixs_unified,colombmatrixs_eigenvalue_unified




colombmatrixs_unified,colombmatrixs_eigenvalue_unified = xyz_to_colombmatrix(xyzs0,
                                                                             elementid,2.0,dimension=atomnum_max)


sio.savemat('molecular.mat',{'id':mo_id,'xyzs0':xyzs0,'atomnum':atomnum,'elementid':elementid,
                             'colomb_matrix':colombmatrixs_unified,
                             'colomb_matrix_eigenvalues':colombmatrixs_eigenvalue_unified,
                             'r2':r2,'cv':cv,'u0':u0,
                             'homo':homo,
                             'lumo':lumo})


'''size_id = np.size(mo_id)
x=[]
for i in mo_id:
    mysql_query="SELECT x FROM xyzs0  WHERE  id = mo_id[i][0] ;"
    x_i=mysql_cursor(mysql_query)
    x.append(x_i)
x=np.array(x)


 sio.savemat('molecular.mat',{'id':mo_id,'x':x})'''
