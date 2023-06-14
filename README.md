# Nonlocal_Singular_BVP_DL

### Describing each file as follows.  

PINNModel  
    - \_\_init\_\_.py  
    - CalTool.py : There are functions which need to be modified according to the desired equations.  
    - ItofSin.py  
    - PtChange.py   
          
          
bvp4c_version : Simulation the profile of the solution by bvp4c in Matlab.<br><br><br>
main.py : There are super-parameters which need to be modified according to the desired configurations.<br><br><br>
main_stregth_bdd.py : There are super-parameters which need to be modified according to the desired configurations.<br><br><br>
sample.txt : Sampling f(x)=12(x-1/2)^2 in [0,1] by sample.R(rejection sampling algorithm).<br><br><br>
**The different between main.py and main_stregth_bdd.py is the function named "net_NS_u". The latter emphasize performance with boundary points within the interior.<br><br><br><br><br><br>

### Where we customize the configuration
#### According different case, we can customize three functions in CalTool.py:
![In_CalTool.py](https://raw.githubusercontent.com/efef31016/Nonlocal_Singular_BVP_DL/master/PINN_nonlocal/figure/change03.png).
#### Furthermore, we also can customize more information of model as following:
![In_main.py1](https://raw.githubusercontent.com/efef31016/Nonlocal_Singular_BVP_DL/master/PINN_nonlocal/figure/change02.png)
![In_main.py2](https://raw.githubusercontent.com/efef31016/Nonlocal_Singular_BVP_DL/master/PINN_nonlocal/figure/change02.png)
