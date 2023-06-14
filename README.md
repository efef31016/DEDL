# Nonlocal_Singular_BVP_DL

#### Describing each file as follows.  

--PINNModel  
    --- \_\_init\_\_.py  
    --- CalTool.py : There are functions which need to be modified according to the desired equations.  
    --- ItofSin.py  
    --- PtChange.py   
          
          
--bvp4c_version : Simulation the profile of the solution by bvp4c in Matlab.  


--main.py : There are super-parameters which need to be modified according to the desired configurations.  
        
--main_stregth_bdd.py : There are super-parameters which need to be modified according to the desired configurations.  

--sample.txt : Sampling f(x)=12(x-1/2)^2 in [0,1] by sample.R(rejection sampling algorithm).  

The different between main.py and main_stregth_bdd.py is the function named "net_NS_u". The latter emphasize performance with boundary points within the interior.
