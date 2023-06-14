# Nonlocal_Singular_BVP_DL

___PINNModel___<br>
    - \_\_init\_\_.py<br>
    - CalTool.py : There are functions which need to be modified according to the desired equations.<br>
    - ItofSin.py<br>
    - PtChange.py<br><br><br>
___bvp4c_version___ : Simulation the profile of the solution by bvp4c in Matlab.<br><br><br>
___main.py___ : There are super-parameters which need to be modified according to the desired configurations.<br><br><br>
___main_stregth_bdd.py___ : There are super-parameters which need to be modified according to the desired configurations.<br><br><br>
___sample.txt___ : Sampling f(x)=12(x-1/2)^2 in [0,1] by ___sample.R___(rejection sampling algorithm).<br><br><br>
**The different between main.py and main_stregth_bdd.py is the function named "net_NS_u". The latter emphasize performance with boundary points within the interior.**<br><br><br>
## Where we customize the configuration

#### According different case, we can customize three functions in ___CalTool.py___:<br><br>
![In_CalTool.py](https://raw.githubusercontent.com/efef31016/Nonlocal_Singular_BVP_DL/master/PINN_nonlocal/figure/change03.png)
---
#### Furthermore, we also can customize more information of model in ___main.py___.:<br><br>
![In_main.py1](https://raw.githubusercontent.com/efef31016/Nonlocal_Singular_BVP_DL/master/PINN_nonlocal/figure/change02.png)
![In_main.py2](https://raw.githubusercontent.com/efef31016/Nonlocal_Singular_BVP_DL/master/PINN_nonlocal/figure/change02.png)<br><br>

___main_stregth_bdd.py___ can also be used similarly.


<span style="font-size: larger;">This text is larger.</span>
