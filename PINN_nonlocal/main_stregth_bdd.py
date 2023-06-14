import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from datetime import datetime
from sympy import Symbol
import random
import PINNModel.CalTool as PCal
import PINNModel.PtChange as PPtC
from PINNModel import ItofSin

os.environ["CUDA_VISIBLE_DIVICES"] = "0"

start_all = time.time()
np.random.seed(609)
tf.set_random_seed(609)

# f = open('sample.txt')
# x_sample = []
# for line in f.readlines():
#     num = line.split("\n")
#     x_sample.append(float(num[0]))
# f.close
# plt.hist(x_sample)
# x_sample = np.array(x_sample).reshape(-1,1)

pts = 1000
# 左右邊界的定義域(點)
dm = [0,1]
test_points = dm[0] + (dm[1]-dm[0])*np.random.rand(pts)
test_points = np.sort(test_points)
test_points = test_points.reshape(-1,1)

# test different epsilon
epsilon_save = {}
mode = "vw"

for epsilon in [0.05]:

    # 查看運算時間
    start = time.time()

    # save informations 
    folder_path = '論文結果整理'
    date_string = datetime.now().strftime('%Y%m%d%H%M%S')
    date_string += "_%.3f"%epsilon
    epsilon_save[epsilon] = {}
    epsilon_save[epsilon]["allfcn"] = []
    epsilon_save[epsilon]["loss_terms"] = []
    epsilon_save[epsilon]["grad_terms_r"] = []
    epsilon_save[epsilon]["grad_terms_ub"] = []
    epsilon_save[epsilon]["adaptive_cons"] = []
    epsilon_save[epsilon]["model"] = []
    epsilon_save[epsilon]["date_string"] = date_string
    
    if not os.path.exists(folder_path + "/" + date_string):
        os.makedirs(folder_path + "/" + date_string)

    if mode == "vw":
        dotimes = 2
    else:
        dotimes = 1
        
    for lc in range(dotimes):

        # linear combination
        if lc == 0:
            comb = [1,0]
        else:
            comb = [0,1]

        class PhysicsInformedNN:
            def __init__(self, x, xg0, xg1, layers, allfcn, input_weight = np.array([0]), learning_rate_cus=0.001, weight_loss=[1,1,1], method="no", mode="vw"):
                
                '''
                parameters : explanation; type, additional
                
                x : training points; nparray, shape = [[??]]
                
                layers : hidden layer; nparray, shape = [1,?,?,?,1]
                
                allfcn : function parameters; dictionary, the keys must include "eps", "afcn", "bfcn", "mu0", "mu1", "g0fcn", "g1fcn", "l0", "l1"
                
                input_weight : initial weights; nparray, shape follows the setting of "layers", default : np.array([0])
                
                learning_rate_cus : start learning rate for adam, float32, default : 0.001
                
                weight_loss : the weights of loss values in this order left, right and equation; list, default : [1,1,1]
                
                method : do you want using annealing scheme; str, "yes" or "no"(default)
                
                mode : how to approximate u; str, "u" or "vw"(default)
                
                '''
                self.lb = x.min(0)
                self.ub = x.max(0)

                self.x = x
                self.xg0 = xg0
                self.xg1 = xg1
                self.eps = allfcn['eps']
                self.afcn = allfcn['afcn']
                self.bfcn = allfcn['bfcn']
                self.mu0 = allfcn['mu0']
                self.mu1 = allfcn['mu1']
                self.g0fcn = allfcn['g0fcn']
                self.g1fcn = allfcn['g1fcn']
                self.l0 = allfcn['l0']
                self.l1 = allfcn['l1']
                self.method = method
                self.layers = layers
                
                self.input_weight = input_weight
                self.save_weight = []
                self.weights, self.biases = self.initialize_NN(layers)        

                config = tf.compat.v1.ConfigProto(
                    intra_op_parallelism_threads = 20,
                    inter_op_parallelism_threads = 20,
                    device_count={"CPU":10},
                    allow_soft_placement=True,
                    log_device_placement=True)
                
                self.sess = tf.Session(config=config)

                self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
                self.xg0_tf = tf.placeholder(tf.float32, shape=[None, self.xg0.shape[1]])
                self.xg1_tf = tf.placeholder(tf.float32, shape=[None, self.xg1.shape[1]])
                self.y_pred = self.neural_net(self.x_tf, self.weights, self.biases)
                # self.y_pred = self.neural_net_relative(self.x_tf, self.weights, self.biases)

                if mode == "vw":
                    self.y_loss = self.net_NS_vw(self.x_tf, self.eps, self.afcn, self.bfcn, weight_loss)
                    self.loss_r = self.y_loss[2]
                    self.loss_ubs = tf.convert_to_tensor(self.y_loss[0] + self.y_loss[1])
                else:
                    self.y_loss = self.net_NS_u(self.x_tf, self.xg0_tf, self.xg1_tf, self.eps, self.afcn, self.bfcn, weight_loss)
                    self.loss_r = self.y_loss[2]
                    self.loss_ub1 = self.y_loss[0] + self.y_loss[1]
                    self.loss_ub2 = self.y_loss[3] + self.y_loss[4]
                    self.loss_ubs = tf.convert_to_tensor([self.loss_ub1, self.loss_ub2])
                

                if self.method == "yes":
                    self.beta = 0.9
                    self.adaptive_constant_val = np.ones(self.loss_ubs.shape, dtype=np.float64)
                    self.adaptive_constant_tf = tf.placeholder(tf.float32, shape=self.adaptive_constant_val.shape)
                    self.loss = self.loss_r + tf.reduce_sum(tf.multiply(self.adaptive_constant_tf, self.loss_ubs))
                else:
                    self.adaptive_constant_tf = 1
                    self.loss = self.loss_r + tf.reduce_sum(self.loss_ubs)
                    self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, method = 'L-BFGS-B', 
                                                                            options = {'maxiter': 50000,
                                                                                        'maxfun': 50000,
                                                                                        'maxcor': 400,
                                                                                        'maxls': 50,
                                                                                        'ftol' : 1.0 * np.finfo(float).eps})
                # tf.gradients 無法同時計算兩項量的梯度
                if mode == "vw":
                    self.grad_r = []
                    self.grad_ubs = []

                    for i in range(len(self.layers) - 1):
                        self.grad_r.append(tf.gradients(self.loss_r, self.weights[i])[0])
                        self.grad_ubs.append(tf.gradients(self.loss_ubs, self.weights[i])[0])


                    if self.method == "yes":
                        self.max_grad_res_list = []
                        self.mean_grad_bcs_list = []

                        for i in range(len(self.layers) - 1):
                            self.max_grad_res_list.append(tf.reduce_max(tf.abs(self.grad_r[i])))
                            self.mean_grad_bcs_list.append(tf.reduce_mean(tf.abs(self.grad_ubs[i])))

                        self.max_grad_res = tf.reduce_max(tf.stack(self.max_grad_res_list))
                        self.mean_grad_bcs = tf.reduce_mean(tf.stack(self.mean_grad_bcs_list))
                        self.adaptive_constant = self.max_grad_res / self.mean_grad_bcs

                else:
                    self.grad_r = []
                    self.grad_ub1 = []
                    self.grad_ub2 = []

                    for i in range(len(self.layers) - 1):
                        self.grad_r.append(tf.gradients(self.loss_r, self.weights[i])[0])
                        self.grad_ub1.append(tf.gradients(self.loss_ub1, self.weights[i])[0])
                        self.grad_ub2.append(tf.gradients(self.loss_ub2, self.weights[i])[0])


                    if self.method == "yes":
                        self.max_grad_res_list = []
                        self.mean_grad_bcs_list1 = []
                        self.mean_grad_bcs_list2 = []

                        for i in range(len(self.layers) - 1):
                            self.max_grad_res_list.append(tf.reduce_max(tf.abs(self.grad_r[i])))
                            self.mean_grad_bcs_list1.append(tf.reduce_mean(tf.abs(self.grad_ub1[i])))
                            self.mean_grad_bcs_list2.append(tf.reduce_mean(tf.abs(self.grad_ub2[i])))

                        self.max_grad_res = tf.reduce_max(tf.stack(self.max_grad_res_list))
                        self.mean_grad_bc1 = tf.reduce_mean(tf.stack(self.mean_grad_bcs_list1))
                        self.mean_grad_bc2 = tf.reduce_mean(tf.stack(self.mean_grad_bcs_list2))
                        lambda1 = self.max_grad_res / self.mean_grad_bc1
                        lambda2 = self.max_grad_res / self.mean_grad_bc2
                        self.adaptive_constant = tf.convert_to_tensor([lambda1, lambda2])


                self.global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = learning_rate_cus
                self.learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step, 1000, 0.9, staircase=False)
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

                init = tf.global_variables_initializer()
                self.sess.run(init)

            def simp(self, y, x):
                dx = (x[-1] - x[0]) / (int(x.shape[0]) - 1)
                return (y[0] + y[-1] + 4*tf.reduce_sum(y[1:-1:2]) + 2*tf.reduce_sum(y[2:-1:2])) * dx / 3
        
            def dis(self, x, y):
                return tf.reduce_mean(tf.square(x-y))
            
            def initialize_NN(self, layers):  

                weights = []
                biases = []
                num_layers = len(layers)

                for l in range(0,num_layers-1):
                    W = self.Xavier_init(size=[layers[l], layers[l+1]], counts = l)
                    b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), name='bias', dtype=tf.float32)
                    weights.append(W)
                    biases.append(b)

                return weights, biases

            
            def Xavier_init(self, size, counts):

                if len(self.input_weight) == 1:
                    in_dim = size[0]
                    out_dim = size[1]        
                    inte = np.sqrt(6/(in_dim + out_dim)) 
                    return tf.Variable(tf.random.uniform([in_dim, out_dim], minval=-inte, maxval=inte), name='weight', dtype=tf.float32)

                else:
                    ck = input('第%d層隱藏層要使用輸入之權重?(Y/N)  '% (counts+1) )

                    if ck == 'N':
                        in_dim = size[0]
                        out_dim = size[1]        
                        inte_stddev = np.sqrt(2/(in_dim + out_dim)) 
                        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=inte_stddev), name='weight', dtype=tf.float32)
                    else:
                        return self.input_weight[counts]

                    
            def gelu(self, x):
                return x * 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))
                    
            # fully-connected
            def neural_net(self, x, weights, biases):
                num_layers = len(weights) + 1

                H = tf.convert_to_tensor(x, tf.float32)

                for l in range(0,num_layers-2):
                    W = weights[l]
                    b = biases[l]
                    H = tf.math.tanh(tf.add(tf.matmul(H, W), b))

                W = weights[-1]
                b = biases[-1]
                Y = tf.add(tf.matmul(H, W), b)

                return Y

            # fully-connected little different
            def neural_net_relative(self, x, weights, biases):

                num_layers = len(weights) + 1

                H = tf.convert_to_tensor(x, tf.float32)

                for l in range(0,num_layers-2):

                    W = weights[l]
                    b = biases[l]

                    if l==0:
                        H = tf.math.tanh(tf.add(tf.matmul(H, W), b)) + H
                    else:
                        H = tf.math.tanh(tf.add(tf.matmul(H, W), b))

                W = weights[-1]
                b = biases[-1]
                Y = tf.add(tf.matmul(H, W), b)
                
                return Y    


            def net_NS_vw(self, x, eps, afcn, bfcn, weights):

                y = self.neural_net(x, self.weights, self.biases)
                yt = tf.gradients(y,x)[0]
                ytt = tf.gradients(yt,x)[0]

                # v(x) and w(x) are same
                # for "lambda function of multiple inputs"
                afcn = tf.map_fn(afcn, x)
                afcn = tf.reshape(afcn, shape=[-1,1])
                bfcn = tf.map_fn(bfcn, x)
                bfcn = tf.reshape(bfcn, shape=[-1,1])
                equ = eps**2*ytt + eps*afcn*yt - bfcn*y

                return weights[0] * self.dis(y[0], comb[0]), weights[1] * self.dis(y[-1], comb[1]), weights[2] * self.dis(equ, 0)

            def net_NS_u(self, x, xg0, xg1, eps, afcn, bfcn, weights):

                y = self.neural_net(x, self.weights, self.biases)
                yt = tf.gradients(y,x)[0]
                ytt = tf.gradients(yt,x)[0]
                afcn = tf.map_fn(afcn, x)
                afcn = tf.reshape(afcn, shape=[-1,1])
                bfcn = tf.map_fn(bfcn, x)
                bfcn = tf.reshape(bfcn, shape=[-1,1])
                y_equ = eps**2*ytt + eps*afcn*yt - bfcn*y
                
                yg0 = self.neural_net(xg0, self.weights, self.biases)
                g0u = tf.map_fn(self.g0fcn, xg0)
                g0u = tf.reshape(g0u, shape=[-1,1])
                y_bddl = self.mu0 + self.simp(g0u*yg0, self.xg0) / eps

                yg1 = self.neural_net(xg1, self.weights, self.biases)
                g1u = tf.map_fn(self.g1fcn, xg1)
                g1u = tf.reshape(g1u, shape=[-1,1])
                y_bddr = self.mu1 + self.simp(g1u*yg1, self.xg1) / eps
                
                y_bddlint = eps**2*ytt[0] + eps*afcn*yt[0] - bfcn*y[0]
                y_bddrint = eps**2*ytt[-1] + eps*afcn*yt[-1] - bfcn*y[-1]

                return weights[0] * self.dis(y_bddl, y[0]), weights[1] * self.dis(y[-1], y_bddr), weights[2] * self.dis(y_equ , 0), \
                    weights[3] * self.dis(y_bddlint, 0), weights[4] * self.dis(y_bddrint, 0)

            
            def callback(self, it, loss_value, loss_terms, elapsed):
                print('It: %d, Loss: %.3e, loss_r: %.3e, loss_ub: %.3e Time: %.2f' %
                    (it, loss_value, loss_terms[-1][0]+loss_terms[-1][1], loss_terms[-1][2], elapsed))


            def annealing_update(self, xf_dict):
                
                adaptive_constant_value = self.sess.run(self.adaptive_constant, xf_dict)
                self.adaptive_constant_val = adaptive_constant_value * (1 - self.beta) \
                                            + self.beta * self.adaptive_constant_val

                return self.adaptive_constant_val
                

            def train(self, nIter):
                
                if self.method == "yes":
                    xf_dict = {self.x_tf: self.x, self.xg0_tf : self.xg0, self.xg1_tf : self.xg1, self.adaptive_constant_tf: self.adaptive_constant_val}
                else:
                    xf_dict = {self.x_tf: self.x, self.xg0_tf : self.xg0, self.xg1_tf : self.xg1}
                    
                start_time = time.time()
                loss=[]
                loss_terms = []
                grad_terms_r = []
                grad_terms_ub = []
                adaptive_cons = []

                for it in range(nIter):

                    self.sess.run(self.train_op, xf_dict)
                    loss_ = self.sess.run(self.y_loss, xf_dict)
                    if mode == "vw":
                        loss_terms.append([loss_[i] for i in range(3)])
                    else:
                        loss_terms.append([loss_[i] for i in range(5)])

                    if it % 10 == 0:

                        elapsed = time.time() - start_time
                        loss_value = self.sess.run(self.loss, xf_dict)

                        grad_terms_r.append(self.sess.run(self.grad_r, xf_dict))
                        if mode == "vw":
                            grad_terms_ub.append(self.sess.run(self.grad_ubs, xf_dict))
                        else:
                            grad_terms_ub.append(self.sess.run(self.grad_ub1, xf_dict))

                        if self.method == "yes":
                            adaptive_cons.append(self.annealing_update(xf_dict))
                            self.callback(it, loss_value, loss_terms, elapsed)
                        else:
                            self.callback(it, loss_value, loss_terms, elapsed)
                            pass

                        start_time = time.time()
                    loss.append(loss_value)

                
                if self.method == "no":
                        self.optimizer.minimize(self.sess, feed_dict = xf_dict, fetches = [self.loss],
                                                loss_callback = self.callback(it, loss_value, loss_terms, elapsed))

                return loss, loss_terms, grad_terms_r, grad_terms_ub, adaptive_cons


            def predict(self, x_star):
                
                if self.method == "yes":
                    y_star = self.sess.run(self.y_pred, feed_dict = {self.x_tf: x_star, self.adaptive_constant_tf: self.adaptive_constant_val})
                else:
                    y_star = self.sess.run(self.y_pred, feed_dict = {self.x_tf: x_star})

                return y_star


            
#=================================================================MAIN=================================================================#
        if __name__ == "__main__": 

            points = 10000
                
            layers = [1,30,30,1]
            allfcn = {'eps': epsilon, 'afcn' : lambda x : 1, 'bfcn' : lambda x : 1,
                    'mu0' : 1, 'mu1' : 1,
                    'g0fcn' : lambda x : 1, 'g1fcn' : lambda x : 1,
                    'l0' : 1/2, 'l1':1/2}
            
            if "x_sample" not in locals().keys():
                x_train = np.linspace(dm[0],dm[1],points).reshape(-1,1)
                xg0 = np.linspace(allfcn["l0"], dm[1], x_train.shape[0]).reshape(-1,1)
                xg1 = np.linspace(dm[0], allfcn["l1"], x_train.shape[0]).reshape(-1,1)
            else:
                x_train = x_sample

            learning_rate_cus = 0.001
            method = "no"
            mode = mode
            train_times = 2000
            if lc == 0:
                weight_loss_v = [1, 1, 1]
                # weight_loss_v = [1,1,100,1,1]  # u
                model = PhysicsInformedNN(x_train, xg0, xg1, layers, allfcn, learning_rate_cus = learning_rate_cus, weight_loss=weight_loss_v, method=method, mode=mode)
            else:
                weight_loss_w = [1, 1, 1]
                model = PhysicsInformedNN(x_train, xg0, xg1, layers, allfcn, learning_rate_cus = learning_rate_cus, weight_loss=weight_loss_w, method=method, mode=mode)
            
            loss, loss_terms, grad_terms_r, grad_terms_ub, adaptive_cons = model.train(train_times)
            epsilon_save[epsilon]["loss_terms"].append(loss_terms)
            epsilon_save[epsilon]["grad_terms_r"].append(grad_terms_r)
            epsilon_save[epsilon]["grad_terms_ub"].append(grad_terms_ub)
            epsilon_save[epsilon]["adaptive_cons"].append(adaptive_cons)
            epsilon_save[epsilon]["model"].append(model)
    

    # record informations
    with open(folder_path + '/%s/record_%s.txt'%(date_string,date_string), 'w') as f:
        f.write('\n\n==========================PROBLEM_INFORMATION==========================\n\n')
        f.write('eps^2u''(x) + epsa(x)u(x) - b(x)u(x)=0,\n u(0) = mu0 + int_{l0}^{1}g0(x)u(x)dx/eps, u(1) = mu1 + int_{0}^{l1}g1(x)u(x)dx/eps\n')
        f.write('eps = %f\n'%allfcn['eps'])
        f.write('a(x)=' + str(allfcn['afcn'](Symbol('x'))))
        f.write('\n')
        f.write('b(x)=' + str(allfcn['bfcn'](Symbol('x'))))
        f.write('\nμ_0 = %f\n'%allfcn['mu0'])
        f.write('μ_1 = %f\n'%allfcn['mu1'])
        f.write('g0(x)=' + str(allfcn['g0fcn'](Symbol('x'))))
        f.write('\n')
        f.write('g1(x)=' + str(allfcn['g1fcn'](Symbol('x'))))
        f.write('\n')
        f.write('\n\n==========================MODEL_INFORMATION==========================\n\n')
        f.write('\nlayers :')
        f.write(str(layers))
        f.write("\nannealing scheme : %s"%method)
        f.write("\nu or vw : %s"%mode)
        f.write('\nstart learning rate of Adam : ')
        f.write(str(learning_rate_cus))
        if mode == "vw":
            f.write('\nweight of left boundary point_v : %s\n'%weight_loss_v[0])
            f.write('weight of right boundary point_v : %s\n'%weight_loss_v[1])
            f.write('weight of interior points_v : %s\n'%weight_loss_v[2])
            f.write('loss_value_left_v : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][0][-1][0])
            f.write('loss_value_right_v : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][0][-1][1])
            f.write('loss_value_equ_v : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][0][-1][2])
            f.write('\nweight of left boundary point_w : %s\n'%weight_loss_w[0])
            f.write('weight of right boundary point_w : %s\n'%weight_loss_w[1])
            f.write('weight of interior points_w : %s\n'%weight_loss_w[2])
            f.write('loss_value_left_w : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][1][-1][0])
            f.write('loss_value_right_w : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][1][-1][1])
            f.write('loss_value_equ_w : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][1][-1][2])
        else:
            f.write('\nweight of left boundary point_v : %s\n'%weight_loss_v[0])
            f.write('weight of right boundary point_v : %s\n'%weight_loss_v[1])
            f.write('weight of interior points_v : %s\n'%weight_loss_v[2])
            f.write('weight of right boundary point_aux_v : %s\n'%weight_loss_v[3])
            f.write('weight of left boundary point_aux_v : %s\n'%weight_loss_v[4])
            f.write('\nloss_value_left_u : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][0][-1][0])
            f.write('loss_value_right_u : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][0][-1][1])
            f.write('loss_value_equ_u : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][0][-1][2])
            f.write('loss_value_left_aux_u : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][0][-1][3])
            f.write('loss_value_right_aux_u : %010.5e\n'%epsilon_save[epsilon]["loss_terms"][0][-1][4])

        f.write('\n\n==========================TRAIN_INFORMATION==========================\n\n')
        if "x_sample" not in locals().keys():
            f.write('the number of points for training : %d\n'%points)
        else:
            f.write('the number of points for training(sample) : %d\n'%len(x_sample))
        f.write("train_times : %d\n"%train_times)
        f.write('training points : \n')
        f.write(str(x_train))

    f.close()
    print("spend : %.5f seconds"%(time.time()-start))
    epsilon_save[epsilon]["time"] = time.time()-start
    

# =================TEST================= #
# test_points = x_train

if mode == "vw":
    start = time.time()
    for item in epsilon_save.items():

        allfcn["eps"] = item[0]
        vw = item[1]["model"]
        g01 = [allfcn['g0fcn'],allfcn['g1fcn']]

        print("allfcn :", allfcn)

        low = np.array(allfcn['l0'])
        upp = np.array(allfcn['l1'])
        pts = 10**5

        print("\n\ng01 :", g01,"\n\nvw :", vw,"\n\nlow :", low,"\n\nupp :", upp,"\n\npts :", pts)
        AMat = ItofSin.IntegralofSingular(g01,vw,low,upp,pts)
        A_pred, ub_solve_inv = AMat.get_A(allfcn['eps'], np.array([allfcn['mu0'],allfcn['mu1']]))
        with open(folder_path + '/%s/record_%s.txt'%(item[1]["date_string"], item[1]["date_string"]), 'a') as f:
            f.write("\n\n=================AMATRIX_PRED=================\n\n")
            f.write(str(A_pred)+"\n\n")
            f.close()
        u_solve_inv = AMat.solve_u(test_points, ub_solve_inv, [vw[0].predict(test_points), vw[1].predict(test_points)])
        epsilon_save[item[0]]["solve_inv"] = [ub_solve_inv, u_solve_inv]
        plt.plot(test_points, u_solve_inv)
        plt.title("calculating boundary by inverse matrix")
        plt.show()

        epsilon_save[item[0]]["time"] += (time.time()-start)
        start = time.time()
    md = "M1"
    print("spend : %d" % (time.time()-start))
    
else:
    print("We use mode %s"%mode)



if mode == "vw":
    label = ["loss_l", "loss_r", "loss_equ"]
    label_ = ["v", "w"]
    ncols = 3
else:
    label = ["loss_l", "loss_r", "loss_equ", "loss_l_equ", "loss_r_equ"]
    label_ = ["u"]
    ncols = 5


for item in epsilon_save.items():

    # loss_value
    fig, axes = plt.subplots(nrows=len(label_), ncols=ncols, figsize=(15,8))
    for idx, ax in enumerate(axes.ravel()):
        ax.plot([i[idx%ncols] for i in item[1]["loss_terms"][idx//ncols]])
        ax.set_title(label[idx%ncols] + "_" + label_[idx//ncols])

    plt.savefig(folder_path + '/%s/loss_terms.jpg'%(item[1]["date_string"]))
    plt.show()

    # grad_histogram
    try:
        grad_v_r = item[1]["grad_terms_r"][0]
        grad_v_ub = item[1]["grad_terms_ub"][0]
        grad_w_r = item[1]["grad_terms_r"][1]
        grad_w_ub = item[1]["grad_terms_ub"][1]
        PPtC.grad_hist(grad_v_r, grad_v_ub, 100, layers, label_[0], folder_path, item[1]["date_string"])
        PPtC.grad_hist(grad_w_r, grad_w_ub, 100, layers, label_[1], folder_path, item[1]["date_string"])
    except:
        PPtC.grad_hist(grad_v_r, grad_v_ub, 100, layers, label_[0], folder_path, item[1]["date_string"])


    # grad_change
    try:
        grad_plot_v_r = PPtC.grad_plot(item[1]["grad_terms_r"][0], layers)
        grad_plot_v_ub = PPtC.grad_plot(item[1]["grad_terms_ub"][0], layers)
        grad_plot_w_r = PPtC.grad_plot(item[1]["grad_terms_r"][1], layers)
        grad_plot_w_ub = PPtC.grad_plot(item[1]["grad_terms_ub"][1], layers)
        grad_plot_terms = [grad_plot_v_r, grad_plot_v_ub, grad_plot_w_r, grad_plot_w_ub]
        label_2 = ["grad_change_v_r", "grad_change_v_ub", "grad_change_w_r", "grad_change_w_ub"]
    except:
        grad_plot_terms = [grad_plot_v_r, grad_plot_v_ub]
        label_2 = ["grad_change_u_r", "grad_change_u_ub"]
    

    fig, axes = plt.subplots(nrows=len(label_), ncols=2, figsize=(15,12))
    for idx, ax in enumerate(axes.ravel()):
        for i in grad_plot_terms[idx]:
            ax.plot(i)
        ax.set_title(label_2[idx])
        ax.set_xlabel("ite")
        ax.set_ylabel(r'$\bigtriangledown$W', rotation=0)
    plt.savefig(folder_path + '/%s/grad_change.jpg'%(item[1]["date_string"]))
    plt.show()

    if method == "yes":
        if mode == "vw":
            plt.figure()
            plt.plot(item[1]["adaptive_cons"][0])
            plt.plot(item[1]["adaptive_cons"][1])
            plt.xlabel("ite")
            plt.ylabel("λ", rotation=0)
            plt.title("adaptive_constant_change")
            plt.legend(["v","w"])
            plt.savefig(folder_path + '/%s/adap_const.jpg'%item[1]["date_string"])
            plt.show()

        else:
            plt.figure()
            plt.plot(np.array(item[1]["adaptive_cons"][0])[:,0])
            plt.plot(np.array(item[1]["adaptive_cons"][0])[:,1])
            plt.xlabel("ite")
            plt.ylabel("λ", rotation=0)
            plt.title("adaptive_constant_change")
            plt.legend(["ub, ub_aux"])
            plt.savefig(folder_path + '/%s/adap_const.jpg'%item[1]["date_string"])
            plt.show()
    else:
        pass

# =================EXACT================= #

mu = [allfcn['mu0'],allfcn['mu1']]

for item in epsilon_save.items():
    
    eps = item[0]
    
    A = -1+np.sqrt(5)
    B = -1-np.sqrt(5)
    Cv = np.exp(A/(2*eps)) / (np.exp(A/(2*eps)) - np.exp(B/(2*eps)))
    Cw = 1 / (np.exp(B/(2*eps)) - np.exp(A/(2*eps)))
    epsilon_save[eps]["Cvw"] = [Cv,Cw]
    alp = np.exp(A/(2*eps)) - np.exp(A/(4*eps))
    bet = np.exp(B/(2*eps)) - np.exp(B/(4*eps))
    A11 = 1 - ((1-Cv) * 2/A * alp + Cv * 2/B * bet)
    A12 = Cw * 2/A * alp - Cw * 2/B * bet
    gam = np.exp(A/(4*eps))-1
    et = np.exp(B/(4*eps))-1
    A21 = (Cv-1) * 2/A * gam - Cv *2/B * et
    A22 = 1 - (-Cw * 2/A * gam + Cw * 2/B * et)
    A, ub_exact, condA = PCal.u0u1(A11,A12,A21,A22,mu)
    epsilon_save[eps]["ub_exact"] = ub_exact
    with open(folder_path + '/%s/record_%s.txt'%(item[1]["date_string"], item[1]["date_string"]), 'a') as f:
        f.write("\n\n=================AMATRIX_TRUE=================\n\n")
        f.write(str(A)+"\n\n")
        f.write("\n\n==========================CONDITION_NUMBER==========================\n\n")
        f.write("\nκ(A) = %.5f\n" % condA)
        f.write('\n\n==========================EXACT_SOLUTION==========================\n\n')
        f.write('u exact solution : \n')
        f.write(str(PCal.u_exact_direct(test_points, eps)))
        f.write('\n\n==========================PREDICT_SOLUTION==========================\n\n')
        if mode== "vw":
            f.write('u predicted solution : \n')
            f.write(str(item[1]["solve_inv"][1]))
        else:
            pass
        f.close()
        
    plt.figure()
    plt.plot(test_points, PCal.u_exact_direct(test_points, eps), "r")
    plt.plot(test_points, ub_exact[0]*PCal.V(test_points,eps) + ub_exact[1]*PCal.W(test_points,eps))
    plt.legend(["direct calculation", "calculation by v and w"])
    plt.title("check exact solution is right")
    plt.savefig(folder_path + '/' + item[1]["date_string"] + '/' + 'compare_truth_solution.jpg')
    plt.show()


#======================Result======================#
for item in epsilon_save.items():
    
    eps = item[0]
    date_string = item[1]["date_string"]
    v_exact = PCal.V(test_points, eps)
    w_exact = PCal.W(test_points, eps)

    if mode == "vw":
        
        plt.figure()
        vw = item[1]["model"]
        plt.plot(test_points, v_exact, 'r', linewidth=3)
        plt.plot(test_points, vw[0].predict(test_points), 'o', markersize=0.5)
        plt.legend(['exact','approximation'])
        plt.title('v_error')
        plt.savefig(folder_path + '/' + date_string + '/' + 'v_error_%s.jpg'%(date_string))
        plt.show()

        plt.figure()
        plt.plot(test_points, w_exact, 'r', linewidth=3)
        plt.plot(test_points, vw[1].predict(test_points), 'o', markersize=0.5)
        plt.legend(['exact','approximation'])
        plt.title('w_error')
        plt.savefig(folder_path + '/' + date_string + '/' + 'w_error_%s.jpg'%(date_string))
        plt.show()

        plt.figure()
        plt.plot(test_points, PCal.u_exact_direct(test_points, eps), 'r', linewidth=3)
        plt.plot(test_points, item[1]["solve_inv"][1], 'o', markersize=0.5)
        plt.legend(['exact', 'approximation'])
        plt.title("calculating boundary by inverse matrix")
        plt.savefig(folder_path + '/' + date_string + '/' + 'u_error_%s.jpg'%(date_string))
        plt.show()

        vw_error = PCal.two_norm(v_exact, vw[0].predict(test_points)), PCal.two_norm(w_exact, vw[1].predict(test_points))
        u_error = PCal.two_norm(PCal.u_exact_direct(test_points, eps), item[1]["solve_inv"][1])
        print("error_v and error_w (two norm) :", vw_error)
        print("error_u (two norm) :", u_error)
        print("ub_app :", item[1]["solve_inv"][0])
        print("ub_exact :", item[1]["ub_exact"])

        with open(folder_path + '/%s/record_%s.txt'%(date_string,date_string), 'a') as f:
            f.write('\n\n==========================ERROR_EFFICIENCY==========================\n\n')
            f.write("v_error : %010.5e\n"%vw_error[0])
            f.write("w_error : %010.5e\n"%vw_error[1])
            f.write("u_error : %010.5e\n\n"%u_error[0])
            f.write("\ntime : %.5f seconds.\n"%epsilon_save[epsilon]["time"])
            f.close()
            
    else:
        u = item[1]["model"]
        u_pred = u[0].predict(test_points)
        u_exact = PCal.u_exact_direct(test_points, eps)
        fig, ax = plt.subplots(1,2, figsize=(15, 12))
        ax[0].set_title('$\hat{u}$')
        ax[0].plot(test_points, u_pred, 'o', markersize=0.5)
        ax[1].set_title('$\hat{u}$ and u')
        ax[1].plot(test_points, u_exact, 'r', linewidth=3)
        ax[1].plot(test_points, u_pred, 'o', markersize=0.5)
        ax[1].legend(['exact', 'approximation'])
        plt.savefig(folder_path + '/' + date_string + '/' + 'u_error_%s.jpg'%(date_string))

        u_error = PCal.two_norm(u_exact, u_pred)
        print("error_u (two norm) :", u_error)

        with open(folder_path + '/%s/record_%s.txt'%(date_string,date_string), 'a') as f:
            f.write('\n\n==========================ERROR_U==========================\n\n')
            f.write("u_error : %010.5e\n\n"%u_error[0])
            f.write("\ntraining_time : %.5f seconds."%epsilon_save[epsilon]["time"])
            f.close() 

print("total spend : %d" % (time.time()-start_all))