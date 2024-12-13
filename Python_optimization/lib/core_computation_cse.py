#import numpy as np
import math
import autograd.numpy as np  # Use autograd's numpy wrapper
from autograd import jacobian


class OptimizationProcess:

    """
    Class for optimization process

    Each optimization process, cost function regarding to one instance of magnetic data is needed to optimize the parameters to fit the magnetic data solution.
    Inside optimization process, each iteration in the algorithm will change the parameters feed and compute the cost value. Therefore, it is required to wrap the cost function to each data.
    Magnetic model, jacobian based on https://ieeexplore.ieee.org/document/8809206

    9DOF of jacobian is already implemented. However, cost function is not. But only make the other 3DOF as arguments should be enough to modify the cost function.
    """
    def __init__(self,N_sensor,N_magnets):
        self.N_magnets = N_magnets
        #self.params = np.zeros((N_magnets, 6))  # Each magnet has 6 parameters: x, y, z, theta, phi, m_j
        self.Gx = 0.0
        self.Gy = 0.0
        self.Gz = 0.0
        self.N_sensor = N_sensor
        
        ###Parameters is mostly only useful when multi-step optimization is required, then parameters could be stored here
        
        self.sensor_pos = np.loadtxt('sensor_position_30.txt').reshape(-1,1)*1e-3 #1e-3 mm -> m
        
    def B_model(self, x_i, y_i, z_i, theta, phi, m_j):
        r_sq = x_i**2 + y_i**2 + z_i**2
        r_sq_52 = r_sq**(5/2)
        r_sq_32 = r_sq**(3/2)
        product_sum = np.sin(theta)*np.cos(phi)*x_i + np.sin(theta)*np.sin(phi)*y_i + np.cos(theta)*z_i
        Bx = m_j*(3*x_i*product_sum/r_sq_52 - np.sin(theta)*np.cos(phi)/r_sq_32)
        By = m_j*(3*y_i*product_sum/r_sq_52 - np.sin(theta)*np.sin(phi)/r_sq_32)
        Bz = m_j*(3*z_i*product_sum/r_sq_52 - np.cos(theta)/r_sq_32)
        return Bx, By, Bz
    

    def cost_9DOF(self, data):
        #print("Initialized cost_func_factory")
        def cost_func (param):
            '''
            Parameters
            ----------
            param: Array
                Vector of 9 DOF, x,y,z,theta,phi,m_j,Gx,Gy,Gz
            
            Returns
            ----------
            result: Array
                Vector of cost [x1,y1,z1,x2...] 3xN
            '''
            
            sensor_B = data
            Gx = param[6*self.N_magnets]
            Gy = param[6*self.N_magnets+1]
            Gz = param[6*self.N_magnets+2]
            cost_value = np.zeros([3*self.N_sensor], dtype=np.float64)
            for i in range(0, np.size(sensor_B), 3):
                for j in range(self.N_magnets):
                    #idx = j*9
                    idx = j*6
                    x_i = self.sensor_pos[i] - param[idx]
                    y_i = self.sensor_pos[i+1] - param[idx+1]
                    z_i = self.sensor_pos[i+2] - param[idx+2]
                    theta = param[idx+3]
                    phi = param[idx+4]
                    m_j = param[idx+5]
                    
                    #Gx = param[idx+6]
                    #Gy = param[idx+7]
                    #Gz = param[idx+8]


                    Bx, By, Bz =  self.B_model(x_i,y_i,z_i,theta,phi,m_j)

                    cost_value[i] += Bx
                    cost_value[i+1] += By
                    cost_value[i+2] += Bz
                
                cost_value[i] += Gx - sensor_B[i//3,i%3]
                cost_value[i+1] += Gy - sensor_B[i//3,i%3+1] 
                cost_value[i+2] += Gz - sensor_B[i//3,i%3+2]
            
            #print("Parameters inside cost: ", param[0:3])
            
            return cost_value
        return cost_func
    
    

    def jacobian_9DOF (self,param):
        
        '''
        Parameters
        ----------
        param : Array
            Vector of 9 DOF, x,y,z,theta,phi, mj, Gx, Gy, Gz (constant)
        
        Returns
        -------
        result : Array
            Combonents of jacobian, [3*N,9*M]

        '''
        J_mtx = np.zeros([3*self.N_sensor,6*self.N_magnets+3])
        #J_mtx = np.zeros([3*self.N_sensor,9*self.N_magnets])
        D = np.identity(3)
        for i in range(0, 3*self.N_sensor, 3):
            for j in range(self.N_magnets):
                #idx = j*9
                idx = j*6
                x_i = self.sensor_pos[i] - param[idx]   ### Should not combine with cost function xi, yi, zi, because iteration of jacobian and cost function are different
                y_i = self.sensor_pos[i+1] - param[idx+1]
                z_i = self.sensor_pos[i+2] - param[idx+2]
                theta = param[idx+3]
                phi = param[idx+4]
                m_j = param[idx+5]
                #print(f"xi[{i}]: ", x_i)
                #print(f"yi[{i}]: ", y_i)
                #print(f"zi[{i}]: ", z_i)
                #print(f"theta[{i}]: ", theta)
                #print(f"phi[{i}]: ", phi)
                #print(f"mj[{i}]: ", m_j)
            
                sub_J = np.zeros([3,6])

                r_sq = x_i**2 + y_i**2 + z_i**2
                r_sq_52 = r_sq**(5/2)
                r_sq_72 = r_sq**(7/2)
                ma_7 = m_j/r_sq_72
                ma_5 = m_j/r_sq_52
                a_5 = 1/r_sq_52

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                # Sub-Jacobian components
                common_1 = 3 * (4 * x_i**2 - y_i**2 - z_i**2)
                common_2 = 3 * (-1 * x_i**2 + 4 * y_i**2 - z_i**2)
                common_3 = 15 * x_i * y_i * z_i
                common_4 = 3 * (-1 * x_i**2 - y_i**2 + 4 * z_i**2)
                
                common_5 = 2 * x_i**2 - y_i**2 - z_i**2
                common_6 = -1 * x_i**2 + 2 * y_i**2 - z_i**2
                common_7 = -1 * x_i**2 - y_i**2 + 2 * z_i**2
                common_8 = 3 * x_i * y_i
                common_9 = 3 * x_i * z_i
                common_10 = 3 * y_i * z_i

                sub_J[0,0] = ma_7 * (3*x_i*(2*x_i**2-3*y_i**2-3*z_i**2)*sin_theta*cos_phi + common_1*y_i*sin_theta*sin_phi + common_1*z_i*cos_theta)
                sub_J[1,0] = ma_7 * (common_1*y_i*sin_theta*cos_phi + common_2*x_i*sin_theta*sin_phi + common_3*cos_theta)
                sub_J[2,0] = ma_7 * (common_1*z_i*sin_theta*cos_phi + common_3*sin_theta*sin_phi + common_4*x_i*cos_theta)
                sub_J[0,1] = sub_J[1,0]
                sub_J[1,1] = ma_7 * (common_2*x_i*sin_theta*cos_phi + 3*y_i*(-3*x_i**2+2*y_i**2-3*z_i**2)*sin_theta*sin_phi + common_2*z_i*cos_theta)
                sub_J[2,1] = ma_7 * (common_3*sin_theta*cos_phi + common_2*z_i*sin_theta*sin_phi + common_4*y_i*cos_theta)
                sub_J[0,2] = sub_J[2,0]
                sub_J[1,2] = sub_J[2,1]
                sub_J[2,2] = ma_7 * (common_4*x_i*sin_theta*cos_phi + common_4*y_i*sin_theta*sin_phi + 3*z_i*(-3*x_i**2-3*y_i**2+2*z_i**2)*cos_theta)

                sub_J[0,3] = ma_5 * (common_5*cos_phi*cos_theta + common_8*sin_phi*cos_theta - common_9*sin_theta)
                sub_J[1,3] = ma_5 * (common_8*cos_phi*cos_theta + common_6*sin_phi*cos_theta - common_10*sin_theta)
                sub_J[2,3] = ma_5 * (common_9*cos_phi*cos_theta + common_10*sin_phi*cos_theta - common_7*sin_theta)
                sub_J[0,4] = ma_5 * (common_8*sin_theta*cos_phi - common_5*sin_theta*sin_phi)
                sub_J[1,4] = ma_5 * (common_6*sin_theta*cos_phi - common_8*sin_theta*sin_phi)
                sub_J[2,4] = ma_5 * (common_10*sin_theta*cos_phi - common_9*sin_theta*sin_phi)
                sub_J[0,5] = a_5 * (common_5*sin_theta*cos_phi + common_8*sin_theta*sin_phi + common_9*cos_theta)
                sub_J[1,5] = a_5 * (common_8*sin_theta*cos_phi + common_6*sin_theta*sin_phi + common_10*cos_theta)
                sub_J[2,5] = a_5 * (common_9*sin_theta*cos_phi + common_10*sin_theta*sin_phi +common_7*cos_theta)

                J_mtx[i:i+3,idx:idx+6] = sub_J
                #J_mtx[i:i+3,idx+6:idx+9] = D    
            J_mtx[i:i+3,6*self.N_magnets:6*self.N_magnets+3] = D  

        
        return J_mtx