# Read data from lis3mdl sensor with using PCA9548A multiplexer

import time
import machine
from PCA9548A import PCA9548A
import lis3mdl 



class PCA_LIS_magnetometer:
    
    # i2c information
    PORT_1 = const(0)
    PORT_2 = const(1)
    
    # Multiplexer pins
    SDA_MP1  = const (4) # PCA pin SDA connected to pin GPIO4 on esp32
    SCL_MP1  = const (5) # PCA pin SCL connected to pin GPIO5 on esp32
    MP1_ADDR = const (0x70) ## PCA pins A0,A1,A2 connected to 0,0,0 => 0x70
    NUM_SENSORS_MP1 = const (8) 
    
    SDA_MP2 = const (6) # PCA pin SDA connected to pin GPIO6 on esp32
    SCL_MP2 = const (7) # PCA pin SCL connected to pin GPIO7 on esp32
    MP2_ADDR = const (0x71) ## PCA pins A0,A1,A2 connected to 1,0,0 => 0x71
    NUM_SENSORS_MP2 = const(8)
    
    def __init__(self):
        self.mp1 = PCA9548A(PORT_1, scl=SCL_MP1, sda=SDA_MP1, address=MP1_ADDR) ### Here, port is not specify, could have problem for later
        self.mp2 = PCA9548A(PORT_2, scl=SCL_MP2, sda=SDA_MP2, address=MP2_ADDR)
        self.mp1.enable_only_channel(0)
        self.lis1 = lis3mdl.LIS3MDL(self.mp1.get_i2c()) ##the 1 correspond to the first multiplexer
        self.mp2.enable_only_channel(0)
        self.lis2 = lis3mdl.LIS3MDL(self.mp2.get_i2c()) ##the 2 correspond to the second multiplexer
        print("Multiplexer initialized")
        
        for i in range(0, NUM_SENSORS_MP1): #lis3mdl does not need to be initialize, but could be use to change default settings
            self.mp1.enable_only_channel(i)
            print("Current sensor number: ", i)
            self.lis1 = lis3mdl.LIS3MDL(self.mp1.get_i2c())
            
            self.lis1.scale_range = lis3mdl.SCALE_4_GAUSS
            
            #print(self.lis1.scale_range) ## print string
            ctrl_reg2_value = self.lis1.read_ctrl_reg2()
            print(f"CTRL_REG2 value: {ctrl_reg2_value:08b}") # Print the value in binary format

            
            self.lis1.data_rate = lis3mdl.RATE_155_HZ
            
            #print(self.lis1.data_rate) ## print string
            ctrl_reg1_value = self.lis1.read_ctrl_reg1()
            print(f"CTRL_REG1 value: {ctrl_reg1_value:08b}")  # Print the value in binary format
            #print(ctrl_reg1_value) # Print the value in decimal format
            #print(self.mp1.i2c.readfrom_mem(0x1C, 0x20, 8)[0])  # 0x1C is sensor's i2c address, same usage as function "read_ctrl_reg1()"
            
            #print(lis3mdl.RATE_0_625_HZ) ## 0
            #print(lis3mdl.RATE_1_25_HZ) ## 2
            #print(lis3mdl.RATE_155_HZ) ## 98
            
            #ctrl_reg3_value = self.lis1.read_ctrl_reg3()
            #print(f"CTRL_REG3 value: {ctrl_reg3_value:08b}") # Print the value in binary format

            ctrl_reg4_value = self.lis1.read_ctrl_reg4()
            print(f"CTRL_REG4 value: {ctrl_reg4_value:08b}") # Print the value in binary format

            
        for i in range(0, NUM_SENSORS_MP2): #lis3mdl does not need to be initialize, but could be use to change default settings
            self.mp2.enable_only_channel(i)
            print("Current sensor number: ", i)
            self.lis2 = lis3mdl.LIS3MDL(self.mp2.get_i2c())
            self.lis2.scale_range = lis3mdl.SCALE_4_GAUSS
            self.lis2.data_rate = lis3mdl.RATE_155_HZ
            
            #ctrl_reg1_value = self.lis2.read_ctrl_reg1()
            #print(f"CTRL_REG1 value: {ctrl_reg1_value:08b}")  # Print the value in binary format

    def get_sensordata(self):
        data = []
        for i in range (0, NUM_SENSORS_MP1):
            self.mp1.enable_only_channel(i)
            
            #mag_x, mag_y, mag_z = self.lis1.magnetic
            #mag_x, mag_y, mag_z = self.lis1.magnetic_raw
            mag_x, mag_y, mag_z = self.lis1.offset_magnetic_raw
            
            #print("Current sensor number", i+1)
            #print(f"X:{mag_x:0.2f}, Y:{mag_y:0.2f}, Z:{mag_z:0.2f} uT")
            #print("")
            data.append([mag_x, mag_y, mag_z])
            #print(data)
            
        for i in range (0, NUM_SENSORS_MP2):
            self.mp2.enable_only_channel(i)
            
            #mag_x, mag_y, mag_z = self.lis2.magnetic
            #mag_x, mag_y, mag_z = self.lis2.magnetic_raw
            mag_x, mag_y, mag_z = self.lis2.offset_magnetic_raw
            
            #print("Current sensor number", i+1+8)
            #print(f"X:{mag_x:0.2f}, Y:{mag_y:0.2f}, Z:{mag_z:0.2f} uT")
            #print("")
            data.append([mag_x, mag_y, mag_z])
            #print(data)
            
        #print(data)
        #print("The data dimension is: ", len(data), len(data[0]))
        return data
            
            
            
            
            
            
            
            
            
            
            
