"""
Based on https://github.com/hbldh/bleak/blob/develop/examples/async_callback_with_queue.py

Copyright (c) 2024 Medability
"""
"""
Script that uses bleak to interact with BLE components and asyncio to perform action while reading data
async def callback_handler()        : Responsible for receiving data from BLE, scaling is applied here
async def run_queue_consumer        : Main area for tasks

Perform magnet localization
This script directly performs 9DOF optimization to retrieve 3 position, 3 orientation and magnetic strength
Optimization algorithm using 'trf'.

"""

import argparse
import time
import asyncio
import logging
import struct
from collections import deque 
from operator import sub

from lib.calibration_simple import *
from lib.calibration_ellipsoid_fit import *
from lib.core_computation import *
from lib.utility import *
from lib.plot import *
import scipy as sp
import time

from bleak import BleakClient, BleakScanner

import numpy as np

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#UUIDs of all sensor characteristics
ALL_SENSOR_CHAR_UUIDS = (
"2aa10000-88b8-11ee-8501-0800200c9a66",
)

first_run = True
background = []
count = 0
for_time_signal = []
count_cost = 0


# Sample data
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

class DeviceNotFoundError(Exception):
    pass

async def run_ble_client(args: argparse.Namespace, queue: asyncio.Queue):
    logger.info("starting scan...")

    if args.address:
        logger.info("searching for device with address '%s'", args.address)
        device = await BleakScanner.find_device_by_address(
            args.address, cb=dict(use_bdaddr=args.macos_use_bdaddr)
        )
        if device is None:
            logger.error("could not find device with address '%s'", args.address)
            raise DeviceNotFoundError
    else:
        device = await BleakScanner.find_device_by_name(
            args.name, cb=dict(use_bdaddr=args.macos_use_bdaddr)
        )
        if device is None:
            logger.error("could not find device with name '%s'", args.name)
            raise DeviceNotFoundError

    logger.info("connecting to device...")

    async def callback_handler(_, d):
        #print("Callback called.")
        global first_run
        global background
        while not queue.empty():
            queue.get_nowait()

        #logger.info("received data...")
        values = []
        for uuid in ALL_SENSOR_CHAR_UUIDS:
            data = await client.read_gatt_char(uuid)
            #print (data.hex())
            n = int(len(data)/2)   
            # Unpack data to flat list
            v = struct.unpack("<"+str(n)+"h", data)
            if not first_run:
                v = list(map(sub, v, background))
            if first_run:
                background = [3370, 1928, -2431, 937, -1552, -4287, -245, 217, 
                              -3784, -1140, -1873, -12072, 795, 740, -3986, 646, 
                              -1088, -2154, -1433, -3581, -3436, 1859, 757, -5984, 
                              789, -392, -3793, 1039, -206, -1780, -315, -661, 
                              -3716, 2872, 688, -2249, -392, -556, -5844, -2296, 
                              -1913, -5386, 1188, -668, -2698, -476, -1756, -1937] ## raw background data for 4x4 spacing 30mm device
                first_run = False
            # Rearrange into list of 3-value lists
            vl = [v[i:i+3] for i in range(0, len(v), 3)]
            for item in vl:
                x,y,z = item
                x_s = float(x)/6842*100
                y_s = float(y)/6842*100
                z_s = float(z)/6842*100
                values.append((x_s,y_s,z_s))
        
        await queue.put((time.time(), np.array((values)))) #* 1e-2)) ### e-3 Adjusted for the *1000 from esp32, then times 10, it shd be in microTesla

    
    async with BleakClient(device) as client:
        logger.info("connected")
        #await client.start_notify(args.characteristic, callback_handler)
        # invoke callback_handler when receiving notify on first sensor characteristic
        characteristic_sensor1 = ALL_SENSOR_CHAR_UUIDS[0]
        #print (characteristic_sensor1)
        await client.start_notify(characteristic_sensor1, callback_handler)
        await asyncio.sleep(1.0)
        #paired = await client.pair(protection_level=2)
        #print(f"Paired: {paired}")
         
        while True:
            try:
                for uuid in ALL_SENSOR_CHAR_UUIDS:
                    print (uuid)
                    data = await client.read_gatt_char(uuid)
                    print (data)
                await asyncio.sleep(1.0)
            except KeyboardInterrupt:
                break
                # data will be empty on EOF (e.g. CTRL+D on *nix)
                #if not data:
                #    break

        await client.stop_notify(characteristic_sensor1)

    # Send an "exit command to the consumer"
    #await queue.put((time.time(), None))
    await queue.put((time.time(), None))

    logger.info("disconnected")
   

tracking_latencies = []
async def run_queue_consumer(queue: asyncio.Queue):
    
    logger.info("Starting queue consumer")
    
###Define global variable here
    N_data = 100
    N_sensor = 16

    #Optimization conditions for sensor separation of 30mm
    x0_9DOF = np.array([0.045,0.045,0.035,3.0,3.0,0.0,10.0,10.0,-50.0])          
    bounds_9DOF = ([-0.02,-0.02,-0.02,0.0,0.0,-1,-np.inf,-np.inf,-np.inf],[0.14,0.14,0.14,2*np.pi,2*np.pi,1,np.inf,np.inf,np.inf])
    grid_canvas = create_grid_canvas()


###Initialization of optimization, calibration object
    opt = OptimizationProcess(N_sensor)
    calibrator = EllipsoidCalibrator(N_sensor,N_data)


###Calibration retrieve from previous calibration
    calibrator.load_coefficient()
    
###Get static background noise
    epoch, data = await queue.get()
    print("data before calibrate",data)
    data = calibrator.apply(data)
    background_data = data
    print(f"Background_data retreived:", background_data)
    
###Countdown
    countdown=10
    for i in range(countdown):
        print("Localization start in ",countdown-i)
        time.sleep(0.5)

###Initialization for rolling median
    data_queue = deque([])
    for i in range(1):
        epoch,data = await queue.get()
        data_queue.append(data)
        time.sleep(0.1)
        
    iteration_count = 0
    while True:
        start_time = time.time()
        epoch, data = await queue.get()
        data_queue.append(data)
        data_queue.popleft()
        #print(len(data_queue))
        #print("data queue: ", data_queue)
        #print(np.shape(data_queue)) #(3,16,3)
        
        ## no calibration, no median
        #cost = opt.cost_9DOF(data_queue[0]) ## (make data_queue length 1)

        ## only taking median, without calibration ##
        #data = np.median(data_queue,axis=0)###Rolling median
        #cost = opt.cost_9DOF(data)          ## only median

        ## only calibration, without taking median ##
        #calibrated_data = calibrator.apply(data)
        #cost = opt.cost_9DOF(calibrated_data)

        ## taking median before calibration ##
        #data = np.median(data_queue,axis=0)###Rolling median
        #print('original rolling median data:', data)     
        #print(data)
        #calibrated_data = calibrator.apply(data)# - background_data #uT -> mG
        #cost = opt.cost_9DOF(calibrated_data)

        ## calibration before taking median ##
        cali_data = []
        for i in range(len(data_queue)):
            cali_data.append(calibrator.apply(data_queue[i]))  
        #print('cali data list: ', cali_data)
        calibrated_data = np.median(cali_data, axis=0)
        #print("calibrated data:", calibrated_data)
        #print('original data:', data)
        cost = opt.cost_9DOF(calibrated_data)

        result = sp.optimize.least_squares(cost,x0_9DOF,opt.jacobian_9DOF,bounds=bounds_9DOF,method='trf')
        end_time = time.time()

        # Calculate the latency
        latency = end_time - start_time
        tracking_latencies.append(latency)
        
        # Periodically plot and save the latency distribution
        #iteration_count += 1
        #if iteration_count % 200 == 0:  # Adjust the interval as needed
        #    plot_latency_distribution(tracking_latencies, filename=f'latency_distribution_{iteration_count}_22.png')

        ##print(f"Number of func evaluated is {result.nfev}, and number of jac evaluated is {result.njev}, success: {result.success}")
        print(f"x:{result.x[0]*1000}, y:{result.x[1]*1000}, z:{result.x[2]*1000}")###Plotting result back in mm
        print(f"theta:{result.x[3]}, phi:{result.x[4]}, mj:{result.x[5]}")
        print(f"Gx:{result.x[6]}, Gy:{result.x[7]}, Gz:{result.x[8]}")
        ##print(f"48 cost value:{result.fun}")
        print(f"cost:{result.cost}")
        
        plot_cv2localization_30_reg(result.x[0:3]*1000,grid_canvas) 
        
        x0_9DOF = result.x


from scipy.stats import gaussian_kde
# After the loop or when you decide to stop tracking, plot the latency distribution
def plot_latency_distribution(latencies, filename):
    # Calculate the KDE
    kde = gaussian_kde(latencies)
    latency_range = np.linspace(min(latencies), max(latencies), 1000)
    prob_density = kde(latency_range)

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(latency_range, prob_density, color='blue', lw=2)
    plt.fill_between(latency_range, prob_density, color='lightblue', alpha=0.5)
    plt.title('Probability Density of Tracking Latency')
    plt.xlabel('Tracking Latency (seconds)')
    plt.ylabel('Probability Density')
    plt.grid(True)
    # Save the plot as an image file
    plt.savefig(filename)
    plt.show()      

async def main(args: argparse.Namespace):
    queue = asyncio.Queue(maxsize=100)
    client_task = run_ble_client(args, queue)
    consumer_task = run_queue_consumer(queue)
    print("main is called.")

    try:
        #await client_task 
        await asyncio.gather(client_task, consumer_task)
    except DeviceNotFoundError:
        pass

    logger.info("Main method done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    device_group = parser.add_mutually_exclusive_group(required=False)

    device_group.add_argument(
        "--name",
        metavar="<name>",
        help="the name of the bluetooth device to connect to",
    )
    device_group.add_argument(
        "--address",
        default="68:B6:B3:3E:45:2E",
        metavar="<address>",
        help="the address of the bluetooth device to connect to",
    )

    parser.add_argument(
        "--macos-use-bdaddr",
        action="store_true",
        help="when true use Bluetooth address instead of UUID on macOS",
    )

    parser.add_argument(
        "-d",
        "--debug",  
        action="store_true",
        help="sets the logging level to debug",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)-15s %(name)-8s %(levelname)s: %(message)s",
    )

    asyncio.run(main(args))