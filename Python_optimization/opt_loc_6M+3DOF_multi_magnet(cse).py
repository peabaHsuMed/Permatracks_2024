"""
Based on https://github.com/hbldh/bleak/blob/develop/examples/async_callback_with_queue.py

Copyright (c) 2024 Medability
"""
"""
Script that uses bleak to interact with BLE components and asyncio to perform action while reading data
async def callback_handler()        : Responsible for receiving data from BLE, scaling is applied here
async def run_queue_consumer        : Main area for tasks

Perform magnet localization
This script directly performs (6M+3)DOF optimization with CSE(common subexpression elimination) applied to retrieve 3 position, 3 orientation and magnetic strength
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
from lib.core_computation_cse import *
from lib.utility import *
from lib.plot import *
import scipy as sp
import time

from bleak import BleakClient, BleakScanner

import numpy as np


logger = logging.getLogger(__name__)

#UUIDs of all sensor characteristics
ALL_SENSOR_CHAR_UUIDS = (
"2aa10000-88b8-11ee-8501-0800200c9a66",
)

first_run = True
background = []

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
    N_data = 500
    N_sensor = 16
    N_magnet = 2

    #Optimization conditions for magnet separation of 30mm
    x0_9DOF = np.array([0.045,0.045,0.035,3.0,3.0,0.0,10.0,10.0,-50.0])   
    x0_9DOF_2 = np.array([0.045,0.045,0.005,3.0,3.0,0.0, 0.02,0.02,0.005,3.0,3.0,0.0,10.0,10.0,-50.0])   ## 2 magnets, 9dof
    x0_9DOF_3 = np.array([0.045,0.045,0.005,3.0,3.0,0.0, 0.02,0.02,0.005,3.0,3.0,0.0, 0.065,0.065,0.005,3.0,3.0,0.0,10.0,10.0,-50.0])   ## 3 magnets, 9dof
    x0_9DOF_4 = np.array([0.045,0.045,0.005,3.0,3.0,0.0, 0.02,0.02,0.005,3.0,3.0,0.0, 0.065,0.065,0.005,3.0,3.0,0.0, 0.085,0.085,0.005,3.0,3.0,0.0,10.0,10.0,-50.0])   ## 4 magnets, 9dof
    bounds_9DOF = ([-0.02,-0.02,-0.02,0.0,0.0,-1,-np.inf,-np.inf,-np.inf],[0.14,0.14,0.14,2*np.pi,2*np.pi,1,np.inf,np.inf,np.inf])
    bounds_9DOF_2 = ([-0.02,-0.02,-0.02,0.0,0.0,-1, -0.02,-0.02,-0.02,0.0,0.0,-1,-np.inf,-np.inf,-np.inf],[0.14,0.14,0.14,2*np.pi,2*np.pi,1, 0.14,0.14,0.14,2*np.pi,2*np.pi,1,np.inf,np.inf,np.inf])
    bounds_9DOF_3 = ([-0.02,-0.02,-0.02,0.0,0.0,-1, -0.02,-0.02,-0.02,0.0,0.0,-1, -0.02,-0.02,-0.02,0.0,0.0,-1,-np.inf,-np.inf,-np.inf],[0.14,0.14,0.14,2*np.pi,2*np.pi,1, 0.14,0.14,0.14,2*np.pi,2*np.pi,1, 0.14,0.14,0.14,2*np.pi,2*np.pi,1,np.inf,np.inf,np.inf])
    bounds_9DOF_4 = ([-0.02,-0.02,-0.02,0.0,0.0,-1, -0.02,-0.02,-0.02,0.0,0.0,-1, -0.02,-0.02,-0.02,0.0,0.0,-1, -0.02,-0.02,-0.02,0.0,0.0,-1,-np.inf,-np.inf,-np.inf],[0.14,0.14,0.14,2*np.pi,2*np.pi,1, 0.14,0.14,0.14,2*np.pi,2*np.pi,1, 0.14,0.14,0.14,2*np.pi,2*np.pi,1, 0.14,0.14,0.14,2*np.pi,2*np.pi,1,np.inf,np.inf,np.inf])
    grid_canvas = create_grid_canvas()

###Initialization of optimization, calibration object
    opt = OptimizationProcess(N_sensor, N_magnet)
    calibrator = EllipsoidCalibrator(N_sensor,N_data)

###Calibration retrieve from previous calibration
    calibrator.load_coefficient()
    
###Get static background noise
    epoch, data = await queue.get()
    data = calibrator.apply(data)
    background_data = data
    print(f"Background_data retreived")
    
###Countdown
    countdown=10
    for i in range(countdown):
        print("Localization start in ",countdown-i)
        time.sleep(0.5)

###Initialization for rolling median
    data_queue = deque([])
    for i in range(3):
        epoch,data = await queue.get()
        data_queue.append(data)
        time.sleep(0.1)
    
    iteration_count=0
    while True:
        epoch, data = await queue.get()

        data_queue.append(data)
        data_queue.popleft()
        data = np.median(data_queue,axis=0)###Rolling median
        
        data = calibrator.apply(data)# - background_data #uT -> mG
        #print(data) ## 16*3 array
        #data = data - background_data
        

        cost = opt.cost_9DOF(data)
        

        start_time = time.time()
        #result = sp.optimize.least_squares(cost,x0_9DOF,opt.jacobian_9DOF,bounds=bounds_9DOF,method='trf')
        result = sp.optimize.least_squares(cost,x0_9DOF_2,opt.jacobian_9DOF,bounds=bounds_9DOF_2,method='trf')
        #result = sp.optimize.least_squares(cost,x0_9DOF_3,opt.jacobian_9DOF,bounds=bounds_9DOF_3,method='trf')
        #result = sp.optimize.least_squares(cost,x0_9DOF_4,opt.jacobian_9DOF,bounds=bounds_9DOF_4,method='trf')
        end_time = time.time()
        latency = end_time - start_time
        #print(latency*1000)
        tracking_latencies.append(latency)

        #iteration_count+=1
        #if iteration_count % 200 == 0:
        #    plot_latency_distribution(tracking_latencies, filename=f'latency_distribution_{iteration_count}_{N_magnet}magnets_2.png')


        print(f"Number of func evaluated is {result.nfev}, and number of jac evaluated is {result.njev}, success: {result.success}")
        for i in range(N_magnet):
            print(f"magnet {i}: x:{result.x[i*6]*1000}, y:{result.x[i*6+1]*1000}, z:{result.x[i*6+2]*1000}")###Plotting result back in mm
            
        print(f"Gx: {result.x[-3]}, Gy: {result.x[-2]}, Gz: {result.x[-1]}")

        #plot_cv2grey(data)
        plot_cv2localization_30_multi_magnets(result.x,grid_canvas,dof=6) 

        x0_9DOF_2 = result.x
        
from scipy.stats import gaussian_kde
def plot_latency_distribution(latencies, filename):
    kde = gaussian_kde(latencies)
    latency_range = np.linspace(min(latencies), max(latencies), 1000)
    prob_density = kde(latency_range)

    plt.figure(figsize=(10,6))
    plt.plot(latency_range,prob_density,color='blue',lw=2)
    plt.fill_between(latency_range, prob_density, color='lightblue', alpha=0.5)
    plt.title('Probability Density of Tracking Latency')
    plt.xlabel('Tracking Latency (seconds)')
    plt.ylabel('Probability Density')
    plt.grid(True)
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

