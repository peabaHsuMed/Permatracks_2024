"""
Based on https://github.com/hbldh/bleak/blob/develop/examples/async_callback_with_queue.py

Copyright (c) 2024 Medability
"""
"""
Script that uses bleak to interact with BLE components and asyncio to perform action while reading data
async def callback_handler()        : Responsible for receiving data from BLE, scaling is applied here
async def run_queue_consumer        : Main area for tasks

Perform magnet localization
This script directly performs 6DOF optimization to retrieve 3 position, 3 orientation and magnetic strength
Optimization algorithm could be chosen 'trf' & 'lm' by commenting different lines, and rolling median filter, direct deduct of static background could be chosen by commenting the line

"""

import argparse
import time
import asyncio
import logging
import struct
from collections import deque 
from operator import sub

#from lib import *
from lib.calibration_simple import *
from lib.calibration_ellipsoid_fit import *
from lib.core_computation import *
from lib.utility import *
from lib.plot import *
import scipy as sp
import time

from bleak import BleakClient, BleakScanner

import numpy as np
import cv2 
import keyboard
import csv
from pynput import mouse  # For detecting mouse clicks





logger = logging.getLogger(__name__)

#UUIDs of all sensor characteristics
ALL_SENSOR_CHAR_UUIDS = (
"2aa10000-88b8-11ee-8501-0800200c9a66",
#"2aa10001-88b8-11ee-8501-0800200c9a66", ## this line should be commented out but don't remove it
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
                background = v
                first_run = False
            # Rearrange into list of 3-value lists
            vl = [v[i:i+3] for i in range(0, len(v), 3)]
            for item in vl:
                x,y,z = item
                x_s = float(x)/6842*100#*100
                y_s = float(y)/6842*100#*100
                z_s = float(z)/6842*100#*100
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
   

async def run_queue_consumer(queue: asyncio.Queue):
    
    logger.info("Starting queue consumer")
    
###Define global variable here
    N_data = 100
    N_sensor = 16

    '''
    #Optimization conditions for magnet separation of 40mm
    x0 = np.array([0.060,0.060,0.005,1.0,1.0,0.0])            #x,y,z,theta,phi
    x0_5DOF = np.array([0.060,0.060,0.005,1.0,1.0])   
    bounds = ([-0.02,-0.02,-0.02,0.0,0.0,-np.inf],[0.14,0.14,0.14,2*np.pi,2*np.pi,np.inf])
    bounds_5DOF = ([-0.02,-0.02,-0.02,0.0,0.0],[0.14,0.14,0.14,2*np.pi,2*np.pi])
    '''
    #Optimization conditions for magnet separation of 30mm
    x0 = np.array([0.045,0.045,0.005,1.0,1.0,0.0])            #x,y,z,theta,phi,mj
    x0_5DOF = np.array([0.045,0.045,0.005,1.0,1.0])        
    bounds = ([-0.02,-0.02,-0.02,0.0,0.0,-np.inf],[0.14,0.14,0.14,2*np.pi,2*np.pi,np.inf])
    bounds_5DOF = ([-0.02,-0.02,-0.02,0.0,0.0],[0.14,0.14,0.14,2*np.pi,2*np.pi])
    grid_canvas = create_grid_canvas()

###Initialization of optimization, calibration object
    opt = OptimizationProcess(N_sensor)
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
    
    #print(f"Fininished initializing data_queue, it is: {data_queue}")
    while True:
        epoch, data = await queue.get()

        data_queue.append(data)
        data_queue.popleft()
        data = np.median(data_queue,axis=0)###Rolling median
        
        #print(f"Data AFTER 3point median: {data}")
        #print(f"data_queue: {data_queue}")
        #print(data)
        data = calibrator.apply(data)# - background_data #uT -> mG
        #print(data)
        #data = data - background_data
        cost = opt.cost_6DOF(data)

        #result = sp.optimize.least_squares(cost,x0,opt.jacobian_6DOF,method='lm')
        result = sp.optimize.least_squares(cost,x0,opt.jacobian_6DOF,bounds=bounds,method='trf')
    
        #print(f"Iteration {i} is: {result.x}")
        print(f"Number of func evaluated is {result.nfev}, and number of jac evaluated is {result.njev}, success: {result.success}")
        #print(f"x: {result.x[0]} y: {result.x{1} z: {result.x[2]}")
        print(f"x:{result.x[0]*1000}, y:{result.x[1]*1000}, z:{result.x[2]*1000}")###Plotting result back in mm

        plot_cv2grey(data)
        plot_cv2localization_30(result.x[0:2],grid_canvas) 
        #print(result.x[0:3])
        #x0[2:] = result.x[2:] ### as x&y are fix, no need to change
        #if not (abs(result.x[0]) > 0.2) or not(abs(result.x[1])>0.2):
        x0 = result.x
        #m_j_temp.append(result.x[5])
        #time.sleep(0.1)
        

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

    #parser.add_argument(
    #    "characteristic",
    #    metavar="<notify uuid>",
    #    help="UUID of a characteristic that supports notifications",
    #)

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

