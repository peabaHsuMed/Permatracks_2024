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
Optimization algorithm could be chosen 'differential evolution' & 'dual annealing' by commenting different lines

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
import csv

logger = logging.getLogger(__name__)

#UUIDs of all sensor characteristics
ALL_SENSOR_CHAR_UUIDS = (
"2aa10000-88b8-11ee-8501-0800200c9a66",
#"2aa10001-88b8-11ee-8501-0800200c9a66" ## this line should be commented out but don't remove it
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

def save_to_csv(filename,data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(['x_s', 'y_s', 'z_s'])
        for i in range(len(data)):
            writer.writerow(data[i])
        #print(data)    
    print("Values saved to csv file")


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
   

async def run_queue_consumer(queue: asyncio.Queue):
    
    logger.info("Starting queue consumer")
    
###Define global variable here
    N_data = 100
    N_sensor = 16

    #Optimization conditions for sensor separation of 30mm
    grid_canvas = create_grid_canvas()
    bounds_diff = [(-0.02, 0.14), (-0.02, 0.14), (-0.02, 0.14), (0, 2*np.pi), (0, 2*np.pi), (-1, 1), (-100, 100), (-100, 100), (-100, 100)]


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
    
    #print(f"Fininished initializing data_queue, it is: {data_queue}")
    while True:
        epoch, data = await queue.get()
        data_queue.append(data)
        data_queue.popleft()
        #print(len(data_queue))
        #print("data queue: ", data_queue)
        #print(np.shape(data_queue)) #(3,16,3)
        

        ## calibration before taking median ##
        cali_data = []
        for i in range(len(data_queue)):
            cali_data.append(calibrator.apply(data_queue[i]))  
        #print('cali data list: ', cali_data)
        calibrated_data = np.median(cali_data, axis=0)
        #print("calibrated data:", calibrated_data)
        #print('original data:', data)

        cost_diff = opt.cost_9DOF_diff(calibrated_data)

        result = sp.optimize.differential_evolution(cost_diff,bounds_diff,maxiter=100,popsize=15) ## very very slow
        #result = sp.optimize.dual_annealing(cost_diff,bounds_diff,maxiter=100) ## very slow

        ##print(f"Iteration {i} is: {result.x}")
        #print(f"Number of func evaluated is {result.nfev}, and number of jac evaluated is {result.njev}, success: {result.success}")
        print(f"x:{result.x[0]*1000}, y:{result.x[1]*1000}, z:{result.x[2]*1000}")###Plotting result back in mm
        print(f"theta:{result.x[3]}, phi:{result.x[4]}, mj:{result.x[5]}")
        print(f"Gx:{result.x[6]}, Gy:{result.x[7]}, Gz:{result.x[8]}")
        ##print(f"48 cost value:{result.fun}")
        #print(f"cost:{result.cost}")
        #if result.cost > 5000:
        #    global count_cost
        #    count_cost+=1
        #    print(f"cost:{result.cost}")
        #    print(count_cost)
        
        plot_cv2localization_30_reg(result.x[0:3]*1000,grid_canvas) 
        

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

