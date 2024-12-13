"""
Copyright (c) 2024 Medability
"""
"""
Script that uses bleak to interact with BLE components and asyncio to perform action while reading data
async def callback_handler()        : Responsible for receiving data from BLE, scaling is applied here
async def run_queue_consumer        : Main area for tasks

This script is to print and collect raw sensor data
"""

import numpy as np

import argparse
import time
import asyncio
import logging
import struct
from collections import deque 
from operator import sub

from lib.calibration_simple import *
from lib.calibration_ellipsoid_fit import *
from lib.utility import *
from lib.plot import *
import time

from bleak import BleakClient, BleakScanner

import keyboard
import csv

logger = logging.getLogger(__name__)

#UUIDs of all sensor characteristics
ALL_SENSOR_CHAR_UUIDS = (
"2aa10000-88b8-11ee-8501-0800200c9a66",
#"2aa10000-88b8-11ee-8501-0800200c9a66" ## this line should be commented out but don't remove it
)


## raw background data for 4x4 spacing 30mm device
background_30 = [
    3370, 1928, -2431, 937, -1552, -4287, -245, 217, 
    -3784, -1140, -1873, -12072, 795, 740, -3986, 646, 
    -1088, -2154, -1433, -3581, -3436, 1859, 757, -5984, 
    789, -392, -3793, 1039, -206, -1780, -315, -661, 
    -3716, 2872, 688, -2249, -392, -556, -5844, -2296, 
    -1913, -5386, 1188, -668, -2698, -476, -1756, -1937
] 

## raw background data for 4x4 spacing 40mm device 
background_40 = [
    3869, 5390, -5436, 118, -2141, -6459, 1587, -1233, 
    -6244, 334, -2822, -4747, 2233, -1236, -5002, -181, 
    -2065, -3383, 2449, -284, -4150, 97, -2539, -5658, 
    1383, -1384, -4852, 638, -2024, -5058, 1371, -1234, 
    -2553, 2581, -595, -4054, -1180, -2506, -3287, 3210, 
    908, -5636, 3132, 586, -4272, 635, -2528, -2201
]


## device address for 4x4 spacing 30mm device
address_30 = "68:B6:B3:3E:45:2E" 

## device address for 4x4 spacing 40mm device
address_40 = "68:B6:B3:3E:52:3A" 


device = 1 ## 0: spacing 30mm, 1: spacing 40mm

if(device == 0):
    background = background_30
    address = address_30
elif(device == 1):
    background = background_40
    address = address_40

values_to_save = []
def save_to_csv(filename,data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(data)):
            writer.writerow(data[i])
    print("Values saved to raw_sensor_data_subtract_background(uT).csv")

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
        while not queue.empty():
            queue.get_nowait()

        values = []
        for uuid in ALL_SENSOR_CHAR_UUIDS:
            data = await client.read_gatt_char(uuid)
            n = int(len(data)/2)   

            # Unpack data to flat list
            v = struct.unpack("<"+str(n)+"h", data)
            #print("v:\n",v)
            
            v = list(map(sub, v, background))
            #print("v after sub background: \n", v)

            # Rearrange into list of 3-value lists
            vl = [v[i:i+3] for i in range(0, len(v), 3)]
            for item in vl:
                x,y,z = item
                x_s = float(x)/6842*100#*100
                y_s = float(y)/6842*100#*100
                z_s = float(z)/6842*100#*100
                values.append((x_s,y_s,z_s))
        
            # Save values when space is pressed
            if keyboard.is_pressed('space'):
                global values_to_save
                print("Space pressed! Saving values to CSV...")
                values_to_save.append(values)
                print('value list length beforeeeeeeeeeeeeee: ',len(values_to_save))
                if len(values_to_save) == 5:
                    save_to_csv('raw_sensor_data_subtract_background(uT).csv',values_to_save)
                    print('5 times this point alreadyyyyyyyyyyyyy!!!!!!')
                    values_to_save = []
                    print('value list length afterrrrrrrrrrrrrrr: ',len(values_to_save))
                    time.sleep(0.5)

        await queue.put((time.time(), np.array((values)))) #* 1e-2)) ### e-3 Adjusted for the *1000 from esp32, then times 10, it shd be in microTesla
    
    async with BleakClient(device) as client:
        logger.info("connected")
        # invoke callback_handler when receiving notify on first sensor characteristic
        characteristic_sensor1 = ALL_SENSOR_CHAR_UUIDS[0]
        await client.start_notify(characteristic_sensor1, callback_handler)
        await asyncio.sleep(1.0)
         
        while True:
            try:
                for uuid in ALL_SENSOR_CHAR_UUIDS:
                    print (uuid)
                    data = await client.read_gatt_char(uuid)
                    print (data)
                await asyncio.sleep(1.0)
            except KeyboardInterrupt:
                break

        await client.stop_notify(characteristic_sensor1)

    # Send an "exit command to the consumer"
    await queue.put((time.time(), None))

    logger.info("disconnected")


tracking_latencies = []
async def run_queue_consumer(queue: asyncio.Queue):
    
    logger.info("Starting queue consumer")
    
###Define global variable here
    N_data = 500
    N_sensor = 16

###Initialization of optimization, calibration object
    calibrator = EllipsoidCalibrator(N_sensor,N_data)

###Calibration retrieve from previous calibration
    calibrator.load_coefficient()
    
###Get static background noise
    epoch, data = await queue.get()
    data = calibrator.apply(data)
    
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
    
    while True:
        epoch, data = await queue.get()

        data_queue.append(data)
        data_queue.popleft()
        data = np.median(data_queue,axis=0)###Rolling median
        #print("raw sensor data after rolling median (uT):\n",data)
        data = calibrator.apply(data)
        print("data after calibration:\n",data) ## 16*3 array
        
        #plot_cv2grey(data)
        


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
        default=address, 
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