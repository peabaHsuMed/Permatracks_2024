"""
Based on https://github.com/hbldh/bleak/blob/develop/examples/async_callback_with_queue.py

Copyright (c) 2024 Medability
"""
"""
Script that uses bleak to interact with BLE components and asyncio to perform action while reading data
async def callback_handler()        : Responsible for receiving data from BLE, scaling is applied here
async def run_queue_consumer        : Main area for tasks

Perform magnet localization
This script directly uses simple NN regression model to perform localization of one magnet, retrieving position values in x, y, z directions
"""

import argparse
import time
import asyncio
import logging
import struct
from collections import deque 
from operator import sub

from lib.utility import *
from lib.plot import *
import time

from bleak import BleakClient, BleakScanner

import numpy as np
import keyboard
import csv

from keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib


logger = logging.getLogger(__name__)

#UUIDs of all sensor characteristics
ALL_SENSOR_CHAR_UUIDS = (
"2aa10000-88b8-11ee-8501-0800200c9a66",
)


first_run = True
background = []
values_to_save = []
count = 0
results_to_save = []

class DeviceNotFoundError(Exception):
    pass

def save_to_csv(filename,data):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(data)):
            writer.writerow(data[i])
    print("Values saved to sensor_data.csv")


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
        global first_run
        global background
        while not queue.empty():
            queue.get_nowait()

        values = []
        for uuid in ALL_SENSOR_CHAR_UUIDS:
            data = await client.read_gatt_char(uuid)
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
            

            # Save values when space is pressed
            if keyboard.is_pressed('space'):
                global values_to_save
                print("Space pressed! Saving values to CSV...")
                values_to_save.append(values)
                print('value list length beforeeeeeeeeeeeeee: ',len(values_to_save))
                if len(values_to_save) == 5:
                    save_to_csv('data_acquisition.csv',values_to_save)
                    print('5 times this point alreadyyyyyyyyyyyyy!!!!!!')
                    values_to_save = []
                    print('value list length afterrrrrrrrrrrrrrr: ',len(values_to_save))
                    time.sleep(0.5)

        await queue.put((time.time(), values)) #* 1e-2)) ### e-3 Adjusted for the *1000 from esp32, then times 10, it shd be in microTesla
    

    async with BleakClient(device) as client:
        logger.info("connected")
        # invoke callback_handler when receiving notify on first sensor characteristic
        characteristic_sensor1 = ALL_SENSOR_CHAR_UUIDS[0]
        #print (characteristic_sensor1)
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
   
async def run_queue_consumer(queue: asyncio.Queue):
    model = load_model('z_39_to_79_model_lr0.001_epochs200_batch16_folds15.h5', custom_objects={'mse': MeanSquaredError()}) 
    scaler = joblib.load('scaler_z3979.joblib') 

    logger.info("Starting queue consumer")
    
    global x_data, y_data, z_data

    grid_canvas = create_grid_canvas()
    
###Get static background noise
    epoch, data = await queue.get()
    
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
        #print("dataaaaaaaaaaaaaa: ", data) # raw data [(1,2,3),(4,5,6),...,(46,47,48)]
        flattened_data = [item for sublist in data for item in sublist] # 1d [1,2,...,48]
        #print("flattened data: ", flattened_data)
        flattened_data_2d = [flattened_data] # 2d [[1,2,...,48]]
        #print("flattened data 2d: ", flattened_data_2d)
        X = np.array(flattened_data_2d) # turn to numpy array for model.predict() # [[1 2 ... 48]]
        #print("X np arrayyyyyyyyyyy: ", X)
        X = scaler.transform(X)
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXX: ",X)
        result = model.predict(X) # [[1 2 3]]
        result = result[0] # [1 2 3]
        #print(result)

        print(f"x: {result[0]}, y: {result[1]}, z: {result[2]}") # already in mm 

        #plot_cv2grey(data)
        plot_cv2localization_30(result[0:3],grid_canvas) 
    

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
        default="68:B6:B3:3E:45:2E" ,
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
