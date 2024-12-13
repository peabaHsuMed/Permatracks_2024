"""
Based on https://github.com/hbldh/bleak/blob/develop/examples/async_callback_with_queue.py

Copyright (c) 2024 Medability
"""
"""
Script that uses bleak to interact with BLE components and asyncio to perform action while reading data
async def callback_handler()        : Responsible for receiving data from USB, scaling is applied here
async def run_queue_consumer        : Main area for tasks

Perform magnet localization
This script directly uses simple regression model to perform localization of one magnet, retrieving position values in x, y, z directions
"""

import numpy as np
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

import serial  # Import the pyserial module

from keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import joblib

logger = logging.getLogger(__name__)

# Global variables for data
first_run = True
background = []

# Open the serial port
ser = serial.Serial('COM3', 115200, timeout=1) 

class DeviceNotFoundError(Exception):
    pass

async def run_usb_client(args: argparse.Namespace, queue: asyncio.Queue):
    logger.info("Starting USB communication...")

    async def callback_handler():
        global first_run
        global background
        
        values = []
        start_time_usb = time.time()
        print("Start time: ", start_time_usb)
        numeric_values = []

        #raw_data = ser.read(96)  # Read 96 bytes for 48 signed 16-bit integers
    
        # Print the raw data received
        #print("Raw data from USB:", raw_data.hex())

        # Read 16 lines from the USB, assuming each line contains 3 values
        
        for _ in range(16):
            data = ser.readline().decode('utf-8').strip()
            #print(f"Received data from USB line: {data}")

            # Attempt to process numeric data from the line
            try:
                line_values = [float(x) for x in data.split()]
                if len(line_values) == 3:
                    numeric_values.extend(line_values)
                else:
                    print(f"Skipping line, expected 3 values but got {len(line_values)}")
            except ValueError as e:
                print(f"Error processing line: {e}")
                continue
        
        # Calculate the number of 16-bit integers
        #n = int(len(raw_data) / 2)
        #print("n: ",n)
        # Unpack the data into signed 16-bit integers
        #numeric_values = struct.unpack("<" + str(n) + "h", raw_data)
        #print("Collected numeric values:", numeric_values)

        if len(numeric_values) != 48:
            print(f"Error: Expected 48 values but got {len(numeric_values)}")
            return  # Handle incomplete data as necessary

        # If background subtraction is needed like in the BLE version
        if not first_run:
            numeric_values = list(map(sub, numeric_values, background))
        if first_run:
            background = [3370, 1928, -2431, 937, -1552, -4287, -245, 217, 
                              -3784, -1140, -1873, -12072, 795, 740, -3986, 646, 
                              -1088, -2154, -1433, -3581, -3436, 1859, 757, -5984, 
                              789, -392, -3793, 1039, -206, -1780, -315, -661, 
                              -3716, 2872, 688, -2249, -392, -556, -5844, -2296, 
                              -1913, -5386, 1188, -668, -2698, -476, -1756, -1937] ## raw background data for 4x4 spacing 30mm device
            first_run = False
        #print("numerical value after sub background:\n", numeric_values)
        # Rearrange the flat list into groups of 3 (x, y, z)
        #numeric_values = [x // 10 for x in numeric_values]
        #print("numerical value after divided by 10:\n", numeric_values)
        vl = [numeric_values[i:i+3] for i in range(0, len(numeric_values), 3)]
        for item in vl:
            x, y, z = item
            x_s = float(x) / 6842 * 100  # Adjust as per your needs
            y_s = float(y) / 6842 * 100
            z_s = float(z) / 6842 * 100
            values.append((x_s, y_s, z_s))

        #print("Processed values (scaled):", values)

        end_time_usb = time.time()
        usb_transmission_latency = end_time_usb - start_time_usb
        print(f"USB Transmission Latency: {usb_transmission_latency} seconds")

        # Put the processed values into the queue
        await queue.put((time.time(), np.array((values))))
        await asyncio.sleep(0.01)
        
    # Continuously read data from USB
    while True:
        try:
            #data = ser.readline()
            await callback_handler()
            #await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            break

    logger.info("Disconnected from USB")


tracking_latencies = []
async def run_queue_consumer(queue: asyncio.Queue):
    model = load_model('z_39_to_79_model_lr0.001_epochs200_batch16_folds15.h5', custom_objects={'mse': MeanSquaredError()}) ## best for now
    scaler = joblib.load('scaler_z3979.joblib') ## best for now

    logger.info("Starting queue consumer")

    grid_canvas = create_grid_canvas()

    epoch, data = await queue.get()
    #data = calibrator.apply(data)
    background_data = data
    print("Background data retrieved")

    countdown = 10
    for i in range(countdown):
        print("Localization start in", countdown - i)
        time.sleep(0.5)

    data_queue = deque([])
    for i in range(3):
        epoch, data = await queue.get()
        data_queue.append(data)
        time.sleep(0.1)

    iteration_count = 0
    while True:
        epoch, data = await queue.get()
        #print("dataaaaaaaaaaaaaa: ", data) # raw data [(1,2,3),(4,5,6),...,(46,47,48)]
        flattened_data = [item for sublist in data for item in sublist] # 1d [1,2,...,48]
        #print("data ffffffffff: ", flattened_data)
        flattened_data_2d = [flattened_data] # 2d [[1,2,...,48]]
        #print("data flattened 22222222222222d: ", flattened_data_2d)
        X = np.array(flattened_data_2d) # turn to numpy array for model.predict() # [[1 2 ... 48]]
        #print("X np arrayyyyyyyyyyy: ", X)
        X = scaler.transform(X)
        #print("XXXXXXXXXXXXXXXXXXXXXXXXXXX: ",X)
        result = model.predict(X) # [[1 2 3]]
        result = result[0] # [1 2 3]
        #print(result)

        print(f"x: {result[0]}, y: {result[1]}, z: {result[2]}") # already in mm 
        plot_cv2localization_30(result[0:3],grid_canvas) 

async def main(args: argparse.Namespace):
    queue = asyncio.Queue(maxsize=100)
    client_task = run_usb_client(args, queue)
    consumer_task = run_queue_consumer(queue)

    try:
        await asyncio.gather(client_task, consumer_task)
    except DeviceNotFoundError:
        pass

    logger.info("Main method done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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