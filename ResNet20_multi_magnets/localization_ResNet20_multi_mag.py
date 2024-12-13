"""
Based on https://github.com/hbldh/bleak/blob/develop/examples/async_callback_with_queue.py

Copyright (c) 2024 Medability
"""
"""
Script that uses bleak to interact with BLE components and asyncio to perform action while reading data
async def callback_handler()        : Responsible for receiving data from BLE, scaling is applied here
async def run_queue_consumer        : Main area for tasks

Perform magnet localization
This script directly uses ResNet20 model to perform localization of two magnets, retrieving position values in x, y, z directions
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

import joblib


from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, add, GlobalAveragePooling2D, Dense, ZeroPadding2D, MaxPooling2D, ZeroPadding3D
from tensorflow.keras.models import Model
import keras

logger = logging.getLogger(__name__)

#UUIDs of all sensor characteristics
ALL_SENSOR_CHAR_UUIDS = (
"2aa10000-88b8-11ee-8501-0800200c9a66",
#"2aa10001-88b8-11ee-8501-0800200c9a66" ## this line should be commented out but don't remove it
)

first_run = True
background = []
values_to_save = []
count = 0
results_to_save = []

class DeviceNotFoundError(Exception):
    pass

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True):
    x = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', data_format='channels_first')(inputs)
    if batch_normalization:
        x = BatchNormalization(axis=1)(x)
    x = Activation(activation)(x)
    return x

def resnet_v1(input_shape, depth, num_classes=3):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (e.g. 20, 32, 44, etc.)')
    
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    
    inputs = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(2, 2), data_format='channels_first')(inputs)
    x = resnet_layer(inputs=x, num_filters=num_filters)
    
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if stack > 0 and res_block == 0:
                x = Conv2D(num_filters, kernel_size=1, strides=strides, padding='same', data_format='channels_first')(x)
            
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2
        if stack < 2:  # Apply MaxPooling at the end of each stack except the last one
            x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_first')(x)

    x = GlobalAveragePooling2D(data_format='channels_first')(x)
    outputs = Dense(num_classes, activation=None)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


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
            

        await queue.put((time.time(), values)) #* 1e-2)) ### e-3 Adjusted for the *1000 from esp32, then times 10, it shd be in microTesla
    
    async with BleakClient(device) as client:
        logger.info("connected")
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

    await queue.put((time.time(), None))

    logger.info("disconnected")
   

async def run_queue_consumer(queue: asyncio.Queue):
    model = keras.saving.load_model('ResNet20_model_superposition_height_44_292118_samples_3ori_2mag.keras')
    scalerx = joblib.load('scaler_ResNet20_superposition_z44_3ori_292118samples_x.joblib')
    scalery = joblib.load('scaler_ResNet20_superposition_z44_3ori_292118samples_y.joblib')
    scalerz = joblib.load('scaler_ResNet20_superposition_z44_3ori_292118samples_z.joblib')

    logger.info("Starting queue consumer")

    #Optimization conditions for sensor separation of 30mm
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
    
    while True:
        epoch, raw_data = await queue.get()
        #print("dataaaaaaaaaaaaaa: ", data) # raw data [(1,2,3),(4,5,6),...,(46,47,48)]
        data_queue.append(raw_data)
        data_queue.popleft()
        raw_data = np.median(data_queue,axis=0)###Rolling median

        flattened_data = [item for sublist in raw_data for item in sublist] # 1d [1,2,...,48]
        #print(np.shape(flattened_data))
        
        data_x_matrix = []
        data_y_matrix = []
        data_z_matrix = []
        data_x = []
        data_y = []
        data_z = []
        for j in range(16): 
            data_x.append(flattened_data[j*3])
        data_x_matrix.append(np.array(data_x).reshape((4, 4)))

        for j in range(16):
            data_y.append(flattened_data[j*3+1])
        data_y_matrix.append(np.array(data_y).reshape((4, 4)))

        for j in range(16):
            data_z.append(flattened_data[j*3+2])
        data_z_matrix.append(np.array(data_z).reshape((4, 4)))
        
        data_x_matrix = np.array(data_x_matrix)
        data_y_matrix = np.array(data_y_matrix)
        data_z_matrix = np.array(data_z_matrix)

        data_x = scalerx.transform(data_x_matrix.reshape(-1, 16)).reshape(-1, 4, 4)
        data_y = scalery.transform(data_y_matrix.reshape(-1, 16)).reshape(-1, 4, 4)
        data_z = scalerz.transform(data_z_matrix.reshape(-1, 16)).reshape(-1, 4, 4)

        data = np.stack((data_x, data_y, data_z), axis=1)
        #print(np.shape(data))
        # Initial guess using ResNet predictions on the sample
        initial_guess = model.predict(data) ## [[x,y,z]]
        initial_guess = initial_guess*0.001
        initial_guess = [item for sublist in initial_guess for item in sublist] ## [x,y,z]
        plot_cv2localization_30_multi_magnets(np.array(initial_guess),grid_canvas) ## multi magnets
        for i in range(len(initial_guess)//3):
            print(f"x{i}:{initial_guess[i*3]*1000}, y{i}:{initial_guess[i*3+1]*1000}, z{i}:{initial_guess[i*3+2]*1000}")
            
        

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
