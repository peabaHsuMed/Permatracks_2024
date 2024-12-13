"""
Based on https://github.com/hbldh/bleak/blob/develop/examples/async_callback_with_queue.py

Copyright (c) 2024 Medability
"""
"""
Script that uses bleak to interact with BLE components and asyncio to perform action while reading data
async def callback_handler()        : Responsible for receiving data from BLE, scaling is applied here
async def run_queue_consumer        : Main area for tasks

Retrieve data to pre-defined sample number, then perform ellipsoid fitting with calibrator from lib.calibration_ellipsoid_fit
Calibration coefficient for soft-iron scaling and hard-iron deduction will be saved externally
"""
import argparse
import time
import asyncio
import logging
import struct
from operator import sub

from lib.calibration_simple import *
from lib.calibration_ellipsoid_fit import *
from lib.utility import *
from lib.plot import *
import time

from bleak import BleakClient, BleakScanner

import numpy as np




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
        print("Callback called.")
        global first_run
        global background
        while not queue.empty():
            queue.get_nowait()

        logger.info("received data...")
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
                
        #await queue.put((time.time(), d))

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
    N_data = 150
    N_sensor = 16
    
###Implementation
    calibrator = EllipsoidCalibrator(N_sensor,N_data)
    
###Calibration data retrieval
    for i in range(N_data):
        # Use awaiasyncio.wait_for(queue.get(), timeout=1.0) if you want a timeout for getting data.
        await asyncio.wait_for(queue.get(), timeout=20.0)
        epoch, data = await queue.get()
        calibrator.retrieve_data(i,data)
        print(f"Iteration {i} of calibration data: {calibrator.cal_data[i,:,:]}")
        
        if data is None:
            logger.info(
                "Got message from client about disconnection. Exiting consumer loop..."
            )
            break
        else:
            plot_cv2grey(data)
            time.sleep(0.1)
        
###Calculate calibration coefficients
    calibrator.assemble_coefficient()

    print(f"Calibration finished")
    print(f"Coefficient b: {calibrator.b}")
    print(f"Coefficient A_1: {calibrator.A_1}")

###Countdown
    countdown=15
    for i in range(countdown):
        print("Calibration finish. ",countdown-i)
        time.sleep(0.5)

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
        #default="68:B6:B3:3E:45:2E", ## spacing 30mm device
        default="68:B6:B3:3E:52:3A", ## spacing 40mm device
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

