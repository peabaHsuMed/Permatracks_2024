
#Based on: https://github.com/micropython/micropython/blob/master/examples/bluetooth/ble_temperature.py
# This example demonstrates a simple temperature sensor peripheral.
#
# The sensor's local value updates every second, and it will notify
# any connected central every 10 seconds.

# This example demonstrates the low-level bluetooth module. For most
# applications, we recommend using the higher-level aioble library which takes
# care of all IRQ handling and connection management. See
# https://github.com/micropython/micropython-lib/tree/master/micropython/bluetooth/aioble
# and in particular the temp_sensor.py example included with aioble.

# Copyright Medability GmbH

import bluetooth
import random
import struct
import time
from ble_advertising import advertising_payload

from micropython import const

#from QMC5883 import QMC5883L
#from magnetometers import Magnetometers
from PCA_LIS import PCA_LIS_magnetometer

_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_INDICATE_DONE = const(20)
_IRQ_MTU_EXCHANGED = const(21)

_FLAG_READ = const(0x0002)
_FLAG_NOTIFY = const(0x0010)
_FLAG_INDICATE = const(0x0020)

# org.bluetooth.service.environmental_sensing
_ENV_SENSE_UUID = bluetooth.UUID(0x181A)
# org.bluetooth.characteristic.Magnetic_flux_density_3D
_MAG3D_CHAR = (
    bluetooth.UUID(0x2AA1),
    _FLAG_READ | _FLAG_NOTIFY | _FLAG_INDICATE,
)

# Custom UUID documentation, see: https://www.espruino.com/About+Bluetooth+LE
_CUSTOM_CHAR_SENSOR1  = (bluetooth.UUID("2aa10000-88b8-11ee-8501-0800200c9a66"), _FLAG_READ | _FLAG_NOTIFY | _FLAG_INDICATE,)    
#_CUSTOM_CHAR_SENSOR2  = (bluetooth.UUID("2aa10001-88b8-11ee-8501-0800200c9a66"), _FLAG_READ | _FLAG_NOTIFY | _FLAG_INDICATE,)    
 
    
#_ENV_SENSE_SERVICE = (
#    _ENV_SENSE_UUID,
#    (_CUSTOM_CHAR_SENSOR1, _CUSTOM_CHAR_SENSOR2, _CUSTOM_CHAR_SENSOR3, _CUSTOM_CHAR_SENSOR4,
#     _CUSTOM_CHAR_SENSOR5, _CUSTOM_CHAR_SENSOR6, _CUSTOM_CHAR_SENSOR7, _CUSTOM_CHAR_SENSOR8,
#     _CUSTOM_CHAR_SENSOR9, _CUSTOM_CHAR_SENSOR10, _CUSTOM_CHAR_SENSOR11, _CUSTOM_CHAR_SENSOR12,
#     _CUSTOM_CHAR_SENSOR13, _CUSTOM_CHAR_SENSOR14, _CUSTOM_CHAR_SENSOR15, _CUSTOM_CHAR_SENSOR16,
#     ),
#)

_ENV_SENSE_SERVICE = (
    _ENV_SENSE_UUID,
    (_CUSTOM_CHAR_SENSOR1,),
)

# org.bluetooth.characteristic.gap.appearance.xml
_ADV_APPEARANCE_GENERIC_DISPLAY = const(320)


class BLEMagnetometer:
    def __init__(self, ble, name="mpy-magnetometer"):
        self._ble = ble
        self._ble.active(True)
        self._ble.irq(self._irq)
        self._ble.config(mtu=500)
        #self._ble.config(rxbuf=200)
        #((self._handle,self._handle2),) = self._ble.gatts_register_services((_ENV_SENSE_SERVICE,))
        handle_tuple = self._ble.gatts_register_services((_ENV_SENSE_SERVICE,))
        self._handles = handle_tuple[0]
        self._ble.gatts_set_buffer(self._handles[0], 500, True)
        print (handle_tuple)
        print (self._handles)
        self._connections = set()
        self._payload = advertising_payload(
            name=name, services=[_ENV_SENSE_UUID], appearance=_ADV_APPEARANCE_GENERIC_DISPLAY
        )
        self._advertise()

    def _irq(self, event, data):
        # Track connections so we can send notifications.
        if event == _IRQ_CENTRAL_CONNECT:
            conn_handle, _, _ = data
            self._connections.add(conn_handle)
        elif event == _IRQ_CENTRAL_DISCONNECT:
            conn_handle, _, _ = data
            self._connections.remove(conn_handle)
            # Start advertising again to allow a new connection.
            self._advertise()
        elif event == _IRQ_GATTS_INDICATE_DONE:
            conn_handle, value_handle, status = data
        elif event == _IRQ_MTU_EXCHANGED:
            # ATT MTU exchange complete (either initiated by us or the remote device).
            print("MTU_EXCHANGED")
            #conn_handle, mtu = data
            #self.ble.gattc_exchange_mtu(conn_handle)

    def set_magnetometer(self, x, y, z, handle_num=0, notify=False, indicate=False):
        # Note: In the future we want to send an array of xyz measurements for a sensor grid
        #
        # According to: https://btprodspecificationrefs.blob.core.windows.net/gatt-specification-supplement/GATT_Specification_Supplement.pdf
        # Data is sint16
        # The Magnetic Flux Density - 3D characteristic is used to represent measurements of magnetic flux densityfor three orthogonal axes: X, Y, and Z. Note that 1 x 10-7 Tesla equals 0.001 Gauss.
        # Base Unit: org.bluetooth.unit.magnetic_flux_density.tesla
        # Represented values: M = 1, d = -7, b = 0
        # Unit is 10-7 Tesla.
        # Write the local value, ready for a central to read.
        self._ble.gatts_write(self._handles[handle_num], struct.pack("<hhh", int(x * 1000), int(y * 1000), int(z * 1000)))
        
        if notify or indicate:
            for conn_handle in self._connections:
                if notify:
                    # Notify connected centrals.
                    self._ble.gatts_notify(conn_handle, self._handles[handle_num])
                if indicate:
                    # Indicate connected centrals.
                    self._ble.gatts_indicate(conn_handle, self._handles[handle_num])

    def set_magnetometer_array_GATT_specification(self, xyz_array, handle_num=0, notify=False, indicate=False):
        # According to: https://btprodspecificationrefs.blob.core.windows.net/gatt-specification-supplement/GATT_Specification_Supplement.pdf
        # Data is sint16
        # The Magnetic Flux Density - 3D characteristic is used to represent measurements of magnetic flux densityfor three orthogonal axes: X, Y, and Z. Note that 1 x 10-7 Tesla equals 0.001 Gauss.
        # Base Unit: org.bluetooth.unit.magnetic_flux_density.tesla
        # Represented values: M = 1, d = -7, b = 0
        # Unit is 10-7 Tesla.
        #
        # Write array of sensor readings to the characterstic
        #print(len(xyz_array))
        struct_format = "<" + "h" * len(xyz_array)
        #print(struct_format)
        byte_data = bytearray()
        for xyz in xyz_array:
            x, y, z = xyz
            byte_data += struct.pack("<hhh", int(x * 1000), int(y * 1000), int(z * 1000))
        #print(byte_data.hex())    
        self._ble.gatts_write(self._handles[handle_num], byte_data)
        if notify or indicate:
            for conn_handle in self._connections:
                if notify:
                    # Notify connected centrals.
                    self._ble.gatts_notify(conn_handle, self._handles[handle_num])
                if indicate:
                    # Indicate connected centrals.
                    self._ble.gatts_indicate(conn_handle, self._handles[handle_num])
                    
    def set_magnetometer_array(self, xyz_array, handle_num=0, notify=False, indicate=False):
        # According to: https://btprodspecificationrefs.blob.core.windows.net/gatt-specification-supplement/GATT_Specification_Supplement.pdf
        # Data is sint16
        # The Magnetic Flux Density - 3D characteristic is used to represent measurements of magnetic flux densityfor three orthogonal axes: X, Y, and Z. Note that 1 x 10-7 Tesla equals 0.001 Gauss.
        # Base Unit: org.bluetooth.unit.magnetic_flux_density.tesla
        # Represented values: M = 1, d = -7, b = 0
        # Unit is 10-7 Tesla.
        #
        # Write array of sensor readings to the characterstic
        # Currently, data is retrieved as raw signed 16 bits sensor reading, make sure it is in integer, then pass through BLE
        # Scaling is done in consumer end, to ensure the data transferred is consistent
        #
        #print(len(xyz_array))
        struct_format = "<" + "h" * len(xyz_array)
        #print(struct_format)
        byte_data = bytearray()
        for xyz in xyz_array:
            x, y, z = xyz
            byte_data += struct.pack("<hhh", int(x), int(y), int(z))
        #print(byte_data.hex())    
        self._ble.gatts_write(self._handles[handle_num], byte_data)
        if notify or indicate:
            for conn_handle in self._connections:
                if notify:
                    # Notify connected centrals.
                    self._ble.gatts_notify(conn_handle, self._handles[handle_num])
                if indicate:
                    # Indicate connected centrals.
                    self._ble.gatts_indicate(conn_handle, self._handles[handle_num])

                    
    def notify(self, handle_num):
        for conn_handle in self._connections:
            # Notify connected centrals.
            self._ble.gatts_notify(conn_handle, self._handles[handle_num])

    def _advertise(self, interval_us=500000):
        self._ble.gap_advertise(interval_us, adv_data=self._payload)


def demo():
    ble = bluetooth.BLE()
    ble_magnetometer = BLEMagnetometer(ble)
    

    magnetometers = PCA_LIS_magnetometer()
    i = 0

    while True:
        
        # Read all sensors' data and append it to an array of sensor values
        # x, y axis is reversed, and y axis is fliped, to address the different coordinate system of the sensor and what optimization defined.
        sensor_data = magnetometers.get_sensordata()        
        xyz_array = []
        for idx, item in enumerate(sensor_data):
            #x,y,z,status,temp = item
            x,y,z = item
            xyz_array.append((-y,x,z))
            print (-y,x,z)
            #print (idx)
            #ble_magnetometer.set_magnetometer(x, y, z, idx, notify=False, indicate=False)
        
        # Write sensor data to characteristic and notify connected centrals 
        ble_magnetometer.set_magnetometer_array(xyz_array, notify=True, indicate=False)
        #print()        
        time.sleep_ms(33) #original 33


if __name__ == "__main__":
    demo()
