# Modified from: https://github.com/adafruit/Adafruit_CircuitPython_TCA9548A/blob/main/adafruit_tca9548a.py
# SPDX-FileCopyrightText: 2018 Carter Nelson for Adafruit Industries
#
# SPDX-License-Identifier: MIT
#
# Copyright Medability

import time
import machine
from micropython import const

# Default I2C address
PCA9548A_ADDR = const(0x70)

class PCA9548A:
    """Class which provides interface to PCA9548A I2C multiplexer."""

    def __init__(self, port, scl, sda, address=PCA9548A_ADDR):
        self.i2c = machine.I2C(port, scl=machine.Pin(scl), sda=machine.Pin(sda), freq=100000)
        self.address = address
        self.channels = 8
        self.command = bytearray(1)
        
        # disable all channels
        self.disable_all_channels()

    def get_channels(self): 
        try:
            channels = self.i2c.readfrom(self.address, 1)
            #print(bin(int.from_bytes(channels, 'b')))
            return channels
        except:
            return None
        
    def enable_channel(self, channel):
        if not 0 <= channel < self.channels:
            raise IndexError("Channel must be an integer in the range: 0-"+str(self.channels))
        channels = int.from_bytes(self.get_channels(), 'b')
        if channels is None:
            return False
        # set channel bit 
        channels = channels | (1 << channel)
        self.command[0] = channels
        self.i2c.writeto(self.address, self.command)

    def enable_only_channel(self, channel):
        if not 0 <= channel < self.channels:
            raise IndexError("Channel must be an integer in the range: 0-"+str(self.channels))
        # overwrite channel byte
        self.command[0] = (1 << channel)
        self.i2c.writeto(self.address, self.command)

    def disable_channel(self, channel):
        if not 0 <= channel < self.channels:
            raise IndexError("Channel must be an integer in the range: 0-"+str(self.channels))
        channels = int.from_bytes(self.get_channels(), 'b')
        if channels is None:
            return False
        # unset channel bit 
        channels = channels & ~(1 << channel)
        self.command[0] = channels
        self.i2c.writeto(self.address, self.command)
        
    def disable_all_channels(self):
        # disable all channels
        self.command[0] = 0x00
        self.i2c.writeto(self.address, self.command)         
        
    def get_i2c(self):
        return self.i2c
