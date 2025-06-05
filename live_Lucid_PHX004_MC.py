# Live Acquisition script for LUCID PHX004_MC Camera
# Author: Moritz Jünnemann
# Date: 2025.06.05
from arena_api.system import system
from arena_api.buffer import *

import ctypes
import numpy as np
import cv2
import time
import subprocess
from typing import Tuple, Dict, Any, List
import datetime
import os


# PROGRAM SETTINGS CLASS: #################################################################
class Settings:
    EXP_TIME_US = 20000.0
    FPS = 1 / (EXP_TIME_US * 1e-6) * 0.9  # Frame rate based on exposure time, set it 10% lower

    # Set the Binning:
    BINNING = 4

    CAMERA_IP = "10.236.10.220"
    PIXEL_FORMAT = 'Mono12'  # check what Mono12p means
    BYTES_PER_PIXEL = 2  # Mono12 = 12 bits uses 2 bytes

    PATH_FEATURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_features")

    # Will be assigned throughout the code:
    TIMESTAMP = None


##########################################################################
# MAIN FUNCTION ########################################################
##########################################################################
def main():
    devices = create_devices_with_tries_ip()
    device = system.select_device(devices)

    # create a timestamp for the output files, format YYYY-MM-DD_HH-MM-SS
    Settings.TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # save all the features/node values to a file
    filename = os.path.join(Settings.PATH_FEATURES, f"{Settings.TIMESTAMP}_features_start.txt")
    save_features_to_file(device, filename)

    # save the streamable node values to a file
    filename = os.path.join(Settings.PATH_FEATURES, f"{Settings.TIMESTAMP}_features_streamable_start.txt")
    device.nodemap.write_streamable_node_values_to(file_name=filename)

    # setup the device:
    num_channels, nodes, selected_nodes, initial_settings = setup(device)

    print("\nLIVE ACQUISITION STARTED, press ESC to stop ---------------------\n")

    prev_frame_time = time.time()

    try:
        with device.start_stream():
            while True:
                curr_frame_time = time.time()

                # Get the buffer:
                buffer = device.get_buffer()
                item = BufferFactory.copy(buffer)
                device.requeue_buffer(buffer)

                # buffer_bytes_per_pixel = int(len(item.data) / (item.width * item.height))

                # convert the buffer to an array
                array = (ctypes.c_ubyte * num_channels * item.width * item.height).from_address(
                    ctypes.addressof(item.pbytes))

                # create a numpy array from the buffer
                npndarray = np.ndarray(buffer=array, dtype=np.uint8,
                                       shape=(item.height, item.width, Settings.BYTES_PER_PIXEL))

                # convert the numpy array to a 16 bit image
                image_16bit = npndarray.view(np.uint16).reshape(item.height, item.width)

                # Increase the size of the image by repeating pixels for display
                # Upscale factor (e.g., 4x in each dimension)
                # Repeat pixels along each axis
                if Settings.BINNING > 1:
                    image_16bit = np.repeat(np.repeat(image_16bit, Settings.BINNING, axis=0), Settings.BINNING, axis=1)

                # print the maximum value of the image
                print(f"Max value of image: {np.max(image_16bit)}")

                # Convert to 8-bit for display (assuming 12-bit data: max value 4095)
                image_display = cv2.convertScaleAbs(image_16bit, alpha=(255.0 / 4095.0))

                # Display the image with FPS text
                fps = "fps: " + str(int(1 / (curr_frame_time - prev_frame_time)))
                cv2.putText(image_display, fps, (7, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

                cv2.imshow('Lucid', image_display)

                BufferFactory.destroy(item)

                prev_frame_time = curr_frame_time

                # Check for ESC key to exit
                key = cv2.waitKey(1)
                if key == 27:
                    break

    finally:
        device.stop_stream()
        restore_initial_settings(nodes, selected_nodes, initial_settings)
        cv2.destroyAllWindows()
        system.destroy_device()
        print(f'Destroyed all created devices')


################################################################################
# Camera and System Setup ################################################
################################################################################
def create_devices_with_tries_ip():
    tries = 0
    tries_max = 6

    print("CONNECTING TO CAMERA: -----------------------------------------------")
    print("\n")
    print(f'Waiting for device to be connected...\n')
    while tries < tries_max:
        devices = system.create_device()
        if not devices:
            print(f'\tTry {tries + 1} of {tries_max}: Running ping command  twice to check device connection...')

            # ping twice:
            ping_host(Settings.CAMERA_IP)
            ping_host(Settings.CAMERA_IP)

            tries += 1
        else:
            print("\n")
            print(f'SUCCESSFULLY CONNECTED TO DEVICE')
            print(f'Created {len(devices)} device(s)\n')
            print("--------------------------------------------------------------")
            print("\n\n")
            return devices
    else:
        raise Exception(f'No device found! Please connect a device and run '
                        f'the example again.')


def setup(device) -> Tuple[int, Dict[str, Any], List[str], Dict[str, Any]]:
    """
    Setup pixel format, fixed exposure time, frame rate, and stream parameters.
    Also stores and returns original exposure and frame rate settings.

    :param: device: The camera device to configure.
    :return:
        - num_channels: Number of channels in the image.
        - nodes : Dictionary of selected nodes.
        - selected_parameters: List of selected parameters.
        - initial_settings: Dictionary of initial settings.
    """
    selected_parameters: List[str] = ['PixelFormat', 'ExposureAuto', 'ExposureTime',
                           'AcquisitionFrameRateEnable', 'AcquisitionFrameRate',
                           'TriggerSelector', 'TriggerMode', 'TriggerSource',
                           'BinningHorizontal', 'BinningVertical']

    nodemap = device.nodemap

    # get the nodes ONCE: !!!
    nodes = nodemap.get_node(selected_parameters)

    nodes['PixelFormat'].value = Settings.PIXEL_FORMAT

    # === Store initial settings ===
    initial_settings = {}
    for param in selected_parameters:
        initial_settings[param] = nodes[param].value

    # === Set Acquisition Frame Rate to slightly above the 1/exposure time ===
    print(f"\nCAMERA SETUP: ---------------------------------------------------\n")
    nodes['AcquisitionFrameRateEnable'].value = True

    nodes['AcquisitionFrameRate'].value = Settings.FPS  # Set to a fixed frame rate

    # print the minimum frame rate
    print(f"AcquisitionFrameRate limits: min = {nodes['AcquisitionFrameRate'].min} FPS, "
          f"max = {nodes['AcquisitionFrameRate'].max} FPS")

    print(f"AcquisitionFrameRate set to {nodes['AcquisitionFrameRate'].value} FPS\n")

    # === Disable Auto Exposure and Set Fixed Exposure Time ===
    nodes['ExposureAuto'].value = 'Off'

    min_exp = nodes['ExposureTime'].min
    max_exp = nodes['ExposureTime'].max

    print(f"ExposureTime limits: min = {min_exp} µs, max = {max_exp} µs")

    if Settings.EXP_TIME_US > max_exp or Settings.EXP_TIME_US < min_exp:
        print(f"WARNING: Desired ExposureTime {Settings.EXP_TIME_US} µs is out of bounds!")

    if Settings.EXP_TIME_US > max_exp:
        nodes['ExposureTime'].value = max_exp
    elif Settings.EXP_TIME_US < min_exp:
        nodes['ExposureTime'].value = min_exp
    else:
        nodes['ExposureTime'].value = Settings.EXP_TIME_US

    print(f"ExposureTime set to {nodes['ExposureTime'].value} µs")

    # === Trigger setup ===
    # Needed if trigger was enabled before by other script
    nodes['TriggerSelector'].value = 'FrameStart'
    nodes['TriggerMode'].value = 'Off'
    # nodes['TriggerSource'].value = 'Software'

    print(f"TriggerSelector set to {nodes['TriggerSelector'].value}")
    print(f"TriggerMode set to {nodes['TriggerMode'].value}")

    # Set up the Binning:
    nodes['BinningHorizontal'].value = Settings.BINNING
    nodes['BinningVertical'].value = Settings.BINNING

    print(f"BinningHorizontal set to {nodes['BinningHorizontal'].value}")
    print(f"BinningVertical set to {nodes['BinningVertical'].value}")

    # Stream settings  ################################################
    tl_stream_nodemap = device.tl_stream_nodemap
    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True

    num_channels = 3  # Not used for Mono12 but still required
    return num_channels, nodes, selected_parameters, initial_settings


def restore_initial_settings(nodes: Dict[str, Any], selected_parameters: List[str], initial_settings: Dict[str, Any]) -> None:
    """
    Restores original exposure, frame rate, and trigger settings.
    :param nodes: Dictionary of selected nodes.
    :param selected_parameters: List of selected parameters to restore.
    :param initial_settings: Dictionary containing the initial settings.
    """
    print(f"\n\nRESTORING CAMERA SETTINGS: -------------------------------------\n")

    # restore the initial settings
    for param in selected_parameters:
        nodes[param].value = initial_settings[param]


def save_features_to_file(device, filename):
    """
    Save all features of the camera to a file.
    """
    # get all available nodes/features
    nodemap = device.nodemap
    features = nodemap.feature_names

    # save the features to a file
    with open(filename, 'w') as f:
        for feature in features:
            node = nodemap.get_node(feature)
            # check if the attribute "value" exists
            try:
                value = node.value
            except:
                value = "N/A"
            f.write(f"{feature}: {value}\n")

    print(f"Saved all features to {filename}")


############################################################
# Other functions ########################################
############################################################
def ping_host(ip: str, count: int = 3) -> None:
    """
    Pings a given IP address and prints the output.
    Prints when the ping process has finished.

    :param ip: IP address to ping.
    :param count: Number of ping packets to send.
    """
    result = subprocess.run(['ping', '-c', str(count), ip])
    print("Ping process finished with return code:", result.returncode)


if __name__ == '__main__':
    main()

