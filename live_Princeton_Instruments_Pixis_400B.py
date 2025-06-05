# Live Acquisition script for Princeton Instruments Pixis 400B
# Author: Moritz Jünnemann
# Date: 2025.06.05
# THIS SCRIPT REQUIRES A HARDWARE TRIGGER CONNECTED TO AND TRIGGERING THE CAMERA!!!
import numpy as np
import time
import matplotlib.pyplot as plt
from pylablib.devices import PrincetonInstruments


# SETTINGS #####################################################################
class Settings:
    EXP_TIME_MS = 20
    BINNING = (1, 400)
    SPECTRA_SHAPE = (1, 1340)
    REFRESH_TIME_S = 0.1
    ENERGY_FILE = r"C:\Users\Moritz\Desktop\Pixis_data\Spec.txt"


# MAIN FUNCTION ################################################################
def main():
    # Load energy axis from file
    try:
        with open(Settings.ENERGY_FILE, "r") as f:
            energy_eV = np.loadtxt(f)
    except Exception as e:
        print(f"Failed to load energy axis from file: {e}")
        return

    if energy_eV.shape[0] != Settings.SPECTRA_SHAPE[1]:
        print(f"Energy axis length ({energy_eV.shape[0]}) does not match spectrum length ({Settings.SPECTRA_SHAPE[1]})")
        return

    print("Connected devices:")
    print(PrincetonInstruments.list_cameras())

    cam = PrincetonInstruments.PicamCamera('2105050003')
    print("Camera connected.")

    # Setup camera
    cam.set_attribute_value("Exposure Time", Settings.EXP_TIME_MS)
    cam.set_roi(hbin=Settings.BINNING[0], vbin=Settings.BINNING[1])
    cam.set_attribute_value("Trigger Determination", "Positive Polarity")
    cam.set_attribute_value("Trigger Response", "Readout Per Trigger")
    cam.set_attribute_value("Clean Until Trigger", False)
    cam.set_attribute_value("Shutter Timing Mode", "Always Open")
    cam.set_attribute_value("Shutter Closing Delay", 0)

    time.sleep(0.2)

    cam.setup_acquisition(mode="sequence", nframes=1000)
    cam.start_acquisition()

    print("Starting live display... (Press Ctrl+C to stop)")

    # Setup matplotlib
    plt.ion()
    fig, ax = plt.subplots()
    y = np.zeros_like(energy_eV)
    line, = ax.plot(energy_eV, y)
    ax.set_title("Live XUV Spectrum")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Counts")
    fig.canvas.draw()
    fig.canvas.flush_events()

    try:
        while True:
            data = cam.read_newest_image()

            if data is None:
                time.sleep(Settings.EXP_TIME_MS / 1000 / 10)
                continue

            spectrum = data.ravel().astype(np.uint16)

            if spectrum.shape[0] != energy_eV.shape[0]:
                print(f"Unexpected spectrum shape: {spectrum.shape}")
                continue

            # Update plot
            try:
                line.set_ydata(spectrum)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(Settings.REFRESH_TIME_S)
            except Exception as e:
                print(f"Plot update error: {e}")
                continue

    except KeyboardInterrupt:
        print("User stopped with Ctrl+C.")

    finally:
        if cam.acquisition_in_progress():
            cam.stop_acquisition()
        cam.clear_acquisition()
        cam.close()
        print("Camera disconnected.")
        plt.ioff()
        plt.show()


# RUN SCRIPT ##################################################################
if __name__ == "__main__":
    main()
