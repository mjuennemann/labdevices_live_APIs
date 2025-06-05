# Live Acquisition script for Ocean Optics spectrometer
# Author: Moritz Jünnemann
# Date: 2025.06.05
import matplotlib.pyplot as plt
from seabreeze.spectrometers import Spectrometer


def main():
    # Connect to spectrometer
    spec = Spectrometer.from_first_available()

    # Set integration time (in microseconds)
    integration_time_ms = 20
    spec.integration_time_micros(int(integration_time_ms * 1e3))

    # Get the wavelength axis
    wavelengths = spec.wavelengths()

    # Initial intensities
    intensities = spec.intensities()

    # Set up the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(20, 10))  # Set figure size here
    line, = ax.plot(wavelengths, intensities, linewidth=2)
    ax.set_xlabel("Wavelength (nm)", fontsize=24)
    ax.set_ylabel("Intensity", fontsize=24)
    ax.set_title("Live Spectrum", fontsize=24)
    # set the tick size:
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)

    ax.set_ylim(0, int(2**14 + 100))  # Adjust as needed
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    # set a grid:
    ax.grid()
    fig.canvas.draw()
    fig.show()

    try:
        while True:
            # Get new spectrum (blocks until ready)
            intensities = spec.intensities()

            # Update plot
            line.set_ydata(intensities)
            fig.canvas.draw()
            fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("Stopped by user")


if __name__ == "__main__":
    main()
