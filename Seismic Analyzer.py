import sys
import os
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QCheckBox,
    QSpinBox,
    QMessageBox,
    QMainWindow,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)
from PySide6.QtGui import QPixmap
import pyqtgraph as pg
from pyqtgraph import mkPen, GraphicsLayoutWidget
from numpy import (
    fft,
    abs,
    concatenate,
    zeros,
    log10,
    where,
    cumsum,
    zeros_like
)
import pandas as pd

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths to the UI files
ui_file_path = os.path.join(script_dir, "ui_files", "mainwindow.ui")
statistics_ui_file_path = os.path.join(script_dir, "ui_files", "statistics.ui")

# Load the UI files
uiclass, baseclass = pg.Qt.loadUiType(ui_file_path)
statistics_uiclass, statistics_baseclass = pg.Qt.loadUiType(statistics_ui_file_path)


class StatisticsWindow(QMainWindow, statistics_uiclass):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.labelFileDuration = self.findChild(QLabel, "label_Duration")
        self.tableStats = self.findChild(QTableWidget, "table_Stats")

        # Set the number of rows and columns for the table
        self.tableStats.setRowCount(6)
        self.tableStats.setColumnCount(6)
        # Initially, the window is not locked
        self.is_locked = False

    def lockWindow(self):
        # Disable widgets or set them as read-only
        self.tableStats.setEditTriggers(
            QAbstractItemView.NoEditTriggers
        )  # Make the table read-only
        self.is_locked = True

    def unlockWindow(self):
        # Enable widgets to allow changes
        self.tableStats.setEditTriggers(
            QAbstractItemView.AllEditTriggers
        )  # Allow editing in the table
        self.is_locked = False

    def populateStatsTable(self, data, sample_rate):
        # Calculate and populate the first row with max values
        max_values = data.max()
        for col, max_value in enumerate(max_values):
            item = QTableWidgetItem(f"{max_value:.3f}")
            self.tableStats.setItem(0, col, item)

        # Calculate and populate the second row with min values
        min_values = data.min()
        for col, min_value in enumerate(min_values):
            item = QTableWidgetItem(f"{min_value:.3f}")
            self.tableStats.setItem(1, col, item)

        # Calculate and populate the third row with max velocities (using trapezoidal rule)
        velocities = self.cumtrapz(386.088 * data, sample_rate)
        max_velocities = velocities.max().values

        for col, max_velocity in enumerate(max_velocities):
            item = QTableWidgetItem(f"{max_velocity:.3f}")
            self.tableStats.setItem(2, col, item)

        # Calculate and populate the fourth row with min velocities (using trapezoidal rule)
        min_velocities = velocities.min().values
        for col, min_velocity in enumerate(min_velocities):
            item = QTableWidgetItem(f"{min_velocity:.3f}")
            self.tableStats.setItem(3, col, item)

        # Calculate and populate the fifth row with max displacements (using trapezoidal rule)
        displacements = self.cumtrapz(velocities, sample_rate)
        max_displacements = displacements.max().values
        for col, max_disp in enumerate(max_displacements):
            item = QTableWidgetItem(f"{max_disp:.3f}")
            self.tableStats.setItem(4, col, item)

        # Calculate and populate the sixth row with min displacements (using trapezoidal rule)
        min_displacements = displacements.min().values
        for col, min_disp in enumerate(min_displacements):
            item = QTableWidgetItem(f"{min_disp:.3f}")
            self.tableStats.setItem(5, col, item)

    def cumtrapz(self, data, sample_rate):
        # Initialize an empty DataFrame to store the integrated data
        int_data = pd.DataFrame()

        for col in data.columns:
            orig_data = data[col].values
            int_col = zeros_like(orig_data)

            for i in range(1, len(orig_data)):
                dt = (
                    1.0 / sample_rate
                )  # Sample rate is passed as an argument from spinbox
                int_col[i] = (
                    int_col[i - 1] + 0.5 * (orig_data[i] + orig_data[i - 1]) * dt
                )

            # Append the integrated column to the result DataFrame
            int_data[col] = int_col

        return int_data

    def updateFileDuration(self, sample_rate, file_length):
        # Calculate the file duration
        duration_seconds = file_length / sample_rate
        # Update the QLabel's text to display the file duration i.e. length of EQ record
        self.labelFileDuration.setText(f"{duration_seconds:.2f} seconds")


class PlotWindow(QMainWindow):
    def __init__(self, title, plot_types):
        super().__init__()
        self.setWindowTitle(title)
        self.plotWidget = GraphicsLayoutWidget()
        self.plotWidget.setMinimumSize(
            800, 600
        )  # Set a minimum size for the plot widget
        self.setCentralWidget(self.plotWidget)

        self.plots = {}  # Dictionary to store plots for different types

        # Define a blue line color
        line_color = (0, 0, 255)  # Blue

        # Create separate axes for Time History, FFT, and ASD stacked vertically
        for plot_type in plot_types:
            if plot_type == "Time History":
                self.time_history_plot = self.plotWidget.addPlot(
                    title="Time History", row=1, col=1
                )
                self.plots["Time History"] = self.time_history_plot.plot(
                    pen=mkPen(color=line_color)
                )
                self.time_history_plot.showGrid(x=True, y=True)  # Show gridlines
                self.time_history_plot.setLabel("left", "Amplitude (g)")  # Y-axis label
                self.time_history_plot.setLabel("bottom", "Time (s)")  # X-axis label
            elif plot_type == "FFT":
                self.fft_plot = self.plotWidget.addPlot(title="FFT", row=2, col=1)
                self.plots["FFT"] = self.fft_plot.plot(pen=mkPen(color=line_color))
                self.fft_plot.showGrid(x=True, y=True)  # Show gridlines
                self.fft_plot.setLabel("left", "Amplitude (g)")  # Y-axis label
                self.fft_plot.setLabel("bottom", "Frequency (Hz)")  # X-axis label
            elif plot_type == "ASD":
                self.asd_plot = self.plotWidget.addPlot(title="ASD", row=3, col=1)
                self.plots["ASD"] = self.asd_plot.plot(pen=mkPen(color=line_color))
                self.asd_plot.showGrid(x=True, y=True)  # Show gridlines
                self.asd_plot.setLabel("left", "Amplitude (dB)")  # Y-axis label
                self.asd_plot.setLabel("bottom", "Frequency (Hz)")  # X-axis label

    def update_plot_data(self, x_data, y_data, plot_type):
        if plot_type in self.plots:
            self.plots[plot_type].setData(x_data, y_data)


class MainWindow(QMainWindow, uiclass):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # Connect file button signal to slot
        self.fileButton.clicked.connect(self.openFileDialog)
        # Access the checkbox and spinbox widgets
        self.checkboxTimeCheck = self.findChild(QCheckBox, "checkBox_TimeCheck")
        self.spinboxSampleRate = self.findChild(QSpinBox, "spinBox_SampleRate")
        # Connect the "Generate" button to the generatePlots function
        self.generateButton.clicked.connect(self.generatePlots)
        # Create a list to store references to plot windows - ensures plot windows don't close
        # immediately after falling out of scope
        self.plot_windows = []
        # Connect the "Statistics" button to the openStatisticsWindow function
        self.pushButton_Stats.clicked.connect(self.openStatisticsWindow)
        self.loaded_data = None  # Store the loaded data in this variable

        # Load the logo image and set it as a pixmap for a QLabel
        self.logo_label = QLabel(self)
        self.logo_label.setGeometry(
            0, 0, 90, 90
        )  # Adjust the size and position as needed
        # Construct the relative path to the image file (replace 'logos' and 'ucsd-logo-png-transparent.png' with your actual folder and file names)
        image_file_path = os.path.join(
            script_dir, "logos", "ucsd-logo-png-transparent.png"
        )

        # Load the image using the relative path
        pixmap = QPixmap(image_file_path)
        self.logo_label.setPixmap(pixmap)

        # Place the logo at the bottom right corner
        window_width = self.width()
        window_height = self.height()
        logo_width = self.logo_label.width()
        logo_height = self.logo_label.height()
        self.logo_label.move(window_width - logo_width, window_height - logo_height)

        # Set fixed window size
        self.setFixedSize(475, 315)

    # Open the statistics window
    def openStatisticsWindow(self):
        if not self.filePath.text():
            self.showErrorMessage("Please select a file.")
            return
        self.statistics_window = StatisticsWindow()
        # Pass the sample rate and file length as arguments
        sample_rate = self.spinboxSampleRate.value()
        if sample_rate == 0:
            self.showErrorMessage("Please set a nonzero sample rate.")
            return
        file_length = self.loaded_data.shape[0]  # Calculate or obtain the file length
        self.statistics_window.updateFileDuration(sample_rate, file_length)
        self.statistics_window.populateStatsTable(self.loaded_data, sample_rate)
        # Lock the statistics window
        self.statistics_window.lockWindow()
        self.statistics_window.show()

    # Select time history file
    def openFileDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "Text Files (*.txt)"
        )
        if fileName:
            self.filePath.setText(fileName)
            # Load data from the file using pandas and store it in self.loaded_data
            self.loaded_data = self.load_data_from_file(fileName)

    # Generate plots based on the selected options
    def generatePlots(self):
        # Check if a file is selected
        if self.loaded_data is None:
            self.showErrorMessage("Please select a file.")
            return
        # Check if at least one of Time History, FFT, or ASD checkboxes is checked
        if not (
            self.checkBox_Time.isChecked()
            or self.checkBox_FFT.isChecked()
            or self.checkBox_ASD.isChecked()
        ):
            self.showErrorMessage(
                "Select at least one of Time History, FFT, or ASD checkboxes."
            )
            return

        # Check if at least one of the axes checkboxes is checked
        if not (
            self.checkBox_DOFX.isChecked()
            or self.checkBox_DOFY.isChecked()
            or self.checkBox_DOFZ.isChecked()
            or self.checkBox_DOFRx.isChecked()
            or self.checkBox_DOFRy.isChecked()
            or self.checkBox_DOFRz.isChecked()
        ):
            self.showErrorMessage("Select at least one of the DOF checkboxes.")
            return
        # Check if the Sample Rate spinbox has a nonzero value OR if there is a checkbox to check if a user-selected file has a time column
        if self.spinboxSampleRate.value() == 0:
            self.showErrorMessage("Set a nonzero Sample Rate.")
            return

        # Check which degree of freedom checkboxes are selected
        self.selected_degree_of_freedom = []
        if self.checkBox_DOFX.isChecked():
            self.selected_degree_of_freedom.append("X")
        if self.checkBox_DOFY.isChecked():
            self.selected_degree_of_freedom.append("Y")
        if self.checkBox_DOFZ.isChecked():
            self.selected_degree_of_freedom.append("Z")
        if self.checkBox_DOFRx.isChecked():
            self.selected_degree_of_freedom.append("Rx")
        if self.checkBox_DOFRy.isChecked():
            self.selected_degree_of_freedom.append("Ry")
        if self.checkBox_DOFRz.isChecked():
            self.selected_degree_of_freedom.append("Rz")

        # Check which plot type checkboxes are selected
        self.selected_plot_types = []
        if self.checkBox_Time.isChecked():
            self.selected_plot_types.append("Time History")
        if self.checkBox_FFT.isChecked():
            self.selected_plot_types.append("FFT")
        if self.checkBox_ASD.isChecked():
            self.selected_plot_types.append("ASD")

        # Create a separate window for each degree of freedom
        for dof in self.selected_degree_of_freedom:
            plot_window = PlotWindow(f"{dof} Plots", self.selected_plot_types)

            # Create and add the plots to the PlotWidget
            for plot_type in self.selected_plot_types:
                x_data, y_data = self.get_plot_data(self.loaded_data, dof, plot_type)
                plot_window.update_plot_data(x_data, y_data, plot_type)

            plot_window.show()
            self.plot_windows.append(plot_window)  # Store the reference

    # Retrieve data from the file using pandas
    # Determine delimiter automatically
    # Determine if there is a time column and column labels
    def load_data_from_file(self, file_path):
        # Load data from the text file using pandas and automatically detect the delimiter
        try:
            data = pd.read_csv(file_path, delimiter=None, header=None, engine="python")
        except pd.errors.EmptyDataError:
            self.showErrorMessage("The selected file is empty.")
            return None
        except pd.errors.ParserError:
            self.showErrorMessage(
                "Unable to determine the delimiter in the selected file."
            )
            return None

        # Determine the most likely delimiter based on the number of columns
        possible_delimiters = [",", "\t", " "]

        delimiter_counts = {}
        for delimiter in possible_delimiters:
            count = 0
            for col in data.columns:
                if data[col].str.contains(delimiter).any():
                    count += 1
            delimiter_counts[delimiter] = count

        # Sort delimiters by column count in descending order
        sorted_delimiters = sorted(
            delimiter_counts, key=delimiter_counts.get, reverse=True
        )

        # Select the delimiter with the highest count
        selected_delimiter = sorted_delimiters[0]

        # Use the selected delimiter to load the data
        data = pd.read_csv(file_path, delimiter=selected_delimiter, header=None)
        num_columns = len(data.columns)
        if num_columns != 6:
            self.showErrorMessage("The selected file does not have 6 columns.")
            self.filePath.clear()  # Clear the file_path text field
            return None
        col_names = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
        data.columns = col_names

        return data

    # Get the data to plot based on the selected plot type
    def get_plot_data(self, data, dof, plot_type):
        if plot_type == "FFT":
            sample_rate = self.spinboxSampleRate.value()
            fft_result = fft.fft(data[dof])
            # Implement logic for FFT data (replace with your own)
            x_data = fft.fftfreq(
                len(data[dof]), 1 / sample_rate
            )  # Calculate frequency data
            frequency_axis = fft.fftfreq(len(data[dof]), 1 / sample_rate)
            x_data = frequency_axis[
                : len(frequency_axis) // 2
            ]  # Get half of the frequency data
            y_data = (
                2.0 / len(fft_result) * abs(fft_result[: len(frequency_axis) // 2])
            )  # Calculate amplitude data

        elif plot_type == "Time History":
            # Implement logic for Time History data (replace with your own)
            sample_rate = self.spinboxSampleRate.value()
            x_data = data.index / sample_rate  # Calculate time based on sample rate
            y_data = data[dof]
        elif plot_type == "ASD":
            # Calculate PSD using NumPy
            fs = self.spinboxSampleRate.value()  # Sample rate
            n = len(data[dof])
            freq = fft.fftfreq(n, 1 / fs)
            fft_result = fft.fft(data[dof])
            psd = (1 / (fs * n)) * abs(fft_result) ** 2
            one_sided_psd = 2 * psd[: n // 2]
            x_data = freq[: n // 2]

            # Handle zero values in one_sided_psd
            one_sided_psd = where(one_sided_psd == 0, 1e-10, one_sided_psd)

            y_data = 10 * log10(one_sided_psd)  # Convert to dB
            # Apply moving average to smooth the data
            y_data = self.moving_average(y_data, 10)

            # Trim y_data if needed to match x_data's length
            # Trim or pad y_data to match the length of x_data
            if len(y_data) < len(x_data):
                # Pad y_data with zeros at the beginning to match the length of x_data
                y_data = concatenate((zeros(len(x_data) - len(y_data)), y_data))
            elif len(y_data) > len(x_data):
                # Trim y_data to match the length of x_data
                y_data = y_data[len(y_data) - len(x_data) :]

        else:
            x_data = []
            y_data = []

        return x_data, y_data

    # Add the moving_average function outside the class
    def moving_average(self, data, window_size):
        cum_sum = cumsum(data)
        cum_sum[window_size:] = cum_sum[window_size:] - cum_sum[:-window_size]
        return cum_sum[window_size - 1 :] / window_size

    # Show an error message box to guide the user
    def showErrorMessage(self, message):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setText(message)
        msgBox.setWindowTitle("Error")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()
