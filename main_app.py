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

from load_rpc import loadrpc

pg.setConfigOption('foreground', 'k')

# Needed for executable
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths to the UI files
ui_file_path = resource_path(os.path.join("ui_files", "mainwindow.ui"))
statistics_ui_file_path = resource_path(os.path.join("ui_files", "statistics.ui"))

# Load the UI files
uiclass, baseclass = pg.Qt.loadUiType(ui_file_path)
statistics_uiclass, statistics_baseclass = pg.Qt.loadUiType(statistics_ui_file_path)
image_file_path = resource_path(os.path.join("logos", "ucsd-logo-png-transparent.png"))


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

        # <<< ADD THIS LINE to set the background to white ('w')
        self.plotWidget.setBackground('w')

        self.plotWidget.setMinimumSize(
            800, 600
        )
        self.setCentralWidget(self.plotWidget)

        self.plots = {}

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
                self.time_history_plot.showGrid(x=True, y=True)
                self.time_history_plot.setLabel("left", "Amplitude (g)")
                self.time_history_plot.setLabel("bottom", "Time (s)")
            elif plot_type == "FFT":
                self.fft_plot = self.plotWidget.addPlot(title="FFT", row=2, col=1)
                self.plots["FFT"] = self.fft_plot.plot(pen=mkPen(color=line_color))
                self.fft_plot.showGrid(x=True, y=True)
                self.fft_plot.setLabel("left", "Amplitude (g)")
                self.fft_plot.setLabel("bottom", "Frequency (Hz)")
            elif plot_type == "ASD":
                self.asd_plot = self.plotWidget.addPlot(title="ASD", row=3, col=1)
                self.plots["ASD"] = self.asd_plot.plot(pen=mkPen(color=line_color))
                self.asd_plot.showGrid(x=True, y=True)
                self.asd_plot.setLabel("left", "Amplitude (dB)")
                self.asd_plot.setLabel("bottom", "Frequency (Hz)")

    def update_plot_data(self, x_data, y_data, plot_type):
        if plot_type in self.plots:
            self.plots[plot_type].setData(x_data, y_data)


class MainWindow(QMainWindow, uiclass):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.fileButton.clicked.connect(self.openFileDialog)

        self.checkBox_DOFX = self.findChild(QCheckBox, "checkBox_DOFX")
        self.checkBox_DOFY = self.findChild(QCheckBox, "checkBox_DOFY")
        self.checkBox_DOFZ = self.findChild(QCheckBox, "checkBox_DOFZ")
        self.checkBox_DOFRx = self.findChild(QCheckBox, "checkBox_DOFRx")
        self.checkBox_DOFRy = self.findChild(QCheckBox, "checkBox_DOFRy")
        self.checkBox_DOFRz = self.findChild(QCheckBox, "checkBox_DOFRz")
        self.dof_checkboxes = [
            self.checkBox_DOFX, self.checkBox_DOFY, self.checkBox_DOFZ,
            self.checkBox_DOFRx, self.checkBox_DOFRy, self.checkBox_DOFRz
        ]

        self.spinboxSampleRate = self.findChild(QSpinBox, "spinBox_SampleRate")
        self.generateButton.clicked.connect(self.generatePlots)
        self.plot_windows = []
        self.pushButton_Stats.clicked.connect(self.openStatisticsWindow)
        self.loaded_data = None

        self.logo_label = QLabel(self)
        self.logo_label.setGeometry(0, 0, 90, 90)
        image_file_path = resource_path(os.path.join("logos", "ucsd-logo-png-transparent.png"))
        pixmap = QPixmap(image_file_path)
        self.logo_label.setPixmap(pixmap)

        window_width = self.width()
        window_height = self.height()
        logo_width = self.logo_label.width()
        logo_height = self.logo_label.height()
        self.logo_label.move(window_width - logo_width, window_height - logo_height)
        self.setFixedSize(475, 315)

    def update_dof_checkboxes(self, num_columns):
        """Enable or disable DOF checkboxes based on the number of data columns."""
        if num_columns == 1:
            # Single DOF file loaded: Disable and uncheck ALL boxes.
            for checkbox in self.dof_checkboxes:
                checkbox.setEnabled(False)
                checkbox.setChecked(False)
        else:
            # Multi-DOF file loaded: Re-enable all checkboxes.
            for checkbox in self.dof_checkboxes:
                checkbox.setEnabled(True)

    def openFileDialog(self):
        file_filter = "All Supported Files (*.txt *.tim);;Text Files (*.txt);;Binary RPC Files (*.tim)"
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter)

        if fileName:
            self.filePath.setText(fileName)
            if fileName.lower().endswith(".txt"):
                self.loaded_data = self.load_data_from_text_file(fileName)
            elif fileName.lower().endswith(".tim"):
                self.loaded_data = self.load_data_from_rpc_file(fileName)
            else:
                self.showErrorMessage("Unsupported file type selected.")
                self.loaded_data = None

            if self.loaded_data is not None:
                self.update_dof_checkboxes(self.loaded_data.shape[1])
            else:
                # If loading failed, reset checkboxes to enabled state
                self.update_dof_checkboxes(6)

    def load_data_from_rpc_file(self, file_path):
        try:
            rpc_data = loadrpc(file_path)
            if rpc_data['x'] is None or rpc_data['x'].size == 0:
                self.showErrorMessage("Failed to load data from RPC file.")
                return None
            if rpc_data.get('delta_t') and rpc_data['delta_t'] > 0:
                sample_rate = 1.0 / rpc_data['delta_t']
                self.spinboxSampleRate.setValue(int(round(sample_rate)))

            data_array = rpc_data['x']
            num_channels = data_array.shape[1]
            col_names = ["X", "Y", "Z", "Rx", "Ry", "Rz"]

            if num_channels <= len(col_names):
                df = pd.DataFrame(data_array, columns=col_names[:num_channels])
            else:
                cols = col_names + [f"Ch{i + 7}" for i in range(num_channels - 6)]
                df = pd.DataFrame(data_array, columns=cols)
            return df
        except Exception as e:
            self.showErrorMessage(f"An error occurred while reading the RPC file: {e}")
            return None

    def load_data_from_text_file(self, file_path):
        try:
            header_rows_to_skip = 0
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 10: break
                    try:
                        line.strip().replace(',', ' ').split()[0]
                        float(line.strip().replace(',', ' ').split()[0])
                        break
                    except (ValueError, IndexError):
                        header_rows_to_skip += 1

            data = pd.read_csv(
                file_path, header=None, engine="python",
                delim_whitespace=True, skiprows=header_rows_to_skip
            )
        except Exception as e:
            self.showErrorMessage(f"Error reading text file: {e}")
            return None

        num_columns = len(data.columns)
        if num_columns == 1:
            col_names = ["X"]
        elif num_columns == 6:
            col_names = ["X", "Y", "Z", "Rx", "Ry", "Rz"]
        else:
            self.showErrorMessage(f"Text file has {num_columns} columns. Please use a file with 1 or 6 columns.")
            self.filePath.clear()
            return None

        data.columns = col_names
        return data

    def generatePlots(self):
        if self.loaded_data is None:
            self.showErrorMessage("Please select a file.")
            return
        if self.spinboxSampleRate.value() == 0:
            self.showErrorMessage("Set a nonzero Sample Rate.")
            return

        self.selected_plot_types = []
        if self.checkBox_Time.isChecked(): self.selected_plot_types.append("Time History")
        if self.checkBox_FFT.isChecked(): self.selected_plot_types.append("FFT")
        if self.checkBox_ASD.isChecked(): self.selected_plot_types.append("ASD")

        if not self.selected_plot_types:
            self.showErrorMessage("Select at least one of Time History, FFT, or ASD checkboxes.")
            return

        num_columns = self.loaded_data.shape[1]

        if num_columns == 1:
            # For a single-DOF file, we don't need to check the boxes.
            # We plot the one column that exists (internally named 'X').
            self.selected_degree_of_freedom = ['X']
        else:
            # For multi-DOF files, check which boxes the user selected.
            self.selected_degree_of_freedom = []
            if self.checkBox_DOFX.isChecked(): self.selected_degree_of_freedom.append("X")
            if self.checkBox_DOFY.isChecked(): self.selected_degree_of_freedom.append("Y")
            if self.checkBox_DOFZ.isChecked(): self.selected_degree_of_freedom.append("Z")
            if self.checkBox_DOFRx.isChecked(): self.selected_degree_of_freedom.append("Rx")
            if self.checkBox_DOFRy.isChecked(): self.selected_degree_of_freedom.append("Ry")
            if self.checkBox_DOFRz.isChecked(): self.selected_degree_of_freedom.append("Rz")

            if not self.selected_degree_of_freedom:
                self.showErrorMessage("Select at least one of the DOF checkboxes.")
                return

        for dof in self.selected_degree_of_freedom:
            window_title = "DOF Plots" if num_columns == 1 else f"{dof} Plots"

            plot_window = PlotWindow(window_title, self.selected_plot_types)
            for plot_type in self.selected_plot_types:
                x_data, y_data = self.get_plot_data(self.loaded_data, dof, plot_type)
                plot_window.update_plot_data(x_data, y_data, plot_type)
            plot_window.show()
            self.plot_windows.append(plot_window)

    def openStatisticsWindow(self):
        if not self.filePath.text():
            self.showErrorMessage("Please select a file.")
            return
        self.statistics_window = StatisticsWindow()
        sample_rate = self.spinboxSampleRate.value()
        if sample_rate == 0:
            self.showErrorMessage("Please set a nonzero sample rate.")
            return
        file_length = self.loaded_data.shape[0]
        self.statistics_window.updateFileDuration(sample_rate, file_length)
        self.statistics_window.populateStatsTable(self.loaded_data, sample_rate)
        self.statistics_window.lockWindow()
        self.statistics_window.show()

    def get_plot_data(self, data, dof, plot_type):
        if plot_type == "FFT":
            sample_rate = self.spinboxSampleRate.value()
            fft_result = fft.fft(data[dof])
            x_data = fft.fftfreq(len(data[dof]), 1 / sample_rate)
            frequency_axis = fft.fftfreq(len(data[dof]), 1 / sample_rate)
            x_data = frequency_axis[: len(frequency_axis) // 2]
            y_data = (2.0 / len(fft_result)) * abs(fft_result[: len(frequency_axis) // 2])
        elif plot_type == "Time History":
            sample_rate = self.spinboxSampleRate.value()
            x_data = data.index / sample_rate
            y_data = data[dof]
        elif plot_type == "ASD":
            fs = self.spinboxSampleRate.value()
            n = len(data[dof])
            freq = fft.fftfreq(n, 1 / fs)
            fft_result = fft.fft(data[dof])
            psd = (1 / (fs * n)) * abs(fft_result) ** 2
            one_sided_psd = 2 * psd[: n // 2]
            x_data = freq[: n // 2]
            one_sided_psd = where(one_sided_psd == 0, 1e-10, one_sided_psd)
            y_data = 10 * log10(one_sided_psd)
            y_data = self.moving_average(y_data, 10)
            if len(y_data) < len(x_data):
                y_data = concatenate((zeros(len(x_data) - len(y_data)), y_data))
            elif len(y_data) > len(x_data):
                y_data = y_data[len(y_data) - len(x_data):]
        else:
            x_data, y_data = [], []
        return x_data, y_data

    def moving_average(self, data, window_size):
        cum_sum = cumsum(data)
        cum_sum[window_size:] = cum_sum[window_size:] - cum_sum[:-window_size]
        return cum_sum[window_size - 1:] / window_size

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
