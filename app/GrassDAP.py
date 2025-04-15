import sys
import os
import subprocess
import webbrowser

#from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, QUrl, Slot, Signal
from PySide6.QtWidgets import QSplashScreen
from PySide6.QtGui import QPixmap
from PySide6.QtGui import QIcon
from PySide6.QtCore import QThread
# from PySide6.QtCore import QSize
# from PySide6.QtGui import Qt

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineQuick import QtWebEngineQuick
# from PySide6.QtCore import QTimer
# import PySide6.QtCore as QtCore
# from PySide6.QtCore import Qt, QFileSystemWatcher, QSettings, Property
# from PySide6.QtCore import QStringListModel

# import pandas as pd
# import numpy as np

from bgremover import DamageClassifier
from bgremover import BackgroundRemover

USE_RESOURCES = True  # Set to True to use resources.qrc

RES_PREFIX = ""

if USE_RESOURCES:
    import rc_resources
    RES_PREFIX = ":/"


class Worker(QThread):
    finished = Signal()  # Signal emitted when the thread finishes processing

    def __init__(self, input_folder, model, output_file):
        super().__init__()
        self.input_folder = input_folder
        self.model = model
        self.output_file = output_file

    def run(self):
        """Long-running task."""
        import time
        print("Processing started...")
        self.damage_classifier =  DamageClassifier()
        print(self.input_folder, self.model, self.output_file)
        self.damage_classifier.batch_processing(self.input_folder, self.model, self.output_file)
        print("Processing finished!")
        self.finished.emit()

class WorkerSeg(QThread):
    finished = Signal()  # Signal emitted when the thread finishes processing

    def __init__(self, input_folder, output_folder):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder

    def run(self):
        """Long-running task."""
        import time
        print("Processing started...")
        self.background_remover =  BackgroundRemover()
        print(self.input_folder, self.output_folder)
        self.background_remover.batch_processing(self.input_folder, self.output_folder)
        print("Processing finished!")
        self.finished.emit()

class ProcessorInterface(QObject):
    msg = Signal(str)
    finished = Signal()
    finishedSeg = Signal()

    def initialize(self):
        #self.damage_classifier =  DamageClassifier()
        #self.background_remover = BackgroundRemover()
        pass


    @Slot()
    def execute(self):
        print('Execute')

    @Slot()
    def click(self):
        print('click')

    @Slot()
    def download(self):
        print('download')

    # @Slot(str, str, str)
    # def process(self, input_folder, model, output_file):

    #     input_folder = input_folder.replace("file:///","")
    #     output_file = output_file.replace("file:///","")
    #     # self.damage_classifier =  DamageClassifier()
    #     # self.damage_classifier.batch_processing(input_folder, model, output_file)
    #     # print('process')
    #     # self.finished.emit()

    @Slot(str, str, str)
    def process(self, input_folder, model, output_file):
        """Start the background processing in a separate thread."""
        input_folder = input_folder.replace("file:///","")
        output_file = output_file.replace("file:///","")
        self.worker = Worker(input_folder, model, output_file)
        self.worker.finished.connect(self.onProcessFinished)
        self.worker.start()

    @Slot(str, str)
    def process_seg(self, input_folder, output_folder):
        """Start the background processing in a separate thread."""
        input_folder = input_folder.replace("file:///","")
        output_folder = output_folder.replace("file:///","")
        self.worker = WorkerSeg(input_folder, output_folder)
        self.worker.finished.connect(self.onProcessFinishedSeg)
        self.worker.start()

    @Slot(str)    
    def onProcessFinished(self):
        """Handle the process completion."""
        self.finished.emit()
        print("Process finished signal emitted.")

    @Slot(str)    
    def onProcessFinishedSeg(self):
        """Handle the process completion."""
        self.finishedSeg.emit()
        print("Process finished signal emitted.")

    @Slot(str)
    def openOutputFile(self, output_file):
        """Open the output file in Excel."""
        output_file = output_file.replace("file:///","")
        if os.path.exists(output_file):
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(output_file)
                elif os.name == 'posix':  # macOS/Linux
                    subprocess.run(["open", output_file])  # macOS
                    # subprocess.run(["xdg-open", output_file])  # Linux (uncomment if needed)
            except Exception as e:
                print(f"Failed to open file: {e}")
        else:
            print(f"Output file not found: {output_file}")

    @Slot(str)
    def openOutputFolder(self, output_folder):
        """Open the output folder in the default file explorer."""
        if os.path.isdir(output_folder): # Check if it's a valid directory
            try:
                if os.name == 'nt':  # Windows
                    os.startfile(output_folder)
                elif os.name == 'posix':  # macOS/Linux
                    if sys.platform == "darwin": # macOS
                        subprocess.run(["open", output_folder])
                    else: # Linux
                        subprocess.run(["xdg-open", output_folder])
                else:
                    print(f"Unsupported OS: {os.name}")
            except Exception as e:
                print(f"Failed to open folder: {e}")
        else:
            print(f"Output folder not found or is not a directory: {output_folder}")

    @Slot(str)
    def open_url(self, url):
        """Open website in default web browser"""
        webbrowser.open(url)
    


if __name__ == "__main__":

    #app = QGuiApplication(sys.argv)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(RES_PREFIX + "icon.png"))

    # Create the splash screen.  Use a QPixmap for image loading.
    splash_pix = QPixmap(RES_PREFIX + "gd_logo_small.png")
    if not splash_pix.isNull(): # check if the image loaded correctly.
        splash = QSplashScreen(splash_pix)
        splash.show()
        #QTimer.singleShot(3000)
        app.processEvents()  # Ensure the splash screen is displayed.
        
    else:
        splash = None # if image didn't load, don't show a splash

    processorInterface = ProcessorInterface()

    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.rootContext().setContextProperty("processorInterface", processorInterface)
    #engine.load(QUrl("view.qml"))
    engine.load(RES_PREFIX + "view.qml")


    if engine.rootObjects():
        if splash:
            splash.finish(None)
    else:
        print("QML load failed")
        sys.exit(1)

    sys.exit(app.exec())