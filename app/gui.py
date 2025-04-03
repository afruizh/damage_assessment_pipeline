import sys
import os

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
# from PySide6.QtCore import QTimer
# import PySide6.QtCore as QtCore
# from PySide6.QtCore import Qt, QFileSystemWatcher, QSettings, Property
# from PySide6.QtCore import QStringListModel

# import pandas as pd
# import numpy as np

from bgremover import DamageClassifier


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

class ProcessorInterface(QObject):
    msg = Signal(str)
    finished = Signal()

    def initialize(self):
        self.damage_classifier =  DamageClassifier()


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

    @Slot(str)    
    def onProcessFinished(self):
        """Handle the process completion."""
        self.finished.emit()
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

if __name__ == "__main__":

    #app = QGuiApplication(sys.argv)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.png"))

    # Create the splash screen.  Use a QPixmap for image loading.
    splash_pix = QPixmap("gd_logo_small.png")
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
    engine.load(QUrl("view.qml"))

    if engine.rootObjects():
        if splash:
            splash.finish(None)
    else:
        print("QML load failed")
        sys.exit(1)

    

    sys.exit(app.exec())