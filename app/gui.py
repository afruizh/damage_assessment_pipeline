# https://stackoverflow.com/questions/71379394/qml-zoom-in-on-an-image-with-x-and-y-but-center-it-when-zooming-out
# https://stackoverflow.com/questions/55310051/displaying-pandas-dataframe-in-qml
# https://stackoverflow.com/questions/77705359/qml-image-inside-flickable-zoom-shifts-and-clips
# https://forum.qt.io/topic/80720/qml-image-comparison-slider


import sys
import os

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from PySide6.QtCore import QUrl

import PySide6.QtCore as QtCore

from PySide6.QtCore import Qt, QFileSystemWatcher, QSettings, Signal, Property, Slot

from PySide6.QtCore import QObject, QUrl, Slot, Signal

from PySide6.QtCore import QStringListModel, QUrl

import pandas as pd
import numpy as np

import sys

from PySide6.QtCore import QObject, Signal, Slot, QThread

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

    processorInterface = ProcessorInterface()

    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.rootContext().setContextProperty("processorInterface", processorInterface)
    engine.load(QUrl("view.qml"))

    sys.exit(app.exec())