import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Rectangle {
    id: page1
    
    //color: "transparent"
    border.color: "lightgray"
    border.width: 1
    radius: 10
    clip: true
    // gradient: Gradient {
    //     GradientStop { position: 0.0; color: "lightgray" }
    //     GradientStop { position: 1.0; color: "transparent" }
    // }

    ColumnLayout {
        spacing: 10
        anchors.fill: parent
        anchors.margins: 20
        

        // Folder Selection
        RowLayout {
            spacing: 5
            Label {
                text: "Select Folder:"
                width: 100
                Layout.minimumWidth: 80
            }
            TextField {
                id: folderPath
                Layout.fillWidth: true
                placeholderText: "Folder path..."
                width: 200
            }
            Button {
                text: "Browse"
                onClicked: {
                    folderDialog.open()
                }
            }
        }

        FolderDialog {
            id: folderDialog
            title: "Select Folder"
            onAccepted: {
                folderPath.text = folderDialog.currentFolder.toString().replace("file:///", "").replace("file://", "//")
            }
        }

        // Model Selection
        RowLayout {
            spacing: 5
            Label {
                text: "Select Model:"
                width: 100
                Layout.minimumWidth: 80
            }
            ComboBox {
                id: modelDropdown
                Layout.minimumWidth: 200
                model: ["Regnet", "Resnet18", "Resnet152", "Googlenet"]
            }
        }

        // Output Filename Selection
        RowLayout {
            spacing: 5
            Label {
                text: "Output File:"
                width: 100
                Layout.minimumWidth: 80
            }
            TextField {
                id: outputFilePath
                Layout.fillWidth: true
                placeholderText: "Output file..."
                width: 200
            }
            Button {
                text: "Browse"
                onClicked: {
                    fileDialog.open()
                }
            }
        }

        FileDialog {
            id: fileDialog
            fileMode: FileDialog.SaveFile
            title: "Select Output File"
            onAccepted: {
                outputFilePath.text = fileDialog.currentFile.toString().replace("file:///", "").replace("file://", "//")
            }
            nameFilters: ["Excel files (*.xlsx)"]
        }

        // Process Button
        RowLayout {
            Button {
                Layout.fillWidth: true
                text: "Process"
                // onClicked: {
                //     console.log("Processing with folder: " + folderPath.text + ", model: " + modelDropdown.currentText + ", output: " + outputFilePath.text)
                //     loadingIndicator.visible = true  // Show loading indicator
                //     processorInterface.process(folderPath.text, modelDropdown.currentText, outputFilePath.text)
                // }
                onClicked: {
                    let params = {
                        "task": "batch_damage_classification"
                        , "input_folder": folderPath.text.toString().replace("file:///", "").replace("file://", "//")
                        , "model": modelDropdown.currentText
                        , "output_file": outputFilePath.text.toString().replace("file:///", "").replace("file://", "//")
                    };
                    loadingIndicator.visible = true  // Show loading indicator
                    processorInterface.process(params)
                }
            }
        }

        Rectangle {
            color: "transparent"
            Layout.fillHeight: true
        }
    }

    

    Rectangle {
        id: loadingIndicator
        anchors.fill: parent
        color:"transparent"
        visible: false  // Initially hidden

        BusyIndicator {
            anchors.centerIn: parent
            width: 100
            height: 100
        }

        MouseArea {
            anchors.fill: parent
            hoverEnabled: true  // Enables capturing hover events
            acceptedButtons: Qt.AllButtons  // Block all mouse buttons
            // onClicked: {
            //     console.log("Rectangle clicked")
            // }
            // onPositionChanged: {
            //     console.log("Hover detected")
            // }
        }


    }

    Connections {
        target: processorInterface
        // function onFinished() {
        //     loadingIndicator.visible = false  // Show loading indicator
        //     infoDialog.open()
        // }
        function onFinished(results) {
            if (results.task === "batch_damage_classification") {
                loadingIndicator.visible = false  // Show loading indicator
                infoDialog.open()
            }
        }
    }

    MessageDialog {
        id: infoDialog
        title: qsTr("Process Completed")
        text: qsTr("The process has finished successfully.")
        buttons: MessageDialog.Ok | MessageDialog.Open
        onButtonClicked: function (button, role) {
            switch (button) {
            case MessageDialog.Open:
                processorInterface.openOutputFile(outputFilePath.text)
                break;
            }
        }
    }
}