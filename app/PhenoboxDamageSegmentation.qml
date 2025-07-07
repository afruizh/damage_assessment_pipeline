import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import Qt.labs.settings

Rectangle {
    
    //color: "transparent"
    border.color: "lightgray"
    border.width: 1
    radius: 10
    clip: true
    // gradient: Gradient {
    //     GradientStop { position: 0.0; color: "lightgray" }
    //     GradientStop { position: 1.0; color: "transparent" }
    // }

    Settings {
        id: appSettings
        property string lastInputFolder: ""
        property string lastOutputFolder: ""
    }

    ColumnLayout {
        spacing: 10
        anchors.fill: parent
        anchors.margins: 20
        

        // Folder Selection
        RowLayout {
            spacing: 5
            Label {
                text: "Input Folder:"
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
                    if (isLinux) {
                        const folder = FolderHelper.pickFolder(appSettings.lastInputFolder || "")
                        if (folder !== "") {
                            folderPath.text = folder
                            appSettings.lastInputFolder = folder
                        }
                    } else {
                        folderDialog.open()
                    }
                }
            }
        }

        FolderDialog {
            id: folderDialog
            title: "Select Folder"
            currentFolder: appSettings.lastInputFolder || ""
            onAccepted: {
                folderPath.text = folderDialog.currentFolder.toString().replace("file:///", "").replace("file://", "//")
                appSettings.lastInputFolder = folderDialog.currentFolder
            }
        }

        // Output Filename Selection
        RowLayout {
            spacing: 5
            Label {
                text: "Output Folder:"
                width: 100
                Layout.minimumWidth: 80
            }
            TextField {
                id: outputFolderPath
                Layout.fillWidth: true
                placeholderText: "Output folder..."
                width: 200
            }
            Button {
                text: "Browse"
                onClicked: {
                    //outputFolderDialog.open()

                    if (isLinux) {
                        const folder = FolderHelper.pickFolder(appSettings.lastInputFolder || "")
                        if (folder !== "") {
                            outputFolderPath.text = folder
                            appSettings.lastOutputFolder = folder
                        }
                    } else {
                        outputFolderDialog.open()
                    }
                }
                
            }
        }

        FolderDialog {
            id: outputFolderDialog
            title: "Select Folder"
            currentFolder: appSettings.lastOutputFolder || ""
            onAccepted: {
                outputFolderPath.text = outputFolderDialog.currentFolder.toString().replace("file:///", "").replace("file://", "//")
                appSettings.lastOutputFolder = outputFolderDialog.currentFolder
            }
        }



        // Process Button
        RowLayout {
            Button {
                Layout.fillWidth: true
                text: "Process"
                // onClicked: {
                //     console.log("Processing with folder: " + folderPath.text + ", output: " + outputFolderPath.text)
                //     loadingIndicator.visible = true  // Show loading indicator
                //     processorInterface.process_seg(folderPath.text, outputFolderPath.text)
                // }
                // onClicked: {
                //     console.log("Processing with folder: " + folderPath.text + ", output: " + outputFolderPath.text)
                //     progressBar.value = 0 // Reset progress bar
                //     progressLabel.text = "" // Reset label
                //     loadingIndicator.visible = true  // Show loading indicator
                //     processorInterface.process(folderPath.text, outputFolderPath.text)
                // }
                onClicked: {
                    let params = {
                        "task": "batch_phenobox_damage_segmentation"
                        , "input_folder": folderPath.text.toString().replace("file:///", "").replace("file://", "//")
                        , "output_folder": outputFolderPath.text.toString().replace("file:///", "").replace("file://", "//")
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
        // function onFinishedSeg() {
        //     loadingIndicator.visible = false  // Show loading indicator
        //     infoDialog.open()
        // }
        function onFinished(results) {
            if (results.task === "batch_phenobox_damage_segmentation") {
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
                processorInterface.openOutputFolder(outputFolderPath.text)
                break;
            }
        }
    }
}