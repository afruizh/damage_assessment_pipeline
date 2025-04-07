import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

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
            onAccepted: {folderPath.text = folderDialog.currentFolder
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
                    outputFolderDialog.open()
                }
            }
        }

        FolderDialog {
            id: outputFolderDialog
            title: "Select Folder"
            onAccepted: {outputFolderPath.text = outputFolderDialog.currentFolder
            }
        }



        // Process Button
        RowLayout {
            Button {
                text: "Process"
                onClicked: {
                    console.log("Processing with folder: " + folderPath.text + ", output: " + outputFolderDialog.text)
                    loadingIndicator.visible = true  // Show loading indicator
                    processorInterface.processSeg(folderPath.text, outputFolderPath.text)
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
        function onFinished() {
            loadingIndicator.visible = false  // Show loading indicator
            infoDialog.open()
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