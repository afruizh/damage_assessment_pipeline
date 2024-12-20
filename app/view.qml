import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import Qt5Compat.GraphicalEffects
import QtQuick.Effects

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 600
    height: 600
    title: qsTr("Phenotyping Pipeline")
    minimumWidth: 600    

    // menuBar: MenuBar {
    //     Menu {
    //         title: qsTr("File")
    //         Action {
    //             text: qsTr("Open")
    //             onTriggered: {
    //                 fileDialog.open()
    //             }
    //         }
    //         Action {
    //             text: qsTr("Exit")
    //             onTriggered: Qt.quit()
    //         }
    //     }
    // }

    // FileDialog {
    //     id: fileDialog
    //     title: qsTr("Select a File")
    //     nameFilters: ["Images (*.png *.jpg *.jpeg)"]
    //     onAccepted: {
    //         console.log("Selected file: " + fileDialog.file)
    //     }
    // }

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

    

    Connections {
        target: processorInterface
        onFinished: {
            loadingIndicator.visible = false  // Show loading indicator
            infoDialog.open()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10
        anchors.margins: 20

        Rectangle {
            Layout.preferredHeight: 150
            //Layout.preferredWidth: parent.width
            Layout.fillWidth: true
            color: "transparent"

            RowLayout {
                spacing: 20
                anchors.fill: parent

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "transparent"

                    ColumnLayout {
                        anchors.fill:parent
                        spacing: 1
                        Text {
                            text: '<div style="display: flex; align-items: center;"><div style="text-align: left;"><h1>Phenotyping pipeline</h1><p>Modular phenotyping pipeline.</p><h3>Tropical Forages Program</h3><p><b>Authors: </b>Andres Felipe Ruiz-Hurtado, Juan Andrés Cardoso Arango</p><p></p></div>'
                            //verticalAlignment: Text.AlignVCenter
                        }
                        
                    }
                }

                Rectangle {
                    Layout.preferredWidth: 200
                    Layout.fillHeight: true
                    color: "transparent"

                    Rectangle {
                        id: logoRectangle
                        color: "white"
                        radius: 10
                        anchors.fill:parent
                        anchors.margins: 10

                        RowLayout {
                            anchors.fill:parent
                            anchors.centerIn: parent
                            Image {
                                Layout.fillWidth:true
                                Layout.fillHeight:true
                                anchors.fill: parent
                                source: "logo.png"  // Replace with your logo file
                                anchors.centerIn: parent
                                fillMode: Image.PreserveAspectFit
                                mipmap: true
                            }
                        }
                    }
                    MultiEffect {
                        source: logoRectangle
                        anchors.fill: logoRectangle
                        shadowBlur: 1.0
                        shadowEnabled: true
                        shadowColor: "gray"
                        shadowVerticalOffset: 0
                        shadowHorizontalOffset: 0
                    }
                }
            }
    
        }


        TabBar {
            id: tabBar
            TabButton { text: qsTr("Batch Processing")
                width: 200
             }
            // TabButton { text: qsTr("Damage Classification")
            //     width: 200
            //  }
            // TabButton { text: qsTr("Color Checker Detection") 
            //     width: 200
            //     }
            // TabButton { text: qsTr("Color Calibration") 
            //     width: 200
            // }
            // TabButton { text: qsTr("Plant Segmentation") 
            //     width: 200
            // }
        }

        StackLayout {
            id: stackLayout
            currentIndex: tabBar.currentIndex
            Layout.fillHeight: True

            Rectangle {
                id: page1
                anchors.fill: parent
                color: "transparent"
                border.color: "black"
                border.width: 1
                Layout.fillHeight: True
                Layout.fillWidth: True

                Item {
                    anchors.fill: parent
                    anchors.margins: 20


                ColumnLayout {
                    spacing: 10
                    anchors.fill: parent

                    // Folder Selection
                    Row {
                        spacing: 5
                        Label {
                            text: "Select Folder:"
                            width: 100
                        }
                        TextField {
                            id: folderPath
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

                    // Model Selection
                    Row {
                        spacing: 5
                        Label {
                            text: "Select Model:"
                            width: 100
                        }
                        ComboBox {
                            id: modelDropdown
                            width: 200
                            model: ["Regnet", "Resnet18", "Resnet152", "Googlenet"]
                        }
                    }

                    // Output Filename Selection
                    Row {
                        spacing: 5
                        Label {
                            text: "Output File:"
                            width: 100
                        }
                        TextField {
                            id: outputFilePath
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
                        onAccepted: outputFilePath.text = fileDialog.currentFile
                        nameFilters: ["Excel files (*.xlsx)"]
                    }

                    // Process Button
                    Row {
                        Button {
                            text: "Process"
                            onClicked: {
                                console.log("Processing with folder: " + folderPath.text + ", model: " + modelDropdown.currentText + ", output: " + outputFilePath.text)
                                loadingIndicator.visible = true  // Show loading indicator
                                processorInterface.process(folderPath.text, modelDropdown.currentText, outputFilePath.text)
                            }
                        }
                    }

                    Rectangle {
                        color: "transparent"
                        Layout.fillHeight: true
                    }
                }

                }
            }

            // Item {
            //     ColumnLayout {
            //         spacing: 20

            //         RowLayout {
            //             spacing: 10

            //             Rectangle {
            //                 Layout.fillWidth: true
            //                 Layout.preferredHeight: 200
            //                 color: "#333"
            //                 Text {
            //                     anchors.centerIn: parent
            //                     text: qsTr("Drop Image Here\nor\nClick to Upload")
            //                     color: "#fff"
            //                     horizontalAlignment: Text.AlignHCenter
            //                     wrapMode: Text.WordWrap
            //                 }
            //             }

            //             Rectangle {
            //                 Layout.fillWidth: true
            //                 Layout.preferredHeight: 200
            //                 color: "#222"
            //                 Text {
            //                     anchors.centerIn: parent
            //                     text: qsTr("Output")
            //                     color: "#fff"
            //                     horizontalAlignment: Text.AlignHCenter
            //                 }
            //             }
            //         }

            //         ComboBox {
            //             Layout.fillWidth: true
            //             model: ["Regnet", "Model 2", "Model 3"]
            //             currentIndex: 0
            //             editable: false
            //             textRole: "Choose Model"
            //         }

            //         RowLayout {
            //             spacing: 10
            //             Button {
            //                 text: qsTr("Clear")
            //                 onClicked: console.log("Clear button clicked")
            //             }
            //             Button {
            //                 text: qsTr("Submit")
            //                 onClicked: console.log("Submit button clicked")
            //             }
            //         }

            //         RowLayout {
            //             spacing: 10
            //             Repeater {
            //                 model: 3  // Replace with the number of examples
            //                 Rectangle {
            //                     width: 100
            //                     height: 100
            //                     color: "#444"
            //                     Text {
            //                         anchors.centerIn: parent
            //                         text: qsTr("Example " + (index + 1))
            //                         color: "#fff"
            //                         font.pixelSize: 12
            //                         horizontalAlignment: Text.AlignHCenter
            //                         wrapMode: Text.WordWrap
            //                     }
            //                 }
            //             }
            //         }
            //     }
            // }

            // Item {
            //     Text { text: qsTr("Color Checker Detection Tab Content") }
            // }

            // Item {
            //     Text { text: qsTr("Color Calibration Tab Content") }
            // }

            // Item {
            //     Text { text: qsTr("Plant Segmentation Tab Content") }
            // }
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
            onClicked: {
                console.log("Rectangle clicked")
            }
            onPositionChanged: {
                console.log("Hover detected")
            }
        }


    }
}