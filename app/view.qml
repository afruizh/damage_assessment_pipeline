import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import Qt5Compat.GraphicalEffects
import QtQuick.Effects

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 620
    height: 630
    title: qsTr("Phenotyping Pipeline")
    minimumWidth: 630

    menuBar: MenuBar {
        Menu {
            title: qsTr("&File")
            MenuSeparator { }
            Action { text: qsTr("&Quit") }
        }
        Menu {
            title: qsTr("&Help")
            Action { 
                text: qsTr("&About")
                onTriggered: aboutWindow.visible = true
             }
        }
    } 

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

     Window {
        id: aboutWindow
        title: qsTr("About")
        width: 630
        height: 200
        visible: false
        modality: Qt.ApplicationModal
        flags: Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint

        ColumnLayout {
                        anchors.fill:parent
                        spacing: 1

        Rectangle {
            Layout.margins: 20
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
                            text: '<div style="text-align: left;"><h1>Phenotyping pipeline</h1><p>Modular phenotyping pipeline.</p><h3>Authors</h3><p>Tropical Forages Program, CIAT.</p><h3>Acknowledgments</h3><p>This work was partially funded by Accelerated Breeding Initiative of CGIAR.</p><p></p></div>'
                            //verticalAlignment: Text.AlignVCenter
                        }
                        
                    }
                }

                Rectangle {
                    Layout.preferredWidth: 200
                    Layout.fillHeight: true
                    color: "transparent"

                    Rectangle {
                        id: logoRectangle2
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
                        source: logoRectangle2
                        anchors.fill: logoRectangle2
                        shadowBlur: 1.0
                        shadowEnabled: true
                        shadowColor: "gray"
                        shadowVerticalOffset: 0
                        shadowHorizontalOffset: 0
                    }
                }
            }
    
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

        // Title
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
                            text: '<div style="text-align: left;"><h1>Phenotyping pipeline</h1><p>Modular phenotyping pipeline.<p></p></div>'
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


        

        // Main Layout
    Column {
        anchors.fill: parent
        spacing: 10

        // Styled TabBar
        TabBar {
            id: tabBar
            width: parent.width

            TabButton {
                text: qsTr("Home")

            }
            TabButton {
                text: qsTr("Settings")

            }
            TabButton {
                text: qsTr("Profile")

            }
        }

        // StackLayout to switch tabs
        StackLayout {
            currentIndex: tabBar.currentIndex
            anchors.fill: parent

            Item {
                Column {
                    anchors.centerIn: parent
                    spacing: 10

                    Text {
                        text: qsTr("Welcome to the Home Tab!")
                        font.pixelSize: 18
                        color: "#2c3e50"
                    }
                }
            }

            Item {
                Column {
                    anchors.centerIn: parent
                    spacing: 10

                    Text {
                        text: qsTr("Settings Page")
                        font.pixelSize: 18
                        color: "#2c3e50"
                    }

                    Switch {
                        text: qsTr("Enable Notifications")
                    }
                }
            }

            Item {
                Column {
                    anchors.centerIn: parent
                    spacing: 10

                    Text {
                        text: qsTr("User Profile")
                        font.pixelSize: 18
                        color: "#2c3e50"
                    }

                    Button {
                        text: qsTr("Edit Profile")
                        onClicked: console.log("Edit Profile Clicked")
                    }
                }
            }
        }
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