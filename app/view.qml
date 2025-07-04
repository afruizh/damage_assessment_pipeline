import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import Qt5Compat.GraphicalEffects
import QtQuick.Effects


ApplicationWindow {
    id: mainWindow
    visible: true
    width: 850
    height: 630
    title: qsTr(Qt.application.name + " - Damage Assessment Pipeline v" + Qt.application.version)
    minimumWidth: 630

    // Rectangle {
    //     anchors.fill: parent        
    //     gradient: Gradient {
    //         GradientStop { position: 0.0; color: "#cfcfcf" }
    //         GradientStop { position: 1.0; color: "transparent" }
    //     }
    // }

    menuBar: MenuBar {
        Menu {
            title: qsTr("&File")
            MenuSeparator { }
            Action { 
                text: qsTr("&Quit") 
                onTriggered: Qt.quit()
            }
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
                                text: '<div style="text-align: left;"><h1>' + Qt.application.name + ' v' + Qt.application.version + '</h1><p>Modular Damage Assessment Pipeline using AI models.</p><h3>Authors</h3><p>' + Qt.application.organization + '.</p><h3>Acknowledgments</h3><p>This work was partially funded by Accelerated Breeding Initiative of CGIAR.</p><p></p></div>'
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
                                    //anchors.fill: parent
                                    source: "logo.png"  // Replace with your logo file
                                    //anchors.centerIn: parent
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

                    RowLayout {
                        anchors.fill:parent
                        //anchors.alignment: Qt.AlignVCenter
                        spacing: 1

                        Image {
                            Layout.preferredWidth: 100
                            source: "gd_logo_small.png"  // Replace with your logo file
                            //anchors.centerIn: parent
                            fillMode: Image.PreserveAspectFit
                            mipmap: true
                            anchors.verticalCenter: parent.verticalCenter
                        }

                        Text {
                            Layout.alignment: Qt.AlignVCenter
                            text: '<div style="text-align: left;"><h1>GrassDamageAI</h1><p>Modular Damage Assessment Pipeline using AI models.</p></div>'
                            anchors.verticalCenter: parent.verticalCenter
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
                                //anchors.fill: parent
                                source: "logo.png"  // Replace with your logo file
                                //anchors.centerIn: parent
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
        ColumnLayout {
            Layout.fillHeight: true
            Layout.fillWidth: true
            spacing: 10

            // Styled TabBar
            TabBar {
                id: tabBar
                Layout.fillWidth: true

                TabButton { 
                    text: qsTr("Damage Classification Batch Processing")
                }

                TabButton { 
                    text: qsTr("Plant Segmentation Batch Processing")
                }

                TabButton { 
                    text: qsTr("Phenobox Damage Segmentation Batch Processing")
                }

                TabButton { 
                    text: qsTr("Webapp")
                }


                // TabButton {
                //     text: qsTr("Home")

                // }
                // TabButton {
                //     text: qsTr("Settings")

                // }
                // TabButton {
                //     text: qsTr("Profile")

                // }
            }

            // StackLayout to switch tabs
            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: tabBar.currentIndex

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true


                    ClassificationPage {
                        id: classificationPage
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                    }
                    
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true


                    SegmentationPage {
                        id: segmentationPage
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                    }
                    
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true


                    PhenoboxDamageSegmentation {
                        id: phenoboxDamageSgmentationPage
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                    }
                    
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.fillHeight: true


                    WebappPage {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                    }
                    
                }

                // Item {
                //     Column {
                //         anchors.centerIn: parent
                //         spacing: 10

                //         Text {
                //             text: qsTr("Welcome to the Home Tab!")
                //             font.pixelSize: 18
                //             color: "#2c3e50"
                //         }
                //     }
                    
                // }

                // Item {
                //     Column {
                //         anchors.centerIn: parent
                //         spacing: 10

                //         Text {
                //             text: qsTr("Settings Page")
                //             font.pixelSize: 18
                //             color: "#2c3e50"
                //         }

                //         Switch {
                //             text: qsTr("Enable Notifications")
                //         }
                //     }
                // }

                // Item {
                //     Column {
                //         anchors.centerIn: parent
                //         spacing: 10

                //         Text {
                //             text: qsTr("User Profile")
                //             font.pixelSize: 18
                //             color: "#2c3e50"
                //         }

                //         Button {
                //             text: qsTr("Edit Profile")
                //             onClicked: console.log("Edit Profile Clicked")
                //         }
                //     }
                // }
            }

            ProcessProgress {
                id: processProgress
                Layout.fillWidth: true
                //Layout.fillHeight: true
                Layout.preferredHeight: 200
                statusText: "Waiting..."
                total: 0
                completed: 0
                percent: 0
                timeElapsed: "0s"
                logs: [] // Placeholder for logs, can be updated dynamically
            }

            Button {
                id: cancelButton
                text: "Cancel"
                Layout.alignment: Qt.AlignHCenter
                Layout.fillWidth: true
                onClicked: {
                    processorInterface.cancelProcessing()
                }
            }
            
        }


    }

    
}