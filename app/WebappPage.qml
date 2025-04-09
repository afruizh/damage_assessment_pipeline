import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs
import QtWebEngine

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

        Button {
            Layout.fillWidth: true
            text: "Open Website"
            onClicked: {
                processorInterface.open_url("https://huggingface.co/spaces/anfruizhu/phenotyping_pipeline")
            }
        }

        WebEngineView {
            id: webView
            url: "https://huggingface.co/spaces/anfruizhu/phenotyping_pipeline"
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
        
}