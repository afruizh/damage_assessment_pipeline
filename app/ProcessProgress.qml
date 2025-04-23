import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: progressPanel
    width: 400
    height: 250

    property string statusText: "Waiting..."
    property int total: 0
    property int completed: 0
    property int percent: 0
    property string timeElapsed: "0s"
    property var logs: []

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        GroupBox {
            title: "Log"
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 5

                TextArea {
                    id: logArea
                    text: logs.join("\n")
                    readOnly: true
                    wrapMode: TextEdit.WrapAnywhere
                    font.family: "Monospace"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                }

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: "Status:"
                        font.bold: true
                    }
                    Label {
                        text: statusText
                    }

                    Label {
                        text: "Progress:"
                        font.bold: true
                    }
                    Label {
                        text: percent + "%"
                    }

                    Label {
                        text: "   Time:"
                        font.bold: true
                    }
                    Label {
                        text: timeElapsed
                    }
                }

                ProgressBar {
                    id: progressBar
                    Layout.fillWidth: true
                    from: 0
                    to: 100
                    value: percent
                }
            }
        }
    }

    Connections {
        target: processorInterface
        function onProgressUpdated(info) {
            // Example dict: {status: "Completed (3/3)", percent: 100, time: "4s", logs: ["..."]}
            if (info.status !== undefined)
                statusText = info.status
            if (info.percent !== undefined)
                percent = info.percent
            if (info.time !== undefined)
                timeElapsed = info.time
            if (info.logs !== undefined)
                logs = info.logs
        }
    }
}