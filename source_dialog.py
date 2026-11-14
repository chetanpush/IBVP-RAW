from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QPushButton,
    QButtonGroup,
    QFileDialog,
    QMessageBox,
)


class SourceSelectDialog(QDialog):
    """
    Startup dialog letting the user pick where IBVAP reads video from:
    webcam, an HTTP(S)/RTSP network stream (IP camera or CCTV feed), or a
    sample video file. After the dialog is accepted, call get_selection()
    to get (video_source, is_live_source).
    """

    def __init__(self, default_sample_path="samples/sample2.mp4"):
        super().__init__()
        self.setWindowTitle("IBVAP - Select Video Source")
        self.setMinimumWidth(440)

        self.selected_source = None
        self.selected_is_live = None

        layout = QVBoxLayout(self)

        self.webcam_radio = QRadioButton("Webcam")
        self.stream_radio = QRadioButton("HTTP(S) / RTSP stream (IP camera, CCTV)")
        self.sample_radio = QRadioButton("Sample video file")
        self.webcam_radio.setChecked(True)

        self.group = QButtonGroup(self)
        for btn in (self.webcam_radio, self.stream_radio, self.sample_radio):
            self.group.addButton(btn)
            layout.addWidget(btn)

        webcam_row = QHBoxLayout()
        webcam_row.addWidget(QLabel("Camera index:"))
        self.webcam_index_input = QLineEdit("0")
        webcam_row.addWidget(self.webcam_index_input)
        layout.addLayout(webcam_row)

        stream_row = QHBoxLayout()
        stream_row.addWidget(QLabel("Stream URL:"))
        self.stream_url_input = QLineEdit("https://")
        stream_row.addWidget(self.stream_url_input)
        layout.addLayout(stream_row)

        file_row = QHBoxLayout()
        self.file_path_input = QLineEdit(default_sample_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_file)
        file_row.addWidget(self.file_path_input)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self.on_start)
        layout.addWidget(start_btn)

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select sample video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if path:
            self.file_path_input.setText(path)

    def on_start(self):
        if self.webcam_radio.isChecked():
            text = self.webcam_index_input.text().strip()

            try:
                self.selected_source = int(text)
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid input", "Camera index must be a number (e.g. 0)."
                )
                return

            self.selected_is_live = True

        elif self.stream_radio.isChecked():
            url = self.stream_url_input.text().strip()

            if not url:
                QMessageBox.warning(self, "Invalid input", "Enter a stream URL.")
                return

            self.selected_source = url
            self.selected_is_live = True

        else:
            path = self.file_path_input.text().strip()

            if not path:
                QMessageBox.warning(self, "Invalid input", "Choose a video file.")
                return

            self.selected_source = path
            self.selected_is_live = False

        self.accept()

    def get_selection(self):
        return self.selected_source, self.selected_is_live
