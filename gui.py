from PyQt5.QtWidgets import QHBoxLayout, QSpinBox, QMessageBox, QDialog, QPushButton, QCheckBox, QMainWindow, QTabWidget, QWidget, QVBoxLayout, \
    QTextEdit, QLineEdit, QLabel, QVBoxLayout, QSizePolicy, QSpacerItem
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QTransform

#from paz.backend.camera import VideoPlayer
#from paz.backend.camera import Camera
#from paz.pipelines import DetectMiniXceptionFER
from paz.backend.image import show_image, resize_image, draw_rectangle
from paz.backend.image.opencv_image import convert_color_space, BGR2RGB
import numpy as np
import json
import math

import cProfile
import pstats

from emili_core_old_with_logging import time_since


def create_chat_evaluation():
    class ChatEvaluation(QDialog):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Chat Evaluation")
            self.setGeometry(300, 300, 400, 300)
            self.layout = QVBoxLayout()

            # Helpfulness Rating
            self.layout.addWidget(QLabel("Helpfulness:"))
            self.helpfulness_rating = QSpinBox()
            self.helpfulness_rating.setRange(1, 10)
            self.layout.addWidget(self.helpfulness_rating)
            self.layout.addWidget(QLabel("Provides valuable assistance to the original poster"))
            self.layout.addWidget(QLabel("1 = no assistance, 10 = excellent assistance"))

            # Repetitiveness Rating
            self.layout.addWidget(QLabel("Repetitiveness:"))
            self.repetitiveness_rating = QSpinBox()
            self.repetitiveness_rating.setRange(1, 10)
            self.layout.addWidget(self.repetitiveness_rating)
            self.layout.addWidget(QLabel("Avoids unnecessary repetition"))
            self.layout.addWidget(QLabel("1 = excessively repetitive, 10 = not repetitive"))

            # Easy-to-understand Rating (renamed to Intent)
            self.layout.addWidget(QLabel("Intent:"))
            self.intent_rating = QSpinBox()
            self.intent_rating.setRange(1, 10)
            self.layout.addWidget(self.intent_rating)
            self.layout.addWidget(QLabel("Accurately grasps the users’s intent"))
            self.layout.addWidget(QLabel("1 = misinterprets the user’s intent, 10 = perfectly understands the user’s "
                                         "intent"))

            # Submit Button
            self.submit_button = QPushButton("Submit")
            self.submit_button.clicked.connect(self.submit_evaluation)
            self.layout.addWidget(self.submit_button)

            self.setLayout(self.layout)

        def submit_evaluation(self):
            self.evaluation = {
                "Helpfulness": self.helpfulness_rating.value(),
                "Repetitiveness": self.repetitiveness_rating.value(),
                "Intent": self.intent_rating.value()
            }
            self.accept()  # Close the dialog

    dialog = ChatEvaluation()
    result = dialog.exec_()
    return dialog.evaluation if result == QDialog.Accepted else None


def create_emotion_survey(title="Emotion Survey", pre_chat=False):
    class EmotionSurvey(QDialog):
        def __init__(self):
            super().__init__()
            self.selected_emotions = {}  # Store selected emotions for both questions
            self.setWindowTitle(title)
            self.setGeometry(300, 300, 600, 400)
            self.layout = QVBoxLayout()

            # Main horizontal layout to hold the two sections side by side
            horizontal_layout = QHBoxLayout()

            # First question: Current feelings
            current_layout = QVBoxLayout()
            prompt_label_current = QLabel("How are you feeling right now? (Choose as many as you like)")
            current_layout.addWidget(prompt_label_current)

            self.current_emotion_checkboxes = []
            emotions = [
                "Surprised", "Excited", "Angry", "Proud", "Sad", "Annoyed", "Grateful", "Lonely",
                "Afraid", "Terrified", "Guilty", "Impressed", "Disgusted", "Hopeful", "Confident",
                "Furious", "Anxious", "Anticipating", "Joyful", "Nostalgic", "Disappointed",
                "Prepared", "Jealous", "Content", "Devastated", "Embarrassed", "Caring",
                "Sentimental", "Trusting", "Ashamed", "Apprehensive", "Faithful"
            ]

            for emotion in emotions:
                checkbox = QCheckBox(emotion)
                self.current_emotion_checkboxes.append(checkbox)
                current_layout.addWidget(checkbox)

            # Add spacer to push the current feelings section to the top
            current_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

            # Add current layout to the main horizontal layout
            horizontal_layout.addLayout(current_layout)

            # Second question: Desired feelings (only if pre_chat is True)
            if pre_chat:
                desired_layout = QVBoxLayout()
                prompt_label_desired = QLabel("How would you like to be feeling? (Choose as many as you like)")
                desired_layout.addWidget(prompt_label_desired)

                self.desired_emotion_checkboxes = []
                desired_emotions = [
                    "Surprised", "Excited", "Proud", "Grateful",
                    "Impressed", "Hopeful", "Confident", "Anticipating", "Caring",
                    "Joyful", "Prepared", "Content", "Trusting", "Faithful"
                ]
                for emotion in desired_emotions:
                    checkbox = QCheckBox(emotion)
                    self.desired_emotion_checkboxes.append(checkbox)
                    desired_layout.addWidget(checkbox)

                # Add spacer to push the desired feelings section to the top
                desired_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

                # Add desired layout to the main horizontal layout
                horizontal_layout.addLayout(desired_layout)

            # Add the horizontal layout to the main layout
            self.layout.addLayout(horizontal_layout)

            self.submit_button = QPushButton("Submit")
            self.submit_button.clicked.connect(self.submit_survey)
            self.layout.addWidget(self.submit_button)

            self.setLayout(self.layout)

        def submit_survey(self):
            current_emotions = [cb.text() for cb in self.current_emotion_checkboxes if cb.isChecked()]
            if pre_chat:
                desired_emotions = [cb.text() for cb in self.desired_emotion_checkboxes if cb.isChecked()]
            else:
                desired_emotions = []

            if current_emotions:
                self.selected_emotions['current'] = current_emotions
                if pre_chat:
                    self.selected_emotions['desired'] = desired_emotions
                self.accept()  # Close the survey dialog
            else:
                QMessageBox.warning(self, "Incomplete Survey", "Please select at least one emotion for each question.")

    dialog = EmotionSurvey()
    result = dialog.exec_()
    return dialog.selected_emotions if result == QDialog.Accepted else None


class VideoPlayerWorker(QObject):
    frameReady = pyqtSignal(np.ndarray)  # signal for new frame
    finished = pyqtSignal()  # signal for thread termination

    def __init__(self, start_time, image_size, pipeline, camera, topic='image'):
        super().__init__()
        self.start_time = start_time
        self.image_size = image_size
        self.pipeline = pipeline  # specifies what to do with each frame
        self.camera = camera
        self.topic = topic
        self.last_frame_sent = 0
        self.stop_flag = False

    def step(self):
        if self.camera.is_open() is False:
            raise ValueError('Camera has not started. Call ``start`` method.')

        frame = self.camera.read()  # shape: [height, width, 3], dtype: uint8. Macbook camera height=720, width=1280
        if frame is None:
            print('No camera input.')
            return None
        frame = convert_color_space(frame, BGR2RGB)
        return self.pipeline(
            frame)  # FER pipeline returns a dictionary with keys 'image' and 'boxes2D' (bounding boxes for faces)

    def run(self):  # this is where the main thread ends up living its lonely life

        profiler = cProfile.Profile()
        profiler.enable()

        self.camera.start()
        while not self.stop_flag:
            output = self.step()  #  dictwith keys 'image' and 'boxes2D' (bounding boxes for faces)
            image = output[self.topic]  # typically, self.topic = 'image'
            if image is None:
                continue
            image = resize_image(image, tuple(
                self.image_size))  # image is a numpy array of shape [width,height,3] and dtype uint8
            self.frameReady.emit(image)
        self.camera.stop()

        profiler.disable()
        stats = pstats.Stats(profiler)
        print("VideoPlayerWorker profiler output:")
        stats.strip_dirs().sort_stats('cumulative').print_stats(10)  # Adjust as needed to view more or fewer lines

    def stop(self):
        self.stop_flag = True
        self.finished.emit()


class DisplaySignal(QObject):
    fresh_scores = pyqtSignal(
        list)  # Signal to display fresh emotion scores, carries list payload with time-series of emotion scores
    tick = pyqtSignal()  # timer to refresh frame


class Visualizer(QMainWindow):  # GUI for real-time FER visualizer
    def __init__(self, start_time, dims, colors, speed, pipeline, end_session_event, camera_id=0):
        super().__init__()
        self.start_time = start_time
        self.display_width = dims[0]
        self.display_height = dims[1]
        self.display_size = QSize(self.display_height, self.display_width)
        self.is_full_screen = False
        self.x0 = self.display_width // 2
        self.y0 = self.display_height // 2
        self.end_session_event = end_session_event
        self.camera_id = camera_id
        self.colors = colors  # expects an np array of shape (7,3) representing an RGB color for each basic emotion
        self.speed = speed  # tunnel expansion rate in pixels per second, recommend 25-50
        self.interval = 1000 // speed  # ms per pixel
        self.pipeline = pipeline
        self.num_bins = max(self.x0, self.y0) + 1
        #self.time_series = [] # list of [time, scores] pairs (moved to main)
        #self.binned_time_series = [] # averaged over bins of length (moved to main)

        self.setWindowTitle("Real-time Emotion Visualizer")
        self.resize(*dims)  # unpack [width, height]
        self.setMinimumSize(1, 1)  # Allow the user to shrink the window
        self.move(100, 100)  # window position: (0,0) is top left

        # Main layout
        main_layout = QVBoxLayout()

        # Tab widget for different tabs
        self.tab_widget = QTabWidget()
        #        self.tab_widget.setStyleSheet("QWidget { background-color: black; }")
        main_layout.addWidget(self.tab_widget)
        #self.tab_widget.currentChanged.connect(self.update_images) # called when the user switches tabs

        self.central_widget = QWidget()
        self.central_widget.setLayout(main_layout)
        self.setCentralWidget(self.central_widget)
        #        self.central_widget.setStyleSheet("background-color: black;")

        self.signal = DisplaySignal()
        self.init_FER_tab()  # tab for displaying the real-time video feed
        self.init_visualizer_tab()  # tab for displaying the visualization of emotion scores
        self.signal.fresh_scores.connect(self.redraw_visualizer)  # redraw the display in the visualizer tab

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.redraw_visualizer)
        self.timer.start(40)  # calls redraw_visualizer every 40 ms

        self.resizeTimer = QTimer(self)  # timout to prevent frequent window resizes when user is dragging the window
        self.resizeTimer.setSingleShot(True)
        self.resizeTimer.timeout.connect(self.handle_resize)
        self.resizeTimer.setInterval(100)  # ms between resize events

    def init_FER_tab(self):
        self.FER_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.FER_image = QLabel()
        layout.addWidget(self.FER_image)
        layout.setAlignment(self.FER_image, Qt.AlignCenter)
        self.FER_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # allow user to resize
        #self.FER_tab.setStyleSheet("background-color: black;")

        self.FER_tab.setLayout(layout)
        self.tab_widget.addTab(self.FER_tab, "FER")

    def init_visualizer_tab(self):
        self.visualizer_tab = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.visualizer_image = QLabel()
        layout.addWidget(self.visualizer_image)
        layout.setAlignment(self.visualizer_image, Qt.AlignCenter)
        self.visualizer_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # allow user to resize
        #self.visualizer_tab.setStyleSheet("background-color: black;")

        self.visualizer_tab.setLayout(layout)
        self.tab_widget.addTab(self.visualizer_tab, "Tunnel")

    def resizeEvent(self, event):

        print("(resizeEvent) event.size() ", event.size())
        self.resizeTimer.start()
        super().resizeEvent(event)

    def handle_resize(self):  # called when user drags the window to change its size
        # note: only the currently active tab records the new size!
        # following keeps all tabs the same size as the currently active tab.

        active_tab = self.tab_widget.currentWidget()
        active_tab_size = active_tab.size()
        print(f"(handle_resize) active_tab_size: {active_tab_size}")

        # Update the size of the inactive tab to match the active tab
        if active_tab == self.FER_tab:
            self.visualizer_tab.resize(active_tab_size)
        else:
            self.FER_tab.resize(active_tab_size)

        visualizer_tab_size = self.visualizer_tab.size()
        print("(resizeEvent) visualizer_tab_size ", visualizer_tab_size)
        #visualizer_image_size = self.visualizer_image.size()
        #print("(resizeEvent) visualizer_image_size ",visualizer_image_size)
        FER_tab_size = self.FER_tab.size()
        print("(resizeEvent) FER_tab_size ", FER_tab_size)

        # following has weird behavior without the -2 (triggers repeated resizeEvent calls, why?)
        self.display_width = active_tab_size.width() - 2
        self.display_height = active_tab_size.height() - 2
        self.display_size = QSize(self.display_width, self.display_height)
        self.x0 = self.display_width // 2
        self.y0 = self.display_height // 2
        self.num_bins = max(self.x0, self.y0) + 1

    def keyPressEvent(self, event):
        keystroke = event.key()
        print("(keyPressEvent) event ", event)
        print("     keystroke: ", keystroke)
        # logic to enter/exit full screen mode
        if not self.resizeTimer.isActive():  # ignore resize events for a brief period after a resize
            if not self.is_full_screen:  # Enter full screen when 'f' is pressed
                if keystroke == Qt.Key_F:
                    self.enterFullScreen()
            else:  # Exit full screen when 'Esc' or 'f' is pressed
                if keystroke == Qt.Key_F or keystroke == Qt.Key_Escape:
                    self.exitFullScreen()

    def enterFullScreen(self):
        self.setWindowFlags(
            Qt.Window | Qt.FramelessWindowHint | Qt.CustomizeWindowHint)  # don't show window and tab titles in fullscreen mode
        self.showFullScreen()
        self.is_full_screen = True
        QTimer.singleShot(100, self.setFocusToActiveTab)  # Delay focus setting to ensure transition completion

    #            self.tab_widget.setFocus()
    #            self.resizeTimer.start()

    def exitFullScreen(self):
        self.setWindowFlags(Qt.Window)  # show window and tab titles in normal mode
        self.showNormal()
        self.is_full_screen = False
        QTimer.singleShot(100, self.setFocusToActiveTab)  # Delay focus setting to ensure transition completion
        # self.tab_widget.setFocus()
        # self.resizeTimer.start()

    def setFocusToActiveTab(self):  # Assuming tab_widget is the QTabWidget and each tab is a QWidget
        current_tab = self.tab_widget.currentWidget()
        if current_tab is not None:
            current_tab.setFocus()

    def redraw_visualizer(self):  # expects a list of [time, scores] pairs in chronological order

        if len(self.pipeline.binned_time_series) == 0:
            return  # no data to display

        #print("(redraw_visualizer) len(self.pipeline.binned_time_series) ",len(self.pipeline.binned_time_series))
        binned_time_series = self.pipeline.binned_time_series[-self.num_bins:]  # get the most recent binned time series
        #print("(redraw_visualizer) self.num_bins, len(binned_time_series) ",self.num_bins,len(binned_time_series))
        #print("(redraw_visualizer) binned_time_series: ",binned_time_series)

        # pad time series with most recent data to fill the display
        current_time = time_since(self.start_time)
        last_scores = binned_time_series[-1][1]  # most recent scores
        bin_end_time = binned_time_series[-1][0] + self.interval  # most recent bin end time
        while bin_end_time < current_time:  # catch up to current_time
            #print("(redraw_visualizer) catching up, empty bin")
            binned_time_series.append([bin_end_time, last_scores])  # pad data with most recent scores
            bin_end_time += self.interval

        # draw the tunnel
        image = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        for timestamp, scores in reversed(binned_time_series):  # draw the most recent scores first
            # print("(redraw_visualizer) item",item)
            # timestamp = item[0]
            # scores = item[1]
            # if(len(item)>2):
            #     print("(redraw_visuzlizer) long item! item[2], len(item)",item[2],len(item))
            radius = (current_time - timestamp) // self.interval  # most recent data at center, 25 pixels per second
            #print("(redraw_visuzlizer) radius, timestamp, scores/1e6: ",radius,timestamp,scores/1e6)
            x_min, x_max = self.x0 - radius, self.x0 + radius
            y_min, y_max = self.y0 - radius, self.y0 + radius
            if (x_min < 0 and y_min < 0):  # draw partial rectangles too
                break
            combined_color = self.colors.T @ (scores / 1e6)  # matrix multiplication (3,7) @ (7,1) = (3,1)
            #print("(redraw_visualizer) scores/1e6: ",scores/1e6)
            #print("(redraw_visualizer) combined_color: ",combined_color)
            #print(f"(redraw_visualizer) {(x_min, y_min)}, {(x_max, y_max)}, {combined_color.tolist()}")
            image = draw_rectangle(image, (x_min, y_min), (x_max, y_max), combined_color.tolist(),
                                   1)  # corner, corner, color, thickness

        #print("(redraw_visualizer) image: ",image)
        #print("(redraw_visualizer) np.amax(image): ",np.amax(image))
        # Convert the numpy array image to QPixmap and display it on a QLabel
        bytesPerLine = 3 * self.display_width
        qImg = QImage(image.data, self.display_width, self.display_height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.visualizer_image.setPixmap(pixmap)  # pixmap will be displayed in the Visualizer tab of the GUI

    def display_frame(self, image):  # display what the camera sees, marked up with FER boxes
        # Convert the numpy array image to QPixmap and display it on a QLabel
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        #print(f"(display_frame) pixmap.size() {pixmap.size()}") # 800, 450
        # Create a QTransform for horizontal flipping. todo: flip elsewhere so the text doesn't reverse!
        #reflect = QTransform()
        #reflect.scale(-1, 1)  # Scale by -1 on the X axis for horizontal flip
        #reflected_pixmap = pixmap.transformed(reflect)

        resized_pixmap = pixmap.scaled(self.display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        #print(f"(display_frame) resized_pixmap.size() {resized_pixmap.size()}")
        self.FER_image.setPixmap(resized_pixmap)

        #self.FER_image.setPixmap(pixmap) #pixmap will be displayed in the FER tab of the GUI
        #self.FER_image.setPixmap(reflected_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):  # called when user closes the GUI window
        self.end_session_event.set()  # Signal other threads that the session should end
        event.accept()  # Continue the closing process


# Define a signal class to handle new chat messages
class ChatSignal(QObject):
    new_message = pyqtSignal(dict)  # Signal to display a new user message, carries dict payload with message
    update_transcript = pyqtSignal(
        list)  # Signal to update the transcript display, carries list payload with transcript


class ChatApp(QMainWindow):  # GUI for LLM video chat
    def __init__(self, start_time, chat_window_dims, user_chat_name, assistant_chat_name, chat_queue, chat_timestamps,
                 new_chat_event, end_session_event):
        super().__init__()
        self.start_time = start_time
        self.user_chat_name = user_chat_name
        self.assistant_chat_name = assistant_chat_name
        self.chat_queue = chat_queue
        self.chat_timestamps = chat_timestamps
        self.new_chat_event = new_chat_event
        self.end_session_event = end_session_event

        self.setWindowTitle("EMILI: Emotionally Intelligent Listener")
        self.resize(*chat_window_dims)  # unpack [width, height]
        self.move(100, 100)  # window position: (0,0) is top left

        # Main layout
        main_layout = QVBoxLayout()

        # Tab widget for different tabs
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Shared input bar at the bottom
        self.chat_input = QLineEdit()
        self.chat_input.setFixedHeight(72)  # Set the height to accommodate three lines of text
        self.chat_input.setStyleSheet(
            "QLineEdit { height: 80px; font-size: 24px; }")  # Adjust the height and font-size as needed
        self.chat_input.returnPressed.connect(self.act_on_user_input)  # function to call when user presses Enter
        main_layout.addWidget(self.chat_input)

        # Central widget setup
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.signal = ChatSignal()
        self.init_chat_tab()
        self.init_FER_tab()
        self.init_transcript_tab()
        self.signal.new_message.connect(self.display_new_message)
        self.signal.update_transcript.connect(self.update_transcript_display)

    def closeEvent(self, event):  # called when user closes the GUI window
        self.end_session_event.set()  # Signal other threads that the session should end
        event.accept()  # Continue the closing process

    def act_on_user_input(self):
        user_input = self.chat_input.text().rstrip('\n')  # remove trailing newline
        if user_input:
            self.signal.new_message.emit(
                {"role": "user", "content": user_input})  # Signal chat pane to display user message
            self.chat_input.clear()
            self.chat_timestamps.put(time_since(self.start_time))  # milliseconds since start of session
            self.chat_queue.put(user_input)  # pass user message to the assembler thread
            self.new_chat_event.set()  # Signal new chat to the assembler thread

    def display_frame(self, image):
        # Convert the numpy array image to QPixmap and display it on a QLabel
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)

        # Create a QTransform for horizontal flipping. todo: flip elsewhere so the text doesn't reverse!
        #reflect = QTransform()
        #reflect.scale(-1, 1)  # Scale by -1 on the X axis for horizontal flip
        #reflected_pixmap = pixmap.transformed(reflect)

        #image_label will be displayed in the FER tab of the GUI
        self.image_label.setPixmap(pixmap)
        #self.image_label.setPixmap(reflected_pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def init_FER_tab(self):
        self.FER_tab = QWidget()
        layout = QVBoxLayout()

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.FER_tab.setLayout(layout)
        self.tab_widget.addTab(self.FER_tab, "FER")

    def init_transcript_tab(self):
        self.transcript_tab = QWidget()  # Create a new tab widget
        layout = QVBoxLayout()  # Use a vertical layout

        # Create a read-only QTextEdit widget to display the transcript
        self.transcript_display = QTextEdit()
        self.transcript_display.setReadOnly(True)
        layout.addWidget(self.transcript_display)  # Add the QTextEdit to the layout

        self.transcript_tab.setLayout(layout)  # Set the layout for the transcript tab
        self.tab_widget.addTab(self.transcript_tab, "Transcript")  # Add the transcript tab to the main tab widget

    def init_chat_tab(self):
        self.chat_tab = QWidget()
        layout = QVBoxLayout()

        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setStyleSheet("QTextEdit { font-size: 18pt; }")
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)

        # # User input area: moved to main window
        # self.chat_input = QLineEdit()
        # self.chat_input.returnPressed.connect(self.act_on_user_input) # function to call when user presses Enter
        # layout.addWidget(self.chat_input)

        self.chat_tab.setLayout(layout)
        self.tab_widget.addTab(self.chat_tab, "Chat")

    def display_new_message(self, message):  # Display new message in the chat tab
        sender = message["role"]
        content = message["content"]
        if sender == "user":
            sender = self.user_chat_name
            text = f"{sender}: {content}"  # todo: color by user emotion
            text = f"<span style='font-size:18pt;'>{sender}: {content}</span><br>"
            self.chat_display.append(text)
        elif message["role"] == "assistant":
            sender = self.assistant_chat_name
            colorful_text = f"<span style='font-size:18pt;'>{sender}: <span style='color:green;'>{content}</span></span><br>"
            self.chat_display.append(colorful_text)  # todo: check for verbose

    def update_transcript_display(self, full_transcript):
        # Convert the JSON data to a pretty-printed string
        transcript_json = json.dumps(full_transcript, indent=4, sort_keys=False)  # newlines escape as '\\n'
        transcript_json = transcript_json.replace('\\n', '\n')  # Replace escaped newlines with actual newlines
        scroll_position = self.transcript_display.verticalScrollBar().value()  # Save the current scroll position
        self.transcript_display.setPlainText(transcript_json)  # renders as plain text, no HTML
        self.transcript_display.verticalScrollBar().setValue(scroll_position)  # Restore the scroll position

        # transcript_html = transcript_json.replace('\\n', '<br>') # render line breaks
        # self.transcript_display.setHtml(transcript_html)
