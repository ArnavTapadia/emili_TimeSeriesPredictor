# video chat with OpenAI models (pipe real-time emotion logs along with user's chats)

from PyQt5.QtWidgets import QApplication, QDialog # GUI uses PyQt
from PyQt5.QtCore import QThread # videoplayer lives in a QThread
from gui import ChatApp, VideoPlayerWorker, create_emotion_survey, create_chat_evaluation
from emili_core_old_with_logging import * # core threading logic

import sys
import argparse
from paz.backend.camera import Camera
import threading
import time
from datetime import datetime
import os

from openai import OpenAI
client = OpenAI()

if __name__ == "__main__":

    app = QApplication(sys.argv)

    # Show the Pre-Chat Emotion Survey
    pre_survey_emotions = create_emotion_survey(title="Pre-Chat Emotion Survey", pre_chat=True)

    if pre_survey_emotions is not None:

        # Initialize model parameters and paths
        model_name = "claude-3-sonnet-20240229"  # start with a good model
        vision_model_name = "gpt-4-vision-preview"  # can this take regular text inputs too?
        secondary_model_name = "gpt-3.5-turbo-0125"  # switch to a cheaper model if the conversation gets too long
        max_context_length = 16000
        start_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()  # all threads can access this, no need to pass it!

        transcript_path = "transcript"  # full and condensed transcripts are written here at end of session
        if not os.path.exists(transcript_path):
            os.makedirs(transcript_path)

        # Ensure the directory exists
        directory = f'{transcript_path}/{start_time_str}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save pre-chat survey emotions to a file
        with open(f'{directory}/pre_chat_emotions_{start_time_str}.json', 'w') as f:
            json.dump(pre_survey_emotions, f)

        snapshot_path = "snapshot"  # snapshots of camera frames sent to OpenAI are written here
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if (use_tts):
            tts_path = "tts_audio"  # temporary storage for text-to-speech audio files
            if not os.path.exists(tts_path):
                os.makedirs(tts_path)

        parser = argparse.ArgumentParser(description='Real-time face classifier')
        parser.add_argument('-c', '--camera_id', type=int, default=0, help='Camera device ID')
        parser.add_argument('-o', '--offset', type=float, default=0.1,
                            help='Scaled offset to be added to bounding boxes')
        args = parser.parse_args()
        camera = Camera(args.camera_id)

        chat_window_dims = [600, 600]  # width, height
        gui_app = ChatApp(start_time, chat_window_dims, user_chat_name, assistant_chat_name, chat_queue,
                          chat_timestamps, new_chat_event, end_session_event)

        pipeline = Emolog(start_time, [args.offset, args.offset],
                          f'{directory}/Emili_raw_{start_time_str}.txt')  # video processing pipeline
        user_id = 100000  # set your user ID here

        tick_thread = threading.Thread(target=tick)
        tick_thread.start()

        EMA_thread = threading.Thread(target=EMA_thread, args=(start_time, snapshot_path, pipeline), daemon=True)
        EMA_thread.start()

        use_anthropic = True  # use anthropic API for emotion detection
        print(f"Anthropic value: {use_anthropic}")  # Add this line
        sender_thread = threading.Thread(
            target=sender_thread,
            args=(model_name, vision_model_name, secondary_model_name, max_context_length, gui_app, transcript_path,
                start_time_str, start_time, use_anthropic),
            daemon=True)
        
        sender_thread.start()

        assembler_thread = threading.Thread(target=assembler_thread,
                                            args=(start_time, snapshot_path, pipeline, user_id), daemon=True)
        assembler_thread.start()

        print(f"Video chat with {model_name} using emotion labels sourced from on-device camera.")
        print(f"Chat is optional, the assistant will respond to your emotions automatically!")
        print(f"Type 'q' to end the session.")

        gui_app.show()  # Start the GUI

        print("Started GUI app.")
        print("gui_app.thread()", gui_app.thread())
        print("QThread.currentThread()", QThread.currentThread())

        video_dims = [800, 450]  # width, height (16:9 aspect ratio)
        video_thread = QThread()  # video thread: OpenCV is safe in a QThread but not a regular thread
        video_worker = VideoPlayerWorker(
            start_time,
            video_dims,
            pipeline,  # applied to each frame of video
            camera)
        video_worker.moveToThread(video_thread)

        video_thread.started.connect(video_worker.run)  # connect signals and slots
        video_worker.finished.connect(video_thread.quit)
        video_worker.finished.connect(video_worker.deleteLater)
        video_thread.finished.connect(video_thread.deleteLater)
        video_worker.frameReady.connect(gui_app.display_frame)

        video_thread.start()
        print("Started video thread.")
        app.exec_()  # start the GUI app. This should run in the main thread. Lines after this only execute if user closes the GUI.

        print("GUI app closed by user.")
        video_thread.quit()

        # Show the Post-Chat Emotion Survey
        post_survey_emotions = create_emotion_survey(title="Post-Chat Emotion Survey")
        if post_survey_emotions is not None:
            # Save post-chat survey emotions to a file
            with open(f'{directory}/post_chat_emotions_{start_time_str}.json', 'w') as f:
                json.dump(post_survey_emotions, f)

            # Show the Chat Evaluation Form
            chat_evaluation = create_chat_evaluation()
            if chat_evaluation is not None:
                # Save chat evaluation to a file
                with open(f'{directory}/chat_evaluation_{start_time_str}.json', 'w') as f:
                    json.dump(chat_evaluation, f)

        #   timer_thread.join()
        #   print("Timer thread joined.") # won't join while sleeping
        print("Video thread closed.")
        new_chat_event.set()  # signal assembler thread to stop waiting
        assembler_thread.join()
        print("Assembler thread joined.")
        new_message_event.set()  # signal sender thread to stop waiting
        sender_thread.join()
        print("Sender thread joined.")
        tick_event.set()  # signal tick and EMA threads to stop waiting
        EMA_thread.join()
        print("EMA thread joined.")
        tick_thread.join()
        print("Tick thread joined.")



        print("Session ended.")
    else:
        print("Survey was not completed. Exiting.")
        sys.exit()
