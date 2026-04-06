import sys
import os
import subprocess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QWidget, QLabel, QFileDialog, QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class AnalysisThread(QThread):
    """Thread to run the analysis in the background"""
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.process = None
        self._is_running = True
        
    def stop(self):
        """Stop the analysis thread and any running subprocesses"""
        self._is_running = False
        if self.process and self.process.poll() is None:
            self.process.terminate()  # Try to terminate gracefully first
            try:
                self.process.wait(timeout=2)  # Wait for process to terminate
            except subprocess.TimeoutExpired:
                self.process.kill()  # Force kill if not responding
                
    def run(self):
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(script_dir, 'HornetTracker.py')
            
            # Store the process reference so we can terminate it if needed
            self.process = subprocess.Popen(
                [sys.executable, script_path, '--video', self.video_path],
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW
            )
            
            # First, get total frames from the video to set up progress tracking
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Emit total frames first (negative value to indicate it's the total)
            self.progress_signal.emit(-total_frames)
            
            # Track frames processed
            frames_processed = 0
            
            # Read output in real-time to update progress
            while self._is_running:
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                if output:
                    # Look for the frame analysis pattern
                    if output.strip().startswith('0: 384x640'):
                        frames_processed += 1
                        self.progress_signal.emit(frames_processed)
                    print(output.strip())
                
                # Small delay to prevent busy waiting
                self.msleep(10)
            
            # Check if we were stopped or if the process completed
            if not self._is_running:
                self.finished_signal.emit(False, "Analysis was cancelled by user.")
            else:
                # Check if the process completed successfully
                return_code = self.process.poll()
                if return_code == 0:
                    self.finished_signal.emit(True, "Analysis completed successfully!")
                else:
                    error = self.process.stderr.read()
                    self.finished_signal.emit(False, f"Error: {error}")
                
        except Exception as e:
            self.finished_signal.emit(False, f"An error occurred: {str(e)}")


class HornetTrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.analysis_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Hornet Tracker")
        self.setGeometry(100, 100, 500, 250)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # File selection
        self.file_label = QLabel("No video selected")
        self.file_label.setWordWrap(True)
        
        select_btn = QPushButton("Select Video File")
        select_btn.clicked.connect(self.select_video)
        
        # Progress bar
        self.progress_label = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)
        
        # Start analysis button
        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.clicked.connect(self.start_analysis)
        self.analyze_btn.setEnabled(False)
        
        # Add widgets to layout
        layout.addWidget(QLabel("Selected Video:"))
        layout.addWidget(self.file_label)
        layout.addWidget(select_btn)
        layout.addSpacing(20)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addStretch()
        layout.addWidget(self.analyze_btn)
        
        # Set layout
        central_widget.setLayout(layout)
    
    def select_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if file_path:
            self.video_path = file_path
            self.file_label.setText(file_path)
            self.analyze_btn.setEnabled(True)
    
    def start_analysis(self):
        if not self.video_path:
            QMessageBox.warning(self, "Error", "Please select a video file first.")
            return
            
        # Update UI for analysis
        self.progress_label.setText("Analysis in progress...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.analyze_btn.setEnabled(False)
        
        # Create and start analysis thread
        self.analysis_thread = AnalysisThread(self.video_path)
        self.analysis_thread.progress_signal.connect(self.update_progress)
        self.analysis_thread.finished_signal.connect(self.analysis_finished)
        self.analysis_thread.start()
    
    def update_progress(self, value):
        # If value is negative, it's the total frames count
        if value < 0:
            self.total_frames = -value
            self.progress_bar.setRange(0, self.total_frames)
            self.progress_bar.setValue(0)
        else:
            # Update progress based on frames processed
            self.progress_bar.setValue(value)
    
    def analysis_finished(self, success, message):
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        
        if success:
            self.progress_label.setText("Analysis completed successfully!")
            QMessageBox.information(self, "Success", message)
            
            # Results are saved in the 'Results' folder
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
        else:
            # Don't show error message if analysis was cancelled by user
            if message != "Analysis was cancelled by user.":
                self.progress_label.setText("Analysis failed")
                QMessageBox.critical(self, "Error", message)
            else:
                self.progress_label.setText("Analysis cancelled")
    
    def closeEvent(self, event):
        """Handle the window close event"""
        if hasattr(self, 'analysis_thread') and self.analysis_thread and self.analysis_thread.isRunning():
            # Ask for confirmation before closing during analysis
            reply = QMessageBox.question(
                self, 
                'Analysis in Progress',
                'Analysis is still running. Are you sure you want to quit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                # Stop the analysis thread
                self.analysis_thread.stop()
                # Wait for thread to finish (with timeout)
                self.analysis_thread.wait(2000)  # Wait up to 2 seconds
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = HornetTrackerGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
