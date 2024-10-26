# Overview
This repository is part of my 2024 honour's thesis for my Bachelor of Advanced Computing.
Relevant to markers, it contains:
- A PDF with the answers each student gave during the interview
- The code required to operate the prototype
- A link to the test videos used within the thesis

# Interview Answers
The answers to the interview are provided within a single pdf file called InterviewAnswers.pdf

# How to Use/Run the Application
The application used in the thesis can be run using either a connected camera, or with a pre-recorded video. The camera functionality is how students used the application, while the video functionality is intended for testing and demonstration purposes. 
The test videos used in the thesis can be found in the following Google Drive folder: https://drive.google.com/drive/folders/1KxDefE1m49_9DPTFLG43rTilqQev7BQ8?usp=sharing. (Needs to be hosted externally due to GitHub size requirements). 
Functionally, there is no difference between the video and webcam modes of operation. Do note that the program was built to work specifically on my laptop, it remains untested on alternate devices.  

## Controls
The controls for all application modes are identical.
c - clears all previous tracking and images
q - stop current action (note: one q stops the video/webcam feed and displays graphs, a second q closes the graphs and terminates the program).

## Free Mode
This mode uses no task-specific filtering. It was designed for general tasks, but doesn't work particularly well and wasn't used in the thesis.
Webcam: py -3 .\generaltracker.py
Video: py -3 .\generaltracker.py -v "path\to\video.mp4"

## Bouncing Ball
This mode uses specific filtering designed for the 2D ball task. Gravity is hard-coded to be downwards at 9.8ms^-2 outside of bounces.
Webcam: py -3 .\balltracker.p
Video: py -3 .\balltracker.p -v "path\to\video.mp4"

## Pendulum
This mode uses specific filtering designed for the simple pendulum task. It makes use of a moving focal point for the direction of the acceleration vector. 
Webcam: py -3 .\pendulumtracker.py
Video: py -3 .\pendulumtracker.py -v "path\to\video.mp4"

## Centripetal Motion
This mode uses specific filtering designed for the centripetal motion task. Acceleration vectors are hard-coded to point to a specific location on screen. 
Webcam: py -3 .\circletracker.py 
Video: py -3 .\circletracker.py  -v "path\to\video.mp4"
