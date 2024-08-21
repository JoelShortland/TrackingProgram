# https://pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/
# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import math
import matplotlib.pyplot as plt

# construct the argument parse and parse the arguments
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

try:
	buffer = args["buffer"]
except:
	buffer = 64


# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (50, 50, 0) #vmin = 20?
greenUpper = (100, 255, 255)
pts = deque(maxlen=buffer)

# Get webcam going
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

#Set some constants
ball_size = 0.065/2 #in metres, radius, measure properly later.
max_previous_frames = 10 #How many items to keep in trackers below
position_tracker = []
for i in range(10):
	position_tracker.append((0,0))
radius_tracker = [(0,0)]
velocity_tracker = [[0,0]]
 

prev_frame = None #Setup to avoid looking at repeat frames
arrow_stretch_factor = 50 #multiplies the length of all arrows

last_x_speed = 0
last_y_speed = 0

#Graph Tracking
x_velocities = []
y_velocities = []
velocities = []
x_accelerations = []
y_accelerations = []
accelerations = []
timestamps = []

measurement_arrays = []
measurement_arrays.append(x_velocities)
measurement_arrays.append(y_velocities)
measurement_arrays.append(velocities)
measurement_arrays.append(x_accelerations)
measurement_arrays.append(y_accelerations)
measurement_arrays.append(accelerations)
measurement_arrays.append(timestamps)

#Set up trackers for fps
fps = 0
last_frame_time = -1
current_frame_time = -1

#Need to set max/min values for detection, can be swapped depending on camera. 
min_radius = 15
velocity_history = []

#Results file
f = open("logs\\movement_history_" +str(time.time()) + ".csv", "a")
f.write("timestamp,x_vel,y_vel,vel,x_accel,y_accel,accel\n")




# Now we start the tracking loop.
while True:
	# grab the current frame
	frame = vs.read()

	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	#Read each frame only once
	if np.all(frame == prev_frame):
		time.sleep(0.001) #Performance, we dont need to check THAT often. 
		continue
	prev_frame = frame

	current_frame_time = time.time()
	time_step = current_frame_time-last_frame_time
	fps = 1/(time_step)
	print("fps: " + str(fps))
	last_frame_time = current_frame_time

	# resize the frame, blur it, and convert it to the HSV
	# color space
	
	frame = imutils.resize(frame, width=1920, height=1080) #My webcam is 1920x1080


	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	
    # find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use  it to compute the minimum enclosing circle and centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		# only proceed if the radius meets a minimum size
		if radius > min_radius:
			# draw the circle and centroid on the frame, then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

		# update the points queue
		#pts.appendleft(center)
		print(radius)
		#If it turns out I need to track things for longer, store last 10 here
		position_tracker.append((x,y))
		if len(position_tracker) > max_previous_frames:
			position_tracker.pop(0)
		radius_tracker.append((x,y))
		if len(radius_tracker) > max_previous_frames:
			radius_tracker.pop(0)
		pixel_size = ball_size/radius #length of each pixel at the ball in metres


		#Calcs and shid
		#Current Displacement, Velocity = Displacement/Time
		x_speed = (position_tracker[max_previous_frames-1][0] - position_tracker[max_previous_frames-2][0])#/time_step
		y_speed = (position_tracker[max_previous_frames-1][1] - position_tracker[max_previous_frames-2][1])#/time_step
		velocity_tracker.append([x_speed, y_speed])

		#Velocity averaging
		x_sum = 0
		y_sum = 0
		i = 3
		for i in range(i):
			x_sum += velocity_tracker[len(velocity_tracker)-1-i][0]
			y_sum += velocity_tracker[len(velocity_tracker)-1-i][1]
		x_speed = x_sum/i
		y_speed = y_sum/i
		
		#Accel v = u + at, (v-u)/t
		x_accel = (x_speed-last_x_speed)#/time_step
		y_accel = (y_speed-last_y_speed)#/time_step

		if x_speed == 0 or y_speed == 0:
			angle = 0
		elif x_speed > 0 and y_speed > 0:
			angle = math.tan(y_speed/x_speed)
		else:
			angle = math.tan(y_speed/x_speed)

		total_speed = (math.sqrt(x_speed**2 + y_speed**2)/time_step) * pixel_size
		total_accel = (math.sqrt(x_accel**2 + y_accel**2)/time_step) * pixel_size

		#Rounding
		total_speed = round(total_speed, 2)
		total_accel = round(total_accel, 2)

		#Draw
		if radius > min_radius:
			if total_speed > 0.05: #or math.sqrt(x_accel**2 + y_accel**2) > 0.05:
				#Velocity
				ideal_arrow_length = 200
				vel_arrow_length = math.sqrt((x_speed)**2  +  (y_speed)**2)
				speed_scaling_factor = vel_arrow_length/ideal_arrow_length
				accel_arrow_length = math.sqrt((x_accel)**2  +  (y_accel)**2)
				accel_scaling_factor = accel_arrow_length/ideal_arrow_length

				cv2.arrowedLine(frame, (int(x), int(y)), (int(x+(x_speed/speed_scaling_factor)),int(y+(y_speed/speed_scaling_factor))), (0, 255,  0), thickness=5) #Velocity
				cv2.arrowedLine(frame, (int(x), int(y)), (int(x+(x_accel/accel_scaling_factor)),int(y+(y_accel/accel_scaling_factor))), (0, 0,  255), thickness=5) #Accel

		#Update last loops
		last_x_speed = x_speed
		last_y_speed = y_speed

	if len(cnts) == 0:
		x_speed = 0
		y_speed = 0
		x_accel = 0
		y_accel = 0
		total_speed = 0
		total_accel = 0
		pixel_size = 1

	velocities.append(total_speed)
	accelerations.append(total_accel)
	x_velocities.append(x_speed/time_step*pixel_size)
	y_velocities.append(y_speed/time_step*pixel_size)
	x_accelerations.append(x_accel/time_step*pixel_size)
	y_accelerations.append(y_accel/time_step*pixel_size)

	if len(timestamps) == 0:
		timestamps.append(0)
	else:
		timestamps.append(timestamps[len(timestamps)-1]+time_step)
	

	print("Time: " +str(round(timestamps[len(timestamps)-1], 3)) + ", velocity: " + str(total_speed) + ", accel: " + str(total_accel))
	f.write(str(round(timestamps[len(timestamps)-1], 3)) + "," + str(x_speed)+ ","  + str(y_speed)+ ","  + str(total_speed)+ ","  + str(x_accel) + "," + str(y_accel) + "," + str(total_accel) + "\n")


	# show the frame to our screen
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break

	#print("Time to do work: " + str(time.time()-current_frame_time))
	

# Release the Camera
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()
# otherwise, release the camera
else:
	vs.release()

# Close all windows
cv2.destroyAllWindows()

#Stop writing
f.close()


#Post work
#Pre-Processing
for e in measurement_arrays:
	e.pop(0)
	e.pop(0)
	e.pop(0)


#Plotting
fig, axs = plt.subplots(2, 3, figsize=(15, 15))
axs[0, 0].plot(timestamps, x_velocities)
axs[0, 0].set_title('Horizontal (x) Velocity')
axs[0, 0].set(xlabel='Time (s)', ylabel='Speed (m/s)')

axs[0, 1].plot(timestamps, y_velocities, 'tab:orange')
axs[0, 1].set_title('Vertical (y) Velocity')
axs[0, 1].set(xlabel='Time (s)', ylabel='Speed (m/s)')

axs[0, 2].plot(timestamps, velocities, 'tab:red')
axs[0, 2].set_title('Total Velocity')
axs[0, 2].set(xlabel='Time (s)', ylabel='Speed (m/s)')

#Accel
axs[1, 0].plot(timestamps, x_accelerations, 'tab:green')
axs[1, 0].set_title('Horizontal (x) Acceleration')
axs[1, 0].set(xlabel='Time (s)', ylabel='Acceleration (m/s/s)')

axs[1, 1].plot(timestamps, y_accelerations, 'tab:red')
axs[1, 1].set_title('Vertical (y) Acceleration')
axs[1, 1].set(xlabel='Time (s)', ylabel='Acceleration (m/s/s)')

axs[1, 2].plot(timestamps, accelerations, 'tab:red')
axs[1, 2].set_title('Total Acceleration')
axs[1, 2].set(xlabel='Time (s)', ylabel='Acceleration (m/s/s)')


for ax in axs.flat:
    ax.set(xlabel='Time (s)')

fig.set_size_inches(15, 10)
plt.show()