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
############################################################
## FUNCTIONS
############################################################
## We return the average of the last number_to_average entries in the list, if and only if they are actual readings.
def get_x_y_averages(the_list, number_to_average):
    if len(the_list) < number_to_average:
        return (-1, -1)
    else:
        list_slice = the_list[len(the_list)-number_to_average:len(the_list)]
        x_total = 0
        y_total = 0
        for i in list_slice:
            if i == (0, 0): 
                number_to_average-= 1
            else:
                x_total += i[0]
                y_total += i[1]

        if x_total == 0:
            return (-1, -1)
        return (x_total/number_to_average, y_total/number_to_average)
        
#Take in data as np array, return list with no outliers.
def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    return data[s<m]   

############################################################
## THE PROGRAM
############################################################
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=30,
    help="max buffer size")
args = vars(ap.parse_args())

try:
    buffer = args["buffer"]
except:
    buffer = 30

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (40, 30, 0) 
greenUpper = (100, 255, 255)



pts = deque(maxlen=buffer)
echo_draw = False
naive_velocity_draw = True

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
prev_frame = None #Setup to avoid looking at repeat frames
arrow_stretch_factor = 50 #multiplies the length of all arrows

#Set up trackers for fps
fps = 0
last_frame_time = -1
current_frame_time = -1

#Need to set max/min values for detection, can be swapped depending on camera. 
min_radius = 10
min_draw_time = 1

#Trackers
timestamps = []
position_tracker = []
filtered_position_tracker = []
velocity_tracker = []
naive_velocity_tracker = []
acceleration_tracker = []
radius_tracker = []

#Drawing variables
draw_queue = []
since_last_draw = 99999
min_draw_distance = 10000


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
    
    #Time calcs
    current_frame_time = time.time()
    time_step = current_frame_time-last_frame_time
    last_frame_time = current_frame_time
    if len(timestamps) == 0:
        timestamps.append(0)
    else:
        timestamps.append(timestamps[len(timestamps)-1]+time_step)


    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=1920, height=1080) 
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # construct a mask for the color "green", then perform a series of dilations and 
    # erosions to remove any small blobs left in the mask
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
        if len(position_tracker) > 1 and (x,y) == position_tracker[len(position_tracker)-1]:
            timestamps.pop()
            continue

        # only proceed if the radius meets a minimum size
        if radius > min_radius:
            # Draw the circle and its centre on the frame
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            position_tracker.append((x, y))
            
        else:
            print("Ball detected but too small, radius: " + str(radius))
            position_tracker.append((0, 0))
       

        #Movement calculations
        pixel_size = ball_size/radius #length of each pixel at the ball in metres
        
        average_frames = 5
        (x_filtered, y_filtered) = get_x_y_averages(position_tracker, average_frames)
        filtered_position_tracker.append((x_filtered, y_filtered))

        if time_step < 0.001:
            print(time_step)
            timestamps.pop()
            continue

        if x_filtered == -1 or len(filtered_position_tracker) < average_frames+2 or radius < min_radius: #If we dont have readings yet, set 0s
            x_speed, y_speed, x_accel, y_accel = 0, 0, 0, 0
            velocity_tracker.append((0, 0))
            acceleration_tracker.append((0, 0))
            naive_velocity_tracker.append((0, 0))

        else:
            #print("X velocity debug: First val" + str(filtered_position_tracker[len(filtered_position_tracker)-2][0]) + ", Second val: " + str(filtered_position_tracker[len(filtered_position_tracker)-5][0]))
            x_vel_range = []
            y_vel_range = []
            for i in range(average_frames):
                    x_vel_range.append((x-position_tracker[len(position_tracker)-2-i][0])*pixel_size / (timestamps[len(timestamps)-1]-timestamps[len(timestamps)-2-i]))
                    y_vel_range.append((y-position_tracker[len(position_tracker)-2-i][1])*pixel_size  / (timestamps[len(timestamps)-1]-timestamps[len(timestamps)-2-i]))

            x_speed = np.average(reject_outliers(np.array(x_vel_range)))
            y_speed = np.average(reject_outliers(np.array(y_vel_range)))

            x_naive_speed = (x-position_tracker[len(position_tracker)-2][0])/time_step*pixel_size
            y_naive_speed = (y-position_tracker[len(position_tracker)-2][1])/time_step*pixel_size
            #print((x_naive_speed, y_naive_speed))
            #naive_velocity_tracker.append((x_naive_speed, y_naive_speed))

            #x_speed = (filtered_position_tracker[len(filtered_position_tracker)-2][0] - filtered_position_tracker[len(filtered_position_tracker)-5][0])/time_step*pixel_size
            #y_speed = (filtered_position_tracker[len(filtered_position_tracker)-2][1] - filtered_position_tracker[len(filtered_position_tracker)-5][1])/time_step*pixel_size
            velocity_tracker.append((x_speed, y_speed))

            #v = u+at, a = v-u/t
            #s = ut + 1/2at^2, a = 2*(s-ut)/t^2
            if len(velocity_tracker) < 7 or velocity_tracker[len(velocity_tracker)-2] == (0, 0):
                x_accel, y_accel = 0, 0
            else:
                #x_accel = (x_speed-velocity_tracker[len(velocity_tracker)-2][0])/(time_step)
                #y_accel = (y_speed-velocity_tracker[len(velocity_tracker)-2][1])/(time_step)

                x_accel_range = []
                y_accel_range = []
                for i in range(average_frames):
                    x_accel_range.append((x_speed-velocity_tracker[len(velocity_tracker)-2-i][0]) / (timestamps[len(timestamps)-1]-timestamps[len(timestamps)-2-i]))
                    y_accel_range.append((y_speed-velocity_tracker[len(velocity_tracker)-2-i][1]) / (timestamps[len(timestamps)-1]-timestamps[len(timestamps)-2-i]))

                #print(y_accel_range)
                x_accel = np.average(reject_outliers(np.array(x_accel_range)))
                y_accel = np.average(reject_outliers(np.array(y_accel_range)))
                

            acceleration_tracker.append((x_accel, y_accel))

        total_speed = math.sqrt(x_speed**2+y_speed**2)
        total_accel = math.sqrt(x_accel**2+y_accel**2)
        #Check
        print("Time: " +str(round(timestamps[len(timestamps)-1], 3)) + ", velocity: " + str(total_speed) + ", accel: " + str(total_accel))


        if x_speed == 0:
            x_naive_speed = 0
        if y_speed == 0:
            y_naive_speed = 0

        #Add things to queue
        if echo_draw:
            if since_last_draw > min_draw_time or math.sqrt((draw_queue[len(draw_queue)-1][0]-x)**2+(draw_queue[len(draw_queue)-1][1]-y)**2) > min_draw_distance:
                if naive_velocity_draw:    
                    x_speed = x_naive_speed
                    y_speed = y_naive_speed
                draw_queue.append((x, y, x_speed, y_speed, x_accel, y_accel, radius))
                since_last_draw = 0
            else:
                since_last_draw += 1
        else:
            #For single frame drawing, just do everything in here.
            draw_queue.append((x, y, x_speed, y_speed, x_accel, y_accel))
            if radius > min_radius and total_speed != 0 and total_accel != 0:
                if naive_velocity_draw:    
                    x_speed = x_naive_speed
                    y_speed = y_naive_speed
        
                cv2.arrowedLine(frame, (int(x), int(y)), (int(x+x_speed*1000), int(y+y_speed*1000)), (0, 255,  0), thickness=5) #Velocity
                cv2.arrowedLine(frame, (int(x), int(y)), (int(x+x_accel*100), int(y+y_accel*100)), (0, 0,  255), thickness=5)   #Acceleration
                

    #No ball detected, handle the timestamp.                
    else: 
        print("No ball detected")
        #Just fill in the trackers with 0 readings
        position_tracker.append((0, 0))
        filtered_position_tracker.append((-1, -1))
        velocity_tracker.append((0, 0))
        acceleration_tracker.append((0, 0))
        naive_velocity_tracker.append((0, 0))

    #Drawing when in echo mode
    if echo_draw:    
        for i in range(buffer):
            if i >= len(draw_queue):
                break
            queue_item = draw_queue[len(draw_queue)-1-i]
            x = queue_item[0]
            y = queue_item[1]
            x_speed = queue_item[2]
            y_speed = queue_item[3]
            x_accel = queue_item[4]
            y_accel = queue_item[5]
            radius = queue_item[6]
            cv2.arrowedLine(frame, (int(x), int(y)), (int(x+x_speed*100), int(y+y_speed*100)), (0, 255,  0), thickness=3) #Velocity
            cv2.arrowedLine(frame, (int(x), int(y)), (int(x+x_accel*10), int(y+y_accel*10)), (0, 0,  255), thickness=3) #Acceleration
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

    
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    if not args.get("video", False):
        key = cv2.waitKey(1) & 0xFF
    else:
        key = cv2.waitKey(int(1000/35)) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

    if key == ord("c"):
        timestamps = []
        position_tracker = []
        filtered_position_tracker = []
        velocity_tracker = []
        acceleration_tracker = []


    #print("Time to do work: " + str(time.time()-current_frame_time))
    

# Release the Camera
# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()
    cv2.destroyAllWindows()
# otherwise, release the camera
else:
    vs.release()

# Close all windows


x_velocities = []
y_velocities = []
velocities = []
for val in velocity_tracker:
    x_velocities.append(val[0])
    y_velocities.append(val[1])
    velocities.append(math.sqrt(val[0]**2+val[1]**2))

naive_x_velocities = []
naive_y_velocities = []
for val in naive_velocity_tracker:
    naive_x_velocities.append(val[0])
    naive_y_velocities.append(val[1])

x_accelerations = []
y_accelerations = []
accelerations = []
for val in acceleration_tracker:
    x_accelerations.append(val[0])
    y_accelerations.append(val[1])
    accelerations.append(math.sqrt(val[0]**2+val[1]**2))

#Plotting
#Position
x_pos = []
y_pos = []
for (x, y) in position_tracker:
    x_pos.append(x)
    y_pos.append(y)

filtered_x = []
filtered_y = []
for (x, y) in filtered_position_tracker:
    if (x, y) != (-1, -1):
        filtered_x.append(x)
        filtered_y.append(y)

fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs[0, 0].plot(timestamps, x_pos, '-o')
axs[0, 0].set_title('X')
axs[0, 0].set(xlabel='Time (s)', ylabel='pixel')

axs[0, 1].plot(timestamps, y_pos, '-o')
axs[0, 1].set_title('Y')
axs[0, 1].set(xlabel='Time (s)', ylabel='pixel')
axs[0, 1].set_ylim(axs[0, 1].get_ylim()[::-1])

axs[0, 2].plot(filtered_x, filtered_y, '-o')
axs[0, 2].set_title('Position Mapping')
axs[0, 2].set(xlabel='Time (s)', ylabel='pixel')
axs[0, 2].set_xlim(0, 1920)
axs[0, 2].set_ylim(0, 1080)
axs[0, 2].set_ylim(axs[0, 2].get_ylim()[::-1])

axs[1, 0].plot(timestamps, x_velocities, '-o')
axs[1, 0].set_title('Horizontal (x) Velocity')
axs[1, 0].set(xlabel='Time (s)', ylabel='Speed (m/s)')
axs[1, 0].set_ylim(-max(list(map(abs, velocities))), max(list(map(abs, velocities))))

axs[1, 1].plot(timestamps, y_velocities, '-o')
axs[1, 1].set_title('Vertical (y) Velocity')
axs[1, 1].set(xlabel='Time (s)', ylabel='Speed (m/s)')
axs[1, 1].set_ylim(-max(list(map(abs, velocities))), max(list(map(abs, velocities))))

axs[1, 2].plot(timestamps, velocities, '-o')
axs[1, 2].set_title('Total Velocity')
axs[1, 2].set(xlabel='Time (s)', ylabel='Speed (m/s)')
axs[1, 2].set_ylim(0, max(list(map(abs, velocities))))

#Accel
axs[2, 0].plot(timestamps, x_accelerations, '-o')
axs[2, 0].set_title('Horizontal (x) Acceleration')
axs[2, 0].set(xlabel='Time (s)', ylabel='Acceleration (m/s/s)')
axs[2, 0].set_ylim(-max(list(map(abs, accelerations))), max(list(map(abs, accelerations))))

axs[2, 1].plot(timestamps, y_accelerations, '-o')
axs[2, 1].set_title('Vertical (y) Acceleration')
axs[2, 1].set(xlabel='Time (s)', ylabel='Acceleration (m/s/s)')
axs[2, 1].set_ylim(-max(list(map(abs, accelerations))), max(list(map(abs, accelerations))))

axs[2, 2].plot(timestamps, accelerations, '-o')
axs[2, 2].set_title('Total Acceleration')
axs[2, 2].set(xlabel='Time (s)', ylabel='Acceleration (m/s/s)')
axs[2, 2].set_ylim(0, max(list(map(abs, accelerations))))



for ax in axs.flat:
    ax.set(xlabel='Time (s)')

fig.set_size_inches(15, 15)
plt.show()