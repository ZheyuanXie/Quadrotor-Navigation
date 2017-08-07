#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from mav_msgs.msg import RollPitchYawrateThrust
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import sensor_msgs.point_cloud2
import numpy as np
import time, sys
import tf
import cv2
from mpctr import Mpc
from cv_bridge import CvBridge
from path import *
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

class ctrl_node():
    def __init__(self):

    	rospy.on_shutdown(self.onClose)
    	self.enabled = True

        #Callback Storage Variables
        self.pose_vicon=[0]*6  # Vicon Server Feedback: x,y,z,vx,vy,vz
        self.pose_drone=[0]*6  # Drone Sensor Feedback: vx,vy,vz,roll,pitch,yaw
        self.acc=[0]*3
        self.select_point = (0,0,0)
        self.depth_image = None

        self.cmd=[0]*3

        #Controller Variables
        self.mpc = Mpc()
        self.sim_time=0
        self.mpc_time_last_u = 0
        self.mpc_time_last_c = 0
        self.no_area_flag = False
        self.no_area_path = None

        #Image Processing Variables
        self.bridge = CvBridge()  # CvBridge for converting ROS message to OpenCv image
        self.has_area = True
        self.area_center = (0,0)
        self.area_radius = 0
        self.img_shape = (0,0)
        self.target_coordinate = (0,0,0)

        # Open a file for data logging
        self.logfile = open('log.txt','w')

        # Publisher
        self.pubcmd = rospy.Publisher('/firefly/command/roll_pitch_yawrate_thrust',RollPitchYawrateThrust)
        # Subscriber
        rospy.Subscriber('/firefly/odometry_sensor1/odometry',Odometry,self.navdata_callback)
        rospy.Subscriber('/firefly/ground_truth/odometry',Odometry,self.vrpn_callback)
        rospy.Subscriber('/firefly/ground_truth/imu',Imu,self.imu_callback)
        rospy.Subscriber('/firefly/vi_sensor/camera_depth/depth/disparity',Image,self.depth_image_callback)  # Depth disparity image
        rospy.Subscriber('/firefly/vi_sensor/camera_depth/depth/points',PointCloud2,self.point_cloud_callback)  # Point cloud data

        

    def depth_image_callback(self,data):
    	print 'image-in:'+str(self.sim_time)
        #Image decodidng and connversion
        depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")  #32FC1
        res2 = np.float32(depth_image) / 10.0  # scaling
        depth_image_display = cv2.cvtColor(res2,cv2.COLOR_GRAY2RGB) #Convert to 3 Channel for colored displays
        res = np.uint8(res2*255)
        ret , depth_image_thres = cv2.threshold(res, 96, 255, cv2.THRESH_BINARY)  
        self.img_shape = depth_image_thres.shape

        #Find annd Draw all the contours
        contours, hierarchy = cv2.findContours(depth_image_thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(depth_image_display, contours, -1, (0,255,0), 3)
        if not contours:
            print 'no contour'
            cv2.imshow('Image', depth_image_display)
            self.area_center = (0,0,0)
            self.has_area = False
            return

        #Find and Draw the contour of largest area
        largest_area = 0
        largest_contours = None
        for item in contours:
            area = cv2.contourArea(item)
            if area > largest_area:
                largest_area = area
                largest_contours = [item]
        epsilon = 0.02*cv2.arcLength(largest_contours[0],True)
        cnt = cv2.approxPolyDP(largest_contours[0],epsilon,True)
        cv2.drawContours(depth_image_display, largest_contours, -1, (0,0,255), 3)
        
        if (self.has_area == True and largest_area < 5000) or (self.has_area == False and largest_area < 10000):
            print 'area too small'
            cv2.imshow('Image', depth_image_display)
            self.area_center = (0,0,0)
            self.has_area = False
            return

        # Find Largest inscribed circle
        center = (0,0)
        maxdist = 0
        rect_x,rect_y,rect_w,rect_h = cv2.boundingRect(cnt)
        for i in range(rect_x,rect_x+rect_w,5):
            for j in range(rect_y,rect_y+rect_h,5):
                dist = cv2.pointPolygonTest(cnt,(i,j),True)
  
                if (dist > maxdist):
                    maxdist = dist
                    center = (i,j)
        self.area_center = center
        self.area_radius = maxdist
        self.has_area = True
        cv2.circle(depth_image_display, center, int(maxdist), (255,0,0), 1)

        #Display result
        cv2.imshow('Image', depth_image_display)
        cv2.waitKey(2)

    def point_cloud_callback(self,data):
        if self.has_area == False:
            self.select_point = (0,0,0)
            return

        # Generate a list for all the points
        # point_cloud_gen = sensor_msgs.point_cloud2.read_points(data, skip_nans=False)  #  A generator for all the points
        # point_cloud = []
        # for p in point_cloud_gen:
        #     point_cloud.append(p)
            
        # Select corresponding point of the image from the point cloud
        (x, y) = self.area_center
        index = (y - 1) * self.img_shape[1] + x - 1
        self.select_point = self.read_depth(data,x,y)
        print self.select_point
        
        # If no value for the select point, average its left/right neighbour
        if np.isnan(self.select_point[0]):
            x_l , x_r = x, x
            while 1:
                x_l -= 1
                p = self.read_depth(data,x_l,y)
                if not np.isnan(p[0]):
                    break
            while 1:
                x_r += 1
                p = self.read_depth(data,x_r,y)
                if not np.isnan(p[0]):
                    break
            pl=self.read_depth(data,x_l,y)
            pr=self.read_depth(data,x_r,y)
            self.select_point = (0.5*(pl[0]+pr[0]),0.5*(pl[1]+pr[1]),0.5*(pl[2]+pr[2]))

    def read_depth(self,data,x,y):
    	gen = sensor_msgs.point_cloud2.read_points(data,field_names=("x","y","z"),skip_nans=False,uvs=[[x,y]])
    	return next(gen)

    def vrpn_callback(self,data):
    	# this callback is also used for timing
        self.sim_time = data.header.stamp.secs + data.header.stamp.nsecs * 0.000000001

        self.pose_vicon[0] = data.pose.pose.position.x
        self.pose_vicon[1] = data.pose.pose.position.y
        self.pose_vicon[2] = data.pose.pose.position.z
        self.pose_vicon[3] = data.twist.twist.linear.x
        self.pose_vicon[4] = data.twist.twist.linear.y
        self.pose_vicon[5] = data.twist.twist.linear.z

        self.pose_drone[0] = data.twist.twist.linear.x
        self.pose_drone[1] = data.twist.twist.linear.y
        self.pose_drone[2] = data.twist.twist.linear.z

        # command generation and publishing every 0.01s
        if self.enabled:
	        self.do_calculation()
	        self.send_command()
	        self.logfile.write(str(self.sim_time) + " %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n"%(self.pose_vicon[0],self.pose_vicon[1],self.pose_vicon[2],self.cmd[0],self.cmd[1],self.cmd[2],self.cmd[3]))

    def imu_callback(self,data):
		self.acc[0] = data.linear_acceleration.x
		self.acc[1] = data.linear_acceleration.y
		self.acc[2] = data.linear_acceleration.z

    def navdata_callback(self,data):
        quaternion = (
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.pose_drone[3] = euler[0]
        self.pose_drone[4] = euler[1]
        self.pose_drone[5] = euler[2]

    def do_calculation(self):
        select_point = self.select_point

        # Convert to gazebo world conventional coordinate
        target = (select_point[2],-select_point[0],-select_point[1])

        # rotate according to ground plane
        roll = self.pose_drone[3]
        pitch = self.pose_drone[4]
        sin_pitch, cos_pitch = np.sin(pitch), np.cos(pitch)
        sin_roll, cos_roll = np.sin(roll), np.cos(roll)
        p = np.matrix([[select_point[0]],[select_point[1]],[select_point[2]]])
        rx = np.matrix([[1,0,0],[0,cos_pitch,-sin_pitch],[0,sin_pitch,cos_pitch]])
        ry = np.matrix([[cos_roll,-sin_roll,0],[sin_roll,cos_roll,0],[0,0,1]])
        p_g = np.dot(np.dot(rx,ry),p)
        target = (p_g[2],-p_g[0],-p_g[1])

        # if self.mpc_time_last_u == 0 or self.sim_time - self.mpc_time_last_u >=0.1:
        if not self.has_area:
            my_path = real_time_path((0,0,0),speed = 0.1, t_start = self.sim_time, given_height=False)
            self.mpc_u = self.mpc.calc_U(0,0,0,self.pose_drone[0],self.pose_drone[1],self.pose_drone[2],self.sim_time,path = my_path)
        else:
            my_path = real_time_path(target,speed = 0.35, t_start = self.sim_time, given_height=False)
            self.mpc_u = self.mpc.calc_U(0,0,0,self.pose_drone[0],self.pose_drone[1],self.pose_drone[2],self.sim_time,path = my_path)

        # Set RC Command Values
        mpc_c = self.mpc.calc_ctrl(self.pose_drone[3],self.pose_drone[4],self.pose_drone[5])
        self.cmd = [mpc_c[0], mpc_c[1], mpc_c[2], mpc_c[3]]
        if self.has_area:
            self.cmd[2] = target[1] / 25
        else:
            self.cmd[2] = -0.5


    def send_command(self):
        msg = RollPitchYawrateThrust()
        msg.roll = self.cmd[0]
        msg.pitch = self.cmd[1]
        msg.yaw_rate = self.cmd[2]
        msg.thrust.z = self.cmd[3]
        self.pubcmd.publish(msg)

    def onClose(self):
	    self.logfile.close()
	    print "Closing..."
	    self.enabled = False
	    rpytmsg = RollPitchYawrateThrust()
	    self.pubcmd.publish(rpytmsg)
	    set_model_state = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
	    msmsg = ModelState()
	    msmsg.model_name = 'firefly'
	    msmsg.reference_frame = 'world'
	    try:
	    	rep = set_model_state(msmsg)
	    	print 'Model Reset'	
	    except rospy.ServiceException as exc:
	    	print 'Service Error:' + str(exc)
	    self.pubcmd.publish(rpytmsg)

if __name__ == '__main__':
    rospy.init_node('ardrone_ctrl', anonymous=True)
    acn = ctrl_node()
    rospy.spin()
