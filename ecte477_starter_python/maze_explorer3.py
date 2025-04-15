#!/usr/bin/env python3
# Simple Explorer
# Written by Jeff Moscrop
# March 2024

import rclpy
import time
import math
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Point, Vector3, Pose
from nav_msgs.msg import Odometry
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String, ColorRGBA
from tf_transformations import quaternion_from_euler as tquat
from tf_transformations import euler_from_quaternion as teul
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from visualization_msgs.msg import Marker, MarkerArray
from . import transformations as trans

from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

BASE_FRAME =        "base_link"         # Param from the SLAM module

class TurtlebotExplore(Node):

    def __init__(self):
        super().__init__('turtlebot_explorer_node')
        self.initPose = False
        self.setInitPose = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 1)

        qos_policy = QoSProfile(durability=QoSDurabilityPolicy.VOLATILE, reliability=QoSReliabilityPolicy.BEST_EFFORT, history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        
        self.tf_buffer = Buffer()
        self.transform_listener = TransformListener(self.tf_buffer, self)
        
        # If our computer is slow, we must not assume that previous nodes have completed
        # their setup before we get to this point. hence we should wait until there is 
        # a transform available to us before moving on. In the mean time, we spin once
        # to allow background callbacks to function along with our transform listener
        self.get_logger().info("Waiting for initial transform to be available...")
        while not self.tf_buffer.can_transform(BASE_FRAME, 'odom', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.pub_stack_points = self.create_publisher(MarkerArray, '/stack_points', 5)
        self.startSub = self.create_subscription(String, '/start_explore', self.start_callback, 5)
        self.laserSub = self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_policy)
        self.odomSub = self.create_subscription(Odometry, '/odom', self.odom_callback, 5)
        self.move_action = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.goal = None
        self.startSub
        self.laserSub
        self.odomSub
        self.next_move = [[0, 0], [0, 0] , [0, 0]]
        self.px = 0.0
        self.py = 0.0
        self.rotAngle = math.pi/4
        self.rob_dir = 0
        self.compare = False
        self.laser_done = False
        self.odom_done = False
        self.start_command = False
        self.start = True
        self.finished = False
        self.goal_reached = False
        self.goal_stack = []
        self.prev_goals = []
        self.stack_points = MarkerArray()
        self.initial = True
        self.zone = 5.0

        while not self.finished:
            while not self.start_command:
                self.get_logger().info('Waiting for the start command')
                rclpy.spin_once(self, timeout_sec=1.0) # wait for the start command
            if self.goal_reached:
                self.laser_done = False
                self.odom_done = False
                while not self.laser_done or not self.odom_done:
                    self.get_logger().info('Getting sensor update')
                    rclpy.spin_once(self, timeout_sec=1.0)  # wait for lidar and odom callbacks
                if (self.next_move[0][1] == 270):
                    self.set_goal(self.next_move[0])
                if (self.next_move[1][1] == 90):
                    self.set_goal(self.next_move[1])
                if self.next_move[2][0] > 0: 
                    if self.next_move[2][0] > self.zone:
                        self.set_goal(self.next_move[2])

                if not self.goal_stack:
                    if self.start:
                        # There was a bug here. An additional argument was included
                        # despite set_goal only taking 1. From reading the code I
                        # inferred the intended functionality and fixed it.
                        # Before the fix this was causing occasional crashes
                        self.set_goal([0,180])
                    else:
                        self.get_logger().info('Exploration Completed')
                        self.pub_stack_points.publish(self.stack_points)
                        self.finished = True
                        break
                self.setPose(self.goal_stack.pop())
                self.goGoal(self.goal)

    def start_callback(self, data):
        command = data.data
        command = command.lower()
        if command == 'start':
            self.goal_reached = True
            self.start_command = True

    def set_goal(self, move):
        self.start = False
        if move[0] > 2.5:
            move[0] = 2.5
        elif move[0] > 0.25:
            move[0] = move[0] - 0.25
        elif move[0] == 0:
            move[0] = 1.2
        
        theta = math.pi*move[1]/180
        newTheta = self.rob_dir + theta
        move_x = move[0]*math.cos(newTheta)
        move_y = move[0]*math.sin(newTheta)
        
        nOrient = 2*self.rotAngle*(round(newTheta/(2*self.rotAngle)))
        if move[1] < 15 or move[1] > 345:
            if abs(math.cos(nOrient)) == 1.0:
                move_y = 0
            elif abs(math.sin(nOrient)) == 1.0:
                move_x = 0

        if (abs(move_x) > 0.5) or (abs(move_y) > 0.5):
            [qx,qy,qz,qw] = tquat(0,0,nOrient)
            x = self.px + move_x
            y = self.py + move_y
        
            new_goal = [x, y, qz, qw]
            if self.prev_goals:
                self.compare_goal(new_goal)
            if not self.compare:
                self.goal_stack.append(new_goal)
                self.prev_goals.append(new_goal)
                self.set_markers()
            self.compare = False

    def set_markers(self):
        i = len(self.goal_stack)
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = i
        marker_pos = Point()
        marker_pos.x = self.goal_stack[i-1][0]
        marker_pos.y = self.goal_stack[i-1][1]
        marker_pos.z = 0.0
        marker_orient = Quaternion()
        marker_orient.x = 0.0
        marker_orient.y = 1.0
        marker_orient.z = 0.0
        marker_orient.w = 1.0
        marker.pose.position = marker_pos
        marker.pose.orientation = marker_orient
        marker_scale = Vector3()
        marker_scale.x = 0.1
        marker_scale.y = 0.1
        marker_scale.z = 0.1
        marker.scale = marker_scale
        marker_colour = ColorRGBA()
        marker_colour.r = 1.0
        marker_colour.g = 0.0
        marker_colour.b = 0.0
        marker_colour.a = 1.0
        marker.color = marker_colour
        self.stack_points.markers.append(marker)
        self.pub_stack_points.publish(self.stack_points)

    def compare_goal(self,new):
        for index in range(len(self.prev_goals)):
            prev = self.prev_goals[index]
            if (abs(new[0] - prev[0]) < 0.4) and (abs(new[1] - prev[1]) < 0.4):
                self.compare = True
                break

    def goGoal(self, goal):
        self.goal_reached = False
        action_request = NavigateToPose.Goal()
        action_request.pose = goal

        self.move_action.wait_for_server()
        self.get_logger().info('Sending goal request .....')
        send_goal_future = self.move_action.send_goal_async(action_request)

        try:
            rclpy.spin_until_future_complete(self, send_goal_future)
            self.goal_handle = send_goal_future.result()
        except Exception as e:
            self.get_logger().info('Service call failed %r' % (e,))
        
        if not self.goal_handle.accepted:
            self.get_logger().info('Goal rejected')
        else:
            self.get_logger().info('Goal accepted')

        get_result_future = self.goal_handle.get_result_async()

        self.get_logger().info("Waiting for 'Navigate to Pose' action to complete...")
        try:
            rclpy.spin_until_future_complete(self, get_result_future)
            self.get_logger().info('goal reached')
            time.sleep(1)
            self.goal_reached = True
            self.stack_points.markers.pop()
            return
        except Exception as e:
            self.get_logger().info('Service call failed %r' % (e,))

    def lidar_callback(self, msg):
        transformation = []
        while transformation == []:
            transformation = self.get_transform(msg)
        
        angle = msg.angle_min
        self.next_move = [[0, 0], [0, 0], [0, 0]]
        move_left = [0, 0]
        move_right = [0, 0]
        near_wall_ahead = 5.0
        old_x = 0.0
        old_y = 0.0
        direct_ahead = msg.ranges[0]

        if self.initial:
            for index in range(0,180,5):
                temp_zone = abs(msg.ranges[index] + msg.ranges[index+180])/2
                if (temp_zone < self.zone):
                    self.zone = temp_zone
            self.initial = False
        
        for index in range(0, 360, 5):
            scan = msg.ranges[index] 
            # 3D homogeneous point [x; y; z; 1]
            if scan > 10:
                scan = 1000.0
            point = np.array([[math.cos(angle) * scan], [math.sin(angle) * scan], [0.0], [1.0]])
            point = np.matmul(transformation, point)

            # Update angle
            angle += 5*msg.angle_increment

            # Unwrap
            point_x = point[0][0]
            point_y = point[1][0]
            dx = point_x - old_x
            dy = point_y - old_y
            if (abs(dy) > 0.1) and (abs(dx) < 0.1) and (index < 95 or index > 265) and (point_x > 0.8) and (point_x < near_wall_ahead):
                near_wall_ahead = point_x
            elif direct_ahead < near_wall_ahead:
                near_wall_ahead = direct_ahead
            old_x = point_x
            old_y = point_y
            
            if (index < 105) and (index > 75):
                if point_y > 2*self.zone:
                    move_left = [0, 90]
            elif (index < 285) and (index > 255):
                if point_y < -2*self.zone:
                    move_right = [0, 270]

        if (near_wall_ahead > 0.6):
            self.next_move[2] = [near_wall_ahead, 0]
        
        self.next_move[1] = move_left
        self.next_move[0] = move_right

        self.laser_done = True
        time.sleep(1)

    def odom_callback(self, msg):
        # first set the initial pose for AMCL
        if not self.initPose:
            initialPose = PoseWithCovarianceStamped()
            initialPose.header = msg.header
            initialPose.header.frame_id = 'map'
            initialPose.pose = msg.pose
            self.setInitPose.publish(initialPose)
            self.initPose = True

        self.px = msg.pose.pose.position.x
        self.py = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        cOrient = [0,0,qz,qw]
        r,p,y = teul(cOrient)
        self.rob_dir = y

        self.odom_done = True
        time.sleep(1)
    
    def setPose(self, pose):
        msg = PoseStamped()
        msg.header.frame_id = 'map'
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.orientation.z = pose[2]
        msg.pose.orientation.w = pose[3]
        self.goal = msg

    def get_transform(self, data):
        try:
            # This is another refactor that was particularly
            # challenging to fix. Basically we were demanding transforms
            # which corresponded to exact timestamps taken from lidar results
            # However, if nodes did not correctly synchronise in time during
            # startup (Slow PC issue again), it meant we were asking for transforms
            # at times that did not have any. As far as I could tell from reading the
            # code, it did not seem strictly important that we needed this level
            # of time sync. Instead, we can just request the most recent transform.
            # This improved reliability considerably on lower VM core counts
            new_transform = self.tf_buffer.lookup_transform(
                BASE_FRAME,
                data.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            position = new_transform.transform.translation
            quaternion = new_transform.transform.rotation
            return trans.pq_to_se3(position, quaternion)
            
        # Good code practice: we catch specific errors we expect may occur and handle them
        # gracefully rather than crashing. However, we still crash if an unexpected exception
        # occurs. This prevents us from entering an irrecoverable state.
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().info(f'Unable to get transformation: {e}') #Fixed Syntax Error on this line
            return []

        
def main(args=None):
    rclpy.init(args=args)

    tb3_explore = TurtlebotExplore()

    rclpy.spin(tb3_explore)

if __name__ == '__main__':
    main()
