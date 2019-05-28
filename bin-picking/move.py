#!/usr/bin/env python

import numpy as np
import rospy
import sys
import moveit_commander
import tf.transformations
from tf import TransformListener
import time
import copy
from math import sqrt
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.srv import GetPositionIKRequest
from geometry_msgs.msg import PoseStamped, PointStamped, Vector3, Point
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import geometry_msgs.msg
#from move_prediction.msg import PointArr, Goal
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output  as outputMsg
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension

# Global vars
prev_state = None
mode = "position"
mode_timer = time.time()
ik_srv = None
pub = None
grip = None
robot = None
scene = None
group = None
req = None
start_pub = None
curr_pub = None
marker_pub = None
tf_listener = None
predicted_goal = None
confidence = 0.
time_since_prediction = time.time()
command = outputMsg.Robotiq2FGripper_robot_output();

positions_list = [Point()] * 520
start_position_to_return = None
new_test = False

def init():
    global pub
    global grip
    global robot
    global scene
    global group
    global ik_srv
    global req
    global prev_state
    global start_pub
    global positions_list
    global curr_pub
    global marker_pub
    global tf_listener
    global start_position_to_return
    global command

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('robot_control_node')
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group_name = 'manipulator'
    group = moveit_commander.MoveGroupCommander(group_name)
    group.set_max_velocity_scaling_factor(0.5)

    tf_listener = tf.TransformListener()

    prev_state = robot.get_current_state().joint_state.position

    ik_srv = rospy.ServiceProxy('/compute_ik', GetPositionIK)
    pub = rospy.Publisher('/vel_based_pos_traj_controller/command', JointTrajectory, queue_size=1)
    grip = rospy.Publisher('Robotiq2FGripperRobotOutput', outputMsg.Robotiq2FGripper_robot_output)
    rospy.Subscriber("/Pose", Float32MultiArray, callback)
    req = GetPositionIKRequest()
    req.ik_request.timeout = rospy.Duration(1.)
    req.ik_request.attempts = 2
    req.ik_request.avoid_collisions = True
    req.ik_request.group_name = group_name
    command = outputMsg.Robotiq2FGripper_robot_output();
    command.rACT = 1
    command.rGTO = 1
    command.rSP  = 255
    command.rFR  = 255

    box_pose = PoseStamped()
    box_pose.header.frame_id = robot.get_planning_frame()
    box_pose.pose.orientation.w = 1.0
    box_pose.pose.position.z = 0.025
    scene.add_box("box1", box_pose, size=(1, 1, 0.01))

    rospy.sleep(0.5)

    box_pose.pose.position.x = 0
    box_pose.pose.position.y = 0.2
    box_pose.pose.position.z = 0.3375
    #box_pose.pose.orientation.x = pi/4
    scene.add_box("box2", box_pose, size=(2, 0.01, 2))

    rospy.sleep(0.5)

    box_pose.pose.position.x = 0.3
    box_pose.pose.position.y = 0
    box_pose.pose.position.z = 0.3
    #box_pose.pose.orientation.x = pi/4
    scene.add_box("box3", box_pose, size=(0.06, 0.01, 0.6))

    rospy.sleep(0.5)

    box_pose.pose.position.x = -0.344
    box_pose.pose.position.y = 0.01890
    box_pose.pose.position.z = 0.60036
    #box_pose.pose.orientation.x = pi/4
    scene.add_box("box4", box_pose, size=(0.5, 0.5, 0.01))

    rospy.sleep(0.5)

    box_pose.pose.position.x = -0.172
    box_pose.pose.position.y = -0.055
    box_pose.pose.position.z = 0.120
    #box_pose.pose.orientation.x = pi/4
    scene.add_box("box5", box_pose, size=(0.01, 0.6, 0.3))

    rospy.sleep(0.5)

    box_pose.pose.position.x = -0.482
    box_pose.pose.position.y = -0.055
    box_pose.pose.position.z = 0.120
    #box_pose.pose.orientation.x = pi/4
    scene.add_box("box6", box_pose, size=(0.01, 0.6, 0.3))

    rospy.sleep(0.5)

    box_pose.pose.position.x = -0.335
    box_pose.pose.position.y = -0.289
    box_pose.pose.position.z = 0.120
    #box_pose.pose.orientation.x = pi/4
    scene.add_box("box7", box_pose, size=(0.3, 0.01, 0.3))

    rospy.sleep(0.5)

    box_pose.pose.position.x = -0.335
    box_pose.pose.position.y = 0.182
    box_pose.pose.position.z = 0.120
    #box_pose.pose.orientation.x = pi/4
    scene.add_box("box8", box_pose, size=(0.3, 0.01, 0.3))

    rospy.sleep(0.5)


def callback(msg):
    rospy.loginfo(msg) #debug
    global prev_state
    global req
    global pub

    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    goal.header.frame_id = "world"
    goal.pose.position.x = msg.data[0]
    goal.pose.position.y = msg.data[1]
    goal.pose.position.z = msg.data[2]

    quaternion = tf.transformations.quaternion_from_euler(msg[3], msg[4], msg[5])
    pose_goal.pose.orientation.x = quaternion[0]
    pose_goal.pose.orientation.y = quaternion[1]
    pose_goal.pose.orientation.z = quaternion[2]
    pose_goal.pose.orientation.w = quaternion[3]

    req.ik_request.pose_stamped = goal
    resp = ik_srv.call(req)
    print(resp)

    if(resp.error_code.val == -31):
        print('No IK found!')
        return

    diff = np.array(resp.solution.joint_state.position)-np.array(prev_state)
    max_diff = np.max(diff)
    if(max_diff > 1.0):
        print('Singularity detected. Skipping!')
        return

    traj = JointTrajectory()
    point = JointTrajectoryPoint()
    traj.header.frame_id = pose_goal.header.frame_id
    traj.joint_names = resp.solution.joint_state.name
    point.positions = resp.solution.joint_state.position
    point.time_from_start.secs = 20
    traj.points.append(point)
    rospy.sleep(0.5)
    pub.publish(traj)
    print(traj)
    print("should be published")

    # Update previous state of robot to current state
    prev_state = resp.solution.joint_state.position

    rospy.sleep(6)
    command.rPR = 255
    grip.publish(command)
    print("closed")
    rospy.sleep(6)
    command.rPR = 0
    grip.publish(command)
    print("open")


def move():
    global prev_state
    global req
    global pub

    pose_goal = PoseStamped()
    pose_goal.header.stamp = rospy.Time.now()
    pose_goal.header.frame_id = "world"
    pose_goal.pose.orientation.x = 0.889824 #quaternion[0]
    pose_goal.pose.orientation.y = -0.446493 #quaternion[1]
    pose_goal.pose.orientation.z = -0.0939676 #quaternion[2]
    pose_goal.pose.orientation.w = 0.00515905 #quaternion[3]

    pose_goal.pose.position.x = -0.2712
    pose_goal.pose.position.y = -0.185915
    pose_goal.pose.position.z = 0.451176

    req.ik_request.pose_stamped = pose_goal
    resp = ik_srv.call(req)
    print(resp)


    # rospy.logerr("Before check error")

    # Check if valid robot configuration can be found
    if(resp.error_code.val == -31):
        print('No IK found!')
        return

    # Check if we are near a singularity
    # Skip movement if that is the case
    diff = np.array(resp.solution.joint_state.position)-np.array(prev_state)
    max_diff = np.max(diff)
    if(max_diff > 1.0):
        print('Singularity detected. Skipping!')
    #    return

    # Publish new pose as JointTrajectory
    traj = JointTrajectory()
    point = JointTrajectoryPoint()
    #traj.header.stamp = rospy.Time.now()
    traj.header.frame_id = pose_goal.header.frame_id
    traj.joint_names = resp.solution.joint_state.name
    point.positions = resp.solution.joint_state.position

    point.time_from_start.secs = 20
    traj.points.append(point)
    rospy.sleep(0.5)
    pub.publish(traj)
    print(traj)
    print("should be published")

    # Update previous state of robot to current state
    prev_state = resp.solution.joint_state.position



if __name__ == '__main__':
    init()
    #while True:
    move()
        #move2()
        #rospy.sleep(3)

    rospy.spin()
