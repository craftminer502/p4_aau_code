#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib> // for exit function
#include <sstream>

#include <chrono> //for time measurement
#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"
#include "std_msgs/MultiArrayDimension.h"

#include "src/features.hpp"
#include "src/cv.hpp"

ros::Publisher pose_pub;
ros::Publisher feat_pub;

struct Program {
  rs2::colorizer color_map;
  rs2::pipeline pipe;
  rs2::frameset frameset;

  void do_autoexposure(rs2::pipeline pipe) {
    for (unsigned int i = 0; i < 30; ++i)
      pipe.wait_for_frames();
  }

  void configure_sensor(rs2::pipeline pipe) {
      auto profile = pipe.start();
      auto sensor = profile.get_device().first<rs2::depth_sensor>();

      // Set the device to High Accuracy preset of the D400 stereoscopic cameras
      if (sensor && sensor.is<rs2::depth_stereo_sensor>()) {
        sensor.set_option(RS2_OPTION_VISUAL_PRESET,
                          RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY);
      }
  };

  void run(int argc, char *argv[]) {
    //start timing
  	auto time_1 = chrono::steady_clock::now();

    configure_sensor(pipe);

//for(int i = 0; i<10; i++){ //only used for training

    rs2::align align_to_color(RS2_STREAM_COLOR);
    do_autoexposure(pipe);
    frameset = align_to_color.process(pipe.wait_for_frames());

    auto frame_color = frameset.get_color_frame(); // RGB frame
    auto frame_depth = frameset.get_depth_frame(); // depth frame

    if (auto vf = frame_depth.as<rs2::video_frame>()) {

      vf = color_map.process(frame_depth);

      // Query frame size (width and height)
      const int w = vf.as<rs2::video_frame>().get_width();
      const int h = vf.as<rs2::video_frame>().get_height();

      Mat image_rgb(cv::Size(640, 480), CV_8UC3, (void *)frame_color.get_data(),
                    Mat::AUTO_STEP); // convert rgb image to Mat
      Mat image(Size(w, h), CV_8UC3, (void *)vf.get_data(),
                Mat::AUTO_STEP); // convert depth image to Mat

      cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // convert from BGR to RGB
      cv::cvtColor(image_rgb, image_rgb, cv::COLOR_BGR2RGB); // convert from BGR to RGB

      //////////////////////////////////////////////////////////////////////////////
      ///Image proccessing

      image = image(
          Rect(FRAME_SIZE[0], FRAME_SIZE[1], FRAME_SIZE[2],
               FRAME_SIZE[3])); // cut off edges so we focus on the bin only
      image_rgb = image_rgb(
          Rect(FRAME_SIZE[0], FRAME_SIZE[1], FRAME_SIZE[2],
               FRAME_SIZE[3])); // cut off edges so we focus on the bin only

      imshow("depth", image); // show depth image

      int vect_pose = -1; // position of the object/feature struct in the "feat"
                          // vector that we want to grip

      // Image processing with different methods/kernels
      if (run_kernels(image, image_rgb, &vect_pose, frame_depth)) {

        auto feature = FEATURES[FEATURES.size() - vect_pose];

        cout << "Succes" << endl;

        cout << "center pixels: "
             << feature.get_center().x << ", "
             << feature.get_center().y << endl;
        cout << "average color: "
             << feature.color_hue << endl;
        cout << "ratio: "
             << feature.ratio << endl;
        cout << "area: "
             << feature.area << endl;
        cout << "filled: "
             << feature.filled << endl;
        cout << "depth left: "
             << feature.avr_depth_values[1]
             << ", depth right: "
             << feature.avr_depth_values[2]
             << endl;

        find_HTM_and_RPY(vect_pose); // find position and orientation of the
                                     // object. Pass the object position in the
                                     // vector

        //find elapsed time for detecting and locating
       	auto time_2 = chrono::steady_clock::now();
        cout << "Detected and located an object in "
        << chrono::duration_cast<chrono::milliseconds>(time_2 - time_1).count()
        << " ms"<<endl;

        /////////////////////////////////////////// for training:
        //write_files("src/bin-picking/training_data/cat_5.dat", vect_pose); //write training data to file (change name of file)

        /////////////////////////////////////////// for testing
        // run through all the categories and store them in array
        for (unsigned int i = 0; i < N_CATS; i++) {
          ostringstream source;
          source << "src/bin-picking/training_data/cat_" << i << ".dat"; // run from build dir
          read_files(source.str(), i);
        }
        find_cat(vect_pose); // categorize object (the position of the object in
                             // the vector is passed)

         //find elapsed time for classification
         auto time_3 = chrono::steady_clock::now();
         cout << "Classified the object in "
         << chrono::duration_cast<chrono::microseconds>(time_3 - time_2).count()
         << " µs"<<endl;


         //Publish position and orientation
         std_msgs::Float32MultiArray msg_pose;
         msg_pose.data.clear();
         for (int i = 0; i < 3; i++) //add position to array
          {
            msg_pose.data.push_back(feature.xyz_coords[i]);
          }
         for (int i = 0; i < 3; i++) //add orientation to array
          {
             msg_pose.data.push_back(feature.RPY[i]);
         }
         msg_pose.data.push_back(FEATURES[FEATURES.size() - vect_pose].category); //add category to array

         pose_pub.publish(msg_pose);


         //Publish object features for plotting
         std_msgs::Float32MultiArray msg_feat;
         msg_feat.data.clear();

         msg_feat.data.push_back(feature.color_hue);
         msg_feat.data.push_back(feature.ratio/3.24*100);
         msg_feat.data.push_back(feature.area /12150*100);

         feat_pub.publish(msg_feat);

        waitKey(0);
      } else {
        cout << "Failed" << endl;
        waitKey(0);
      }
    }

  //  FEATURES.clear(); //clear before running again (used for training)
//}  //used for training
    ros::spinOnce();
  };
};

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "CV_node");
  ros::NodeHandle n;
  pose_pub = n.advertise<std_msgs::Float32MultiArray>("/Pose", 10);
  feat_pub = n.advertise<std_msgs::Float32MultiArray>("/Feat", 10);

  /*
  From ros.org:
  ROS needs some time to register at the core and to establish all subscriber connections. When you
  just publish one single message, chances are good that it gets lost because the subscriber is not
  connected yet. A quick fix would be to add a sleep right after creation of the publisher.
  */
  ros::Rate poll_rate(100);
  poll_rate.sleep();

  Program p;

    try {
        p.run(argc, argv);

    } catch (const rs2::error &e) {
      std::cerr << "RealSense error calling " << e.get_failed_function() << "("
                << e.get_failed_args() << "):\n    " << e.what() << std::endl;
      return EXIT_FAILURE;
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
      return EXIT_FAILURE;
    }



  return EXIT_SUCCESS;
}
