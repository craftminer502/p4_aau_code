#if !defined(CV_H)


#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <librealsense2/rsutil.h>
#include <opencv2/core.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "features.hpp"


#define CV_H

#define N_CATS 6 // number of different categories

using namespace std;

RNG rng(12345);

struct distances { // struct used for classification
  float dist; // distance from detected object to a point
  int cat; // the category that the point (to which we found the distance)
           // belongs to
};

vector<vector<float>>
    CATEGORIES[N_CATS]; // Global array storing training data for all categories
vector<features>
    FEATURES; // global vector storing structs containing object features
int FRAME_SIZE[4] = {210, 130, 369, 209}; // the area of the bin in the image

// different methods for image segmentation
cv::Mat test0(cv::Mat);
cv::Mat test1(cv::Mat);
cv::Mat test2(cv::Mat);
cv::Mat test3(cv::Mat);
cv::Mat test4(cv::Mat);
cv::Mat test5(cv::Mat);

int find_object(const rs2::depth_frame &frame_depth);
bool check_img(int, cv::Mat, cv::Mat, int *vect_pose,
               const rs2::depth_frame &frame_depth);
bool compareInterval(features, features);
bool sort_distance(distances, distances);
// void print_vector();
bool process_object(cv::Mat, cv::Mat, int *vect_pose,
                    const rs2::depth_frame &frame_depth,
                    vector<vector<Point>> contours);
bool run_kernels(cv::Mat, cv::Mat, int *vect_pose, const rs2::depth_frame &frame_depth);
void find_depth_values(int, const rs2::depth_frame &frame_depth);
void find_HTM_and_RPY(int);
void read_files(string, int); // function for reading training data files (this
                              // is going to be in another node)
void write_files(string, int); // function for writing training data files (this
                               // is going to be in another node)
void find_cat(int); // function that categorizes the object (this is going to be
                    // in another node)



bool check_img(int prior, cv::Mat image, cv::Mat image_rgb, int *vect_pose,
               const rs2::depth_frame &frame_depth) {
  // make copies for backup
  cv::Mat depth_org = image.clone();
  cv::Mat rgb_org = image_rgb.clone();

  // first time we run the function prior = 1. If nothing is found it will
  // increase by 1
  switch (prior) {
  case 1:
    image = test0(image);
    break;
  case 2:
    image = test1(image);
    break;
  case 3:
    image = test2(image);
    break;
  case 4:
    image = test3(image);
    break;
  case 5:
    image = test4(image);
    break;
  case 6:
    image = test5(image);
    break;

  default:
    break;
  }
  // add additional kernels/methods here to improve the system

  vector<vector<Point>>
      contours; // vector storing all found contours in the depth image

  // Find contours (input = image, output = contours):
  findContours(image, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);

  /// Find the rotated rectangles (bounding boxes) for each contour
  vector<RotatedRect> minRect(contours.size());
  for (unsigned int i = 0; i < contours.size(); i++) {
    minRect[i] = minAreaRect(cv::Mat(contours[i]));
  }

  // make an empty matrix for drawing contours
  //cv::Mat drawing = cv::Mat::zeros( image.size(), CV_8UC3 );

  // Draw all found contours
  for (unsigned int i = 0; i < contours.size(); i++) {
    Scalar color =
        Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    //drawContours( drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0,
    //Point() ); //filling the contour with white color
    // drawContours( image_rgb, contours, i, color, 1, 8, vector<Vec4i>(), 0,
    // Point() );

    // Draw rotated rectangles/bounding boxes
    Point2f rect_points[4];
    minRect[i].points(rect_points);
    for (unsigned int j = 0; j < 4; j++) {
      //line( drawing, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
      line(image_rgb, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
    }

    //imshow("CONTOURS", drawing); // show depth image

    // calculate ratio between the length and width of rectangle
    float dist1 = sqrt((rect_points[1].x - rect_points[0].x) *
                           (rect_points[1].x - rect_points[0].x) +
                       (rect_points[1].y - rect_points[0].y) *
                           (rect_points[1].y - rect_points[0].y));
    float dist2 = sqrt((rect_points[2].x - rect_points[1].x) *
                           (rect_points[2].x - rect_points[1].x) +
                       (rect_points[2].y - rect_points[1].y) *
                           (rect_points[2].y - rect_points[1].y));
    float ratio = dist1 / dist2; // dist1 must be bigger than dist2

    if (ratio <
        1) { // move the points if the "width" is bigger than the "length"
      Point2f temp = rect_points[0];

      rect_points[0] = rect_points[1];
      rect_points[1] = rect_points[2];
      rect_points[2] = rect_points[3];
      rect_points[3] = temp;

      // calculate ratio again
      dist1 = sqrt((rect_points[1].x - rect_points[0].x) *
                       (rect_points[1].x - rect_points[0].x) +
                   (rect_points[1].y - rect_points[0].y) *
                       (rect_points[1].y - rect_points[0].y));
      dist2 = sqrt((rect_points[2].x - rect_points[1].x) *
                       (rect_points[2].x - rect_points[1].x) +
                   (rect_points[2].y - rect_points[1].y) *
                       (rect_points[2].y - rect_points[1].y));
      ratio = dist1 / dist2; // dist1 must be bigger than dist2
    }

    // find ratio between area of contour and area of bounding box
    float area = contourArea(contours[i]);
    float area_bounding_box = dist1 * dist2;
    float area_filled = area / area_bounding_box * 100;

    // add data to struct and push to 'feat' vector
    // guess I need a constructor {i, ratio, area_filled, area_bounding_box,
    // rect_points[0],rect_points[1], rect_points[2],rect_points[3]}
    FEATURES.push_back(
        features(i, ratio, area_filled, area_bounding_box, &rect_points));
  }

  // sort the intervals in increasing order of how much of the bounding box is
  // filled by the contour
  sort(FEATURES.begin(), FEATURES.end(), compareInterval);

  // find out which object to choose from the sorted vector
  return process_object(image_rgb, rgb_org, vect_pose, frame_depth, contours);
}

// 6 different methods to process the image
cv::Mat test5(cv::Mat image) {
  inRange(image, Scalar(253, 0, 0), Scalar(255, 80, 255),
          image); // Changed from 200 to 254  and from 255 to 230
  cv::Mat elem = getStructuringElement(
      MORPH_ELLIPSE,
      Size(19,
           19)); // if the kernel is too large the elements will grow together
  // then check if the area of the contour is too large
  morphologyEx(image, image, MORPH_CLOSE, elem);
  morphologyEx(image, image, MORPH_OPEN, elem);
  inRange(image, Scalar(2, 2, 2), Scalar(255, 255, 255),
          image); // everything in between this interval gets white
  return image;
}
cv::Mat test3(cv::Mat image) {
  inRange(image, Scalar(254, 0, 0), Scalar(255, 140, 255),
          image); // Changed from 200 to 254  and from 255 to 230
  cv::Mat elem = getStructuringElement(
      MORPH_ELLIPSE,
      Size(19,
           19)); // if the kernel is too large the elements will grow together
  // then check if the area of the contour is too large
  morphologyEx(image, image, MORPH_CLOSE, elem);
  morphologyEx(image, image, MORPH_OPEN, elem);
  inRange(image, Scalar(2, 2, 2), Scalar(255, 255, 255),
          image); // everything in between this interval gets white
  return image;
}
cv::Mat test2(cv::Mat image) {
  inRange(image, Scalar(254, 0, 0), Scalar(255, 200, 255),
          image); // Changed from 200 to 254  and from 255 to 230
  cv::Mat elem = getStructuringElement(
      MORPH_ELLIPSE,
      Size(19,
           19)); // if the kernel is too large the elements will grow together
  // then check if the area of the contour is too large
  morphologyEx(image, image, MORPH_CLOSE, elem);
  morphologyEx(image, image, MORPH_OPEN, elem);
  inRange(image, Scalar(2, 2, 2), Scalar(255, 255, 255),
          image); // everything in between this interval gets white
  return image;
}
cv::Mat test0(cv::Mat image) {
  inRange(image, Scalar(200, 0, 0), Scalar(255, 255, 255),
          image); // Changed from 200 to 254  and from 255 to 230
  //imshow("depth_threshold", image); // show depth image
  cv::Mat elem = getStructuringElement(
      MORPH_ELLIPSE,
      Size(19,
           19)); // if the kernel is too large the elements will grow together
  // then check if the area of the contour is too large
  morphologyEx(image, image, MORPH_CLOSE, elem);
  //imshow("depth_closing", image); // show depth image
  morphologyEx(image, image, MORPH_OPEN, elem);
  //imshow("depth_opening", image); // show depth image
  inRange(image, Scalar(2, 2, 2), Scalar(255, 255, 255),
          image); // everything in between this interval gets white
  return image;
}
cv::Mat test1(cv::Mat image) {
  inRange(image, Scalar(200, 0, 0), Scalar(255, 255, 255),
          image); // everything in between this interval gets white
  cv::Mat elem = getStructuringElement(
      MORPH_ELLIPSE,
      Size(15,
           15)); // if the kernel is too large the elements will grow together
  // then check if the area of the contour is too large
  cv::Mat elem2 = getStructuringElement(
      MORPH_ELLIPSE,
      Size(9, 9)); // if the kernel is too large the elements will grow together
  morphologyEx(image, image, MORPH_CLOSE, elem);
  morphologyEx(image, image, MORPH_OPEN, elem2);
  erode(image, image, elem); // separate contours grown together
  inRange(image, Scalar(2, 2, 2), Scalar(255, 255, 255),
          image); // everything in between this interval gets white
  return image;
}
cv::Mat test4(cv::Mat image) { // used when objects grow together (when filled% is low)
  inRange(image, Scalar(50, 0, 0), Scalar(255, 255, 255),
          image); // everything in between this interval gets white
  cv::Mat elem = getStructuringElement(
      MORPH_ELLIPSE,
      Size(19,
           19)); // if the kernel is too large the elements will grow together
  // then check if the area of the contour is too large
  // morphologyEx(image, image, MORPH_CLOSE, elem);
  // morphologyEx(image, image, MORPH_OPEN, elem);
  inRange(image, Scalar(2, 2, 2), Scalar(255, 255, 255),
          image); // everything in between this interval gets white
  return image;
}

bool compareInterval(features i1, features i2) {
  return (i1.filled < i2.filled);
}

bool sort_distance(distances i1, distances i2) { return (i1.dist < i2.dist); }

int find_object(const rs2::depth_frame &frame_depth) {
  for (unsigned int i = 1; i <= FEATURES.size(); i++) {
    if (FEATURES[FEATURES.size() - i].ratio > 1.5 &&
        FEATURES[FEATURES.size() - i].ratio <
            4.1) { // check length-width-ratio //lower limit was 1.5 initially
      if (FEATURES[FEATURES.size() - i].area > 2400 &&
          FEATURES[FEATURES.size() - i].area < 15000) { // check contour area

        if (FEATURES[FEATURES.size() - i].filled > 45) { // check filled %

          find_depth_values(i, frame_depth);

          if (FEATURES[FEATURES.size() - i].avr_depth_values[0] > 0.40 &&
              FEATURES[FEATURES.size() - i].avr_depth_values[0] <
                  0.565) { // last check: does the distance to object make sense?
            // cout << "[" << FEATURES[FEATURES.size()-i].contour_number << ", "
            // << FEATURES[FEATURES.size()-i].ratio << ", " <<
            // FEATURES[FEATURES.size()-i].filled<<"%" << ", " <<
            // FEATURES[FEATURES.size()-i].area << ", " <<
            // FEATURES[FEATURES.size()-i].get_points_for_depth(0)<< ", " <<
            // FEATURES[FEATURES.size()-i].get_points_for_depth(1)<< ", " <<
            // FEATURES[FEATURES.size()-i].dist_between_points()<< "]
            // "<<endl<<endl;

            return i;
          } else {
            cout << "wrong distance. Checking next" << endl;
          }
        }
      }
    }
  }
  return -1; // returns -1 if no contour is approved
}

/*
void print_vector(){
  cout << "Intervals sorted by filled% : \n";
  for (unsigned int x=0; x<FEATURES.size(); x++){
     //cout << "[" << FEATURES[x].ratio << ", " << FEATURES[x].filled<<"%" << ",
" << FEATURES[x].area << ", " << FEATURES[x].rect_points[0]<< ", " <<
FEATURES[x].rect_points[1]<< ", " << FEATURES[x].rect_points[2]<< ", " <<
FEATURES[x].rect_points[3]<< "] "<<endl<<endl;
     cout << "[" << FEATURES[x].contour_number << ", " << FEATURES[x].ratio <<
", " << FEATURES[x].filled<<"%" << ", " << FEATURES[x].area << ", " <<
FEATURES[x].get_points_for_depth(0)<< ", " <<
FEATURES[x].get_points_for_depth(1)<< ", " <<
FEATURES[x].dist_between_points()<< "] "<<endl<<endl;
   }
}
*/

bool process_object(cv::Mat image_rgb, cv::Mat rgb_org, int *vect_pose,
                    const rs2::depth_frame &frame_depth,
                    vector<vector<Point>> contours) {

  int i = find_object(frame_depth); // Go through 'feat' vector and find an
                                    // object to grip. Store the number of
                                    // chosen contour/object from the vector

  int numb; // number of chosen contour from the vector "contours"

  if (i != -1) { // found object
    numb = FEATURES[FEATURES.size() - i].contour_number; // storing the position
                                                         // of the contour we
                                                         // want to work with
  } else { // found nothing
    numb = -1;
  }

  // make a black image for drawing the found contour
  cv::Mat contour_filled = cv::Mat::zeros(image_rgb.size(), CV_8UC3);

  if (numb > -1) { // found object
    drawContours(image_rgb, contours, numb, Scalar(0, 0, 255),
                 -1); // add the filled contour to the rgb image (just so we can
                      // see if the found object from depth frame fits the
                      // actual object in rgb frame)
    drawContours(contour_filled, contours, numb, Scalar(255, 255, 255),
                 -1); // fill the contour with white color (used for finding
                      // average color)

    ///////////////////////////////// Find average color
    double avr_red = 0;
    double avr_green = 0;
    double avr_blue = 0;

    float con_area =
        contourArea(contours[numb]); // area (pixels) of chosen contour/object

    for (int x = 0; x < contour_filled.cols; x++) { // Find average color
      for (int y = 0; y < contour_filled.rows; y++) {
        if (contour_filled.at<Vec3b>(Point(x, y))[0] == 255) {
          avr_red += (rgb_org.at<Vec3b>(Point(x, y))[0]) / con_area;
          avr_green += (rgb_org.at<Vec3b>(Point(x, y))[1]) / con_area;
          avr_blue += (rgb_org.at<Vec3b>(Point(x, y))[2]) / con_area;
        }
      }
    }
    cout << "Average rgb: ("<< avr_blue << ", " << avr_green << ", " << avr_red << ")" <<endl;
    ///////////////////////////////// Convert average color to HSV
    Mat3f bgr(Vec3f((int)avr_red, (int)avr_green, (int)avr_blue));
    Mat3f hsv;
    cvtColor(bgr, hsv, CV_BGR2HSV); // convert to hsv

    FEATURES[FEATURES.size() - i].color_hue =
        hsv[0][0][0]; // store color in struct in 'feat' vector

    *vect_pose = i; // store position of struct in vector

    imshow("rgb", image_rgb); // show rgb image with filled contour on it

    return true;
  }

  else { // No object found -> run everything again with other kernels
    FEATURES.clear(); // clear vector before running again

    return false;
  }
}

bool run_kernels(cv::Mat image, cv::Mat image_rgb, int *vect_pose,
                 const rs2::depth_frame &frame_depth) {
  for (unsigned int i = 1; i <= 6; i++) { // trying 6 times for all 6 different kernels
    if (check_img(i, image, image_rgb, vect_pose, frame_depth)) {
      cout << "found something in " << i << ". attempt"
           << endl; // start processing the image using i'th priority kernels
      return true;
    }
  }
  return false;
}

void find_depth_values(int vect_pose, const rs2::depth_frame &frame_depth) {
  float avr_center = 0;
  float avr_left = 0;
  float avr_right = 0;

  int offset_x = FRAME_SIZE[0];
  int offset_y = FRAME_SIZE[1];

  float x_center = FEATURES[FEATURES.size() - vect_pose].get_center().x;
  float y_center = FEATURES[FEATURES.size() - vect_pose].get_center().y;

  float x_left =
      FEATURES[FEATURES.size() - vect_pose].get_points_for_depth(0).x;
  float y_left =
      FEATURES[FEATURES.size() - vect_pose].get_points_for_depth(0).y;

  float x_right =
      FEATURES[FEATURES.size() - vect_pose].get_points_for_depth(1).x;
  float y_right =
      FEATURES[FEATURES.size() - vect_pose].get_points_for_depth(1).y;

  float depth_arr_center[5]; // find five depth values near to center
  float depth_arr_left[5]; // find five depth values near to center left side
  float depth_arr_right[5]; // find five depth values near to center right side

  depth_arr_center[0] =
      frame_depth.get_distance(x_center + offset_x, y_center + offset_y);
  depth_arr_center[1] =
      frame_depth.get_distance(x_center + 5 + offset_x, y_center + offset_y);
  depth_arr_center[2] =
      frame_depth.get_distance(x_center - 5 + offset_x, y_center + offset_y);
  depth_arr_center[3] =
      frame_depth.get_distance(x_center + offset_x, y_center + 5 + offset_y);
  depth_arr_center[4] =
      frame_depth.get_distance(x_center + offset_x, y_center - 5 + offset_y);

  depth_arr_left[0] =
      frame_depth.get_distance(x_left + offset_x, y_left + offset_y);
  depth_arr_left[1] =
      frame_depth.get_distance(x_left + 5 + offset_x, y_left + offset_y);
  depth_arr_left[2] =
      frame_depth.get_distance(x_left - 5 + offset_x, y_left + offset_y);
  depth_arr_left[3] =
      frame_depth.get_distance(x_left + offset_x, y_left + 5 + offset_y);
  depth_arr_left[4] =
      frame_depth.get_distance(x_left + offset_x, y_left - 5 + offset_y);

  depth_arr_right[0] =
      frame_depth.get_distance(x_right + offset_x, y_right + offset_y);
  depth_arr_right[1] =
      frame_depth.get_distance(x_right + 5 + offset_x, y_right + offset_y);
  depth_arr_right[2] =
      frame_depth.get_distance(x_right - 5 + offset_x, y_right + offset_y);
  depth_arr_right[3] =
      frame_depth.get_distance(x_right + offset_x, y_right + 5 + offset_y);
  depth_arr_right[4] =
      frame_depth.get_distance(x_right + offset_x, y_right - 5 + offset_y);

  float counter = 0;
  for (unsigned int i = 0; i < 5; i++) {
    if (depth_arr_center[i] != 0) {
      avr_center += depth_arr_center[i];
      counter++;
    }
  }
  FEATURES[FEATURES.size() - vect_pose].avr_depth_values[0] =
      avr_center / counter;

  counter = 0;
  for (unsigned int i = 0; i < 5; i++) {
    if (depth_arr_left[i] != 0) {
      avr_left += depth_arr_left[i];
      counter++;
    }
  }
  FEATURES[FEATURES.size() - vect_pose].avr_depth_values[1] =
      avr_left / counter;

  counter = 0;
  for (unsigned int i = 0; i < 5; i++) {
    if (depth_arr_right[i] != 0) {
      avr_right += depth_arr_right[i];
      counter++;
    }
  }
  FEATURES[FEATURES.size() - vect_pose].avr_depth_values[2] =
      avr_right / counter;

  // find xyz-position
  float xy_pixel[2] = {x_center + offset_x, y_center + offset_y};
  float xyz_coords[3];

  // Deproject from pixel to point in 3D
  rs2_intrinsics intr = frame_depth.get_profile()
                            .as<rs2::video_stream_profile>()
                            .get_intrinsics();
  rs2_deproject_pixel_to_point(
      xyz_coords, &intr, xy_pixel,
      FEATURES[FEATURES.size() - vect_pose]
          .avr_depth_values[0]); // convert pixels to coordinates

  FEATURES[FEATURES.size() - vect_pose].xyz_coords[0] = xyz_coords[0];
  FEATURES[FEATURES.size() - vect_pose].xyz_coords[1] = xyz_coords[1];
  FEATURES[FEATURES.size() - vect_pose].xyz_coords[2] = xyz_coords[2];
}

void find_HTM_and_RPY(int vect_pose) {

  float v = FEATURES[FEATURES.size() - vect_pose]
                .get_rotation_z(); // rotation angle in radians

  float T_base_cam[4][4] = {
      // HTM from world frame (base) to camera (this is going to be changed when
      // setup is ready)
      {0, -1, 0, -0.3361+0.007}, //  initializers for row indexed by 0
      {-1, 0, 0, 0.0267+0.018}, //  initializers for row indexed by 1
      {0, 0, -1, 0.61943}, //  initializers for row indexed by 2
      {0, 0, 0, 1}  //  initializers for row indexed by 3
  };

  float T_cam_obj[4][4] = {
      // HTM from camera frame to object
      {cos(v), -sin(v), 0, FEATURES[FEATURES.size() - vect_pose].xyz_coords[0]},
      {sin(v), cos(v), 0, FEATURES[FEATURES.size() - vect_pose].xyz_coords[1]},
      {0, 0, 1, FEATURES[FEATURES.size() - vect_pose].xyz_coords[2]},
      {0, 0, 0, 1}};

  // Initializing elements of result matrix to 0.
  for (unsigned int i = 0; i < 4; ++i)
    for (unsigned int j = 0; j < 4; ++j) {
      FEATURES[FEATURES.size() - vect_pose].T_base_obj[i][j] = 0;
    }

  // Multiplying matrixes
  for (unsigned int i = 0; i < 4; ++i)
    for (unsigned int j = 0; j < 4; ++j)
      for (unsigned int k = 0; k < 4; ++k) {
        FEATURES[FEATURES.size() - vect_pose].T_base_obj[i][j] +=
            T_base_cam[i][k] * T_cam_obj[k][j];
      }

  // Needed values for converting to RPY
  float r21 = FEATURES[FEATURES.size() - vect_pose].T_base_obj[1][0];
  float r11 = FEATURES[FEATURES.size() - vect_pose].T_base_obj[0][0];
  float r31 = FEATURES[FEATURES.size() - vect_pose].T_base_obj[2][0];
  float r32 = FEATURES[FEATURES.size() - vect_pose].T_base_obj[2][1];
  float r33 = FEATURES[FEATURES.size() - vect_pose].T_base_obj[2][2];

  // convert to RPY
  FEATURES[FEATURES.size() - vect_pose].RPY[0] = atan(r32 / r33);
  FEATURES[FEATURES.size() - vect_pose].RPY[1] =
      atan(-r31 / sqrt(r32 * r32 + r33 * r33))+3.1415;
  FEATURES[FEATURES.size() - vect_pose].RPY[2] = atan(r21 / r11);

  // print result matrix T_base_obj
  cout << endl << "Result Matrix: " << endl;
  for (unsigned int i = 0; i < 4; i++)
    for (unsigned int j = 0; j < 4; j++) {
      cout << " " << FEATURES[FEATURES.size() - vect_pose].T_base_obj[i][j];
      if (j == 3)
        cout << endl;
    }

  // print rotation represented as RPY
  cout << "RPY: [" << FEATURES[FEATURES.size() - vect_pose].RPY[0] << ", "
       << FEATURES[FEATURES.size() - vect_pose].RPY[1] << ", "
       << FEATURES[FEATURES.size() - vect_pose].RPY[2] << "]" << endl;
}

void read_files(string name, int cat_num) {
  string line;
  fstream myfile(name);
  if (myfile.is_open()) {

    int counter = 0;
    while (getline(myfile, line)) {
      stringstream ss(line);
      float i;

      CATEGORIES[cat_num].push_back(vector<float>());

      while (ss >> i) {
        CATEGORIES[cat_num][counter].push_back(i);
        if (ss.peek() == ';')
          ss.ignore();
      }
      counter += 1;
    }
    myfile.close();
  } else
    cout << "Unable to open file" << endl;
}

void write_files(string name, int vect_pose) {
  // write to file for training
  ofstream myFile;
  myFile.open(name, ios_base::app);
  if (!myFile) { // file couldn't be opened
    cerr << "Error: file could not be opened" << endl;
    exit(1);
  }
  myFile << FEATURES[FEATURES.size() - vect_pose].color_hue; //divide by max value and multiply by 100
  myFile << ";";
  myFile << FEATURES[FEATURES.size() - vect_pose].ratio /3.24*100;
  myFile << ";";
  myFile << FEATURES[FEATURES.size() - vect_pose].area /12150*100
         << endl;

  myFile.close();
}

void find_cat(int vect_pose) {
  vector<float> vect_test; // testing data
  vector<distances> dists; // vector containing distances in feature space from
                           // detected object to training data

  // push test data to vector
  vect_test.push_back(FEATURES[FEATURES.size() - vect_pose].color_hue);
  vect_test.push_back(FEATURES[FEATURES.size() - vect_pose].ratio /3.24*100);
  vect_test.push_back(FEATURES[FEATURES.size() - vect_pose].area /12150*100);

  cout << "Test values: ";
  for(unsigned int i=0; i<vect_test.size(); i++){
      cout << vect_test[i] << ";";
  }
  cout <<""<<endl;

  // calculate all the distances to traning data
  for (unsigned int category = 0; category < N_CATS; category++) {
    for (unsigned int line = 0; line < CATEGORIES[category].size(); line++) {

      float total = 0;
      float diff = 0;

      for (unsigned int y = 0; y < vect_test.size(); y++) {
        diff = CATEGORIES[category][line][y] - vect_test[y];
        total += diff * diff;
        if(y==0){ //we trust the color more than the other features
          total=total*0.5;
        }
      }
      int _cat = category;
      dists.push_back({sqrt(total), _cat});
    }
  }

  // sort distance vector (shortest distance first)
  sort(dists.begin(), dists.end(), sort_distance); // sort the vector by
                                                   // distance

  // make an array to keep track of how many times we identify each category
  int arr[N_CATS];

  for (unsigned int i = 0; i < N_CATS; i++) { // setting all values to 0
    arr[i] = 0;
  }

  for (unsigned int i = 0; i < 5; i++) { // find most frequent of the 5 nearest
                                // neighbours
    if (dists[i].cat == 0) {
      arr[0] += 1;
    } else if (dists[i].cat == 1) {
      arr[1] += 1;
    } else if (dists[i].cat == 2) {
      arr[2] += 1;
    } else if (dists[i].cat == 3) {
      arr[3] += 1;
    } else if (dists[i].cat == 4) {
      arr[4] += 1;
    } else if (dists[i].cat == 5) {
      arr[5] += 1;
    }
  }

  // find position of biggest value (most frequent category)
  int biggest = arr[0];
  int pose = 0;
  // find highest values
  for (unsigned int i = 1; i < N_CATS; i++) {
    if (arr[i] > biggest) {
      biggest = arr[i];
      pose = i;
    }
  }

  switch (pose) {
  case 0:
    cout << "category is: Jolly Cola" << endl;
    break;
  case 1:
    cout << "category is: Topform" << endl;
    break;
  case 2:
    cout << "category is: Small energy drink" << endl;
    break;
  case 3:
    cout << "category is: Red Bull" << endl;
    break;
  case 4:
    cout << "category is: Large energy drink" << endl;
    break;
  case 5:
    cout << "category is: Booster" << endl;
    break;
  default:
    break;
  }

}


#endif // CV_H
