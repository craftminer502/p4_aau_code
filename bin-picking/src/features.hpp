#if !defined(FEATURES_H)
#define FEATURES_H

#include <opencv2/core.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

struct features { // struct to store features belonging to the same
                  // object/contour
  int contour_number; // storing the position of contour in the "contours"
                      // vector (this is used for finding average color from rgb
                      // image)
  float ratio; // ratio between length and width in the bounding box around the
               // contour
  float filled; // how many % of the bounding box is filled by the contour
  float area; // area of the bounding box around the contour
  float color_hue; // average hue value (H in HSV) of object found from
                   // RGB-image
  float xyz_coords[3]; // x-y-z position of object relatively to camera frame
  float avr_depth_values[3]; // storing depth values in meters: ([0] is at
                             // center and [1] and [2] are at the points found
                             // in 'get_points_for_depth' )
  float T_base_obj[4][4]; // Transformation matrix describing the object
                          // position and orientation relatively to base frame
                          // of robot
  float RPY[3]; // describing the object orientation relatively to base frame of
                // robot (represented as RPY)

  Point2f rect_points[4]; // the coordinates (pixels) of the four corners of the
                          // bounding box

  features(int cn, float r, float fil, float a, Point2f (*rp)[4])
      : contour_number(cn), ratio(r), filled(fil), area(a), rect_points(*rp){};

  Point2f get_points_for_depth(int i) { // returns coordinates (pixels) of two
                                        // points in the bounding box
                                        // respectively 25% and 75% of the
                                        // length and 50% of the width

    Point2f points[2];
    float x_1 = (rect_points[0].x + rect_points[1].x) / 2;
    float y_1 = (rect_points[0].y + rect_points[1].y) / 2;

    points[0].x = (rect_points[2].x + x_1) / 2;
    points[0].y = (rect_points[2].y + y_1) / 2;
    points[1].x = (rect_points[3].x + x_1) / 2;
    points[1].y = (rect_points[3].y + y_1) / 2;

    return points[i]; // points;
  }

  float dist_between_points() { // returns distance between the two points found
                                // in the function 'get_points_for_depth'
    return sqrt((get_points_for_depth(1).x - get_points_for_depth(0).x) *
                    (get_points_for_depth(1).x - get_points_for_depth(0).x) +
                (get_points_for_depth(1).y - get_points_for_depth(0).y) *
                    (get_points_for_depth(1).y - get_points_for_depth(0).y));
  }

  Point2f get_center() { // returns coordinates (pixels) of the centour of the
                         // bounding box
    Point2f center[1];

    center[0].x = (rect_points[0].x + rect_points[2].x) / 2;
    center[0].y = (rect_points[0].y + rect_points[2].y) / 2;

    return center[0]; // return position of object center;
  }

  float get_rotation_z() { // returns the rotation of the object about the
                           // Z-axis relatively to camera frame
    float x_A = rect_points[0].x;
    float y_A = rect_points[0].y;
    float x_B = rect_points[1].x;
    float y_B = rect_points[1].y;

    // if angle is exactly 90 degrees -> return
    if (x_A == x_B) {
      return 3.1415; // radians
    }

    float vector_1[2] = {x_A - x_B, y_A - y_B};
    float vector_2[2] = {vector_1[0], 0};

    float dot = vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1];
    float det = vector_1[0] * vector_2[1] - vector_1[1] * vector_2[0];

    return -atan2(det, dot)+(3.1415/2); //*180/3.1415;
  }
};

#endif // FEATURES_H
