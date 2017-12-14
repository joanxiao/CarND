#define _USE_MATH_DEFINES

#ifndef TRAJECTORY_H
#define TRAJECTORY_H
#endif
#include <vector>

using namespace std;

class Trajectory
{
public:
  Trajectory();
  ~Trajectory();

  
  //double distance(double x1, double y1, double x2, double y2);

  //int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y);
  //int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y);
  //vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y);

  vector<vector<double>> generate(double car_s, double ref_x, double ref_y, double ref_yaw, double& target_speed, int lane, bool too_close, vector<double> previous_path_x, vector<double> previous_path_y,
    vector<double> map_waypoints_s, vector<double> map_waypoints_x, vector<double> map_waypoints_y);
  
};

