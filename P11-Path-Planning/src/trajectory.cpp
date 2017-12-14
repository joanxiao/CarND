#include "trajectory.h"
#include "spline.h"

Trajectory::Trajectory()
{
}

Trajectory::~Trajectory()
{
}

double distance(double x1, double y1, double x2, double y2)
{
  return sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
}

int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

  double closestLen = 100000; //large number
  int closestWaypoint = 0;

  for (int i = 0; i < maps_x.size(); i++)
  {
    double map_x = maps_x[i];
    double map_y = maps_y[i];
    double dist = distance(x, y, map_x, map_y);
    if (dist < closestLen)
    {
      closestLen = dist;
      closestWaypoint = i;
    }

  }

  return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

  int closestWaypoint = ClosestWaypoint(x, y, maps_x, maps_y);

  double map_x = maps_x[closestWaypoint];
  double map_y = maps_y[closestWaypoint];

  double heading = atan2((map_y - y), (map_x - x));

  double angle = abs(theta - heading);

  if (angle > M_PI / 4)
  {
    closestWaypoint++;
  }

  return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
  int next_wp = NextWaypoint(x, y, theta, maps_x, maps_y);

  int prev_wp;
  prev_wp = next_wp - 1;
  if (next_wp == 0)
  {
    prev_wp = maps_x.size() - 1;
  }

  double n_x = maps_x[next_wp] - maps_x[prev_wp];
  double n_y = maps_y[next_wp] - maps_y[prev_wp];
  double x_x = x - maps_x[prev_wp];
  double x_y = y - maps_y[prev_wp];

  // find the projection of x onto n
  double proj_norm = (x_x*n_x + x_y*n_y) / (n_x*n_x + n_y*n_y);
  double proj_x = proj_norm*n_x;
  double proj_y = proj_norm*n_y;

  double frenet_d = distance(x_x, x_y, proj_x, proj_y);

  //see if d value is positive or negative by comparing it to a center point

  double center_x = 1000 - maps_x[prev_wp];
  double center_y = 2000 - maps_y[prev_wp];
  double centerToPos = distance(center_x, center_y, x_x, x_y);
  double centerToRef = distance(center_x, center_y, proj_x, proj_y);

  if (centerToPos <= centerToRef)
  {
    frenet_d *= -1;
  }

  // calculate s value
  double frenet_s = 0;
  for (int i = 0; i < prev_wp; i++)
  {
    frenet_s += distance(maps_x[i], maps_y[i], maps_x[i + 1], maps_y[i + 1]);
  }

  frenet_s += distance(0, 0, proj_x, proj_y);

  return{ frenet_s,frenet_d };

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
  int prev_wp = -1;

  while ((prev_wp < (int)(maps_s.size() - 1)) && s > maps_s[prev_wp + 1])
  {
    prev_wp++;
  }

  int wp2 = (prev_wp + 1) % maps_x.size();

  double heading = atan2((maps_y[wp2] - maps_y[prev_wp]), (maps_x[wp2] - maps_x[prev_wp]));
  // the x,y,s along the segment
  double seg_s = (s - maps_s[prev_wp]);

  double seg_x = maps_x[prev_wp] + seg_s*cos(heading);
  double seg_y = maps_y[prev_wp] + seg_s*sin(heading);

  double perp_heading = heading - M_PI / 2;

  double x = seg_x + d*cos(perp_heading);
  double y = seg_y + d*sin(perp_heading);

  return{ x,y };

}

vector<vector<double>> Trajectory::generate(double car_s, double ref_x, double ref_y, double ref_yaw, double& target_speed, int lane, bool too_close, vector<double> previous_path_x, vector<double> previous_path_y,
  vector<double> map_waypoints_s, vector<double> map_waypoints_x, vector<double> map_waypoints_y)
{
  int prev_size = previous_path_x.size();

  vector<double> ptsx;
  vector<double> ptsy;
  if (prev_size < 2)
  {
    double prev_car_x = ref_x - cos(ref_yaw);
    double prev_car_y = ref_y - sin(ref_yaw);

    ptsx.push_back(prev_car_x);
    ptsy.push_back(prev_car_y);

    ptsx.push_back(ref_x);
    ptsy.push_back(ref_y);
  }
  else
  {
    ref_x = previous_path_x[prev_size - 1];
    ref_y = previous_path_y[prev_size - 1];

    double ref_x_prev = previous_path_x[prev_size - 2];
    double ref_y_prev = previous_path_y[prev_size - 2];

    ptsx.push_back(ref_x_prev);
    ptsy.push_back(ref_y_prev);

    ptsx.push_back(ref_x);
    ptsy.push_back(ref_y);
  }


  vector<double> next_wp0 = getXY(car_s + 30, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
  vector<double> next_wp1 = getXY(car_s + 60, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
  vector<double> next_wp2 = getXY(car_s + 90, (2 + 4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

  ptsx.push_back(next_wp0[0]);
  ptsy.push_back(next_wp0[1]);

  ptsx.push_back(next_wp1[0]);
  ptsy.push_back(next_wp1[1]);

  ptsx.push_back(next_wp2[0]);
  ptsy.push_back(next_wp2[1]);

  // transfrom to car's coordinate frame
  for (int i = 0; i < ptsx.size(); i++)
  {
    double shift_x = ptsx[i] - ref_x;
    double shift_y = ptsy[i] - ref_y;

    ptsx[i] = shift_x * cos(-ref_yaw) - shift_y * sin(-ref_yaw);
    ptsy[i] = shift_x * sin(-ref_yaw) + shift_y * cos(-ref_yaw);
  }

  //sort_xy(ptsx, ptsy);
  tk::spline spline;
  spline.set_points(ptsx, ptsy);

  vector<double> next_x_vals;
  vector<double> next_y_vals;

  for (int i = 0; i < prev_size; i++)
  {
    next_x_vals.push_back(previous_path_x[i]);
    next_y_vals.push_back(previous_path_y[i]);
  }

  double target_x = 30.0;
  double target_y = spline(target_x);
  double target_distance = distance(target_x, target_y, 0.0, 0.0);
  double x = 0;

  for (int i = 1; i <= 50 - prev_size; i++)
  {
    if (!too_close && (target_speed < 49.5))
      //if (target_speed < 49.5)
    {
      target_speed += .224;
    }

    double N = target_distance / (0.02 * target_speed / 2.24);
    x += (target_x) / N;
    double y = spline(x);

    double global_x = ref_x + x * cos(ref_yaw) - y * sin(ref_yaw);
    double global_y = ref_y + x * sin(ref_yaw) + y * cos(ref_yaw);

    next_x_vals.push_back(global_x);
    next_y_vals.push_back(global_y);

  }
  
  return{ next_x_vals, next_y_vals };
}


/*bool sort_by_x(vector<double> v1, vector<double> v2)
{
  return (v1[0] < v2[0]);
}

void sort_xy(vector<double> & vx, vector<double> & vy)
{
  vector<vector<double>> v;
  int size = vx.size();
  for (int i = 0; i < size; ++i) {
    vector<double> item = { vx[i], vy[i] };
    v.push_back(item);
  }

  sort(v.begin(), v.end(), sort_by_x);

  vx.clear();
  vy.clear();
  for (int i = 0; i < size; ++i) {
    if (i > 0 && v[i][0] == v[i - 1][0]) {
      continue;
    }
    vx.push_back(v[i][0]);
    vy.push_back(v[i][1]);
  }
}
*/
