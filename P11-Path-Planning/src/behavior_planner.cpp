#include "behavior_planner.h"

BehaviorPlanner::BehaviorPlanner()
{
}


BehaviorPlanner::~BehaviorPlanner()
{
}

// return lane number given d 
int getLane(double car_d) {
  int lane = 1; 

  if (car_d > 0 && car_d < 4) {
    lane = 0;   //left
  }
  else if (car_d >= 4 && car_d <= 8) {
    lane = 1;  //center 
  }
  else if (car_d > 8 && car_d <= 12) {
    lane = 2;   //right
  }
  return lane;
}

// return lane number for target state given current lane
int getTargetLaneForState(int lane, string state)
{
  if (state == "LCL")
  {
    lane -= 1;
  }
  else if (state == "LCR")
  {
    lane += 1;
  }

  return lane;
}

// returns a left or right lane that is safe to change to
int getTargetLane(int lane, double car_s, int prev_size, vector<string> states, vector<vector<double>> sensor_fusion)
{
  for (int i = 0; i < states.size(); i++)
  {
    string state = states[i];
    int target_lane = getTargetLaneForState(lane, state);

    double front_gap = 1e9;
    double back_gap = 1e9;

    for (int j = 0; j < sensor_fusion.size(); j++)
    {
      float other_d = sensor_fusion[j][6];
      int other_lane = getLane(other_d);

      if (other_lane == target_lane)
      {
        double other_vx = sensor_fusion[j][3];
        double other_vy = sensor_fusion[j][4];
        double other_speed = sqrt(other_vx * other_vx + other_vy * other_vy);
        double other_s = sensor_fusion[j][5];

        other_s += (double)prev_size * 0.02 * other_speed;

        if (other_s > car_s) { front_gap = min(other_s - car_s, front_gap); }
        else if (other_s < car_s) { back_gap = min(car_s - other_s, back_gap); }
      }
    }

    // change lane if there is enough gap in the front and back
    if (front_gap > 35 && back_gap > 20) {
      return target_lane;
    }
  }

  // keep lane
  return lane;
}

// decides whether to change lane and to which lane, or slow down
int BehaviorPlanner::checkLaneChange(int lane, double car_s, int prev_size, vector<vector<double>> sensor_fusion)
{
  bool too_close = false;
  bool change_lane = false;
  int target_lane;

  for (int i = 0; i < sensor_fusion.size(); i++)
  {
    float other_d = sensor_fusion[i][6];
    int other_lane = getLane(other_d);

    if (other_lane == lane)
    {
      double other_vx = sensor_fusion[i][3];
      double other_vy = sensor_fusion[i][4];
      double other_speed = sqrt(other_vx * other_vx + other_vy * other_vy);
      double other_s = sensor_fusion[i][5];

      other_s += (double)prev_size * 0.02 * other_speed;

      if ((other_s > car_s) && ((other_s - car_s) < 30))
      {
        too_close = true;

        // decide which lane to change
        vector <string> states;
        //states.push_back("KL");

        if (lane == 0)
        {
          states.push_back("LCR");
        }
        else if (lane == 2)
        {
          states.push_back("LCL");
        }
        else
        {
          states.push_back("LCL");
          states.push_back("LCR");
        }

        target_lane = getTargetLane(lane, car_s, prev_size, states, sensor_fusion);

        if (target_lane != lane) {            
          change_lane = true;
          cout << "change lane from " << lane << " to " << target_lane << endl;
          break;
        }
      }

    }
  }

  if (change_lane) { return target_lane; } // change lane
  else if (too_close) { return -1; } // keep lane, should slow down
  else { return lane; } // keep lane and speed  
}