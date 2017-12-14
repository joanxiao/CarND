#ifndef BEHAVIOR_PLANNER_H
#define BEHAVIOR_PLANNER_H
#endif
#include <vector>
#include <iostream>
#include<sstream>
#include <algorithm>

using namespace std;

class BehaviorPlanner
{
public:
  BehaviorPlanner();
  ~BehaviorPlanner();
  int checkLaneChange(int lane, double car_s, int prev_size, vector<vector<double>> sensor_fusion);
};

