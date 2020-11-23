//to compile: g++ project.cpp -o project
//author: Francesco Tobia, FBK-TeV

#include <stdio.h>
#include <stdlib.h>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <map>

int main(int argc, char** argv)
{
  if (argc != 5) 
    { 
      std::cout << "Use: " << argv[0] << " CALIBRATION.FILE X Y Z" << std::endl; 
      std::cout << "Utility to project a 3D point to the image plane. Camera calibration (pinhole model) is described in CALIBRATION.FILE; (X,Y,Z) is the 3d point; plane coordinates will be printed on std::out" << std::endl; 
      return 0; 
    }

  // 0. data structures

  // intrinsic/internal camera parameters
  double f;       // nominal focal length
  double mu;      // fx = f*mu
  double mv;      // fy = f*mv
  double cx;      // principal point, x coordinate (origin in image upper left corner)
  double cy;      // principal point, y coordinate (origin in image upper left corner)
  double C[3][3]; // camera calibration matrix

  // extrinsic/external camera parameters
  double RT[3][4]; // [R|T] : Euclidean transformation between camera and world coordinates is Xc = R Xw + T 

  // projection matrix: x = PX, where x = (u,v,1), X = (x,y,z,1)
  double P[3][4];  // P = C[R|T] 

  // distortion, Tsai model
  double k1;
  double k2;
  double k3;
  double p1;
  double p2;

  // 1. parse calibration file
  std::ifstream is(argv[1]);
  if (!is.good()) throw std::invalid_argument(std::string("ERROR : can't open ") + argv[1]);

  std::string line;
  std::map<std::string, std::string> m;

  while(is.good())
    {
      getline(is, line);
      std::string::size_type pos = line.find('=');
      if (pos == std::string::npos) continue;

      std::string label = line.substr(0, pos);
      std::string value = line.substr(pos+1, std::string::npos);

      m[label] = value;
    }

  std::map<std::string, std::string>::iterator end = m.end();

  //

  if (m.find("f") == end || m.find("mu") == end || m.find("mv") == end || m.find("u0") == end || m.find("v0") == end) 
    throw std::invalid_argument("ERROR : incomplete intrinsic");

  f = atof(m["f"].c_str());
  mu = atof(m["mu"].c_str());
  mv = atof(m["mv"].c_str());
  cx = atof(m["u0"].c_str());
  cy = atof(m["v0"].c_str());

  C[0][0] = -f * mu;
  C[1][1] = -f * mv;
  C[2][2] = 1;
  C[0][2] = cx;
  C[1][2] = cy;

  //

  if (m.find("R11") == end || m.find("R12") == end || m.find("R13") == end ||
      m.find("R21") == end || m.find("R22") == end || m.find("R23") == end ||
      m.find("R31") == end || m.find("R32") == end || m.find("R33") == end ||
      m.find("T1") == end || m.find("T2") == end || m.find("T3") == end) 
    throw std::invalid_argument("ERROR : incomplete extrinsic");

  double R[3][3];
  double t[3];

  RT[0][0] = R[0][0] = atof(m["R11"].c_str());
  RT[0][1] = R[0][1] = atof(m["R12"].c_str());
  RT[0][2] = R[0][2] = atof(m["R13"].c_str());
  RT[1][0] = R[1][0] = atof(m["R21"].c_str());
  RT[1][1] = R[1][1] = atof(m["R22"].c_str());
  RT[1][2] = R[1][2] = atof(m["R23"].c_str());
  RT[2][0] = R[2][0] = atof(m["R31"].c_str());
  RT[2][1] = R[2][1] = atof(m["R32"].c_str());
  RT[2][2] = R[2][2] = atof(m["R33"].c_str());

  t[0] = atof(m["T1"].c_str());
  t[1] = atof(m["T2"].c_str());
  t[2] = atof(m["T3"].c_str());

  for (int i = 0; i < 3; ++i) // T = -R * t
    {
      RT[i][3] = 0;
      for (int j = 0; j < 3; ++j)
	RT[i][3] += -R[i][j] * t[j]; 
    }

  //

  if (m.find("k1") == end || m.find("k2") == end || m.find("k3") == end || m.find("p1") == end || m.find("p2") == end)
    throw std::invalid_argument("ERROR : incomplete distortion");

  k1 = atof(m["k1"].c_str()); 
  k2 = atof(m["k2"].c_str()); 
  k3 = atof(m["k3"].c_str()); 
  p1 = atof(m["p1"].c_str()); 
  p2 = atof(m["p2"].c_str()); 

  // 2. projection

  for (int i = 0; i < 3; ++i) // P = C * RT 
    for (int j = 0; j < 4; ++j)
      {
	P[i][j] = 0;
	for (int k = 0; k < 3; ++k)
	  P[i][j] += C[i][k] * RT[k][j];
      }

  double v[3];
  double V[4];
  V[0] = atof(argv[2]);
  V[1] = atof(argv[3]);
  V[2] = atof(argv[4]);
  V[3] = 1;

  for (int i = 0; i < 3; ++i) // v = P * V
    {
      v[i] = 0;
      for (int j = 0; j < 4; ++j) v[i] += P[i][j] * V[j];
    }
  
  for (int i = 0; i < 3; ++i)
    v[i] /= v[2];

  // 3. distortion (Tsai)

  double fx = f * mu;
  double fy = f * mv;

  double dx = (v[0] - cx) / fx;
  double dy = (v[1] - cy) / fy;
  double dx2 = dx*dx;
  double dy2 = dy*dy;
  double dxdy = dx*dy;
  double r2 = dx2 + dy2;

  double x = dx * (1 + r2 * (k1 + r2 * (k2 + r2 * k3)));
  double y = dy * (1 + r2 * (k1 + r2 * (k2 + r2 * k3)));

  x += 2*p1 * dxdy + p2 * (r2 + 2*dx2); 
  y += 2*p2 * dxdy + p1 * (r2 + 2*dy2);

  double d[2];
  d[0] = cx + (fx * x);
  d[1] = cy + (fy * y);

  // 4. output

  std::cout << "(x,y) = " << d[0] << "," << d[1] << std::endl;
}
