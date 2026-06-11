#include "Euler_1D.h"
#include <stdio.h>

void Test(int flag) {
  state_p left, right;
  switch (flag) {
  // sod problem
  case 0:
    left.rho = 1.0, left.u = 0.0, left.p = 1.0;
    right.rho = 0.125, right.u = 0.0, right.p = 0.1;
    break;
  // shifted problem
  case 1:
    left.rho = 1.0, left.u = 0.75, left.p = 1.0;
    right.rho = 0.125, right.u = 0.0, right.p = 0.1;
    break;
  // 123 problem
  case 2:
    left.rho = 1.4, left.u = 0.0, left.p = 1.0;
    right.rho = 1.0, right.u = 0.0, right.p = 1.0;
    break;
    // left half of the blast wave problem of Woodward and Colella problem
    //   case 3:
    //     left.rho = 1.0, left.u = 0.0, left.p = 1000.0;
    //     right.rho = 1.0, right.u = 0.0, right.p = 0.01;
    //     break;
    //   // right half of the blast wave problem of Woodward and Colella problem
    //   case 4:
    //     left.rho = 1.0, left.u = 0.0, left.p = 0.01;
    //     right.rho = 1.0, right.u = 0.0, right.p = 100.0;
    //     break;
    //   case 5:
    //     left.rho = 5.99924, left.u = 19.5975, left.p = 460.894;
    //     right.rho = 5.99242, right.u = -6.19633, right.p = 46.0950;
    //     break;
  default:
    printf("Error Flag\n");
    break;
  }
  // Sample
  double RegionL = 0, RegionR = 1.0;
  int N = 128;
  double h = (RegionR - RegionL) / N;
  double T[3] = {0.2, 0.2, 2.0};
  double center[3] = {0.5, 0.25, 0.5};
  char name[20];
  sprintf(name, "res%d.csv", flag);
  FILE *fp = fopen(name, "w");
  fprintf(fp, "Location,Density,Velocity,Pressure,Internal Energy\n");
  for (int i = 0; i < N; i++) {
    double x = RegionL + i * h + 0.5 * h;
    // double center = 0.5 * (RegionL + RegionR);
    double S = (x - center[flag]) / T[flag];
    state_p res = Exact_Riemann_Solver(left, right, S);
    fprintf(fp, "%e,%e,%e,%e,%e\n", x, res.rho, res.u, res.p,
            res.p / (gamma - 1.0) / res.rho);
  }
  fclose(fp);
}

int main(int argc, char const *argv[]) {
  for (int i = 0; i <= 2; i++) {
    printf("Testing case %d\n", i);
    Test(i);
  }
  return 0;
}
