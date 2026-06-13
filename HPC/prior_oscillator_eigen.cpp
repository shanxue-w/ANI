// prior_oscillator_eigen.cpp
// ------------------------------------------------------------
// Compiled prior solver for the linear oscillator (harmonic oscillator):
//     x' = y
//     y' = -x
//
// Uses the analytic flow map over one time step dt:
//     [x_{n+1}]   [ cos(dt)  sin(dt)] [x_n]
//     [y_{n+1}] = [-sin(dt)  cos(dt)] [y_n]
//
// This file is meant to model a legacy HPC prior written in C++/Eigen.
//
// Interface (binary IO):
//   prior_oscillator_eigen <input_bin> <output_bin>
//
// input_bin format (little-endian):
//   int64 N
//   float64 dt
//   then N rows of float64: x, y
//
// output_bin format:
//   int64 N
//   then N rows of float64: x_next, y_next
//
// Build example (adjust EIGEN include path):
//   g++ -O3 -std=c++17 -I /usr/include/eigen3 prior_oscillator_eigen.cpp -o
//   prior_oscillator_eigen
//
// ------------------------------------------------------------
#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool read_all_bytes(const std::string &path, std::vector<char> &buf) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return false;
  f.seekg(0, std::ios::end);
  std::streamsize size = f.tellg();
  f.seekg(0, std::ios::beg);
  if (size <= 0)
    return false;
  buf.resize(static_cast<size_t>(size));
  if (!f.read(buf.data(), size))
    return false;
  return true;
}

static bool write_all_bytes(const std::string &path,
                            const std::vector<char> &buf) {
  std::ofstream f(path, std::ios::binary);
  if (!f)
    return false;
  f.write(buf.data(), static_cast<std::streamsize>(buf.size()));
  return static_cast<bool>(f);
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: prior_oscillator_eigen <input_bin> <output_bin>\n";
    return 1;
  }
  const std::string in_path = argv[1];
  const std::string out_path = argv[2];

  std::vector<char> raw;
  if (!read_all_bytes(in_path, raw)) {
    std::cerr << "Failed to read input file: " << in_path << "\n";
    return 2;
  }
  if (raw.size() < 8 + 8) {
    std::cerr << "Input too small.\n";
    return 3;
  }

  // Parse header
  const int64_t N = *reinterpret_cast<const int64_t *>(raw.data());
  const double dt = *reinterpret_cast<const double *>(raw.data() + 8);

  const size_t expect = 8 + 8 + static_cast<size_t>(N) * 2 * 8;
  if (raw.size() != expect) {
    std::cerr << "Input size mismatch. Got " << raw.size()
              << " bytes, expected " << expect << " bytes.\n";
    return 4;
  }

  const double c = std::cos(dt);
  const double s = std::sin(dt);
  Eigen::Matrix2d R;
  R(0, 0) = c;
  R(0, 1) = s;
  R(1, 0) = -s;
  R(1, 1) = c;

  // Prepare output buffer: int64 N + N*2 doubles
  std::vector<char> out;
  out.resize(8 + static_cast<size_t>(N) * 2 * 8);
  *reinterpret_cast<int64_t *>(out.data()) = N;

  const double *xy = reinterpret_cast<const double *>(raw.data() + 16);
  double *out_xy = reinterpret_cast<double *>(out.data() + 8);

  for (int64_t i = 0; i < N; ++i) {
    Eigen::Vector2d u(xy[2 * i], xy[2 * i + 1]);
    Eigen::Vector2d v = R * u;
    out_xy[2 * i] = v(0);
    out_xy[2 * i + 1] = v(1);
  }

  if (!write_all_bytes(out_path, out)) {
    std::cerr << "Failed to write output file: " << out_path << "\n";
    return 5;
  }
  return 0;
}
