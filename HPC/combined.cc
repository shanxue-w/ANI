// combined.cc
// ------------------------------------------------------------
// Rollout + per-step L2 loss for two inference modes:
//
//   (A) --mode ani
//       uses TorchScript full model: ani_strang_full.ts
//       forward: [1,2] -> [1,2]  (dt_flow baked inside module)
//
//   (B) --mode lie
//       uses TorchScript corrector: lie_corrector_phi.ts
//       step: u_prior = S_dt(u) computed in C++ (harmonic oscillator analytic)
//             u_next  = Phi(u_prior)
//
// Reference trajectory:
//   Van der Pol (mu=1):
//     x' = y
//     y' = mu*(1-x^2)*y - x
//   integrated by RK4 with dt_ref (default 1e-3)
//   sampled every dt_flow (default 1e-1)
//
// Output CSV columns:
//   k, t, l2, pred_x, pred_y, ref_x, ref_y
//
// Build (example):
//   g++ -O3 -std=c++17 combined.cc -o combined \
//      -I /usr/include/eigen3 \
//      -I ${LIBTORCH}/include -I ${LIBTORCH}/include/torch/csrc/api/include \
//      -L ${LIBTORCH}/lib -ltorch_cpu -ltorch -lc10 \
//      -Wl,-rpath,${LIBTORCH}/lib
//
// Run examples:
//   ./combined --mode ani --ts artifacts/ani_strang_full.ts --out loss_ani.csv
//   --steps 200
//   ./combined --mode lie --ts artifacts/lie_corrector_phi.ts --out
//   loss_lie.csv --steps 200
// ------------------------------------------------------------
#include <torch/script.h>
#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Args {
  std::string mode = "ani"; // ani | lie
  std::string ts_path = "";
  std::string out_csv = "loss.csv";
  int steps = 200;
  double dt_flow = 1e-1;
  double dt_ref = 1e-3;
  double mu = 1.0;
  bool random_init = true;
  double x0 = 1.0;
  double y0 = 0.0;
  int seed = 0;
};

static void usage() {
  std::cerr
      << "Usage:\n"
         "  ./combined --mode ani|lie --ts <model.ts> --out <loss.csv> "
         "[options]\n"
         "Options:\n"
         "  --steps <int>        number of flow steps (default 200)\n"
         "  --dt_flow <float>    flow-map step (default 1e-1)\n"
         "  --dt_ref <float>     reference RK4 step (default 1e-3)\n"
         "  --mu <float>         Van der Pol mu (default 1)\n"
         "  --x0 <float> --y0 <float>  initial state if --no_random_init\n"
         "  --seed <int>         RNG seed (default 0)\n"
         "  --no_random_init     use provided x0,y0 instead of random init\n";
}

static bool parse_args(int argc, char **argv, Args &a) {
  for (int i = 1; i < argc; ++i) {
    std::string s(argv[i]);
    auto need = [&](const char *key) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << key << "\n";
        return "";
      }
      return std::string(argv[++i]);
    };

    if (s == "--mode")
      a.mode = need("--mode");
    else if (s == "--ts")
      a.ts_path = need("--ts");
    else if (s == "--out")
      a.out_csv = need("--out");
    else if (s == "--steps")
      a.steps = std::stoi(need("--steps"));
    else if (s == "--dt_flow")
      a.dt_flow = std::stod(need("--dt_flow"));
    else if (s == "--dt_ref")
      a.dt_ref = std::stod(need("--dt_ref"));
    else if (s == "--mu")
      a.mu = std::stod(need("--mu"));
    else if (s == "--x0")
      a.x0 = std::stod(need("--x0"));
    else if (s == "--y0")
      a.y0 = std::stod(need("--y0"));
    else if (s == "--seed")
      a.seed = std::stoi(need("--seed"));
    else if (s == "--no_random_init")
      a.random_init = false;
    else if (s == "--help" || s == "-h") {
      usage();
      return false;
    } else {
      std::cerr << "Unknown arg: " << s << "\n";
      usage();
      return false;
    }
  }
  if (a.ts_path.empty()) {
    std::cerr << "Error: --ts is required.\n";
    usage();
    return false;
  }
  if (!(a.mode == "ani" || a.mode == "lie")) {
    std::cerr << "Error: --mode must be ani or lie.\n";
    return false;
  }
  if (a.steps <= 0 || a.dt_flow <= 0 || a.dt_ref <= 0) {
    std::cerr << "Error: steps/dt_flow/dt_ref must be positive.\n";
    return false;
  }
  const double ratio = a.dt_flow / a.dt_ref;
  const double r_rounded = std::round(ratio);
  if (std::fabs(ratio - r_rounded) > 1e-12) {
    std::cerr << "Error: dt_flow must be an integer multiple of dt_ref.\n";
    return false;
  }
  return true;
}

// -------- Reference dynamics: Van der Pol --------
static inline void rhs_vdp(double x, double y, double mu, double &dx,
                           double &dy) {
  dx = y;
  dy = mu * (1.0 - x * x) * y - x;
}

static inline void rk4_step_vdp(double &x, double &y, double dt, double mu) {
  double k1x, k1y;
  rhs_vdp(x, y, mu, k1x, k1y);

  double x2 = x + 0.5 * dt * k1x;
  double y2 = y + 0.5 * dt * k1y;
  double k2x, k2y;
  rhs_vdp(x2, y2, mu, k2x, k2y);

  double x3 = x + 0.5 * dt * k2x;
  double y3 = y + 0.5 * dt * k2y;
  double k3x, k3y;
  rhs_vdp(x3, y3, mu, k3x, k3y);

  double x4 = x + dt * k3x;
  double y4 = y + dt * k3y;
  double k4x, k4y;
  rhs_vdp(x4, y4, mu, k4x, k4y);

  x += (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
  y += (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y);
}

static inline void ref_step_flow(double &x, double &y, double dt_flow,
                                 double dt_ref, double mu) {
  const int n = static_cast<int>(std::llround(dt_flow / dt_ref));
  for (int i = 0; i < n; ++i)
    rk4_step_vdp(x, y, dt_ref, mu);
}

// -------- Prior step for Lie mode: harmonic oscillator analytic map --------
static inline void prior_step_ho(double &x, double &y, double dt) {
  const double c = std::cos(dt);
  const double s = std::sin(dt);
  const double xn = x * c + y * s;
  const double yn = -x * s + y * c;
  x = xn;
  y = yn;
}

// -------- TorchScript helpers --------
static inline torch::Tensor make_input_tensor(double x, double y) {
  // CPU float64 tensor shape [1,2]
  auto t = torch::empty(
      {1, 2},
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
  t[0][0] = x;
  t[0][1] = y;
  return t;
}

static inline void tensor_to_state(const torch::Tensor &out, double &x,
                                   double &y) {
  // out: [1,2] float64
  auto o = out.to(torch::kCPU);
  x = o[0][0].item<double>();
  y = o[0][1].item<double>();
}

static inline double l2(double x1, double y1, double x2, double y2) {
  const double dx = x1 - x2;
  const double dy = y1 - y2;
  return std::sqrt(dx * dx + dy * dy);
}

int main(int argc, char **argv) {
  Args a;
  if (!parse_args(argc, argv, a))
    return 1;

  // Initial condition
  double x_pred, y_pred;
  double x_ref, y_ref;
  if (a.random_init) {
    std::mt19937 gen(a.seed);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    x_pred = dist(gen);
    y_pred = dist(gen);
  } else {
    x_pred = a.x0;
    y_pred = a.y0;
  }
  x_ref = x_pred;
  y_ref = y_pred;

  // Load TorchScript
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(a.ts_path);
  } catch (const c10::Error &e) {
    std::cerr << "Error loading TorchScript: " << e.what() << "\n";
    return 2;
  }
  model.eval();
  torch::NoGradGuard no_grad;

  std::ofstream ofs(a.out_csv);
  if (!ofs) {
    std::cerr << "Failed to open output file: " << a.out_csv << "\n";
    return 3;
  }
  ofs << "k,t,l2,pred_x,pred_y,ref_x,ref_y\n";

  double t = 0.0;

  // Write step 0 (loss=0 by definition)
  ofs << 0 << "," << t << "," << 0.0 << "," << x_pred << "," << y_pred << ","
      << x_ref << "," << y_ref << "\n";

  for (int k = 1; k <= a.steps; ++k) {
    // 1) Reference: advance by dt_flow using dt_ref RK4
    ref_step_flow(x_ref, y_ref, a.dt_flow, a.dt_ref, a.mu);

    // 2) Prediction: one flow step
    double x_next = x_pred, y_next = y_pred;

    if (a.mode == "ani") {
      // model: u_next = ANI(u)
      double xp = x_pred, yp = y_pred;
      prior_step_ho(xp, yp, a.dt_flow / 2.0);
      torch::Tensor inp = make_input_tensor(xp, yp);
      double dt = a.dt_flow;
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(inp);
      inputs.push_back(dt);
      torch::Tensor out = model.forward(inputs).toTensor();
      tensor_to_state(out, x_next, y_next);
      prior_step_ho(x_next, y_next, a.dt_flow / 2.0);
    } else {
      // lie: u_prior = S_dt(u), u_next = Phi(u_prior)
      double xp = x_pred, yp = y_pred;
      prior_step_ho(xp, yp, a.dt_flow);

      torch::Tensor inp = make_input_tensor(xp, yp);
      double dt = a.dt_flow;
      std::vector<torch::jit::IValue> inputs;
      inputs.push_back(inp);
      inputs.push_back(dt);
      torch::Tensor out = model.forward(inputs).toTensor();
      tensor_to_state(out, x_next, y_next);
    }

    x_pred = x_next;
    y_pred = y_next;

    t += a.dt_flow;
    const double e = l2(x_pred, y_pred, x_ref, y_ref);

    ofs << k << "," << t << "," << e << "," << x_pred << "," << y_pred << ","
        << x_ref << "," << y_ref << "\n";

    if (k % 50 == 0 || k == 1) {
      std::cout << "step " << k << "  t=" << t << "  l2=" << e << "\n";
    }
  }

  std::cout << "Wrote: " << a.out_csv << "\n";
  return 0;
}
