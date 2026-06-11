#define gamma 1.4

// Primitive Variable
typedef struct state
{
    double rho;
    double u;
    double p;
}state_p;

state_p Exact_Riemann_Solver(state_p left, state_p right, double S);