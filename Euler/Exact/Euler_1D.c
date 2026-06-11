#include <math.h>
#include "Euler_1D.h"

double fK(state_p K, double p)
{
    // constant
    double    rho =K.rho;
    double    u   =K.u;
    double    pK  =K.p;
    double    a   =sqrt(gamma*pK/rho);
    double    A   =2.0/(gamma+1.0)/rho;
    double    B   =(gamma-1.0)/(gamma+1.0)*pK;
    if(p > pK){
        return (p-pK)*sqrt(A/(p+B));  // Left Shock
    }
    else{
        double tmp=(gamma-1.0)/2.0/gamma;
        return 2.0*a/(gamma-1.0)*(pow(p/pK,tmp)-1.0); // Left Rarefaction
    }
}

double dfK(state_p K, double p)
{
    // constant
    double    rho =K.rho;
    double    u   =K.u;
    double    pK  =K.p;
    double    a   =sqrt(gamma*pK/rho);
    double    A   =2.0/(gamma+1.0)/rho;
    double    B   =(gamma-1.0)/(gamma+1.0)*pK;
    if(p>pK){
        return sqrt(A/(B+p))*(1.0-0.5*(p-pK)/(B+p));  // Left Shock
    }
    else{
        double tmp=-(gamma+1.0)/2.0/gamma;
        return 1.0/rho/a*pow(p/pK,tmp); // Left Rarefaction
    }
}

// p* is the foot of f(p)
double f(state_p left, state_p right, double p)
{
    return fK(left,p)+fK(right,p)+right.u-left.u;
}

// f(p)'s derivative
double df(state_p left, state_p right, double p)
{
    return dfK(left,p)+dfK(right,p);
}

// using Newton Iteration to find p*
double p_star_solver(state_p left, state_p right)
{
    double p0=0.5*(left.p+right.p); //Initial guess
    double p;
    double epsilon=1e-6, change_radio;
    do{
        p=p0-f(left,right,p0)/df(left,right,p0);
        p=fmax(p,epsilon); // Handle negative p
        change_radio=2.0*fabs(p-p0)/(p+p0);
        p0=p;
    }while(change_radio>epsilon);
    return p;
}

state_p Exact_Riemann_Solver(state_p left, state_p right, double S)
{
    // constant
    double    rho_l =left.rho,             rho_r =right.rho;
    double    ul    =left.u,               ur    =right.u;
    double    pl    =left.p,               pr    =right.p;
    double    al    =sqrt(gamma*pl/rho_l), ar    =sqrt(gamma*pr/rho_r);
    state_p   res;
    //p* and u*
    double p_star=p_star_solver(left,right);
    double u_star=0.5*(left.u+right.u)+0.5*(fK(right,p_star)-fK(left,p_star));
    if(S<u_star) // Left
    {
        if(p_star > pl) // Left Shock Wave
        {
            double tmp=(gamma-1.0)/(gamma+1.0);
            double rho_star_l=rho_l*(p_star/pl+tmp)/(tmp*p_star/pl+1);
            double Sl=ul-al*sqrt(0.5*(gamma+1)/gamma*p_star/pl+0.5*(gamma-1)/gamma);
            if(S>Sl)
            {
                res.p=p_star, res.u=u_star, res.rho=rho_star_l;
                return res;
            }
            else{
                return left;
            }
        }
        else           // Left Rarefaction
        {
            double tmp=p_star/pl;
            double rho_star_l=rho_l*pow(tmp,1.0/gamma);
            double   a_star_l=   al*pow(tmp,0.5*(gamma-1.0)/gamma);
            double Sh=ul-al, St=u_star-a_star_l;
            if(S<Sh)
            {
                return left;
            }
            if(S>St)
            {
                res.p=p_star, res.u=u_star, res.rho=rho_star_l;
                return res;
            }
            // In the Rarefaction Wave
            tmp=(gamma-1.0)/(gamma+1.0);
            res.rho =rho_l*pow(2.0/(gamma+1.0)+tmp/al*(ul-S),2.0/(gamma-1.0));
            res.u   =2.0/(gamma+1.0)*(al+0.5*(gamma-1)*ul+S);
            res.p   =pl*pow(2.0/(gamma+1.0)+tmp/al*(ul-S),2*gamma/(gamma-1.0));
            return res;
        }
    }
    else // Right
    {
        if(p_star>pr) // Right Shock Wave
        {
            double tmp=(gamma-1.0)/(gamma+1.0);
            double rho_star_r=rho_r*(p_star/pr+tmp)/(tmp*p_star/pr+1.0);
            double Sr=ur+ar*sqrt(0.5*(gamma+1.0)/gamma*p_star/pr+0.5*(gamma-1.0)/gamma);
            if(S<Sr)
            {
                res.p=p_star, res.u=u_star, res.rho=rho_star_r;
                return res;
            }
            else{
                return right;
            }
        }
        else
        {
            double tmp=p_star/pr;
            double rho_star_r=rho_r*pow(tmp,1.0/gamma);
            double   a_star_r=   ar*pow(tmp,0.5*(gamma-1.0)/gamma);
            double Sh=ur+ar, St=u_star+a_star_r;
            if(S>Sh)
            {
                return right;
            }
            if(S<St)
            {
                res.p=p_star, res.u=u_star, res.rho=rho_star_r;
                return res;
            }
            // In the Rarefaction Wave
            tmp=(gamma-1.0)/(gamma+1.0);
            res.rho =rho_r*pow(2.0/(gamma+1.0)-tmp/ar*(ur-S),2.0/(gamma-1.0));
            res.u   =2.0/(gamma+1.0)*(-ar+0.5*(gamma-1)*ur+S);
            res.p   =pr*pow(2.0/(gamma+1.0)-tmp/ar*(ur-S),2*gamma/(gamma-1.0));
            return res;
        }
    }
}
