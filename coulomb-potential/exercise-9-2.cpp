/**
 * @file exercise-9-2.cpp
 * @author John Davis
 * @brief Numerical Monte Carlo integration of Carbon coulomb interaction energy (sp2, O(n) vs O(n^6))
 * @date 2021-12-24
 */

#include <cmath>
#include <random>
#include <iostream>

const double a = 0.529177; // Bohr radius (Ang)
const double Z = 6.0; // Carbon atomic number
const double e = 1.60218*pow(10.0,-19.0); // elementary charge (C)
const double e0 = 8.85419*pow(10.0,-12.0); // vacuum permittivity
const double pi = M_PI;

const double C = e*pow(10.0,10.0)*pow(4.0*pi*e0,-1.0); // constant, eV * Ang
const double vol = pow(6.0*a,3); // volume

// Radial function, n=2, l=0
inline double R20(double r) {return pow(Z/(2*a),1.5)*(2.0-Z*r/a)*exp(-Z*r/(2*a));}

// Radial function, n=2, l=1
inline double R21(double r) {return pow(Z/(2*a),1.5)*(Z*r/(a*sqrt(3)))*exp(-Z*r/(2*a));}

// s-orbital
inline double s(double theta, double phi) {return 1.0/sqrt(4*pi);}

// px-orbital
inline double px(double theta, double phi) {return sqrt(3.0/(4*pi))*sin(theta)*cos(phi);}

// py-orbital
inline double py(double theta, double phi) {return sqrt(3.0/(4*pi))*sin(theta)*sin(phi);}

// pz-orbital
inline double pz(double theta, double phi) {return sqrt(3.0/(4*pi))*cos(theta);}

// uniformly distributed random number generation 
inline std::uniform_real_distribution<double> dist(-3.0*a,3.0*a); // distribution

double calculateE(long long int iter)
{
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < iter; i++)
    {
        std::random_device seed;
        std::mt19937_64 gen(seed());
        double x0 = dist(gen);
        double y0 = dist(gen);
        double z0 = dist(gen);
        double r0 = sqrt(pow(x0,2.0)+pow(y0,2.0)+pow(z0,2.0));

        double R20_0 = R20(r0);
        double R21_0 = R21(r0);

        if (r0 != 0.0) // neglect divergences
        {
            double theta0 = atan(z0/r0);
            double phi0 = atan2(y0,x0);

            double s0 = s(theta0,phi0);
            double px0 = px(theta0,phi0);
            double py0 = py(theta0,phi0);
            double pz0 = pz(theta0,phi0);

            double a0 = (R20_0*s0 -sqrt(2.0)*R21_0*py0)/sqrt(3.0);
            double b0 = (R20_0*s0 + sqrt(1.5)*R21_0*px0 + sqrt(0.5)*R21_0*py0)/sqrt(3.0);
            double c0 = (-R20_0*s0 - sqrt(1.5)*R21_0*px0 + sqrt(0.5)*R21_0*py0)/sqrt(3.0);
            double d0 = R21_0*pz0;

            sum -= 4.0*pow(a0,2.0)/r0;
            sum -= 4.0*pow(b0,2.0)/r0;
            sum -= 4.0*pow(c0,2.0)/r0;
            sum -= 4.0*pow(d0,2.0)/r0;

            double x1 = dist(gen);
            double y1 = dist(gen);
            double z1 = dist(gen);
            double r1 = sqrt(pow(x1,2.0)+pow(y1,2.0)+pow(z1,2.0));

            double R20_1 = R20(r1);
            double R21_1 = R21(r1);
            
            if (r1 != 0.0) // neglect divergences
            {
                double theta1 = atan(z1/r1);
                double phi1 = atan2(y1,x1);

                double s1 = s(theta1,phi1);
                double px1 = px(theta1,phi1);
                double py1 = py(theta1,phi1);
                double pz1 = pz(theta1,phi1);

                double a1 = (R20_1*s1 -sqrt(2.0)*R21_1*py1)/sqrt(3.0);
                double b1 = (R20_1*s1 + sqrt(1.5)*R21_1*px1 + sqrt(0.5)*R21_1*py1)/sqrt(3.0);
                double c1 = (-R20_1*s1 - sqrt(1.5)*R21_1*px1 + sqrt(0.5)*R21_1*py1)/sqrt(3.0);
                double d1 = R21_1*pz1;

                double sep = sqrt(pow(x1-x0,2.0)+pow(y1-y0,2.0)+pow(z1-z0,2.0));

                if (sep != 0.0) // neglect divergences
                {
                    sum += pow(a0,2.0)*pow(b1,2.0)*(vol/sep);
                    sum += pow(a0,2.0)*pow(c1,2.0)*(vol/sep);
                    sum += pow(a0,2.0)*pow(d1,2.0)*(vol/sep);
                    sum += pow(b0,2.0)*pow(c1,2.0)*(vol/sep);
                    sum += pow(b0,2.0)*pow(d1,2.0)*(vol/sep);
                    sum += pow(c0,2.0)*pow(d1,2.0)*(vol/sep);
                }
            }
        }
    }
    return (sum*C*vol/iter);
}

int main()
{
    for (long long int n = 10; n < 100000000000000; n*=10)
    {
        double E = calculateE(n);
        std::cout << n << ',' << E << std::endl; // output to csv file
    }
    
    return 0;
}
