/*------------------------------------------------------------------------------------------------*/
/* ___::((xodCudaSpectra_mdl.cpp))::___

   ___::((created by eschei))___

	Purpose: XODMK C++ Audio Processing - SHT Ambisonics audio processing Model

	Revision History: 2024-08-27 - initial
*/

/*------------------------------------------------------------------------------------------------*/

#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include <complex>
#include <fftw3.h>

#include "../include/xodCudaSpectra_mdl.h"


SphHrmTran::SphHrmTran(int lmax, int ntheta, int nphi) 
    : lmax_(lmax), ntheta_(ntheta), nphi_(nphi) {
    // Allocate memory for coefficients
    coeffs_ = new std::complex<double>[(lmax_ + 1) * (2 * lmax_ + 1) * ntheta_ * nphi_];

    // Initialize FFTW plans
    fftw_plan_ = fftw_plan_dft_2d(ntheta_, nphi_, reinterpret_cast<fftw_complex*>(coeffs_),
                                  reinterpret_cast<fftw_complex*>(coeffs_), FFTW_FORWARD, FFTW_ESTIMATE);
}

SphHrmTran::~SphHrmTran() {
    delete[] coeffs_;
    fftw_destroy_plan(fftw_plan_);
}


// Spherical harmonic function
std::complex<double> SphHrmTran::spherical_harmonic(int l, int m, double theta, double phi) {
    double P_lm = associated_legendre(l, std::abs(m), std::cos(theta));
    double norm = std::sqrt((2 * l + 1) / (4 * M_PI)) * std::sqrt(tgamma(l - std::abs(m) + 1) / tgamma(l + std::abs(m) + 1));
    std::complex<double> phase = (m < 0)? std::complex<double>(std::cos(m * phi), -std::sin(m * phi)) :
                                           std::complex<double>(std::cos(m * phi), std::sin(m * phi));
    return norm * P_lm * phase;
}

// Associated Legendre polynomial
double SphHrmTran::associated_legendre(int l, int m, double x) {
    if (m == 0) {
        return legendre_polynomial(l, x);
    } else if (m == 1) {
        return -std::sqrt(1 - x * x) * legendre_polynomial(l, 1);
    } else {
        double P_lm_1 = associated_legendre(l, m - 1, x);
        double P_lm_2 = associated_legendre(l, m - 2, x);
        return ((2 * m - 1) * x * P_lm_1 - (l + m - 1) * P_lm_2) / (m - 1);
    }
}

// Legendre polynomial
double SphHrmTran::legendre_polynomial(int l, double x) {
    if (l == 0) {
        return 1.0;
    } else if (l == 1) {
        return x;
    } else {
        double P_l_1 = legendre_polynomial(l - 1, x);
        double P_l_2 = legendre_polynomial(l - 2, x);
        return ((2 * l - 1) * x * P_l_1 - (l - 1) * P_l_2) / l;
    }
}



void SphHrmTran::transform(const std::vector<double>& input, AmbisonicSHTData& output, int computeYxYyYz) {
    // Perform SHT
    for (int l = 0; l <= lmax_; l++) {
        for (int m = -l; m <= l; m++) {
            // Calculate index of coefficient
            int idx = (l * (lmax_ + 1) + (m + l)) * ntheta_ * nphi_;

            // Initialize coefficient to zero
            coeffs_[idx] = 0.0;

            // Perform integration over theta and phi
            for (int i = 0; i < ntheta_; i++) {
                for (int j = 0; j < nphi_; j++) {
                    // Calculate spherical harmonic function
                    std::complex<double> Y_lm = spherical_harmonic(l, m, i * M_PI / ntheta_, j * 2 * M_PI / nphi_);

                    // Accumulate coefficient
                    coeffs_[idx] += input[i * nphi_ + j] * Y_lm;
                }
            }
        }
    }

    // Perform FFT
    fftw_execute(fftw_plan_);

    // Store results in output struct
    for (int l = 0; l <= lmax_; l++) {
        for (int m = -l; m <= l; m++) {
            for (int c = 0; c < output.channels; c++) {
                output(m, l, c) = static_cast<std::complex<float>>(coeffs_[(l * (lmax_ + 1) + (m + l)) * ntheta_ * nphi_]);
            }
        }
    }

    if (computeYxYyYz) {
        // Allocate memory for h_Yx, h_Yy, h_Yz
        h_Yx_ = new float[ntheta_ * nphi_];
        h_Yy_ = new float[ntheta_ * nphi_];
        h_Yz_ = new float[ntheta_ * nphi_];

        // Compute h_Yx, h_Yy, h_Yz
        float radius = 1.0f; // replace with desired radius value
        for (int i = 0; i < ntheta_; i++) {
            for (int j = 0; j < nphi_; j++) {
                int idx = i * nphi_ + j;
                double theta = i * M_PI / ntheta_;
                double phi = j * 2 * M_PI / nphi_;

                double Y_real = std::real(coeffs_[idx]);
                double Y_imag = std::imag(coeffs_[idx]);

                h_Yx_[idx] = radius * Y_real * std::sin(theta) * std::cos(phi);
                h_Yy_[idx] = radius * Y_real * std::sin(theta) * std::sin(phi);
                h_Yz_[idx] = radius * Y_real * std::cos(theta);
            }
        }
    }
}





// // Constructor
// SHT::SHT(int lmax, int ntheta, int nphi) 
//     : lmax_(lmax), ntheta_(ntheta), nphi_(nphi) {
//     // Allocate memory for coefficients
//     coeffs_ = new std::complex<double>[(lmax_ + 1) * (2 * lmax_ + 1) * ntheta_ * nphi_];

//     // Initialize FFTW plans
//     fftw_plan_ = fftw_plan_dft_2d(ntheta_, nphi_, reinterpret_cast<fftw_complex*>(coeffs_),
//                                   reinterpret_cast<fftw_complex*>(coeffs_), FFTW_FORWARD, FFTW_ESTIMATE);
// }

// // Destructor
// SHT::~SHT() {
//     delete[] coeffs_;
//     fftw_destroy_plan(fftw_plan_);
// }

// // Transform method
// void SHT::transform(const std::vector<double>& input, AmbisonicSHTData& output) {
//     // Perform SHT
//     for (int l = 0; l <= lmax_; l++) {
//         for (int m = -l; m <= l; m++) {
//             // Calculate index of coefficient
//             int idx = (l * (lmax_ + 1) + (m + l)) * ntheta_ * nphi_;

//             // Initialize coefficient to zero
//             coeffs_[idx] = 0.0;

//             // Perform integration over theta and phi
//             for (int i = 0; i < ntheta_; i++) {
//                 for (int j = 0; j < nphi_; j++) {
//                     // Calculate spherical harmonic function
//                     std::complex<double> Y_lm = spherical_harmonic(l, m, i * M_PI / ntheta_, j * 2 * M_PI / nphi_);

//                     // Accumulate coefficient
//                     coeffs_[idx] += input[i * nphi_ + j] * Y_lm;
//                 }
//             }
//         }
//     }

//     // Perform FFT
//     fftw_execute(fftw_plan_);

//     // Store results in output struct
//     for (int l = 0; l <= lmax_; l++) {
//         for (int m = -l; m <= l; m++) {
//             for (int c = 0; c < output.channels; c++) {
//                 output(m, l, c) = static_cast<std::complex<float>>(coeffs_[(l * (lmax_ + 1) + (m + l)) * ntheta_ * nphi_]);
//             }
//         }
//     }
// }



// // Spherical harmonic function
// std::complex<double> SHT::spherical_harmonic(int l, int m, double theta, double phi) {
//     double P_lm = associated_legendre(l, std::abs(m), std::cos(theta));
//     double norm = std::sqrt((2 * l + 1) / (4 * M_PI)) * std::sqrt(tgamma(l - std::abs(m) + 1) / tgamma(l + std::abs(m) + 1));
//     std::complex<double> phase = (m < 0)? std::complex<double>(std::cos(m * phi), -std::sin(m * phi)) :
//                                            std::complex<double>(std::cos(m * phi), std::sin(m * phi));
//     return norm * P_lm * phase;
// }

// // Associated Legendre polynomial
// double SHT::associated_legendre(int l, int m, double x) {
//     if (m == 0) {
//         return legendre_polynomial(l, x);
//     } else if (m == 1) {
//         return -std::sqrt(1 - x * x) * legendre_polynomial(l, 1);
//     } else {
//         double P_lm_1 = associated_legendre(l, m - 1, x);
//         double P_lm_2 = associated_legendre(l, m - 2, x);
//         return ((2 * m - 1) * x * P_lm_1 - (l + m - 1) * P_lm_2) / (m - 1);
//     }
// }

// // Legendre polynomial
// double SHT::legendre_polynomial(int l, double x) {
//     if (l == 0) {
//         return 1.0;
//     } else if (l == 1) {
//         return x;
//     } else {
//         double P_l_1 = legendre_polynomial(l - 1, x);
//         double P_l_2 = legendre_polynomial(l - 2, x);
//         return ((2 * l - 1) * x * P_l_1 - (l - 1) * P_l_2) / l;
//     }
// }






// *** USING 'factorial' instead of 'tgamma'

// // Spherical harmonic function
// std::complex<double> SHT::spherical_harmonic(int l, int m, double theta, double phi) {
//     double P_lm = associated_legendre(l, std::abs(m), std::cos(theta));
//     double norm = std::sqrt((2 * l + 1) / (4 * M_PI)) * std::sqrt(factorial(l - std::abs(m)) / factorial(l + std::abs(m)));
//     std::complex<double> phase = (m < 0) ? std::complex<double>(std::cos(m * phi), -std::sin(m * phi)) :
//                                            std::complex<double>(std::cos(m * phi), std::sin(m * phi));
//     return norm * P_lm * phase;
// }

// // Associated Legendre polynomial
// double SHT::associated_legendre(int l, int m, double x) {
//     if (m == 0) {
//         return legendre_polynomial(l, x);
//     } else if (m == 1) {
//         return -std::sqrt(1 - x * x) * legendre_polynomial(l, 1);
//     } else {
//         double P_lm_1 = associated_legendre(l, m - 1, x);
//         double P_lm_2 = associated_legendre(l, m - 2, x);
//         return ((2 * m - 1) * x * P_lm_1 - (l + m - 1) * P_lm_2) / (m - 1);
//     }
// }

// // Legendre polynomial
// double SHT::legendre_polynomial(int l, double x) {
//     if (l == 0) {
//         return 1.0;
//     } else if (l == 1) {
//         return x;
//     } else {
//         double P_l_1 = legendre_polynomial(l - 1, x);
//         double P_l_2 = legendre_polynomial(l - 2, x);
//         return ((2 * l - 1) * x * P_l_1 - (l - 1) * P_l_2) / l;
//     }
// }

// // Factorial function
// double SHT::factorial(int n) {
//     double result = 1.0;
//     for (int i = 2; i <= n; ++i) {
//         result *= i;
//     }
//     return result;
// }