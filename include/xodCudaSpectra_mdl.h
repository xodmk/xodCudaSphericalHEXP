/*------------------------------------------------------------------------------------------------*/
/* ___::((xodCudaSpectra_mdl.h))::___

   ___::((created by eschei))___

	Purpose: XODMK C++ Audio Processing - header for SHT Ambisonics audio processing Model

    Requirements:  FFTW - sudo apt-get install libfftw3-dev

	Revision History: 2024-08-27 - initial
*/
/*------------------------------------------------------------------------------------------------*/

#ifndef __XODCUDASPECTRA_MDL_H__
#define __XODCUDASPECTRA_MDL_H__

#include <vector>
#include <complex>
#include <fftw3.h>

// *-----------------------------------------------------------------------------------* //
///// Ptr to 3D Array Implementation /////////////////////

struct AmbisonicSHTData {
    size_t order;                   // Order of the spherical harmonics (m)
    size_t degree;                  // Degree of the spherical harmonics (n)
    size_t channels;                // Number of audio channels (c)
    std::complex<float>* data;      // 3D array of complex coefficients (m x n x c)

    AmbisonicSHTData(size_t order, size_t degree, size_t channels) 
        : order(order), degree(degree), channels(channels), 
          data(new std::complex<float>[order * degree * channels]) {}

    ~AmbisonicSHTData() { delete[] data; }

    // get a reference to the 3D array
    std::complex<float>& operator()(size_t m, size_t n, size_t c) {
        return data[m * degree * channels + n * channels + c];
    }

    // get a reference to the 3D array
    const std::complex<float>& operator()(size_t m, size_t n, size_t c) const {
        return data[m * degree * channels + n * channels + c];
    }
};

class AmbisonicSHTData; // forward declaration


class SphHrmTran {
public:
    SphHrmTran(int lmax, int ntheta, int nphi);
    ~SphHrmTran();

    std::complex<double> spherical_harmonic(int l, int m, double theta, double phi);
    double associated_legendre(int l, int m, double x);
    double legendre_polynomial(int l, double x);

    void transform(const std::vector<double>& input, AmbisonicSHTData& output, int computeYxYyYz = 0);

    float* get_h_Yx() const;
    float* get_h_Yy() const;
    float* get_h_Yz() const;

private:
    int lmax_;
    int ntheta_;
    int nphi_;
    std::complex<double>* coeffs_;
    fftw_plan fftw_plan_;
    float* h_Yx_;
    float* h_Yy_;
    float* h_Yz_;
};

#endif // __XODCUDASPECTRA_MDL_H__


// EXAMPLE main
// #include <iostream>
// #include <vector>
// #include <fftw3.h>
// #include "AmbisonicSHTData.h"
// #include "SHT.h"

// int main() {
//     // Parameters for the SHT
//     int lmax = 3;      // Maximum spherical harmonic order
//     int ntheta = 10;   // Number of points in theta (latitude)
//     int nphi = 20;     // Number of points in phi (longitude)
//     int channels = 5;  // Number of audio channels (frequency bins)

//     // Initialize the SHT object
//     SHT sht(lmax, ntheta, nphi);

//     // Create some input data (this would typically be your spatial audio data)
//     std::vector<double> input(ntheta * nphi, 1.0);  // Example input data filled with 1.0

//     // Create an AmbisonicSHTData object to hold the SHT output
//     AmbisonicSHTData output(lmax + 1, 2 * lmax + 1, channels);

//     // Perform the spherical harmonic transform
//     sht.transform(input, output);

//     // Example: Access and print the transformed data
//     for (size_t m = 0; m <= lmax; ++m) {
//         for (size_t n = 0; n < 2 * lmax + 1; ++n) {
//             for (size_t c = 0; c < channels; ++c) {
//                 std::complex<float> coeff = output(m, n, c);
//                 std::cout << "SHT Coefficient [m=" << m << ", n=" << n << ", c=" << c << "] = " 
//                           << coeff.real() << " + " << coeff.imag() << "i" << std::endl;
//             }
//         }
//     }

//     return 0;
// }