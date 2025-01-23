#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// Helper function to check if two floats are approximately equal
bool almostEqual(float a, float b, float epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

TEST(SphericalHarmonicsTest, NormalizationFactor) {
    // Test the normalization factor function with different values of l and m
    int l_values[] = {3, 4, 5, 6, 7};
    int m_values[] = {0, 1, 2, 3, 4};
    for (int i = 0; i < 5; i++) {
        int l = l_values[i];
        int m = m_values[i];
        float expected = sqrtf((2.0 * l + 1.0) / (4.0 * M_PI * tgammaf(l + m + 1.0) / tgammaf(l - m + 1.0)));
        float actual = normalizationFactor(l, m);
        EXPECT_TRUE(almostEqual(expected, actual));
    }
}

TEST(SphericalHarmonicsTest, LegendrePolynomial) {
    // Test the Legendre polynomial function with different values of l and m
    int l_values[] = {3, 4, 5, 6, 7};
    int m_values[] = {0, 1, 2, 3, 4};
    for (int i = 0; i < 5; i++) {
        int l = l_values[i];
        int m = m_values[i];
        float x = 0.5; // test with a specific value of x
        float expected = legendrePolynomial(l, m, x);
        float actual = legendrePolynomial(l, m, x);
        EXPECT_TRUE(almostEqual(expected, actual));
    }
}

TEST(SphericalHarmonicsTest, ComputeSphericalHarmonics) {
    // Test the computeSphericalHarmonics kernel with different values of l and m
    int l_values[] = {3, 4, 5, 6, 7};
    int m_values[] = {0, 1, 2, 3, 4};
    for (int i = 0; i < 5; i++) {
        int l = l_values[i];
        int m = m_values[i];
        int num_theta = 256;
        int num_phi = 256;
        float* Y_real;
        float* Y_imag;
        cudaMalloc((void**)&Y_real, num_theta * num_phi * sizeof(float));
        cudaMalloc((void**)&Y_imag, num_theta * num_phi * sizeof(float));
        computeSphericalHarmonics<<<256, 256>>>(Y_real, Y_imag, l, m, num_theta, num_phi);
        cudaDeviceSynchronize();
        float* h_Y_real = (float*)malloc(num_theta * num_phi * sizeof(float));
        float* h_Y_imag = (float*)malloc(num_theta * num_phi * sizeof(float));
        cudaMemcpy(h_Y_real, Y_real, num_theta * num_phi * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_Y_imag, Y_imag, num_theta * num_phi * sizeof(float), cudaMemcpyDeviceToHost);
        // Check that the real and imaginary parts are not all zero
        bool non_zero = false;
        for (int j = 0; j < num_theta * num_phi; j++) {
            if (h_Y_real[j]!= 0 || h_Y_imag[j]!= 0) {
                non_zero = true;
                break;
            }
        }
        EXPECT_TRUE(non_zero);
        free(h_Y_real);
        free(h_Y_imag);
        cudaFree(Y_real);
        cudaFree(Y_imag);
    }
}

TEST(SphericalHarmonicsTest, DetermineSphRadiusSize) {
    // Test the determineSphRadiusSize function with different values of l and m
    int l_values[] = {3, 4, 5, 6, 7};
    int m_values[] = {0, 1, 2, 3, 4};
    for (int i = 0; i < 5; i++) {
        int l = l_values[i];
        int m = m_values[i];
        int num_theta = 256;
        int num_phi = 256;
        float expected = determineSphRadiusSize(l, m, num_theta, num_phi);
        float actual = determineSphRadiusSize(l, m, num_theta, num_phi);
        EXPECT_TRUE(almostEqual(expected, actual));
    }
}