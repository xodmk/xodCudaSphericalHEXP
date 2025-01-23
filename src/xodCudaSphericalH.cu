#include <cstdint>
#include <iostream>
#include <cassert>
#include <cmath>
#include <complex>
#include <float.h>

# include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>

#include "../include/xodImg_types.h"
#include "../include/xodCudaUtil.h"
#include "../include/xodCudaSphericalH.h"



// Define the number of theta and phi steps
const int num_theta = 256;   // ~300;
const int num_phi   = 256;   // ~300;

extern uint8_t* imgSrcDataCH1_ptr;


// Define sets of l and m values
int l = 7;  // default
int m = 3;  // default
int current_set = 3;

float sphRadiusSize = 1.6;  

//int l_values[] = {3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 11, 11, 13};
//int m_values[] = {3, 4, 3, 5, 4, 5, 3, 4, 1, 3, 2, 3,  1,  4,  3};
//int num_sets = 15;


int l_values[] = {3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4,
                  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6,
                  6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                  7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9,
                  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10,
                  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11,
                  11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
                  11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                  12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13,
                  13, 13, 13, 13, 13, 13};

int m_values[] = {-3, -2, -1, 0, 1, 2, 3, -4, -3, -2, -1, 0, 1,
                  2, 3, 4, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, -6, -5,
                  -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, -7, -6, -5, -4,
                  -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4,
                  -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, -9, -8, -7, -6,
                  -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -10,
                  -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5,
                  6, 7, 8, 9, 10, -11, -10, -9, -8, -7, -6, -5, -4, -3,
                  -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -12, -11,
                  -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4,
                  5, 6, 7, 8, 9, 10, 11, 12, -13, -12, -11, -10, -9, -8,
                  -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  9, 10, 11, 12, 13};

int num_sets = 142;


bool autoRotate = false;        // track the auto rotate state
// track the rotation speed and direction
float rotationSpeed = 1.0f;
float rotationX = 0.0f;
float rotationY = 0.0f;
float rotationZ = 0.0f;

float view_angle_x = 0.0f; 
float view_angle_y = 0.0f;
float view_angle_z = 0.0f;

Colormap currentColormap = JET;

bool displayBackgroundImage = true;
bool displayColorMap = true;

// Pointers for device memory
float *d_Y_real, *d_Y_imag;
float *h_Y_real, *h_Y_imag;
float *h_Yx, *h_Yy, *h_Yz;

float min_h_Y_real;
float max_h_Y_real;

// Function prototypes
void display();
void initCuda();
void cleanupCuda();
void mapToColor(float value, Colormap cmap, float &r, float &g, float &b);
void mapToImage(float Yx, float Yy, float Yz, float &r, float &g, float &b);
//void mapToColor(float value, float min, float max, Colormap cmap,
//                float &r, float &g, float &b);
void getColorFromBackgroundImage(float u, float v, float &r, float &g, float &b);
void reshape(int w, int h);
void keyboard(unsigned char key, int x, int y);
void specialKeys(int key, int x, int y);



// Helper function to compute the normalization factor
__device__ float normalizationFactor(int l, int m) {
    float num = (2.0 * l + 1.0) * tgammaf(l - m + 1.0);
    float den = 4.0 * M_PI * tgammaf(l + m + 1.0);
    return sqrtf(num / den);
}

// Helper function to compute the associated Legendre polynomials
__device__ float legendrePolynomial(int l, int m, float x) {
    float pmm = 1.0;
    if (m > 0) {
        float sign = (m % 2 == 0) ? 1.0 : -1.0;
        pmm = sign * tgammaf(2.0 * m - 1.0) * powf(1.0 - x * x, 0.5 * m);
    }
    if (l == m) return pmm;

    float pmmp1 = x * (2.0 * m + 1.0) * pmm;
    if (l == m + 1) return pmmp1;

    float pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

__global__ void computeSphericalHarmonics(float* Y_real, float* Y_imag, int l, int m, int num_theta, int num_phi) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_theta * num_phi) return;

    int i = tid / num_phi;
    int j = tid % num_phi;

    float theta = M_PI * i / (num_theta - 1);
    float phi = 2 * M_PI * j / (num_phi - 1);

    float N = normalizationFactor(l, m);
    float P = legendrePolynomial(l, abs(m), cosf(theta));
    float phase = m * phi;

    float real_part = N * P * cosf(phase);
    float imag_part = N * P * sinf(phase);

    Y_real[tid] = real_part;
    Y_imag[tid] = imag_part;
}


// Initialize CUDA resources
void initCuda(float radius) {
    size_t size = num_theta * num_phi * sizeof(float);
    cudaMalloc((void**)&d_Y_real, size);
    cudaMalloc((void**)&d_Y_imag, size);

    //float sphRadiusSize = 1.6f;
    //float sphRadiusSize = 0.3f;
    int blockSize = 256;
    int numBlocks = (num_theta * num_phi + blockSize - 1) / blockSize;
    computeSphericalHarmonics<<<numBlocks, blockSize>>>(d_Y_real, d_Y_imag, l, m, num_theta, num_phi);

    h_Y_real = (float*)malloc(size);
    h_Y_imag = (float*)malloc(size);
    cudaMemcpy(h_Y_real, d_Y_real, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y_imag, d_Y_imag, size, cudaMemcpyDeviceToHost);

    h_Yx = (float*)malloc(size);
    h_Yy = (float*)malloc(size);
    h_Yz = (float*)malloc(size);

    min_h_Y_real = FLT_MAX;
    max_h_Y_real = FLT_MIN;

    for (int i = 0; i < num_theta; ++i) {
        for (int j = 0; j < num_phi; ++j) {
            int idx = i * num_phi + j;
            float theta = M_PI * i / (num_theta - 1);
            float phi = 2 * M_PI * j / (num_phi - 1);

            float r = h_Y_real[idx]; // Use the value of the spherical harmonic as the radius

            // Update min and max values
            min_h_Y_real = std::min(min_h_Y_real, r);
            max_h_Y_real = std::max(max_h_Y_real, r);

            h_Yx[idx] = radius * r * sin(theta) * cos(phi);
            h_Yy[idx] = radius * r * sin(theta) * sin(phi);
            h_Yz[idx] = radius * r * cos(theta);
        }
    }
}


__global__ void calculateMaxExtentKernel(int l, int m, int num_theta, int num_phi, int* max_extent) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = num_theta * num_phi;

    if (idx < total_points) {
        int i = idx / num_phi;
        int j = idx % num_phi;

        float theta = M_PI * i / (num_theta - 1);
        float phi = 2 * M_PI * j / (num_phi - 1);

        float r = sqrtf(powf(normalizationFactor(l, m) * legendrePolynomial(l, abs(m), cosf(theta)) * cosf(m * phi), 2) +
                        powf(normalizationFactor(l, m) * legendrePolynomial(l, abs(m), cosf(theta)) * sinf(m * phi), 2));

        float x = r * sinf(theta) * cosf(phi);
        float y = r * sinf(theta) * sinf(phi);
        float z = r * cosf(theta);

        int x_int = (int)(fabsf(x) * 1000.0f); // scale the value to an integer
        int y_int = (int)(fabsf(y) * 1000.0f);
        int z_int = (int)(fabsf(z) * 1000.0f);

        atomicMax(max_extent, x_int);
        atomicMax(max_extent, y_int);
        atomicMax(max_extent, z_int);
    }
}

// Host function to launch the kernel and determine the sphRadiusSize
float determineSphRadiusSize(int l, int m, int num_theta, int num_phi) {
    int* d_max_extent;
    int h_max_extent = 0;

    cudaMalloc(&d_max_extent, sizeof(int));
    cudaMemcpy(d_max_extent, &h_max_extent, sizeof(int), cudaMemcpyHostToDevice);

    int total_points = num_theta * num_phi;
    int blockSize = 256;
    int numBlocks = (total_points + blockSize - 1) / blockSize;

    calculateMaxExtentKernel<<<numBlocks, blockSize>>>(l, m, num_theta, num_phi, d_max_extent);

    cudaMemcpy(&h_max_extent, d_max_extent, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_max_extent);

    float display_window_radius = 1.0f; // Assuming the display window ranges from -1 to 1 in all dimensions
    float sphRadiusSize = display_window_radius / (h_max_extent / 1000.0f); // scale the value back to a float
    return sphRadiusSize;
}



// The glDisable(GL_DEPTH_TEST) and glEnable(GL_DEPTH_TEST) calls are used
// to prevent the background image from interfering with the depth testing of the 3D scene.
void renderSphericalHarmonics() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPushMatrix();

    // Apply view rotation
    glRotatef(view_angle_x, 1.0f, 0.0f, 0.0f);
    glRotatef(view_angle_y, 0.0f, 1.0f, 0.0f);

    if (displayBackgroundImage) {

        // Draw the background image
        glDisable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT));
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glRasterPos2f(0, 0);
        glDrawPixels(IMAGE_WIDTH, IMAGE_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, imgSrcDataCH1_ptr);
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        glEnable(GL_DEPTH_TEST);

        glLineWidth(1.0f); // Set the line width
        glBegin(GL_LINES);
        for (int i = 0; i < num_theta - 1; ++i) {
            for (int j = 0; j < num_phi - 1; ++j) {
                int idx = i * num_phi + j;
                float r, g, b;
                if (displayColorMap) {
                    mapToColor(h_Y_real[idx], currentColormap, r, g, b);
                } else {
                    mapToImage(h_Yx[idx], h_Yy[idx], h_Yz[idx], r, g, b);
                }
                glColor3f(r, g, b);
                glVertex3f(h_Yx[idx], h_Yy[idx], h_Yz[idx]);
                if (j < num_phi - 2) {
                    glVertex3f(h_Yx[i * num_phi + j + 1], h_Yy[i * num_phi + j + 1], h_Yz[i * num_phi + j + 1]);
                }
                if (i < num_theta - 2) {
                    glVertex3f(h_Yx[idx], h_Yy[idx], h_Yz[idx]);
                    glVertex3f(h_Yx[(i + 1) * num_phi + j], h_Yy[(i + 1) * num_phi + j], h_Yz[(i + 1) * num_phi + j]);
                }
            }
        }
        glEnd();

        // Print l and m values
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT));
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glColor3f(1.0f, 1.0f, 1.0f); // White color
        glRasterPos2f(10, glutGet(GLUT_WINDOW_HEIGHT) - 20);
        char text[50];
        sprintf(text, "l = %d, m = %d", l, m);
        for (int i = 0; text[i]!= '\0'; i++) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
        }
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glPopMatrix();
        glutSwapBuffers();

    } else {

        glLineWidth(1.0f); // Set the line width
        glBegin(GL_LINES);
        for (int i = 0; i < num_theta - 1; ++i) {
            for (int j = 0; j < num_phi - 1; ++j) {
                int idx = i * num_phi + j;
                float r, g, b;
                if (displayColorMap) {
                    mapToColor(h_Y_real[idx], currentColormap, r, g, b);
                } else {
                    mapToImage(h_Yx[idx], h_Yy[idx], h_Yz[idx], r, g, b);
                }
                glColor3f(r, g, b);
                glVertex3f(h_Yx[idx], h_Yy[idx], h_Yz[idx]);
                if (j < num_phi - 2) {
                    glVertex3f(h_Yx[i * num_phi + j + 1], h_Yy[i * num_phi + j + 1], h_Yz[i * num_phi + j + 1]);
                }
                if (i < num_theta - 2) {
                    glVertex3f(h_Yx[idx], h_Yy[idx], h_Yz[idx]);
                    glVertex3f(h_Yx[(i + 1) * num_phi + j], h_Yy[(i + 1) * num_phi + j], h_Yz[(i + 1) * num_phi + j]);
                }
            }
        }
        glEnd();

        // Print l and m values
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluOrtho2D(0, glutGet(GLUT_WINDOW_WIDTH), 0, glutGet(GLUT_WINDOW_HEIGHT));
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glColor3f(1.0f, 1.0f, 1.0f); // White color
        glRasterPos2f(10, glutGet(GLUT_WINDOW_HEIGHT) - 20);
        char text[50];
        sprintf(text, "l = %d, m = %d", l, m);
        for (int i = 0; text[i]!= '\0'; i++) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
        }
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glPopMatrix();
        glutSwapBuffers();
    }
}


void mapToImage(float Yx, float Yy, float Yz, float &r, float &g, float &b) {
    // Normalize the h_Y_real value
    float value = (*h_Y_real - min_h_Y_real) / (max_h_Y_real - min_h_Y_real);

    // Clip the value to the range [0, 1]
    value = std::max(0.0f, std::min(value, 1.0f));

    // Calculate the texture coordinates (u, v) for the background image
    float u = (atan2(Yy, Yx) + M_PI) / (2 * M_PI);
    float v = acos(Yz) / M_PI;

    // Get the color from the background image at the texture coordinates (u, v)
    // Assuming you have a function to get the color from the background image
    getColorFromBackgroundImage(u, v, r, g, b);
}

// Function to get the color from the background image
void getColorFromBackgroundImage(float u, float v, float &r, float &g, float &b) {
    // Assuming you have a 2D array to store the background image pixels
    int width = static_cast<int>(IMAGE_WIDTH);
    int height = static_cast<int>(IMAGE_HEIGHT);
    int x = (int)(u * width);
    int y = (int)(v * height);

    // Get the color from the background image at the pixel coordinates (x, y)
    r = imgSrcDataCH1_ptr[y * width * 3 + x * 3 + 0];
    g = imgSrcDataCH1_ptr[y * width * 3 + x * 3 + 1];
    b = imgSrcDataCH1_ptr[y * width * 3 + x * 3 + 2];
}

void mapToColor(float value, Colormap cmap, float &r, float &g, float &b) {
    float min = -0.5f;
    float max = 0.5f;
    value = (value - min) / (max - min); // Normalize the value

    switch (cmap) {
        case JET:
            if (value <= 0.33f) {
                r = 0.0f;
                g = 4.0f * value;
                b = 1.0f;
            } else if (value <= 0.66f) {
                r = 4.0f * (value - 0.33f);
                g = 1.0f;
                b = 1.0f - 4.0f * (value - 0.33f);
            } else {
                r = 1.0f;
                g = 1.0f - 4.0f * (value - 0.66f);
                b = 0.0f;
            }
            break;
        case COOL:
            r = 1.0f - value;
            g = 1.0f - abs(value - 0.5f) * 2.0f;
            b = value;
            break;
        case BONE:
            if (value <= 0.25f) {
                r = 0.0f;
                g = 4.0f * value;
                b = 0.0f;
            } else if (value <= 0.5f) {
                r = 4.0f * (value - 0.25f);
                g = 1.0f;
                b = 0.0f;
            } else if (value <= 0.75f) {
                r = 1.0f;
                g = 1.0f - 4.0f * (value - 0.5f);
                b = 0.0f;
            } else {
                r = 1.0f;
                g = 0.0f;
                b = 4.0f * (value - 0.75f);
            }
            break;
        case VIRIDIS:
            if (value <= 0.2f) {
                r = 0.0f;
                g = 4.0f * value;
                b = 1.0f;
            } else if (value <= 0.4f) {
                r = 4.0f * (value - 0.2f);
                g = 1.0f;
                b = 1.0f - 4.0f * (value - 0.2f);
            } else if (value <= 0.6f) {
                r = 1.0f;
                g = 1.0f - 4.0f * (value - 0.4f);
                b = 0.0f;
            } else if (value <= 0.8f) {
                r = 1.0f - 4.0f * (value - 0.6f);
                g = 0.0f;
                b = 4.0f * (value - 0.6f);
            } else {
                r = 0.0f;
                g = 0.0f;
                b = 1.0f;
            }
            break;
        case PLASMA:
            if (value <= 0.2f) {
                r = 0.0f;
                g = 4.0f * value;
                b = 1.0f;
            } else if (value <= 0.4f) {
                r = 4.0f * (value - 0.2f);
                g = 1.0f;
                b = 1.0f - 4.0f * (value - 0.2f);
            } else if (value <= 0.6f) {
                r = 1.0f;
                g = 1.0f - 4.0f * (value - 0.4f);
                b = 0.0f;
            } else if (value <= 0.8f) {
                r = 1.0f - 4.0f * (value - 0.6f);
                g = 0.0f;
                b = 4.0f * (value - 0.6f);
            } else {
                r = 0.0f;
                g = 0.0f;
                b = 1.0f;
            }
            break;
        case MAGMA:
            if (value <= 0.2f) {
                r = 0.0f;
                g = 4.0f * value;
                b = 1.0f;
            } else if (value <= 0.4f) {
                r = 4.0f * (value - 0.2f);
                g = 1.0f;
                b = 1.0f - 4.0f * (value - 0.2f);
            } else if (value <= 0.6f) {
                r = 1.0f;
                g = 1.0f - 4.0f * (value - 0.4f);
                b = 0.0f;
            } else if (value <= 0.8f) {
                r = 1.0f - 4.0f * (value - 0.6f);
                g = 0.0f;
                b = 4.0f * (value - 0.6f);
            } else {
                r = 0.0f;
                g = 0.0f;
                b = 1.0f;
            }
            break;
        default:
            // Handle unknown colormap
            break;
    }
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 1.0, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    gluLookAt(0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

void keyboard(unsigned char key, int x, int y) {
    if (key == 27) { // ESC key
        cleanupCuda();
        exit(0);
    } else if (key == 'z' || key == 'Z') {
        // Toggle between sets of l and m values
        current_set = (current_set + 1) % num_sets;
        l = l_values[current_set];
        m = m_values[current_set];
        sphRadiusSize = determineSphRadiusSize(l, m, num_theta, num_phi);
        initCuda(sphRadiusSize);
        glutPostRedisplay();
    } else if (key == 'x' || key == 'X') {
        // Toggle between sets of l and m values
        current_set = (current_set - 1 + num_sets) % num_sets;
        l = l_values[current_set];
        m = m_values[current_set];
        sphRadiusSize = determineSphRadiusSize(l, m, num_theta, num_phi);
        initCuda(sphRadiusSize);
        glutPostRedisplay();
    } else if (key == 'a' || key == 'A') {
        // Toggle auto rotate
        autoRotate =!autoRotate;
        if (autoRotate) {
            // Randomly choose a rotation direction
            rotationX = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            rotationY = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            rotationZ = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    } else if (key == 'c' || key == 'C') {
        // Toggle colormap
        switch (currentColormap) {
            case JET:
                currentColormap = COOL;
                break;
            case COOL:
                currentColormap = BONE;
                break;
            case BONE:
                currentColormap = VIRIDIS;
                break;
            case VIRIDIS:
                currentColormap = PLASMA;
                break;
            case PLASMA:
                currentColormap = MAGMA;
                break;
            case MAGMA:
                currentColormap = JET;
                break;
        }
        glutPostRedisplay();
    } else if (key == 'v' || key == 'V') {
        // Toggle background image
        displayColorMap =!displayColorMap;
        glutPostRedisplay();
    } else if (key == 'b' || key == 'B') {
        // Toggle background image
        displayBackgroundImage =!displayBackgroundImage;
        glutPostRedisplay();
    }
}

// void keyboard(unsigned char key, int x, int y) {
//     if (key == 27) { // ESC key
//         cleanupCuda();
//         exit(0);
//     } else if (key == 'z' || key == 'Z') {
//         // Toggle between sets of l and m values
//         current_set = (current_set + 1) % num_sets;
//         l = l_values[current_set];
//         m = m_values[current_set];
//         sphRadiusSize = determineSphRadiusSize(l, m, num_theta, num_phi);
//         initCuda(sphRadiusSize);
//         glutPostRedisplay();
//     } else if (key == 'x' || key == 'X') {
//         // Toggle between sets of l and m values
//         current_set = (current_set - 1 + num_sets) % num_sets;
//         l = l_values[current_set];
//         m = m_values[current_set];
//         sphRadiusSize = determineSphRadiusSize(l, m, num_theta, num_phi);
//         initCuda(sphRadiusSize);
//         glutPostRedisplay();
//     } else if (key == 'a' || key == 'A') {
//         // Toggle auto rotate
//         autoRotate =!autoRotate;
//         if (autoRotate) {
//             // Randomly choose a rotation direction
//             rotationX = (float)rand() / RAND_MAX * 2.0f - 1.0f;
//             rotationY = (float)rand() / RAND_MAX * 2.0f - 1.0f;
//             rotationZ = (float)rand() / RAND_MAX * 2.0f - 1.0f;
//         }
//     } else if (key == 'c' || key == 'C') {
//         // Toggle colormap
//         switch (currentColormap) {
//             case JET:
//                 currentColormap = COOL;
//                 break;
//             case COOL:
//                 currentColormap = BONE;
//                 break;
//             case BONE:
//                 currentColormap = VIRIDIS;
//                 break;
//             case VIRIDIS:
//                 currentColormap = PLASMA;
//                 break;
//             case PLASMA:
//                 currentColormap = MAGMA;
//                 break;
//             case MAGMA:
//                 currentColormap = JET;
//                 break;
//         }
//         glutPostRedisplay();
//     } else if (key == 'b' || key == 'B') {
//         // Toggle background image
//         displayBackgroundImage =!displayBackgroundImage;
//         glutPostRedisplay();
//     }
// }

void specialKeys(int key, int x, int y) {

    float viewAngleScale = 7.7f;

    switch (key) {
        case GLUT_KEY_UP:
            view_angle_x += viewAngleScale;
            break;
        case GLUT_KEY_DOWN:
            view_angle_x -= viewAngleScale;
            break;
        case GLUT_KEY_LEFT:
            view_angle_y -= viewAngleScale;
            break;
        case GLUT_KEY_RIGHT:
            view_angle_y += viewAngleScale;
            break;
    }
    glutPostRedisplay();
}

void timer(int value) {
    if (autoRotate) {
        // Update the view angles based on the rotation speed and direction
        view_angle_x += rotationSpeed * rotationX;
        view_angle_y += rotationSpeed * rotationY;
        view_angle_z += rotationSpeed * rotationZ;
        glutPostRedisplay();
    }
    glutTimerFunc(16, timer, 0); // 16ms = 60fps
}

// Cleanup CUDA resources
void cleanupCuda() {
    free(h_Y_real);
    free(h_Y_imag);
    free(h_Yx);
    free(h_Yy);
    free(h_Yz);
    cudaFree(d_Y_real);
    cudaFree(d_Y_imag);
}