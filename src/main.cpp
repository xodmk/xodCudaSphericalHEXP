/*------------------------------------------------------------------------------------------------*/
/* ___::((imageProcCuda_main.cpp))::___

   ___::((created by eschei))___

    Purpose: CMake CUDA Acceleration Image processing - main.cpp

    // build.sh
    cmake -S . -B build
    cmake --build build
    
    // run.sh
    ./build/xodCudaSphericalH -o ../../_data/output -png ../../_data/imgSrc/galaxyM42_1920x1080.png

    Revision History: 2024-04-27 - initial
*/
/*------------------------------------------------------------------------------------------------*/

#include <iostream>
#include <cstdlib>

#include <fstream>
#include <cassert>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <filesystem>

#include <png.h>
#include "libpng_rdwr.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "../include/xodImg_types.h"
#include "../include/xod_opengl.h"

#include "../include/xodCudaSphericalH.h"

// *-----------------------------------------------------------------------------------* //
///// function decl /////////////////////

std::string getFileName(const std::string& filePath);

// *-----------------------------------------------------------------------------------* //
///// glbl /////////////////////


uint8_t imgSrcDataCH1_p[IMAGE_WIDTH * IMAGE_HEIGHT * PXL_BYTES];
uint8_t (*imgSrcDataCH1_ptr)[IMAGE_WIDTH * IMAGE_HEIGHT * PXL_BYTES] = &imgSrcDataCH1_p;

//uint8_t imgSrcDataCH2_p[IMAGE_WIDTH * IMAGE_HEIGHT * PXL_BYTES];
//uint8_t (*imgSrcDataCH2_ptr)[IMAGE_WIDTH * IMAGE_HEIGHT * PXL_BYTES] = &imgSrcDataCH2_p;

uint8_t resSphericH_gpu[IMAGE_WIDTH * IMAGE_HEIGHT * PXL_BYTES] = {0};
uint8_t (*resSphericHgpu_ptr)[IMAGE_WIDTH * IMAGE_HEIGHT * PXL_BYTES] = &resSphericH_gpu;


// *-----------------------------------------------------------------------------------* //
///// function def /////////////////////

// Extract the filename stem (without extension) using filename stem method
std::string getFileName(const std::string& filePath) {
    std::filesystem::path pathObj(filePath);
    std::string fileName = pathObj.stem().string();
    return fileName;
}

// *-----------------------------------------------------------------------------------* //

// Display callback for GLUT
void display() {
    renderSphericalHarmonics();
}


// *-----------------------------------------------------------------------------------* //

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES"
              << "Options:\n"
              << "\t-h, --help \t\tShow this help message\n"
              << "\t-o         \t\tOutput_Directory\tPath (full or relative) to result output directory\n"
              << "\t-png       \t\tSource Images\n"
              << std::endl;
}

// *-----------------------------------------------------------------------------------* //

int main(int argc, char** argv) 
{
    // *-----------------------------------------------------------------------------------* //
    // XODMK CUDA Accelerated Image Processing
    //
    // Arguments:
    // x       <executable path>       	- executable
    // 2       -o                       - output,
    // 3       <path/>                  - path to an output result directory (must exist)
    // 4       -png                     - Use .png source images
    // 5       <path/imageSrc1.png>     - 1st source image
    //
    //  - Build example:
    // ./build/xodCudaSphericalH -o ../../_data/output -png ../../_data/imgSrc/dragonfly1920x1080.png
    //
    // *-----------------------------------------------------------------------------------* //

    if (argc < 5)
    {
        std::cerr << "args:= -o <output dir> -png <path/sourceImg1.png> <path/sourceImg2.png>" << std::endl;
        return 1;
    }

    std::cout << std::endl
              << "*** BEGIN XODMK CUDA Accelerated Image Processing ***" << std::endl
              << std::endl;

    // Access argv in order
    // int         argIndex    = 1;
    std::string outputDir;

    int argIndex = 1;

    std::string arg = argv[argIndex];

    if ((arg == "-h") || (arg == "--help"))
    {
        show_usage(argv[0]);
        return 0;
    }

    if (arg == "-o")
    {
        if (argIndex + 1 < argc)
        { // Make sure we aren't at the end of argv!
            argIndex++;
            outputDir = argv[argIndex++];
            std::cout << "Output Dir = " << outputDir << std::endl
                      << std::endl;
        }
        else
        { // no output directory specified
            std::cerr << "-o Output_Directory option requires path to output dir" << std::endl;
            return 1;
        }
    }
    else
    {
        std::cerr << "-o Output_Directory + path to output dir must be specified" << std::endl;
        return 1;
    }
    arg = argv[argIndex];

    if (argIndex + 1 < argc)
    { // Make sure we aren't at the end of argv!
        argIndex++;
        // process .png files...
        std::string fileNameCH1 = argv[argIndex];
        //std::string fileNameCH2 = argv[argIndex+ 1];

        std::cout << "Source Image CH1: " << fileNameCH1 << std::endl;
        //std::cout << "Source Image CH2: " << fileNameCH2 << std::endl;

        std::cout << "BEGIN: Read PNG Source Image... " << std::endl;
        readImgFromPngInv<uint8_t, IMAGE_WIDTH, IMAGE_HEIGHT>(fileNameCH1, imgSrcDataCH1_p);
        //readImgFromPng<uint8_t, IMAGE_WIDTH, IMAGE_HEIGHT>(fileNameCH2, imgSrcDataCH2_p);

        // *-----------------------------------------------------------------------------------* //

        std::cout << "END: Read PNG Source Image" << std::endl;
    }
    else
    {
        std::cerr << "-png requires a path to at least one .png image file" << std::endl;
        return 1;
    }


    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(IMAGE_WIDTH, IMAGE_HEIGHT);
    glutCreateWindow("Spherical Harmonics");

    glewInit();
    glEnable(GL_DEPTH_TEST);
    initCuda(1.6);

    srand(time(0));
    glutTimerFunc(16, timer, 0); // 16ms = 60fps

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);

    glEnable(GL_DEPTH_TEST);

    glutMainLoop();

    cleanupCuda();

    return 0;
}
