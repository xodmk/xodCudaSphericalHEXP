/*------------------------------------------------------------------------------------------*/
/* ___::((xod_opengl.cpp))::___

    created by eschei

    Purpose: XODMK OpenGL Functions
    Device: N/A
    Revision History: 2024-05-22 - initial
*/

/*------------------------------------------------------------------------------------------*/
/*---%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---*
 *---%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---*/
/*------------------------------------------------------------------------------------------*/

#include <iostream>
#include <cstring>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "xod_opengl.h"
#include "xodImg_types.h"


extern uint8_t* imgSrcDataCH1_ptr;
extern uint8_t* resSphericHgpu_ptr;


GLuint texIDch1 = 0;
GLuint texIDch2 = 0;
int windowCH1 = 0;
int windowCH2 = 0;




int init_gl(int *argc, char **argv) {
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(IMAGE_WIDTH, IMAGE_HEIGHT);
    glutCreateWindow("CUDA XODMK IMGPROC");

    GLenum GlewInitResult = glewInit();
    if (GLEW_OK != GlewInitResult) {
        std::cerr << "glew init error: " << glewGetErrorString(GlewInitResult) << std::endl;
        return -1;
    }
    initTextures();
    glutTimerFunc(static_cast<int>(REFRESH_DELAY), [](int) { glutPostRedisplay(); }, 0);
    atexit(cleanup_gl);
    return 0;
}


void initTextures() {
    glGenTextures(1, &texIDch1);
    glBindTexture(GL_TEXTURE_2D, texIDch1);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, ImgDataCH1_ptr);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)MAXWIDTH, (int)MAXHEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, imgSrcDataCH1_ptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenTextures(1, &texIDch2);
    glBindTexture(GL_TEXTURE_2D, texIDch2);
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, IMAGE_WIDTH, IMAGE_HEIGHT, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, ImgJulia_ptr);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)MAXWIDTH, (int)MAXHEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, resSphericHgpu_ptr);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void cleanup_gl() {
    std::cout << "Cleaning up GL..." << std::endl;
    glDeleteTextures(1, &texIDch1);
    glDeleteTextures(1, &texIDch2);
    std::cout << "Cleanup GL done" << std::endl;
}

