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





// // *-----------------------------------------------------------------------------------* //
// /// OpenGL /////////////////////

// //uint8_t *pngImgDataCH1_p = NULL;
// //uint8_t *pngImgDataCH2_p = NULL;
// //float *pngImgData_p = NULL;

// //std::unique_ptr<uint8_t[]> openGL_ImgDataCH1_p;
// extern uint8_t* ImgDataCH1_ptr;
// extern uint8_t* ImgJulia_ptr;

// GLuint texIDch1 = 0;
// GLuint texIDch2 = 0;
// int windowCH1 = 0;
// int windowCH2 = 0;

// int cursor_x = 0, cursor_y = 0;
// double cursor_fx = 0, cursor_fy = 0;
// bool wrPng = false;


// // *-----------------------------------------------------------------------------------* //
// ///// function def /////////////////////


// //int init_gl(int *argc, char **argv) {
// int init_gl(int *argc, char **argv) {

//     static int GL_MAXWIDTH  = MAXWIDTH;
//     static int GL_MAXHEIGHT = MAXHEIGHT;

//     //int BorderWidth;
//     int TitleBarHeight;

//     glutInit(argc, argv);                           // Initialize GLUT
//     glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);   // enable double buffering, RGBA format

//     // Magic numbers!
//     //BorderWidth = glutGet(506);
//     TitleBarHeight = glutGet(507);
//     // DEBUG_PRINT...
//     // printf("BorderWidth = %i\n", BorderWidth);
//     // printf("TitleBarHeight = %i\n", TitleBarHeight);

//     if (TitleBarHeight == 0) TitleBarHeight = 60;

//     glutInitWindowPosition(0, 0);
//     glutInitWindowSize(MAXWIDTH, MAXHEIGHT);

//     // *-----------------------------------------------------------------------------------* //
//     ////// Display OpenGL Texture Window //////

//     // Display Window CH1:
//     windowCH1 = glutCreateWindow("XODJulia Clone input IMG");
//     // std::cout << std::endl << "init_gl - windowCH1 = " << windowCH1 << std::endl;
//     glutDisplayFunc(displayCH1);
//     glutSetWindow(windowCH1);
//     glGenTextures(1, &texIDch1);  // Generate one texture index.
//     glBindTexture(GL_TEXTURE_2D, texIDch1);   // create texture object
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)MAXWIDTH, (int)MAXHEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, ImgDataCH1_ptr);
//     //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)MAXWIDTH, (int)MAXHEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, texImg0_p);
//     glBindTexture(GL_TEXTURE_2D, 0);    // deselect texture

//     // Display Window CH2:
//     windowCH2 = glutCreateWindow("XODJulia Print Res IMG");
//     glutDisplayFunc(displayCH2);
//     glutSetWindow(windowCH2);
//     glGenTextures(1, &texIDch2);  // Generate one texture index.
//     glBindTexture(GL_TEXTURE_2D, texIDch2);   // create texture object
//     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)MAXWIDTH, (int)MAXHEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, ImgJulia_ptr);
//     //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, (int)MAXWIDTH, (int)MAXHEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, texImg0_p);
//     glBindTexture(GL_TEXTURE_2D, 0);    // deselect texture


//     // *-----------------------------------------------------------------------------------* //

//     glutKeyboardFunc(keyboardFunc);
//     //glutMouseFunc(mouse_click);
//     //glutMotionFunc(mouse_move);
//     //glutSpecialFunc(special_key);

//     GLenum GlewInitResult = glewInit();
//     if (GLEW_OK != GlewInitResult) {
//         std::cerr << "glew init error: " << glewGetErrorString(GlewInitResult) << std::endl;
//         return -1;
//     }

//     glutTimerFunc(static_cast<int>(REFRESH_DELAY), timerEvent, 0);

//     atexit(cleanup_gl); // set cleanup function

//     return 0;
// }


// void draw_text(char *string, double xpos, double ypos) {
//     glRasterPos2d(xpos, ypos);
//     for (const char* c = string; *c != '\0'; ++c)
//         glutBitmapCharacter(GLUT_BITMAP_9_BY_15, *c);
// }


// // *-----------------------------------------------------------------------------------* //
// ////// ToDo - push these functions into a generic func

// void displayCH1(void) {
//     glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black and opaque
//     glClear(GL_COLOR_BUFFER_BIT);         // Clear the color buffer

//     glBindTexture(GL_TEXTURE_2D, texIDch1);
//     glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (int)MAXWIDTH, (int)MAXHEIGHT, GL_RGB, GL_UNSIGNED_BYTE, ImgDataCH1_ptr);

//     glDisable(GL_DEPTH_TEST);
//     glEnable(GL_TEXTURE_2D);
//     //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

//     glBegin(GL_QUADS);
//         glColor3f(1, 1, 1);
//         glTexCoord2f(0, 0);     glVertex2f(-1,  1);     // top left
//         glTexCoord2f(1, 0);     glVertex2f( 1,  1);     // top right
//         glTexCoord2f(1, 1);     glVertex2f( 1, -1);     // bottom right
//         glTexCoord2f(0, 1);     glVertex2f(-1, -1);     // bottom left
//     glEnd();
//     glBindTexture(GL_TEXTURE_2D, 0);


//     cursor_fx = (cursor_x / static_cast<double>(glutGet(GLUT_WINDOW_WIDTH))) * 2 - 1.0;
//     cursor_fy = 1.0 - (cursor_y / static_cast<double>(glutGet(GLUT_WINDOW_HEIGHT))) * 2;

//     glBegin(GL_LINES);
//         glColor3f(1, 0, 0);
//         glVertex2f(static_cast<float>(cursor_fx), -1.0);
//         glVertex2f(static_cast<float>(cursor_fx), +1.0);
//         glVertex2f(-1.0, static_cast<float>(cursor_fy));
//         glVertex2f(+1.0, static_cast<float>(cursor_fy));
//     glEnd();

//     glutSwapBuffers(); // render now
// }


// void displayCH2(void) {
//     glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Set background color to black and opaque
//     glClear(GL_COLOR_BUFFER_BIT);         // Clear the color buffer

//     glBindTexture(GL_TEXTURE_2D, texIDch2);
//     glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, (int)MAXWIDTH, (int)MAXHEIGHT, GL_RGB, GL_UNSIGNED_BYTE, ImgJulia_ptr);

//     glDisable(GL_DEPTH_TEST);
//     glEnable(GL_TEXTURE_2D);
//     //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//     //glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//     glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

//     glBegin(GL_QUADS);
//         glColor3f(1, 1, 1);
//         glTexCoord2f(0, 0);     glVertex2f(-1,  1);     // top left
//         glTexCoord2f(1, 0);     glVertex2f( 1,  1);     // top right
//         glTexCoord2f(1, 1);     glVertex2f( 1, -1);     // bottom right
//         glTexCoord2f(0, 1);     glVertex2f(-1, -1);     // bottom left
//     glEnd();
//     glBindTexture(GL_TEXTURE_2D, 0);


//     cursor_fx = (cursor_x / (double)glutGet(GLUT_WINDOW_WIDTH)) * 2 - 1.0;
//     cursor_fy = 1.0 - (cursor_y / (double)glutGet(GLUT_WINDOW_HEIGHT)) * 2;

//     glBegin(GL_LINES);
//         glColor3f(1, 0, 0);
//         glVertex2f((float)cursor_fx, -1.0);
//         glVertex2f((float)cursor_fx, +1.0);
//         glVertex2f(-1.0, (float)cursor_fy);
//         glVertex2f(+1.0, (float)cursor_fy);
//     glEnd();

//     glutSwapBuffers(); // render now
// }


// // *-----------------------------------------------------------------------------------* //


// //void timerEvent(int msecs) {
// void timerEvent(int msecs) {
//     // update every REFRESH_DELAY milliseconds
//     glutPostRedisplay();
//     glutTimerFunc(msecs, timerEvent, 0);
// }


// void keyboardFunc(unsigned char Key, int X, int Y)
// {
//     std::cout << "X, Y: " << X << ", " << Y << std::endl;
//     switch (Key) {
//         // case 'T':
//         // case 't':
//         // {
//         //     ActiveIndexBuffer = (ActiveIndexBuffer == 1 ? 0 : 1);
//         //     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IndexBufferId[ActiveIndexBuffer]);
//         //     break;
//         // }
//         case 'P':
//         case 'p':
//         {
//             printf("Write .PNG File...\n");
//             wrPng = true;
//             break;
//         }
//         case 27:    // Escape key
//         {
//             std::cout << "Escape Key Pressed" << std::endl;
//             exit (0);
//             break;
//         }
//     default:
//         break;
//     }
//     glutPostRedisplay();
// }


// void cleanup_gl(void) {
//     std::cout << "Cleaning up GL..." << std::endl;
//     glDeleteTextures(1, &texIDch1);
//     glDeleteTextures(1, &texIDch2);
//     std::cout << "Cleanup GL done" << std::endl;
// }
