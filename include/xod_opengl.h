/*------------------------------------------------------------------------------------------*/
/* ___::((xod_opengl.h))::___

    created by eschei

    Purpose: OpenGL Display functions header
    Revision History: 2024-05-22 - initial
*/

/*------------------------------------------------------------------------------------------*/
/*---%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---*/
/*------------------------------------------------------------------------------------------*/

#ifndef __XOD_OPENGL_H__
#define __XOD_OPENGL_H__


#include <iostream>
#include <string>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "xodImg_types.h"


constexpr int MAXWIDTH = IMAGE_WIDTH;
constexpr int MAXHEIGHT = IMAGE_HEIGHT;

constexpr int REFRESH_DELAY = 56; // ms


// // *-----------------------------------------------------------------------------------* //
// ///// function decl /////////////////////

int init_gl(int* argc, char** argv);
void initTextures();
void cleanup_gl();


// // int init_gl(int *argc, char **argv);
// int init_gl(int *argc, char **argv);

// void draw_text(char *string, double xpos, double ypos);
// void keyboardFunc(unsigned char Key, int X, int Y);
// void timerEvent(int msecs);
// void cleanup_gl(void);

// void displayCH1(void);
// void displayCH2(void);


// *-----------------------------------------------------------------------------------* //

#endif	// end  __XOD_OPENGL_H__
