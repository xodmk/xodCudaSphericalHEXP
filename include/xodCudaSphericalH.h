/*------------------------------------------------------------------------------------------------*/
/* ___::((xodSphericHexp.h))::___

   ___::((created by eschei))___

	Purpose: CMake CUDA Accelerated Image experiments

	Revision History: 2024-04-27 - initial
*/

/*------------------------------------------------------------------------------------------------*/

#ifndef __XODSPHERICHEXP_H__
#define __XODSPHERICHEXP_H__


enum Colormap {
	JET,
	//HSV,
	//HOT,
	COOL,
	//SPRING,
	//SUMMER,
	//AUTUMN,
	//WINTER,
	//PINK,
	BONE,
	VIRIDIS,
	PLASMA,
	//INFERNO,
	MAGMA,
	//CIVIDIS
};

void initCuda(float radius);
void cleanupCuda();

//void renderSphericalHarmonics(Colormap cmap);
void renderSphericalHarmonics();
void reshape(int w, int h);
void keyboard(unsigned char key, int x, int y);
void specialKeys(int key, int x, int y);
void timer(int value);

#endif
