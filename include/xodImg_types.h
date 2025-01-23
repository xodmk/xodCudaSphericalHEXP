/*------------------------------------------------------------------------------------------*/
/* ___::((xodImg_types.h))::___

    created by eschei

	Purpose: XODMK Image Processing Generic Types
	Revision History: 2024-05-22 - initial
*/

/*------------------------------------------------------------------------------------------*/
/*---%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---*/
/*------------------------------------------------------------------------------------------*/


#ifndef __XODIMG_TYPES_H__
#define __XODIMG_TYPES_H__


#include <stdio.h>
#include <stdint.h>
#include <cmath>


// Select One Image Resoltion
#define IMG1920X1080
//#define IMG480X270
//#define IMG1936X1096

static const int BITS_PER_PXL 					= 12;
static const int BITS_COLOR_PXL 				= 3 * BITS_PER_PXL;
static const int PXL_BYTES						= 3; 							// for 8-bit, 3 bytes per RGB pixel

// 8-bit: pxl3<<28+pxl2<<16+pxl1<<4 ; 12-bit: pxl3<<24+pxl2<<12+pxl1
static const int BYTE_ALIGN_DATA 				= 40;
static const int BYTE_ALIGN_STEREO 				= 72;
static const int BYTE_ALIGN_MOSAIC 				= 48;

static const size_t MAX_PXL						= 4095;							// 255


#ifdef IMG1920X1080
static const size_t IMAGE_WIDTH  				= 1920;							// in pixels
static const size_t IMAGE_HEIGHT 				= 1080;							// in pixels

#elif defined(IMG480X270)
static const size_t IMAGE_WIDTH  				= 480;							// in pixels
static const size_t IMAGE_HEIGHT 				= 270;							// in pixels

#elif defined(IMG1936X1096)
static const size_t IMAGE_WIDTH  				= 1936;							// in pixels
static const size_t IMAGE_HEIGHT 				= 1096;							// in pixels
#endif

static const size_t IMAGE_HALFWIDTH				= IMAGE_WIDTH / 2;					// 960 / 240
static const size_t IMAGE_HALFHEIGHT			= IMAGE_HEIGHT / 2;					// 540 / 235

static const size_t IMAGE_SIZE   				= IMAGE_WIDTH * IMAGE_HEIGHT;				// in pixels (2121856)
static const size_t IMAGE_QTRSIZE				= (IMAGE_WIDTH / 2) * (IMAGE_HEIGHT / 2);		// Quarter Image size (530464)

static const size_t N_OF_CHANNELS  				= 3;							// RGB
static const size_t IMAGE_SIZE_BGR 				= IMAGE_SIZE * N_OF_CHANNELS;				// in pixels
static const size_t IMAGE_QTRSIZE_BGR			= IMAGE_QTRSIZE * N_OF_CHANNELS;

// Fixed sensor size for RV30 (1936 x 1096)
static const size_t IMAGE_WIDTH_SENSOR  		= 1936;							// in pixels
static const size_t IMAGE_HEIGHT_SENSOR 		= 1096;							// in pixels
static const size_t IMAGE_SIZE_SENSOR			= IMAGE_WIDTH_SENSOR * IMAGE_HEIGHT_SENSOR;
static const size_t RV3P_BYTE_PER_PIXEL			= 3;							// 24 bits
static const size_t RV3P_IMAGE_SIZE				= IMAGE_SIZE_SENSOR * RV3P_BYTE_PER_PIXEL; 		// in bytes

// outerLeft, outerRight, innerLeft, innerRight
static const size_t IMAGE_COUNT    				= 2;
static const size_t RV3P_FILE_SIZE 				= RV3P_IMAGE_SIZE * IMAGE_COUNT;			// in bytes

// Use for zero-padding frames if necessary
static const size_t FRAME_SIZE_SENSOR			= IMAGE_SIZE_SENSOR;
static const size_t FRAME_SIZE					= IMAGE_SIZE;

// STRIDE:= line width aligned to 64 (1080 -> 1088)
static const size_t STRIDE						= ceil(IMAGE_WIDTH / 64.0) * 64;


// *--------------------------------------------------------------------------------------* //
////// set dimensions for Adaptive Histogram Equalization Function Sub-Blocks

// Sub-Block Dimensions for 1920 x 1080 frame:
// (160 x 135) => 1920/160 = 12, 1080/135 = 8 -> 12*8 = 96 sub-blocks

static const int SUBBLOCK_WIDTH = 160;
static const int SUBBLOCK_HEIGHT = 135;

static const int HALFBLOCK_WIDTH = static_cast<int>(SUBBLOCK_WIDTH / 2);
static const int HALFBLOCK_HEIGHT = static_cast<int>(SUBBLOCK_HEIGHT / 2);

static const int NUM_SUBBLOCK_WIDTH = 12;
static const int NUM_SUBBLOCK_HEIGHT = 8;

static const int NUM_BINS = 256;			// 8-bit histogram => 256 bins

const int LINEBUF_SIZE = 3;
const int KERNEL_SIZE = 3;


// *--------------------------------------------------------------------------------------* //

using uint24_t = uint32_t;
constexpr uint24_t MAX_VALUE = 0x00FFFFFF;

struct uint32x2 {
	uint32_t data_1;
	uint32_t data_2;
};

struct uint32x4 {
	uint32_t data_1;
	uint32_t data_2;
	uint32_t data_3;
	uint32_t data_4;
};

struct Bgr8Data_t {
    uint8_t blue;
    uint8_t green;
    uint8_t red;
};

struct Bgr8x2CHData_t {
    uint8_t blue1;
    uint8_t green1;
    uint8_t red1;
    uint8_t blue2;
    uint8_t green2;
    uint8_t red2;
};


struct demosaic2CH_t {
	uint24_t pxlCH1;
	uint24_t pxlCH2;
};


using pixel_t = uint16_t;  // Assuming a 12-bit pixel stored in a 16-bit type


#endif
