/*------------------------------------------------------------------------------------------------*/
/* ___::((libpng_rdwr.h))::___

   ___::((ZMP))::___
   ___::((created by eschei))___

  Purpose: Read / Write .png image using libpng library
  Device: All
  Revision History: 2023_3_16 - initial

*/

/*------------------------------------------------------------------------------------------------*/
/*---%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---*
*---%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%---*/
/*------------------------------------------------------------------------------------------------*/

#ifndef __LIBPNG_RDWR_H__
#define __LIBPNG_RDWR_H__

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <memory>

#include <png.h>

// *-----------------------------------------------------------------------------------* //
///// datatypes /////////////////////

struct PngInfo {
   int width;
   int height;
   int bitDepth;
   int colorType;
   int interfaceType;
   unsigned int rowBytes;
};

struct UserData {
    std::ofstream file;
};

// *-----------------------------------------------------------------------------------* //
///// function decl /////////////////////

bool readPngImage(const char *file_name, PngInfo* pngInfo,
                  std::vector<std::unique_ptr<uint8_t[]> >& pngRowsIn_pp);

static void user_write_fn(png_structp png_ptr, png_bytep data, png_size_t length);

bool writePngImage(const char *file_name, PngInfo* pngInfo,
                   const std::vector<std::unique_ptr<uint8_t[]> >& imageRowData);


template <class T, const size_t IMG_WIDTH, const size_t IMG_HEIGHT, const int PXL_BYTES = 3>
void readImgFromPng(std::string &fileName, T (&img)[IMG_WIDTH * IMG_HEIGHT * PXL_BYTES]);

template <class T, const size_t IMG_WIDTH, const size_t IMG_HEIGHT, const int PXL_BYTES = 3>
void readImgFromPngInv(std::string &fileName, T (&img)[IMG_WIDTH * IMG_HEIGHT * PXL_BYTES]);

template <typename T, size_t IMG_WIDTH, size_t IMG_HEIGHT, int BITS = 8,
          int PXL_BYTES = 3, int COLOR_TYPE = 2, int ALPHA_LVL = 100>
void writeImgToPng(const std::string& outDir, const std::string& fileName,
                   const T (&img)[IMG_WIDTH * IMG_HEIGHT * PXL_BYTES]);

// *-----------------------------------------------------------------------------------* //
///// function def /////////////////////


bool readPngImage(const char *file_name, PngInfo* pngInfo,
                  std::vector<std::unique_ptr<uint8_t[]> >& pngRowsIn_pp)
{
    std::ifstream file(file_name, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file: " << file_name << std::endl;
        return false;
    }

    png_structp pngData_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!pngData_ptr) {
        std::cerr << "Error: Failed to create PNG read structure." << std::endl;
        return false;
    }

    png_infop pngInfo_ptr = png_create_info_struct(pngData_ptr);
    if (!pngInfo_ptr) {
        std::cerr << "Error: Failed to create PNG info structure." << std::endl;
        png_destroy_read_struct(&pngData_ptr, nullptr, nullptr);
        return false;
    }

    png_set_read_fn(pngData_ptr, reinterpret_cast<png_voidp>(&file), [](png_structp png_ptr, png_bytep data, png_size_t length) {
        auto& file = *reinterpret_cast<std::ifstream*>(png_get_io_ptr(png_ptr));
        file.read(reinterpret_cast<char*>(data), length);
    });

    if (setjmp(png_jmpbuf(pngData_ptr))) {
        std::cerr << "Error: Problem reading PNG file." << std::endl;
        png_destroy_read_struct(&pngData_ptr, &pngInfo_ptr, nullptr);
        return false;
    }

    png_read_png(pngData_ptr, pngInfo_ptr, PNG_TRANSFORM_IDENTITY, nullptr);

    png_uint_32 width, height;
    int bit_depth, color_type, interlace_type;
    png_get_IHDR(pngData_ptr, pngInfo_ptr, &width, &height, &bit_depth, &color_type, &interlace_type, nullptr, nullptr);

    pngInfo->width = static_cast<int>(width);
    pngInfo->height = static_cast<int>(height);
    pngInfo->bitDepth = bit_depth;
    pngInfo->colorType = color_type;
    pngInfo->interfaceType = interlace_type;

    png_bytep* rowData_pp = png_get_rows(pngData_ptr, pngInfo_ptr);
    pngRowsIn_pp.resize(height);

    for (int i = 0; i < height; ++i) {
        pngRowsIn_pp[i].reset(new uint8_t[png_get_rowbytes(pngData_ptr, pngInfo_ptr)]);
        std::memcpy(pngRowsIn_pp[i].get(), rowData_pp[i], png_get_rowbytes(pngData_ptr, pngInfo_ptr));
    }

    png_destroy_read_struct(&pngData_ptr, &pngInfo_ptr, nullptr);
    return true;
}


// This is a conversion function to use libpng API in a c++ context
// This function is required for compatibility between c++ std::ofstream & FILE* expected y png_init_io
static void user_write_fn(png_structp png_ptr, png_bytep data, png_size_t length) {
    UserData* user_data = reinterpret_cast<UserData*>(png_get_io_ptr(png_ptr));
    user_data->file.write(reinterpret_cast<char*>(data), length);
}


bool writePngImage(const char* file_name, PngInfo* pngInfo,
                   const std::vector<std::unique_ptr<uint8_t[]>>& imageRowData)
{
    UserData user_data;

    user_data.file.open(file_name, std::ios::binary);
    if (!user_data.file.is_open()) {
        std::cerr << "Error: Unable to open file: " << file_name << std::endl;
        return false;
    }

    png_structp pngData_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!pngData_ptr) {
        std::cerr << "Error: Failed to create PNG write structure." << std::endl;
        user_data.file.close();
        return false;
    }

    png_infop pngInfo_ptr = png_create_info_struct(pngData_ptr);
    if (!pngInfo_ptr) {
        std::cerr << "Error: Failed to create PNG info structure." << std::endl;
        png_destroy_write_struct(&pngData_ptr, nullptr);
        user_data.file.close();
        return false;
    }

    png_set_write_fn(pngData_ptr, &user_data, user_write_fn, nullptr);

    png_uint_32 width = static_cast<png_uint_32>(pngInfo->width);
    png_uint_32 height = static_cast<png_uint_32>(pngInfo->height);
    int bit_depth = pngInfo->bitDepth;
    int color_type = pngInfo->colorType;
    int interlace_type = pngInfo->interfaceType;

    png_set_IHDR(pngData_ptr, pngInfo_ptr, width, height, bit_depth, color_type,
                 interlace_type, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

    std::vector<png_bytep> pngRowsOut(imageRowData.size());
    for (size_t i = 0; i < imageRowData.size(); ++i) {
        pngRowsOut[i] = static_cast<png_bytep>(imageRowData[i].get());
    }

    png_set_rows(pngData_ptr, pngInfo_ptr, pngRowsOut.data());

    png_write_png(pngData_ptr, pngInfo_ptr, PNG_TRANSFORM_IDENTITY, nullptr);
    png_destroy_write_struct(&pngData_ptr, &pngInfo_ptr);
    user_data.file.close();

    return true;
}


template <class T, const size_t IMG_WIDTH, const size_t IMG_HEIGHT, const int PXL_BYTES = 3>
void readImgFromPng(std::string &fileName, T (&img)[IMG_WIDTH * IMG_HEIGHT * PXL_BYTES]) {
    std::cout << "Read PNG Image: " << fileName << std::endl;

    // Convert string filename to char pointer for C functions
    const char *cstrImgNm = fileName.c_str();

    PngInfo pngInfo;

    // Create a vector of unique pointers to store pixel data
    std::vector<std::unique_ptr<uint8_t[]>> pngRowsRd(IMG_HEIGHT);
    for (size_t i = 0; i < IMG_HEIGHT; ++i) {
        pngRowsRd[i] = std::make_unique<uint8_t[]>(PXL_BYTES * IMG_WIDTH);
    }

    // Read image data from .png file
    bool RDSTATUS = readPngImage(cstrImgNm, &pngInfo, pngRowsRd);

    // Output status and image information
    if (RDSTATUS) {
        std::cout << "Image Read Success: " << RDSTATUS << std::endl;
    } else {
        std::cout << "Image Read Failed: " << RDSTATUS << std::endl;
    }
    std::cout << "Image Width: " << pngInfo.width << std::endl;
    std::cout << "Image Height: " << pngInfo.height << std::endl;
    std::cout << "Row Bytes: " << pngInfo.rowBytes << std::endl;
    std::cout << "Bit Depth: " << pngInfo.bitDepth << std::endl;
    std::cout << "Color Type: " << pngInfo.colorType << std::endl;
    std::cout << "Interface Type: " << pngInfo.interfaceType << std::endl;

    // Load image data
    for (size_t i = 0; i < IMG_HEIGHT; ++i) {
        for (size_t j = 0; j < PXL_BYTES * IMG_WIDTH; ++j) {
            img[j + i * PXL_BYTES * IMG_WIDTH] = static_cast<T>(pngRowsRd[i][j]);
        }
    }
}


template <class T, const size_t IMG_WIDTH, const size_t IMG_HEIGHT, const int PXL_BYTES = 3>
void readImgFromPngInv(std::string &fileName, T (&img)[IMG_WIDTH * IMG_HEIGHT * PXL_BYTES]) {
    std::cout << "Read PNG Image: " << fileName << std::endl;

    // Convert string filename to char pointer for C functions
    const char *cstrImgNm = fileName.c_str();

    PngInfo pngInfo;

    // Create a vector of unique pointers to store pixel data
    std::vector<std::unique_ptr<uint8_t[]>> pngRowsRd(IMG_HEIGHT);
    for (size_t i = 0; i < IMG_HEIGHT; ++i) {
        pngRowsRd[i] = std::make_unique<uint8_t[]>(PXL_BYTES * IMG_WIDTH);
    }

    // Read image data from.png file
    bool RDSTATUS = readPngImage(cstrImgNm, &pngInfo, pngRowsRd);

    // Output status and image information
    if (RDSTATUS) {
        std::cout << "Image Read Success: " << RDSTATUS << std::endl;
    } else {
        std::cout << "Image Read Failed: " << RDSTATUS << std::endl;
    }
    std::cout << "Image Width: " << pngInfo.width << std::endl;
    std::cout << "Image Height: " << pngInfo.height << std::endl;
    std::cout << "Row Bytes: " << pngInfo.rowBytes << std::endl;
    std::cout << "Bit Depth: " << pngInfo.bitDepth << std::endl;
    std::cout << "Color Type: " << pngInfo.colorType << std::endl;
    std::cout << "Interface Type: " << pngInfo.interfaceType << std::endl;

    // Load image data in reverse order to flip the image
    for (size_t i = 0; i < IMG_HEIGHT; ++i) {
        for (size_t j = 0; j < PXL_BYTES * IMG_WIDTH; ++j) {
            img[j + (IMG_HEIGHT - i - 1) * PXL_BYTES * IMG_WIDTH] = static_cast<T>(pngRowsRd[i][j]);
        }
    }
}


template <typename T, size_t IMG_WIDTH, size_t IMG_HEIGHT, int BITS = 8,
          int PXL_BYTES = 3, int COLOR_TYPE = 2, int ALPHA_LVL = 100>
void writeImgToPng(const std::string& outDir, const std::string& fileName,
                   const T (&img)[IMG_WIDTH * IMG_HEIGHT * PXL_BYTES])
{
    std::string outFileName = outDir + "/" + fileName;

    int pngImgWidth = static_cast<int>(IMG_WIDTH);
    int pngImgHeight = static_cast<int>(IMG_HEIGHT);
    int pngImgBitDepth = BITS;
    assert((PXL_BYTES == 3) || (PXL_BYTES == 6)); // assume pixel bit depth either 8 or 16
    int pngImgPxlBytes = PXL_BYTES;
    assert((COLOR_TYPE == 0) || (COLOR_TYPE == 2) || (COLOR_TYPE == 4) || (COLOR_TYPE == 6));
    int pngImgColorType = COLOR_TYPE;
    int pngImgInterfaceType = 0;

    PngInfo pngInfo;
    pngInfo.width = pngImgWidth;
    pngInfo.height = pngImgHeight;
    pngInfo.rowBytes = static_cast<unsigned int>(pngImgWidth * pngImgPxlBytes);
    pngInfo.bitDepth = pngImgBitDepth;
    pngInfo.colorType = pngImgColorType;
    pngInfo.interfaceType = pngImgInterfaceType;

    std::vector<std::unique_ptr<T[]>> imageRowData(pngImgHeight);
    for (int i = 0; i < pngImgHeight; ++i) {
        if ((pngImgColorType == 0) || (pngImgColorType == 4)) {
            imageRowData[i].reset(new T[pngImgWidth]);
        } else if ((pngImgColorType == 2) || (pngImgColorType == 6)) {
            imageRowData[i].reset(new T[PXL_BYTES * pngImgWidth]);
        }
    }

    // Store image data
    for (int i = 0; i < pngImgHeight; ++i) {
        for (int j = 0; j < PXL_BYTES * pngImgWidth; ++j) {
            imageRowData[i][j] = img[j + i * PXL_BYTES * pngImgWidth];
        }
    }
    // convert to *char for function calls
    char *cstrOutFileName = &outFileName[0];
    bool writeStatus = writePngImage(cstrOutFileName, &pngInfo, imageRowData);
    if (!writeStatus) {
        throw std::runtime_error("Failed to write PNG file: " + outFileName);
    }

    std::cout << "PNG file written successfully: " << outFileName << std::endl;
}


#endif

