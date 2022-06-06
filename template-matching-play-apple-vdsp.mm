#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

typedef struct image {
    int width;
    int height;
    float *data;
} image_t;
image_t imageNewInShapeOf(image_t im) {
    return (image_t) {
            .width = im.width,
            .height = im.height,
            .data = (float *) calloc(im.width * im.height, sizeof(float))
    };
}

float imageMean(const image_t im) {
    float ret;
    vDSP_meanv(im.data, 1, &ret, im.width * im.height);
    return ret;
}
float imageSumOfSquares(const image_t im) {
    float ret;
    vDSP_svesq(im.data, 1, &ret, im.width * im.height);
    return ret;
}

void imageAddScalarInPlace(image_t im, const float scalar) {
    vDSP_vsadd(im.data, 1, &scalar, im.data, 1, im.width * im.height);
}
void imageMultiplyScalarInPlace(image_t im, const float scalar) {
    vDSP_vsmul(im.data, 1, &scalar, im.data, 1, im.width * im.height);
}
void imageDivideScalarInPlace(image_t im, const float scalar) {
    vDSP_vsdiv(im.data, 1, &scalar, im.data, 1, im.width * im.height);
}
void imageSubtractImageInPlace(image_t im, const image_t subtrahend) {
    vDSP_vsub(im.data, 1, subtrahend.data, 1, im.data, 1, im.width * im.height);
}
void imageDivideImageInPlace(image_t im, const image_t divisor) {
    vDSP_vdiv(im.data, 1, divisor.data, 1, im.data, 1, im.width * im.height);
}

image_t imageSquare(const image_t im) {
    image_t ret = imageNewInShapeOf(im);
    for (int y = 0; y < im.height; y++) {
        for (int x = 0; x < im.width; x++) {
            int i = y * im.width + x;
            ret.data[i] = im.data[i] * im.data[i];
        }
    }
    return ret;
}
image_t imageSqrt(const image_t im) {
    int n = im.width * im.height;
    image_t ret = imageNewInShapeOf(im);
    vvsqrtf(ret.data, im.data, &n);
    return ret;
}

image_t fftconvolve(image_t f, image_t g) {
    
}

image_t normxcorr2(image_t templ, image_t image) {
    // template = template - np.mean(template)
    imageAddScalarInPlace(templ, -1 * imageMean(templ));

    // image = image - np.mean(image)
    imageAddScalarInPlace(image, -1 * imageMean(image));

    // a1 = np.ones(template.shape)
    image_t a1 = imageNewInShapeOf(templ);
    float one = 1.0f;
    vDSP_vfill(&one, a1.data, 1, a1.width * a1.height);

    // ar = np.flipud(np.fliplr(template))
    image_t ar = imageNewInShapeOf(templ);
    for (int y = 0; y < templ.height; y++) {
        for (int x = 0; x < templ.width; x++) {
            int flippedY = templ.height - 1 - y;
            int flippedX = templ.width - 1 - x;
            ar.data[flippedY * templ.width + flippedX] = templ.data[y * templ.width + x];
        }
    }

    // out = fftconvolve(image, ar.conj())
    image_t outi = fftconvolve(image, ar);
    
    // image = fftconvolve(np.square(image), a1) - np.square(fftconvolve(image, a1)) / np.prod(template.shape)
    image_t imagen = fftconvolve(imageSquare(image), a1);
    image_t subtrahend = imageSquare(fftconvolve(image, a1));
    imageDivideScalarInPlace(subtrahend, subtrahend.width * subtrahend.height);
    imageSubtractImageInPlace(imagen, subtrahend);
    
    // template = np.sum(np.square(template))
    float templateSum = imageSumOfSquares(templ);
    
    // out = out / np.sqrt(image * template)
    imageMultiplyScalarInPlace(image, templateSum);
    imageDivideImageInPlace(outi, imageSqrt(image));

    // return out
    return outi;
}