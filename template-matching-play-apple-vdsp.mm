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


void onMouse(int event, int x, int y, int, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN)
        return;

    image_t* im = (image_t *) userdata;
    std::cout << "(" << x << "," << y << "): " << im->data[im->width * y + x] << std::endl;
}
void imageShow(const cv::String &winname, const image_t* im) {
    cv::Mat imageMat(im->height, im->width, CV_32F, im->data);
    cv::imshow(winname, imageMat);
    cv::setMouseCallback(winname, onMouse, (void *) im);
    // std::cout << winname << std::endl;
    // for (int i = 0; i < 100; i++) {
    //     std::cout << im.data[i] << std::endl;
    // }
}
void dspImageShow(const cv::String &winname, const DSPSplitComplex spl, int width, int height) {
    image_t *im = (image_t *) malloc(sizeof(image_t));
    im->width = width;
    im->height = height;
    im->data = spl.realp;
    imageShow(winname, im);
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
    // size = np.array(f.shape) + np.array(g.shape) - 1
    int width = f.width + g.width - 1;
    int height = f.height + g.height - 1;

    // fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    int log2n0 = ceil(log2f(width));
    int fwidth = 1 << (int) log2n0;
    int log2n1 = ceil(log2f(height));
    int fheight = 1 << (int) log2n1;
    std::cout << "fwidth, fheight: " << fwidth << "," << fheight << std::endl;
    std::cout << "log2n0, log2n1: " << log2n0 << "," << log2n1 << std::endl;

    // f_ = np.fft.fft2(f, fsize)
    DSPSplitComplex f_ = (DSPSplitComplex) {
        .realp = (float *) calloc(fwidth * fheight, sizeof(float)),
        .imagp = (float *) calloc(fwidth * fheight, sizeof(float))
    };
    for (int y = 0; y < f.height; y++) {
        memcpy(&f_.realp[y * fwidth], &f.data[y * f.width], f.width * sizeof(float));
    }
    FFTSetup fftSetup = vDSP_create_fftsetup(log2n0 > log2n1 ? log2n0 : log2n1,
                                             kFFTRadix2);
    vDSP_fft2d_zip(fftSetup,
                   &f_, 1, 0,
                   log2n0, log2n1,
                   kFFTDirection_Forward); // why is it only half height? real vs complex
    // vDSP_fft2d_zip(fftSetup,
    //                 &f_, 1, 0,
    //                 log2n0, log2n1,
    //                 kFFTDirection_Inverse); // why is it only half height? real vs complex
    dspImageShow("f__", f_, fwidth, fheight);

    // g_ = np.fft.fft2(g, fsize)
    DSPSplitComplex g_ = (DSPSplitComplex) {
        .realp = (float *) calloc(fwidth * fheight, sizeof(float)),
        .imagp = (float *) calloc(fwidth * fheight, sizeof(float))
    };
    for (int y = 0; y < g.height; y++) {
        memcpy(&g_.realp[y * fwidth], &g.data[y * g.width], g.width * sizeof(float));
    }
    vDSP_fft2d_zip(fftSetup,
                    &g_, 1, 0,
                    log2n0, log2n1,
                    kFFTDirection_Forward);

    // FG = f_ * g_
    DSPSplitComplex FG = (DSPSplitComplex) {
        .realp = (float *) calloc(fwidth * fheight, sizeof(float)),
        .imagp = (float *) calloc(fwidth * fheight, sizeof(float))
    };
    vDSP_vmul(f_.realp, 1, g_.realp, 1, FG.realp, 1, fwidth * fheight);
    vDSP_vmul(f_.imagp, 1, g_.imagp, 1, FG.imagp, 1, fwidth * fheight);

    // return np.real(np.fft.ifft2(FG))
    vDSP_fft2d_zip(fftSetup,
                    &FG, 1, 0,
                    log2n0, log2n1,
                    kFFTDirection_Inverse);
    image_t ret = (image_t) {
        .width = width,
        .height = height,
        .data = (float *) calloc(width * height, sizeof(float))
    };
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            memcpy(&ret.data[y * width], &FG.realp[y * fwidth], fwidth * sizeof(float));
        }
    }
    imageDivideScalarInPlace(ret, fwidth * fheight); // ifft2 normalization
    return ret;
}

void printImageShape(image_t im) {
    std::cout << im.width << "x" << im.height << std::endl;
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
    image_t divisor = imageSqrt(image);
    for (int y = 0; y < divisor.height; y++) {
        vDSP_vdiv(&outi.data[y * outi.width], 1,
                  &divisor.data[y * divisor.width], 1,
                  &outi.data[y * outi.width], 1,
                  divisor.width);
    }

    // return out
    return outi;
}

image_t toImage(cv::Mat mat) {
    image_t ret = (image_t) {
        .width = mat.cols,
        .height = mat.rows,
        .data = (float *) malloc(mat.cols * mat.rows * sizeof(float))
    };
    for (int y = 0; y < ret.height; y++) {
        for (int x = 0; x < ret.width; x++) {
            int i = ((y * ret.width) + x) * 3;
            uint8 r = mat.data[i];
            uint8 g = mat.data[i + 1];
            uint8 b = mat.data[i + 2];
            ret.data[y * ret.width + x] = (r/255.0)*0.3 + (g/255.0)*0.58 + (b/255.0)*0.11;
        }
    }
    return ret;
}

int main() {
    image_t templ = toImage(cv::imread("template-traffic-lights.png"));
    image_t image = toImage(cv::imread("screen.png"));
    // imageShow("image", &image);

    image_t result = normxcorr2(templ, image);
    int hits = 0;
    for (int y = 0; y < result.height; y++) {
        for (int x = 0; x < result.width; x++) {
            if (result.data[y * result.width + x] > 9e6) {
                hits++;
            }
        }
    }
    std::cout << "hits: " << hits << std::endl;
    imageShow("result", &result);

    cv::waitKey(0);
}
