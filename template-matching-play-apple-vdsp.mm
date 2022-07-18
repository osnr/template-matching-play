#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

void fail(const char* s) {
    fprintf(stderr, "fail: %s", s);
    exit(1);
}

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
void imagePrint(const char* name, const image_t im) {
    printf("%s [%dx%d]:\n", name, im.width, im.height);
    int x0 = 0; int x1 = im.width;
    int y0 = 0; int y1 = im.height;
    for (int y = y0; y < y0 + 5; y++) {
        for (int x = x0; x < x0 + 5; x++) {
            printf("%05.02f(%d)\t", im.data[y*im.width+x], x);
        }
        printf("...\t");
        for (int x = x1 - 5; x < x1; x++) {
            printf("%05.02f(%d)\t", im.data[y*im.width+x], x);
        }
        printf("\n");
    }
    printf("...\n");
    for (int y = y1 - 5; y < y1; y++) {
        for (int x = x0; x < x0 + 100; x++) {
            printf("%05.02f(%d)\t", im.data[y*im.width+x], x);
        }
        printf("...\t");
        for (int x = x1 - 5; x < x1; x++) {
            printf("%05.02f(%d)\t", im.data[y*im.width+x], x);
        }
        printf("\n");
    }
    printf("\n");
}
void imageShow(const cv::String &winname, const image_t im_) {
    image_t *im = (image_t *) malloc(sizeof(im));
    memcpy(im, &im_, sizeof(im_));
    cv::Mat imageMat(im->height, im->width, CV_32F, im->data);
    cv::imshow(cv::format("%s (%dx%d)", winname.c_str(), im->width, im->height), imageMat);
    cv::setMouseCallback(winname, onMouse, (void *) im);
    // std::cout << winname << std::endl;
    // for (int i = 0; i < 100; i++) {
    //     std::cout << im.data[i] << std::endl;
    // }
}
image_t dspToImage(const DSPSplitComplex spl, int width, int height) {
    image_t im;
    im.width = width;
    im.height = height;
    im.data = spl.realp;
    return im;
}
void dspImageShow(const cv::String &winname, const DSPSplitComplex spl, int width, int height) {
    imageShow(winname, dspToImage(spl, width, height));
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
image_t imageDivideScalar(image_t im, const float scalar) {
    image_t ret = imageNewInShapeOf(im);
    memcpy(ret.data, im.data, ret.width * ret.height * sizeof(float));
    imageDivideScalarInPlace(ret, scalar);
    return ret;
}

void imageSubtractImageInPlace(image_t im, const image_t subtrahend) {
    vDSP_vsub(subtrahend.data, 1, im.data, 1, im.data, 1, im.width * im.height);
}
void imageDivideImageInPlace(image_t im, const image_t divisor) {
    if (divisor.width != im.width || divisor.height != im.height) {
        fail("border mismatch");
    }
    vDSP_vdiv(divisor.data, 1, im.data, 1, im.data, 1, im.width * im.height);
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

image_t fftconvolve(const image_t f, const image_t g) {
    // size = np.array(f.shape) + np.array(g.shape) - 1
    int width = f.width + g.width - 1;
    int height = f.height + g.height - 1;

    // fsize = 2 ** np.ceil(np.log2(size)).astype(int)
    int log2n0 = ceil(log2f(width));
    int fwidth = 1 << (int) log2n0;
    int log2n1 = ceil(log2f(height));
    int fheight = 1 << (int) log2n1;

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
                   kFFTDirection_Forward);

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
    vDSP_zvmul(&f_, 1, &g_, 1, &FG, 1, fwidth * fheight, 1);

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
    vDSP_mmov(FG.realp, ret.data, width, height, fwidth, width);
    imageDivideScalarInPlace(ret, fwidth * fheight); // ifft2 normalization
    return ret;
}

void printImageShape(image_t im) {
    std::cout << im.width << "x" << im.height << std::endl;
}

#define s_(x, y) (((x) < 0 || (y) < 0 || (x >= image.width) || (y >= image.height)) ? 0 : s[((y)*image.width) + (x)])
#define s2_(x, y) (((x) < 0 || (y) < 0 || (x >= image.width) || (y >= image.height)) ? 0 : s2[((y)*image.width) + (x)])

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
    // summed-area tables
    double *s = (double*) calloc(image.width * image.height, sizeof(double));
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            s[y*image.width + x] = image.data[y*image.width + x] + s_(x-1, y) + s_(x, y-1) - s_(x-1, y-1);
        }
    }
    double *s2 = (double*) calloc(image.width * image.height, sizeof(double));
    for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
            s2[y*image.width + x] = image.data[y*image.width + x]*image.data[y*image.width + x] + s2_(x-1, y) + s2_(x, y-1) - s2_(x-1, y-1);
        }
    }

    // image[np.where(image < 0)] = 0
    // template = np.sum(np.square(template))    
    // out = out / np.sqrt(image * template)
    float templateSum = imageSumOfSquares(templ);
    image_t denom = imageNewInShapeOf(outi);    
    for (int y = 0; y < denom.height; y++) {
        for (int x = 0; x < denom.width; x++) {
            double imageSum = s_(x, y)
                - s_(x - templ.width, y)
                - s_(x, y - templ.height)
                + s_(x - templ.width, y - templ.height);

            double energy = s2_(x, y)
                - s2_(x - templ.width, y)
                - s2_(x, y - templ.height)
                + s2_(x - templ.width, y - templ.height);

            float d = energy - 1.0f/(templ.width * templ.height) * imageSum * imageSum; // from Briechle (10)
            if (d < 0) d = 0;
            denom.data[y*denom.width + x] = sqrt(d * templateSum);
        }
    }

    image_t denomOld;
    {
        image_t imagen = fftconvolve(imageSquare(image), a1);
        image_t subtrahend = imageSquare(fftconvolve(image, a1));
        imageDivideScalarInPlace(subtrahend, templ.width * templ.height);
        imageSubtractImageInPlace(imagen, subtrahend);

        // image[np.where(image < 0)] = 0
        for (int y = 0; y < imagen.height; y++) {
            for (int x = 0; x < imagen.width; x++) {
                int i = y * imagen.width + x;
                if (imagen.data[i] < 0) {
                    imagen.data[i] = 0;
                }
            }
        }

        imageMultiplyScalarInPlace(imagen, templateSum);
        denomOld = imageSqrt(imagen);
    }

    imageShow("denom", imageDivideScalar(denom, imageMean(denom)));
    imageShow("denomOld", imageDivideScalar(denomOld, imageMean(denomOld)));
    imageShow("outi", outi);
    imageDivideImageInPlace(outi, denom);

    // out[np.where(np.logical_not(np.isfinite(out)))] = 0
    for (int y = 0; y < outi.height; y++) {
        for (int x = 0; x < outi.width; x++) {
            int i = y * outi.width + x;
            if (isinf(outi.data[i]) || isnan(outi.data[i])) {
                outi.data[i] = 0.0f;
            }
        }
    }
    
    // return out
    return outi;
}

image_t toImage(cv::Mat matOrig) {
    cv::Mat mat;
    cv::resize(matOrig, mat, cv::Size(), 0.5, 0.5);
    
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

// int main() {
//     float inputData[] = {
//         1, 2, 3,
//         4, 5, 6,
//         7, 8, 9
//     };
//     image_t input = (image_t) { .data = inputData, .width = 3, .height = 3 };
//     float kernelData[] = {
//         -1, -2, -1,
//         0, 0, 0,
//         1, 2, 1
//     };
//     image_t kernel = (image_t) { .data = kernelData, .width = 3, .height = 3 };

//     image_t output = fftconvolve(input, kernel);
//     imagePrint("output", output);
// }

void hit(cv::Mat& orig, image_t templ, int x, int y) {
    cv::Point origin((x - templ.width)*2, (y - templ.height)*2);
    cv::Point to((x - templ.width + templ.width)*2, (y - templ.height + templ.height)*2);
    cv::rectangle(orig, origin, to, cv::Scalar(255, 0, 255));
}

int main() {
    image_t templ = toImage(cv::imread("template-traffic-lights.png"));
    image_t image = toImage(cv::imread("screen.png"));

    cv::TickMeter tm;
    tm.start();
    image_t result = normxcorr2(templ, image);
    tm.stop();
    std::cout << "Total time: " << tm.getTimeSec() << std::endl;

    cv::Mat orig = cv::imread("screen.png");
    
    if (0) { // max-value strategy
        int maxX, maxY;
        float maxValue = -10000.0f;
        for (int y = 0; y < result.height; y++) {
            for (int x = 0; x < result.width; x++) {
                if (result.data[y * result.width + x] > maxValue) {
                    maxX = x;
                    maxY = y;
                    maxValue = result.data[y * result.width + x];
                }
            }
        }
        printf("maxValue (%d, %d) = %f\n", maxX, maxY, maxValue);
        hit(orig, templ, maxX, maxY);
    }
    imageShow("result", result);

    if (1) { // threshold strategy
        int hits = 0;
        for (int y = 0; y < result.height; y++) {
            for (int x = 0; x < result.width; x++) {
                if (result.data[y * result.width + x] > 0.98) {
                    hits++;
                    hit(orig, templ, x, y);
                }
            }
        }

        std::cout << "hits: " << hits << std::endl;
    }

    cv::imshow("orig", orig);
    while (cv::waitKey(0) != 27) {}

    return 0;
}
