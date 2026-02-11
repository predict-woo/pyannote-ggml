#if !__has_feature(objc_arc)
#error "This file must be compiled with ARC (-fobjc-arc)"
#endif

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "segmentation_coreml_bridge.h"

#include <cstring>
#include <cstdio>

struct segmentation_coreml_context {
    const void * data;
};

struct segmentation_coreml_context * segmentation_coreml_init(const char * path_model) {
    NSString * path_str = [[NSString alloc] initWithUTF8String:path_model];
    NSURL * url = [NSURL fileURLWithPath:path_str];

    NSError * error = nil;

    NSURL * compiledURL = [MLModel compileModelAtURL:url error:&error];
    if (error != nil) {
        fprintf(stderr, "[CoreML-Seg] Failed to compile model: %s\n",
                [[error localizedDescription] UTF8String]);
        return nullptr;
    }

    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsAll;

    MLModel * model = [MLModel modelWithContentsOfURL:compiledURL
                                        configuration:config
                                                error:&error];
    if (error != nil) {
        fprintf(stderr, "[CoreML-Seg] Failed to load model: %s\n",
                [[error localizedDescription] UTF8String]);
        return nullptr;
    }

    auto * ctx = new segmentation_coreml_context;
    ctx->data = CFBridgingRetain(model);

    return ctx;
}

void segmentation_coreml_free(struct segmentation_coreml_context * ctx) {
    if (ctx) {
        CFRelease(ctx->data);
        delete ctx;
    }
}

void segmentation_coreml_infer(
    const struct segmentation_coreml_context * ctx,
    float * audio_data,
    int32_t n_samples,
    float * output,
    int32_t output_size) {

    @autoreleasepool {
        MLModel * model = (__bridge MLModel *)ctx->data;

        NSError * error = nil;

        // Input shape: [1, 1, 160000] with strides [160000, 160000, 1]
        NSArray<NSNumber *> * shape = @[@1, @1, @(n_samples)];
        NSArray<NSNumber *> * strides = @[@(n_samples), @(n_samples), @1];

        MLMultiArray * input = [[MLMultiArray alloc]
            initWithDataPointer:audio_data
                          shape:shape
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:strides
                    deallocator:^(void * _Nonnull bytes) { /* caller owns audio_data */ }
                          error:&error];

        if (error != nil) {
            fprintf(stderr, "[CoreML-Seg] Failed to create input array: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        NSDictionary<NSString *, MLFeatureValue *> * inputDict = @{
            @"waveform": [MLFeatureValue featureValueWithMultiArray:input]
        };

        MLDictionaryFeatureProvider * provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict
                                                             error:&error];
        if (error != nil) {
            fprintf(stderr, "[CoreML-Seg] Failed to create feature provider: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        id<MLFeatureProvider> result = [model predictionFromFeatures:provider
                                                               error:&error];
        if (error != nil) {
            fprintf(stderr, "[CoreML-Seg] Inference failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        MLMultiArray * out_array = [[result featureValueForName:@"log_probabilities"] multiArrayValue];
        if (out_array == nil) {
            fprintf(stderr, "[CoreML-Seg] Output 'log_probabilities' not found\n");
            return;
        }

        const int total = output_size;

        if (out_array.dataType == MLMultiArrayDataTypeFloat16) {
            const uint16_t * fp16 = (const uint16_t *)out_array.dataPointer;
            for (int i = 0; i < total; i++) {
                uint16_t h = fp16[i];
                uint32_t sign = (h >> 15) & 0x1;
                uint32_t exp  = (h >> 10) & 0x1f;
                uint32_t mant = h & 0x3ff;
                uint32_t f;
                if (exp == 0) {
                    if (mant == 0) {
                        f = sign << 31;
                    } else {
                        exp = 1;
                        while (!(mant & 0x400)) { mant <<= 1; exp--; }
                        mant &= 0x3ff;
                        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    f = (sign << 31) | 0x7f800000 | (mant << 13);
                } else {
                    f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                memcpy(&output[i], &f, sizeof(float));
            }
        } else {
            const float * src = (const float *)out_array.dataPointer;
            memcpy(output, src, total * sizeof(float));
        }
    }
}
