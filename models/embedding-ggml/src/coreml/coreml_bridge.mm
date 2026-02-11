#if !__has_feature(objc_arc)
#error "This file must be compiled with ARC (-fobjc-arc)"
#endif

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "coreml_bridge.h"

#include <cstring>
#include <cstdio>

struct embedding_coreml_context {
    const void * data;
};

struct embedding_coreml_context * embedding_coreml_init(const char * path_model) {
    NSString * path_str = [[NSString alloc] initWithUTF8String:path_model];
    NSURL * url = [NSURL fileURLWithPath:path_str];

    NSError * error = nil;

    // .mlpackage must be compiled at runtime before loading
    NSURL * compiledURL = [MLModel compileModelAtURL:url error:&error];
    if (error != nil) {
        fprintf(stderr, "[CoreML] Failed to compile model: %s\n",
                [[error localizedDescription] UTF8String]);
        return nullptr;
    }

    MLModelConfiguration * config = [[MLModelConfiguration alloc] init];
    config.computeUnits = MLComputeUnitsAll;

    MLModel * model = [MLModel modelWithContentsOfURL:compiledURL
                                        configuration:config
                                                error:&error];
    if (error != nil) {
        fprintf(stderr, "[CoreML] Failed to load model: %s\n",
                [[error localizedDescription] UTF8String]);
        return nullptr;
    }

    auto * ctx = new embedding_coreml_context;
    ctx->data = CFBridgingRetain(model);

    return ctx;
}

void embedding_coreml_free(struct embedding_coreml_context * ctx) {
    if (ctx) {
        CFRelease(ctx->data);
        delete ctx;
    }
}

void embedding_coreml_encode(
    const struct embedding_coreml_context * ctx,
    int64_t num_frames,
    float * fbank_data,
    float * embedding_out) {

    @autoreleasepool {
        MLModel * model = (__bridge MLModel *)ctx->data;

        NSError * error = nil;

        // Zero-copy MLMultiArray wrapping the fbank data
        // Shape: [1, num_frames, 80] with strides [num_frames*80, 80, 1]
        NSArray<NSNumber *> * shape = @[@1, @(num_frames), @80];
        NSArray<NSNumber *> * strides = @[@(num_frames * 80), @80, @1];

        MLMultiArray * input = [[MLMultiArray alloc]
            initWithDataPointer:fbank_data
                          shape:shape
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:strides
                    deallocator:^(void * _Nonnull bytes) { /* caller owns fbank_data */ }
                          error:&error];

        if (error != nil) {
            fprintf(stderr, "[CoreML] Failed to create input array: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        NSDictionary<NSString *, MLFeatureValue *> * inputDict = @{
            @"fbank_features": [MLFeatureValue featureValueWithMultiArray:input]
        };

        MLDictionaryFeatureProvider * provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict
                                                             error:&error];
        if (error != nil) {
            fprintf(stderr, "[CoreML] Failed to create feature provider: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        id<MLFeatureProvider> result = [model predictionFromFeatures:provider
                                                               error:&error];
        if (error != nil) {
            fprintf(stderr, "[CoreML] Inference failed: %s\n",
                    [[error localizedDescription] UTF8String]);
            return;
        }

        MLMultiArray * output = [[result featureValueForName:@"embedding"] multiArrayValue];
        if (output == nil) {
            fprintf(stderr, "[CoreML] Output 'embedding' not found\n");
            return;
        }

        if (output.dataType == MLMultiArrayDataTypeFloat16) {
            const uint16_t * fp16 = (const uint16_t *)output.dataPointer;
            // Output shape is [1, 256], skip batch dim
            int offset = 0;
            if (output.count > 256) {
                offset = (int)(output.count - 256);
            }
            for (int i = 0; i < 256; i++) {
                uint16_t h = fp16[offset + i];
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
                memcpy(&embedding_out[i], &f, sizeof(float));
            }
        } else {
            // Float32 output — direct copy, handle [1, 256] shape
            const float * src = (const float *)output.dataPointer;
            int offset = 0;
            if (output.count > 256) {
                offset = (int)(output.count - 256);
            }
            memcpy(embedding_out, src + offset, 256 * sizeof(float));
        }
    }
}
