cmake_minimum_required(VERSION 3.14)
project(pyannote-addon CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cmake-js provides these variables
include_directories(${CMAKE_JS_INC})

# node-addon-api include path
execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Remove quotes from the path
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
include_directories(${NODE_ADDON_API_DIR})

# NAPI version
add_definitions(-DNAPI_VERSION=7)
add_definitions(-DNAPI_DISABLE_CPP_EXCEPTIONS)

# Path to the diarization-ggml project root (4 levels up from this CMakeLists.txt)
# bindings/node/packages/darwin-arm64/ → project root
set(DIARIZATION_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
set(DIARIZATION_BUILD "${DIARIZATION_ROOT}/diarization-ggml/build-static")

# Source files
file(GLOB ADDON_SOURCES "src/*.cpp")

# Create the addon shared library
add_library(${PROJECT_NAME} SHARED ${ADDON_SOURCES} ${CMAKE_JS_SRC})

# Include directories for diarization headers
target_include_directories(${PROJECT_NAME} PRIVATE
    ${DIARIZATION_ROOT}/diarization-ggml/include
    ${DIARIZATION_ROOT}/diarization-ggml/src
    ${DIARIZATION_ROOT}/models/segmentation-ggml/src
    ${DIARIZATION_ROOT}/models/embedding-ggml/src
    ${DIARIZATION_ROOT}/whisper.cpp/ggml/include
    ${DIARIZATION_ROOT}/whisper.cpp/include
    ${DIARIZATION_ROOT}/whisper.cpp/src
    ${DIARIZATION_BUILD}/kaldi-native-fbank/include
)

# Link static libraries from build-static
target_link_libraries(${PROJECT_NAME} PRIVATE
    # Integrated transcription+diarization pipeline
    ${DIARIZATION_BUILD}/libpipeline-lib.a

    # Diarization pipeline
    ${DIARIZATION_BUILD}/libdiarization-lib.a
    
    # Whisper
    ${DIARIZATION_BUILD}/whisper.cpp/src/libwhisper.a
    ${DIARIZATION_BUILD}/whisper.cpp/src/libwhisper.coreml.a

    # Segmentation
    ${DIARIZATION_BUILD}/segmentation-ggml/libsegmentation-core.a
    ${DIARIZATION_BUILD}/segmentation-ggml/libsegmentation-coreml.a
    
    # Embedding
    ${DIARIZATION_BUILD}/embedding-ggml/libembedding-core.a
    ${DIARIZATION_BUILD}/embedding-ggml/libembedding-coreml.a
    
    # GGML (provided by whisper.cpp)
    ${DIARIZATION_BUILD}/whisper.cpp/ggml/src/libggml.a
    ${DIARIZATION_BUILD}/whisper.cpp/ggml/src/libggml-base.a
    ${DIARIZATION_BUILD}/whisper.cpp/ggml/src/libggml-cpu.a
    ${DIARIZATION_BUILD}/whisper.cpp/ggml/src/ggml-metal/libggml-metal.a
    ${DIARIZATION_BUILD}/whisper.cpp/ggml/src/ggml-blas/libggml-blas.a
    
    # kaldi-native-fbank
    ${DIARIZATION_BUILD}/lib/libkaldi-native-fbank-core.a
    ${DIARIZATION_BUILD}/lib/libkissfft-float.a
    
    # cmake-js
    ${CMAKE_JS_LIB}
)

# Apple frameworks
target_link_libraries(${PROJECT_NAME} PRIVATE
    "-framework Accelerate"
    "-framework Foundation"
    "-framework CoreML"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
)

# Compile definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    EMBEDDING_USE_COREML
    SEGMENTATION_USE_COREML
    WHISPER_USE_COREML
    ACCELERATE_NEW_LAPACK
)

# ObjC++ flags for CoreML bridge code (if any .mm files are included)
# The static libs already contain compiled ObjC++, so this is mainly for safety

# Output as .node file
set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)
