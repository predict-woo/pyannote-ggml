cmake_minimum_required(VERSION 3.14)
project(diarization-ggml CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EMBEDDING_COREML "Enable CoreML acceleration" OFF)
option(SEGMENTATION_COREML "Enable CoreML acceleration for segmentation" OFF)

# Add GGML (guarded — may already be added by sub-projects)
if(NOT TARGET ggml)
    add_subdirectory(../ggml ggml)
endif()

# kaldi-native-fbank (guarded, suppress tests/python)
if(NOT TARGET kaldi-native-fbank-core)
    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    add_subdirectory(../embedding-ggml/kaldi-native-fbank kaldi-native-fbank)
endif()

# Add segmentation-core library
add_subdirectory(../segmentation-ggml segmentation-ggml)

# Add embedding-core library
add_subdirectory(../embedding-ggml embedding-ggml)

# Diarization sources
add_executable(diarization-ggml
    src/main.cpp
    src/diarization.cpp
    src/powerset.cpp
    src/aggregation.cpp
    src/plda.cpp
    src/vbx.cpp
    src/clustering.cpp
    src/fastcluster/fastcluster.cpp
    src/rttm.cpp
)

target_include_directories(diarization-ggml PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(diarization-ggml PRIVATE
    segmentation-core
    embedding-core
)

# Accelerate framework for PLDA/VBx linear algebra
if(APPLE)
    target_link_libraries(diarization-ggml PRIVATE "-framework Accelerate")
    target_compile_definitions(diarization-ggml PRIVATE ACCELERATE_NEW_LAPACK)
endif()

# CoreML support
if(EMBEDDING_COREML)
    target_link_libraries(diarization-ggml PRIVATE embedding-coreml)
    target_compile_definitions(diarization-ggml PRIVATE EMBEDDING_USE_COREML)
endif()

if(SEGMENTATION_COREML)
    target_link_libraries(diarization-ggml PRIVATE segmentation-coreml)
    target_compile_definitions(diarization-ggml PRIVATE SEGMENTATION_USE_COREML)
endif()

set_target_properties(diarization-ggml PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "diarization-ggml project configured successfully")
