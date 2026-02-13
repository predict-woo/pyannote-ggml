cmake_minimum_required(VERSION 3.14)
project(diarization-ggml CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(EMBEDDING_COREML "Enable CoreML acceleration" OFF)
option(SEGMENTATION_COREML "Enable CoreML acceleration for segmentation" OFF)
option(WHISPER_COREML "Enable CoreML for Whisper transcription" OFF)
option(DIARIZATION_STATIC "Build all dependencies as static libraries" OFF)

# Whisper.cpp integration (adds ggml as part of its build, so ggml guard below will skip)
if(NOT TARGET whisper)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    add_subdirectory(../whisper.cpp ${CMAKE_CURRENT_BINARY_DIR}/whisper.cpp)
endif()

# Add GGML (guarded — skipped when whisper.cpp already provides it)
if(NOT TARGET ggml)
    if(DIARIZATION_STATIC)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(../ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)
endif()

# kaldi-native-fbank (guarded, suppress tests/python)
if(NOT TARGET kaldi-native-fbank-core)
    set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    if(DIARIZATION_STATIC)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(../models/embedding-ggml/kaldi-native-fbank ${CMAKE_CURRENT_BINARY_DIR}/kaldi-native-fbank)
endif()

# Add segmentation-core library
add_subdirectory(../models/segmentation-ggml ${CMAKE_CURRENT_BINARY_DIR}/segmentation-ggml)

# Add embedding-core library
add_subdirectory(../models/embedding-ggml ${CMAKE_CURRENT_BINARY_DIR}/embedding-ggml)

# Diarization library (shared code)
add_library(diarization-lib STATIC
    src/diarization.cpp
    src/powerset.cpp
    src/aggregation.cpp
    src/plda.cpp
    src/vbx.cpp
    src/clustering.cpp
    src/fastcluster/fastcluster.cpp
    src/rttm.cpp
    src/streaming.cpp
    src/provisional.cpp
)

target_include_directories(diarization-lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(diarization-lib PUBLIC
    segmentation-core
    embedding-core
)

# Accelerate framework for PLDA/VBx linear algebra
if(APPLE)
    target_link_libraries(diarization-lib PUBLIC "-framework Accelerate")
    target_compile_definitions(diarization-lib PUBLIC ACCELERATE_NEW_LAPACK)
endif()

# CoreML support
if(EMBEDDING_COREML)
    target_link_libraries(diarization-lib PUBLIC embedding-coreml)
    target_compile_definitions(diarization-lib PUBLIC EMBEDDING_USE_COREML)
endif()

if(SEGMENTATION_COREML)
    target_link_libraries(diarization-lib PUBLIC segmentation-coreml)
    target_compile_definitions(diarization-lib PUBLIC SEGMENTATION_USE_COREML)
endif()

# Main executable
add_executable(diarization-ggml src/main.cpp)
target_link_libraries(diarization-ggml PRIVATE diarization-lib)
set_target_properties(diarization-ggml PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Streaming test executable
add_executable(streaming_test tests/test_streaming.cpp)
target_link_libraries(streaming_test PRIVATE diarization-lib)
set_target_properties(streaming_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(test_segment_detector tests/test_segment_detector.cpp src/segment_detector.cpp)
target_link_libraries(test_segment_detector PRIVATE diarization-lib)
set_target_properties(test_segment_detector PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(test_aligner tests/test_aligner.cpp src/aligner.cpp)
target_link_libraries(test_aligner PRIVATE diarization-lib)
target_include_directories(test_aligner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
set_target_properties(test_aligner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Transcription library (combines diarization + whisper)
add_library(transcription-lib INTERFACE)
target_link_libraries(transcription-lib INTERFACE diarization-lib whisper)

# Transcription CLI (placeholder — main_transcribe.cpp will be added in Task 8)
# add_executable(transcribe src/main_transcribe.cpp)
# target_link_libraries(transcribe PRIVATE transcription-lib)

# Integration test targets (sources added by later tasks)
add_executable(test_silence_filter tests/test_silence_filter.cpp src/silence_filter.cpp)
target_link_libraries(test_silence_filter PRIVATE transcription-lib)
set_target_properties(test_silence_filter PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
add_executable(test_audio_buffer tests/test_audio_buffer.cpp src/audio_buffer.cpp)
target_include_directories(test_audio_buffer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(test_audio_buffer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(test_transcriber tests/test_transcriber.cpp src/transcriber.cpp)
target_link_libraries(test_transcriber PRIVATE transcription-lib)
target_include_directories(test_transcriber PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(test_transcriber PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "diarization-ggml project configured successfully")
