cmake_minimum_required(VERSION 3.14)
project(segmentation-ggml CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add GGML as subdirectory (guarded for multi-project builds)
if(NOT TARGET ggml)
    add_subdirectory(../../ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)
endif()

# Optional Metal backend support (enabled by default on macOS)
option(GGML_METAL "Enable Metal backend for GPU acceleration" ON)

# Optional CoreML backend for segmentation
option(SEGMENTATION_COREML "Enable CoreML acceleration for segmentation" OFF)

# --- Core library (model/inference code, reusable by other projects) ---
add_library(segmentation-core STATIC
    src/model.cpp
    src/sincnet.cpp
    src/lstm.cpp
)

target_include_directories(segmentation-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ../../ggml/include
)

target_link_libraries(segmentation-core PUBLIC ggml)

if(APPLE)
    target_compile_definitions(segmentation-core PUBLIC ACCELERATE_NEW_LAPACK)
    target_link_libraries(segmentation-core PUBLIC "-framework Accelerate")
endif()

if(GGML_METAL)
    target_compile_definitions(segmentation-core PUBLIC GGML_USE_METAL)
    message(STATUS "Metal backend enabled for segmentation-ggml")
else()
    message(STATUS "Metal backend disabled for segmentation-ggml")
endif()

# --- Executable ---
add_executable(segmentation-ggml src/main.cpp)
target_link_libraries(segmentation-ggml PRIVATE segmentation-core)

# Set output directory for executable
set_target_properties(segmentation-ggml PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# --- Optional CoreML bridge (separate library) ---
if(SEGMENTATION_COREML)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(COREML_FRAMEWORK CoreML REQUIRED)

    add_library(segmentation-coreml
        src/coreml/segmentation_coreml_bridge.h
        src/coreml/segmentation_coreml_bridge.mm
    )
    target_include_directories(segmentation-coreml PUBLIC src/coreml)
    target_link_libraries(segmentation-coreml PRIVATE
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
    )
    set_target_properties(segmentation-coreml PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )

    target_link_libraries(segmentation-ggml PRIVATE segmentation-coreml)
    target_compile_definitions(segmentation-ggml PRIVATE SEGMENTATION_USE_COREML)
    message(STATUS "CoreML acceleration enabled for segmentation")
endif()

message(STATUS "segmentation-ggml project configured successfully")
