cmake_minimum_required(VERSION 3.14)
project(pyannote-addon CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
    add_compile_options(/utf-8)
endif()

include_directories(${CMAKE_JS_INC})

execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
include_directories(${NODE_ADDON_API_DIR})

add_definitions(-DNAPI_VERSION=7)
add_definitions(-DNAPI_DISABLE_CPP_EXCEPTIONS)

option(ADDON_VULKAN "Enable Vulkan GPU support" ON)
option(ADDON_OPENVINO "Enable OpenVINO encoder support" ON)

set(NODE_BINDINGS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(PROJECT_ROOT "${NODE_BINDINGS_ROOT}/../..")
set(WHISPER_ROOT "${PROJECT_ROOT}/whisper.cpp")

if(ADDON_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan" FORCE)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static libraries" FORCE)
add_subdirectory(${WHISPER_ROOT}/ggml ${CMAKE_CURRENT_BINARY_DIR}/ggml)

set(WHISPER_OPENVINO_ENABLED OFF)
if(ADDON_OPENVINO)
    find_package(OpenVINO COMPONENTS Runtime QUIET)
    if(OpenVINO_FOUND)
        set(WHISPER_OPENVINO_ENABLED ON)
    endif()
endif()

add_library(whisper_static STATIC ${WHISPER_ROOT}/src/whisper.cpp)
target_include_directories(whisper_static PUBLIC
    ${WHISPER_ROOT}/include
    ${WHISPER_ROOT}/src
)
target_compile_definitions(whisper_static PRIVATE WHISPER_VERSION="1.0.0")
target_link_libraries(whisper_static PUBLIC ggml)

if(WHISPER_OPENVINO_ENABLED)
    add_library(whisper_openvino STATIC
        ${WHISPER_ROOT}/src/openvino/whisper-openvino-encoder.cpp
    )
    target_include_directories(whisper_openvino PUBLIC
        ${WHISPER_ROOT}/src
        ${WHISPER_ROOT}/src/openvino
        ${WHISPER_ROOT}/ggml/include
    )
    target_link_libraries(whisper_openvino PRIVATE openvino::runtime)
    target_compile_definitions(whisper_static PRIVATE WHISPER_USE_OPENVINO)
    target_link_libraries(whisper_static PRIVATE whisper_openvino)
endif()

find_package(Threads REQUIRED)
target_link_libraries(whisper_static PUBLIC Threads::Threads)

add_library(common_static STATIC
    ${WHISPER_ROOT}/examples/common.cpp
    ${WHISPER_ROOT}/examples/common-ggml.cpp
    ${WHISPER_ROOT}/examples/common-whisper.cpp
    ${WHISPER_ROOT}/examples/grammar-parser.cpp
)
target_include_directories(common_static PUBLIC
    ${WHISPER_ROOT}/examples
    ${WHISPER_ROOT}/include
)
target_link_libraries(common_static PRIVATE whisper_static)
set_target_properties(common_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(${PROJECT_NAME} SHARED
    ${CMAKE_JS_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/addon.cpp
    ${NODE_BINDINGS_ROOT}/native/whisper/whisper_addon_wrapper.cpp
    ${NODE_BINDINGS_ROOT}/native/pipeline/win_pipeline_stubs.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${NODE_BINDINGS_ROOT}/native/whisper
    ${NODE_BINDINGS_ROOT}/native/pipeline
    ${WHISPER_ROOT}/examples
    ${WHISPER_ROOT}/include
    ${WHISPER_ROOT}/src
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${CMAKE_JS_LIB}
    common_static
    whisper_static
    Threads::Threads
)

if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF} /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
endif()

if(MSVC AND ADDON_VULKAN)
    target_link_options(${PROJECT_NAME} PRIVATE "/DELAYLOAD:vulkan-1.dll")
    target_link_libraries(${PROJECT_NAME} PRIVATE delayimp.lib)
endif()

if(MSVC AND WHISPER_OPENVINO_ENABLED)
    target_link_options(${PROJECT_NAME} PRIVATE
        "/DELAYLOAD:openvino.dll"
        "/DELAYLOAD:openvino_ir_frontend.dll"
        "/DELAYLOAD:tbb12.dll"
        "/DELAYLOAD:pugixml.dll"
    )
    target_link_libraries(${PROJECT_NAME} PRIVATE delayimp.lib)
endif()

if(WIN32 AND WHISPER_OPENVINO_ENABLED)
    get_target_property(OPENVINO_RUNTIME_LIB openvino::runtime IMPORTED_LOCATION_RELEASE)
    if(NOT OPENVINO_RUNTIME_LIB)
        get_target_property(OPENVINO_RUNTIME_LIB openvino::runtime IMPORTED_LOCATION)
    endif()

    if(OPENVINO_RUNTIME_LIB)
        get_filename_component(OPENVINO_LIB_DIR "${OPENVINO_RUNTIME_LIB}" DIRECTORY)

        set(OPENVINO_CORE_DLLS
            openvino.dll
            openvino_ir_frontend.dll
            tbb12.dll
            pugixml.dll
        )

        set(OPENVINO_PLUGIN_DLLS
            openvino_intel_cpu_plugin.dll
            openvino_intel_gpu_plugin.dll
            openvino_auto_plugin.dll
        )

        foreach(dll ${OPENVINO_CORE_DLLS})
            set(dll_path "${OPENVINO_LIB_DIR}/${dll}")
            if(EXISTS "${dll_path}")
                add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll_path}"
                        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                )
            endif()
        endforeach()

        foreach(dll ${OPENVINO_PLUGIN_DLLS})
            set(dll_found FALSE)
            set(dll_path "${OPENVINO_LIB_DIR}/${dll}")

            if(EXISTS "${dll_path}")
                set(dll_found TRUE)
            else()
                file(GLOB OPENVINO_SUBDIRS "${OPENVINO_LIB_DIR}/openvino-*")
                foreach(subdir ${OPENVINO_SUBDIRS})
                    if(IS_DIRECTORY "${subdir}")
                        set(subdir_dll_path "${subdir}/${dll}")
                        if(EXISTS "${subdir_dll_path}")
                            set(dll_path "${subdir_dll_path}")
                            set(dll_found TRUE)
                            break()
                        endif()
                    endif()
                endforeach()
            endif()

            if(dll_found)
                add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dll_path}"
                        "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                )
            endif()
        endforeach()
    endif()
endif()

if(MSVC)
    get_filename_component(MSVC_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(MSVC_TOOLS_DIR "${MSVC_COMPILER_DIR}/../../.." ABSOLUTE)

    set(MSVC_REDIST_ARCH "x64")
    set(MSVC_CRT_DLLS
        msvcp140.dll
        vcruntime140.dll
        vcruntime140_1.dll
    )

    file(GLOB MSVC_REDIST_DIRS "${MSVC_TOOLS_DIR}/Redist/MSVC/*/")
    foreach(redist_dir ${MSVC_REDIST_DIRS})
        set(MSVC_CRT_DIR "${redist_dir}${MSVC_REDIST_ARCH}/Microsoft.VC143.CRT")
        if(IS_DIRECTORY "${MSVC_CRT_DIR}")
            foreach(dll ${MSVC_CRT_DLLS})
                set(dll_path "${MSVC_CRT_DIR}/${dll}")
                if(EXISTS "${dll_path}")
                    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${dll_path}"
                            "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
                    )
                endif()
            endforeach()
            break()
        endif()
    endforeach()
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".node"
)
