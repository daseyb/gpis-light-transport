cmake_minimum_required(VERSION 3.20)
project(Tungsten)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

# Disable annoying boost warning
set(Boost_NO_WARN_NEW_VERSIONS 1)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# AVX does not do much benefit at the moment, but breaks compilation on some platforms.
# Disabled for now until AVX becomes important enough to reconsider.
SET(EMBREE_MAX_ISA "SSE4.2" CACHE STRING "Selects highest ISA to support.")
set(USE_AVX FALSE CACHE BOOL "Use AVX.")
set(USE_AVX2 FALSE CACHE BOOL "Use AVX2.")

include(OptimizeForArchitecture)
OptimizeForArchitecture()

if (MSVC)
    # Needed by MSVC, but not added by OptimizeForArchitexture()
    add_definitions(-D__SSE__)
endif()

add_definitions(-DINSTALL_PREFIX="${CMAKE_INSTALL_PREFIX}")

if (USE_AVX)
    message(STATUS "Compiling with AVX support")
    set(__AVX__ 1)
elseif (USE_SSE4_2)
    message(STATUS "Compiling with SSE4.2 support")
elseif (USE_SSSE3)
    message(STATUS "Compiling with SSE3 support")
else()
    message(FATAL_ERROR "The target machine does not support SSE3. At least SSE3 is required")
endif()

if (MSVC)
    add_definitions(-DCONSTEXPR=const -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
else()
    add_definitions(-DCONSTEXPR=constexpr)
endif()

if (MSVC)
  add_compile_options(/bigobj)
endif ()

IF(COMMAND cmake_policy)
  if (POLICY CMP0043)
    cmake_policy(SET CMP0043 NEW)
  endif()
ENDIF(COMMAND cmake_policy)

foreach(flag ${Vc_ARCHITECTURE_FLAGS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
endforeach()

set(EMBREE_STATIC_LIB ON CACHE BOOL "Build Embree as a static library." FORCE)
set(EMBREE_ISPC_SUPPORT OFF CACHE BOOL "Build Embree with support for ISPC applications." FORCE)
set(EMBREE_TUTORIALS OFF CACHE BOOL "Enable to build Embree tutorials" FORCE)
set(EMBREE_STAT_COUNTERS OFF CACHE BOOL "Enables statistic counters." FORCE)
set(EMBREE_RAY_MASK OFF CACHE BOOL "Enables ray mask support." FORCE)
set(EMBREE_BACKFACE_CULLING OFF CACHE BOOL "Enables backface culling." FORCE)
set(EMBREE_INTERSECTION_FILTER ON CACHE BOOL "Enables intersection filter callback." FORCE)
set(EMBREE_INTERSECTION_FILTER_RESTORE ON CACHE BOOL "Restores previous hit when hit is filtered out." FORCE)
set(EMBREE_TASKING_SYSTEM "INTERNAL" CACHE STRING "Selects tasking system" FORCE)
set(EMBREE_STATIC_RUNTIME OFF CACHE BOOL "Use the static version of the C/C++ runtime library." FORCE)
add_subdirectory(src/thirdparty/embree)
add_definitions(-DEMBREE_STATIC_LIB=1)

add_definitions(-DRAPIDJSON_HAS_STDSTRING=1)
add_definitions(-DSTBI_NO_STDIO=1)
add_definitions(-DLODEPNG_NO_COMPILE_DISK=1)
add_definitions(-DUSE_IPV6=1)
add_library(thirdparty STATIC
		src/thirdparty/civetweb/civetweb.c
		src/thirdparty/lodepng/lodepng.cpp
		src/thirdparty/sobol/sobol.cpp
		src/thirdparty/stbi/stb_image.c
		src/thirdparty/miniz/miniz.c
		src/thirdparty/skylight/ArHosekSkyModel.cpp
		src/thirdparty/tribox/tribox.cpp)

add_subdirectory(src/thirdparty/eigen)

set(AUTODIFF_BUILD_TESTS OFF)
set(AUTODIFF_BUILD_PYTHON OFF)
set(AUTODIFF_BUILD_EXAMPLES OFF)
set(AUTODIFF_BUILD_DOCS OFF)
add_subdirectory(src/thirdparty/autodiff)

add_subdirectory(src/thirdparty/spectra)


add_subdirectory(src/thirdparty/libigl)


if (CMAKE_COMPILER_IS_GNUCXX)
    set(CXX_WARNINGS "-Wall -Wextra -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs -Wno-misleading-indentation -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough -Wno-class-memaccess -Wno-sign-compare -Wno-unknown-pragmas -Wno-unused-parameter -Wno-reorder -Wno-parentheses -Wno-pessimizing-move")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_WARNINGS} -fvisibility-inlines-hidden")
endif()

find_package(Boost REQUIRED COMPONENTS math)

set(core_libs core thirdparty embree Eigen3::Eigen autodiff::autodiff Spectra igl::core Boost::math)

include_directories(
	src/core src/thirdparty 
	src/thirdparty/embree/include src 
	src/thirdparty/eigen/
	src/thirdparty/autodiff
	src/thirdparty/spectra/include
	src/thirdparty/libigl/include
	src/thirdparty/facet-forge/include
	${FCPW_EIGEN_INCLUDES})

find_package(Imath)
find_package(OpenEXR)
if (OpenEXR_FOUND AND Imath_FOUND)
    message(STATUS "OpenEXR detected. Building with .exr support")
    add_definitions(-DOPENEXR_AVAILABLE -DIMATH_HALF_NO_LOOKUP_TABLE)
    include_directories(${Imath_INCLUDE_DIRS})
    set(core_libs ${core_libs} OpenEXR::OpenEXR Imath::Imath Imath::ImathConfig)
else()
    message(STATUS "No OpenEXR detected. Building without .exr support")
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	set(core_libs ${core_libs} OpenMP::OpenMP_CXX)
    message("OPENMP FOUND")
else()
    message("Missing OPENMP!")
endif()

find_package(JPEG)
if (JPEG_FOUND)
    message(STATUS "JPEG library detected.")
    add_definitions(-DJPEG_AVAILABLE)
    include_directories(${JPEG_INCLUDE_DIR})
    set(core_libs ${core_libs} ${JPEG_LIBRARIES})
else()
    message(STATUS "No JPEG library detected. Falling back to stb_image JPG decoder")
endif()

find_package(TBB REQUIRED)
find_package(OpenVDB)
if (OpenVDB_FOUND AND OpenEXR_FOUND AND TBB_FOUND)
    message(STATUS "OpenVDB detected. Building with .vdb support")
    add_definitions(-DOPENVDB_AVAILABLE)
    add_definitions(-DOPENVDB_3_ABI_COMPATIBLE)
	add_definitions(-DQT_NO_KEYWORDS)
    include_directories(${OpenVDB_INCLUDE_DIRS} ${Tbb_INCLUDE_DIRS})
    set(core_libs ${core_libs} OpenVDB::openvdb TBB::tbb TBB::tbbmalloc)

    if(UNIX)
        set(core_libs ${core_libs} blosc_static)
    endif()
else()
    if (NOT OpenVDB_FOUND)
        message(STATUS "No OpenVDB detected. Building without .vdb support")
    elseif(NOT OpenEXR_FOUND)
        message(STATUS "No OpenEXR libraries detected. Building without .vdb support")
    else()
        message(STATUS "No TBB detected. Building without .vdb support")
    endif()
endif()

find_package(FFTW3 CONFIG)

if(FFTW3_FOUND)
    message(STATUS "FFTW detected. Building with DFT realization generation support")
	include_directories(${FFTW3_INCLUDE_DIRS})
	set(core_libs ${core_libs} FFTW3::fftw3)
	add_definitions(-DFFTW_AVAILABLE)
endif()

find_package(Ceres CONFIG)

if(Ceres_FOUND)
    message(STATUS "Ceres detected. Building with covariance fitting support.")
	include_directories(${Ceres_INCLUDE_DIRS})
	set(core_libs ${core_libs} Ceres::ceres)
	add_definitions(-DCERES_AVAILABLE)
endif()

file(GLOB_RECURSE Core_SOURCES "src/core/*.cpp")
add_library(core STATIC ${Core_SOURCES})

add_library(facet-forge STATIC src/thirdparty/facet-forge/src/random.cpp)
set(core_libs ${core_libs} facet-forge)

add_executable(obj2json src/obj2json/obj2json.cpp)
target_link_libraries(obj2json ${core_libs})

add_executable(json2xml src/json2xml/json2xml.cpp)
target_link_libraries(json2xml ${core_libs})

add_executable(scenemanip src/scenemanip/scenemanip.cpp)
target_link_libraries(scenemanip ${core_libs})

add_executable(hdrmanip src/hdrmanip/hdrmanip.cpp)
target_link_libraries(hdrmanip ${core_libs})

if (EIGEN3_FOUND)
    file(GLOB_RECURSE denoiser_SOURCES "src/denoiser/*.cpp")
    add_executable(denoiser ${denoiser_SOURCES})
    target_link_libraries(denoiser ${core_libs})
endif()

add_executable(tungsten src/tungsten/tungsten.cpp)
target_link_libraries(tungsten ${core_libs})

set(tests 
	derivative-test realization-gen 
	lod-gen microfacet-gen dft-realization-gen
	intersection-test transmittance-test weight-space-test
	phase-function-sampling bake-csg bin-to-vdb nspsr-test kernel-explorer)

foreach (test IN LISTS tests)
	add_executable(tungsten-${test} src/tungsten-test/${test}.cpp)
	target_link_libraries(tungsten-${test} ${core_libs})
	set_target_properties (tungsten-${test} PROPERTIES 
		FOLDER tests
		VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/data")
endforeach ()



if (WIN32)
    set(socket_libs wsock32 Ws2_32)
else()
    set(socket_libs "")
endif()

add_executable(tungsten_server src/tungsten-server/tungsten-server.cpp)
target_link_libraries(tungsten_server ${core_libs} ${socket_libs})

set(executables obj2json json2xml scenemanip hdrmanip tungsten tungsten_server)
if (EIGEN3_FOUND)
    set(executables ${executables} denoiser)
endif()
set(data_dirs example-scenes materialtest mc-loader)


find_package(OpenGL)
find_package(Qt5 COMPONENTS Widgets OpenGL)
option(BUILD_EDITOR "Build the editor" FALSE) 
if (BUILD_EDITOR AND OPENGL_FOUND AND Qt5Widgets_FOUND)
	set(CMAKE_AUTOMOC ON)
	set(CMAKE_AUTORCC ON)
	set(CMAKE_AUTOUIC ON)
	set(CMAKE_INCLUDE_CURRENT_DIR ON)

    include_directories(${OPENGL_INCLUDE_DIR})
    set(opengl_libs ${OPENGL_LIBRARIES})
	
    file(GLOB_RECURSE Editor_SOURCES "src/editor/*.*pp")
    if (APPLE)
        list(APPEND Editor_SOURCES "src/editor/CoreProfileAttributes.mm")
    endif()
    add_executable(editor ${Editor_SOURCES} src/editor/resources/Tungsten.rc)


    set_target_properties(editor PROPERTIES AUTOMOC TRUE)
    target_link_libraries(editor ${core_libs} ${opengl_libs} Qt5::Core Qt5::Widgets Qt5::OpenGL)
    if (APPLE)
        find_library(APPKIT NAMES AppKit) 
        target_link_libraries(editor ${APPKIT})
    endif()
    message(STATUS "Building editor")
    
    set(data_dirs ${data_dirs} editor shaders)
    set(executables ${executables} editor)
else()
    if (NOT Qt5Widgets_FOUND)
        message(STATUS "Qt5 not found. Editor will not be built")
    endif()
    if (NOT OPENGL_FOUND)
        message(STATUS "OpenGL not found. Editor will not be built")
    endif()
	if (NOT BUILD_EDITOR)
        message(STATUS "Editor build disabled. Editor will not be built")
    endif()
endif()

if (WIN32)
    set(data_prefix "data")
else()
    set(data_prefix "share/tungsten")
endif()

foreach (data_dir ${data_dirs})
   install(DIRECTORY data/${data_dir} DESTINATION ${data_prefix})
   file(GLOB_RECURSE data_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/data "data/${data_dir}/*")
   foreach (data ${data_files})
       configure_file(data/${data} ${CMAKE_CURRENT_BINARY_DIR}/${data_prefix}/${data} COPYONLY)
   endforeach(data)
endforeach()

if (WIN32)
    install(TARGETS ${executables} DESTINATION .)
else()
    install(TARGETS ${executables} DESTINATION bin)
endif()
