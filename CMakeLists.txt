cmake_minimum_required(VERSION 3.24...3.30)

# Sanity check, forgetting to clone submodules is a common omission and results in a poor error message
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tt_metal/third_party/umd/CMakeLists.txt")
    message(FATAL_ERROR "Missing submodules.  Run: git submodule update --init --recursive")
endif()

############################################
# Project setup
############################################

# For single-config generators, default to RelWithDebInfo if unspecified
get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT isMultiConfig)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type")
endif()

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(version)
ParseGitDescribe()
project(
    Metalium
    VERSION ${VERSION_NUMERIC}
    DESCRIPTION "Tenstorrent Metalium"
    HOMEPAGE_URL "https://github.com/tenstorrent/tt-metal"
    LANGUAGES
        C # Some of the jit-build files are plain C
        CXX
)
message(STATUS "Metalium version: ${PROJECT_VERSION}")
message(STATUS "Building Unified Library for all architectures, thanks to blozano-tt")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# FIXME: Why are we setting these ourselves?
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DDEBUG")
# Defining build types is the pervue of the top-level project
if(PROJECT_IS_TOP_LEVEL)
    include(sanitizers)
endif()

if(${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR})
    message(
        FATAL_ERROR
        "CMake generation is not allowed within source directory!! Please set a build folder with '-B'!!"
    )
endif()

if(DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE)
    message(STATUS "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(STATUS "CMAKE_TOOLCHAIN_FILE is not set.")
endif()

include(project_options)
include(unity)
include(clang-tidy)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# CMake 3.19: "smartly dropping dependencies of static and object libraries"
set(CMAKE_OPTIMIZE_DEPENDENCIES TRUE)

if(NOT ENABLE_LIBCXX)
    # required when linking with libstdc++ with clang and gcc
    add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,Clang>:-fsized-deallocation>)
endif()

include(CTest)

get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

# Global settings if we're the top-level project
if(PROJECT_IS_TOP_LEVEL)
    set_property(
        GLOBAL
        PROPERTY
            GLOBAL_DEPENDS_NO_CYCLES
                TRUE
    )

    if(ENABLE_CCACHE)
        include(ccache)
    endif()
endif()

include(compilers)
CHECK_COMPILERS()

# We're not currently using C++20 modules, so don't bother scanning for them
set(CMAKE_CXX_SCAN_FOR_MODULES FALSE)

# Promote all IMPORTED targets discovered by find_package() to a GLOBAL scope
set(CMAKE_FIND_PACKAGE_TARGETS_GLOBAL TRUE)

############################################################################################################################
# Project Options
#   The following options and their defaults impact what artifacts get built
############################################################################################################################
message(STATUS "ENABLE_LIBCXX: ${ENABLE_LIBCXX}")
message(STATUS "Build shared libs: ${BUILD_SHARED_LIBS}")
message(STATUS "Build Python bindings: ${WITH_PYTHON_BINDINGS}")
message(STATUS "Build Programming Examples: ${BUILD_PROGRAMMING_EXAMPLES}")
message(STATUS "Build TT METAL Tests: ${TT_METAL_BUILD_TESTS}")
message(STATUS "Build TTNN Tests: ${TTNN_BUILD_TESTS}")
message(STATUS "Build with Unity builds: ${TT_UNITY_BUILDS}")
message(STATUS "Build with Shared TTNN Sublibraries: ${ENABLE_TTNN_SHARED_SUBLIBS}")
message(STATUS "Build with LightMetal Trace Enabled: ${TT_ENABLE_LIGHT_METAL_TRACE}")

############################################################################################################################

if(ENABLE_BUILD_TIME_TRACE)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        message(STATUS "Adding compile option: -ftime-trace")
        add_compile_options("-ftime-trace")
    else()
        message(FATAL_ERROR "ENABLE_BUILD_TIME_TRACE is only supported with Clang")
    endif()
endif()

############################################################################################################################
# Find all required libraries to build
############################################################################################################################
include(CPM)
if(CMAKE_VERSION VERSION_LESS 3.25)
    # FIXME(14681): `SYSTEM` was introduced in v3.25; remove this when we can require v3.25
    add_subdirectory(third_party)
else()
    add_subdirectory(third_party SYSTEM)
endif()

if(WITH_PYTHON_BINDINGS)
    set(Python3_FIND_STRATEGY LOCATION)
    find_package(
        Python3
        COMPONENTS
            Interpreter
            Development
    )
    message(STATUS "Python3 include dirs: ${Python3_INCLUDE_DIRS}")
endif()

find_library(NUMA_LIBRARY NAMES numa)
if(NOT NUMA_LIBRARY)
    message(FATAL_ERROR "NUMA library not found")
endif()

# Bring in UMD and all it's dependencies
add_subdirectory(tt_metal/third_party/umd)

############################################################################################################################
# Boost Core via CPM
############################################################################################################################
add_library(metal_common_libs INTERFACE)

# Glob the boost_core include dir once (robust-ish)
# AFTER add_library(metal_common_libs INTERFACE)
# Find boost root (example path shown in your logs)
set(BOOST_CORE_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/.cpmcache/boost/1359e136761ab2d10afa1c4e21086c8d824735cd/libs/core/include")

# Sanity check
message(STATUS "Manually including Boost Core headers: ${BOOST_CORE_INCLUDE_DIR}")

target_include_directories(metal_common_libs INTERFACE ${BOOST_CORE_INCLUDE_DIR})

############################################################################################################################
# Constructing interface libs for common compiler flags, header directories, and libraries
#   These interface libs are linked with PUBLIC scope at lowest common target (tt_metal/common) and at tt_metal_libs level
#   in order to propogate to the rest of tt_metal, tt_eager, etc.
############################################################################################################################
target_link_libraries(
    metal_common_libs
    INTERFACE
        dl
        pthread
        atomic
        hwloc
        numa
)

add_compile_options(
    "$<$<STREQUAL:${CMAKE_SYSTEM_PROCESSOR},x86_64>:-mavx2>"
    -fno-lto # FIXME: This seems to be here for ttnn; it should go to TTNN, then.
    -fPIC
    -fvisibility-inlines-hidden
    -Wall
    -Werror
    "$<$<CXX_COMPILER_ID:Clang>:-Wconditional-uninitialized>"
    # The following list is not indicative of what we _desire_ only the status quo to build.
    -Wno-deprecated-declarations
    # Vars that are used only in Asserts will appear as dead stores in builds
    # where the preprocessor has stripped away the assert.
    "$<$<CONFIG:Release>:-Wno-unused-but-set-variable>"
    -Wno-unused-function
    -Wno-unused-variable
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-volatile>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-for-loop-analysis>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-logical-op-parentheses>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-mismatched-tags>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-reorder-ctor>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-unused-lambda-capture>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-unused-local-typedef>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-unused-private-field>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wno-unknown-pragmas>"
    "$<$<CXX_COMPILER_ID:Clang>:SHELL:-Xclang -fno-pch-timestamp>" # ccache-friendly PCH flag
    "$<$<CXX_COMPILER_ID:GNU>:-fpch-preprocess>" # ccache-friendly PCH flag
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-int-in-bool-context>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-non-template-friend>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-parentheses>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-pragmas>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-reorder>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-restrict>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-strict-aliasing>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overread>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-unused-local-typedefs>"
    "$<$<CXX_COMPILER_ID:GNU>:-Wno-volatile>"
)

# Planned to be temporary, remove later.
if(TT_ENABLE_LIGHT_METAL_TRACE)
    add_compile_definitions(TT_ENABLE_LIGHT_METAL_TRACE=1)
else()
    add_compile_definitions(TT_ENABLE_LIGHT_METAL_TRACE=0)
endif()

include(GNUInstallDirs)
# GNUInstallDirs takes PROJECT_DIR verbatim, but directories should always be lowercase
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
string(REPLACE ${PROJECT_NAME} ${PROJECT_NAME_LOWER} CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_DOCDIR})

if(ENABLE_CODE_TIMERS)
    add_compile_definitions(TT_ENABLE_CODE_TIMERS)
endif()
include(tracy)

add_library(metal_common_pch STATIC)
add_library(TT::CommonPCH ALIAS metal_common_pch)
file(GENERATE OUTPUT metal_common_pch.cpp CONTENT "/*dummy pch file*/\n")
target_sources(metal_common_pch PRIVATE metal_common_pch.cpp)

target_precompile_headers(
    metal_common_pch
    PUBLIC
        <fmt/core.h>
        <fmt/format.h>
        <nlohmann/json.hpp>
        <algorithm>
        <cstddef>
        <cstdint>
        <functional>
        <map>
        <memory>
        <tuple>
        <unordered_map>
        <variant>
        <vector>
)

target_link_libraries(
    metal_common_pch
    PRIVATE
        nlohmann_json::nlohmann_json
        fmt::fmt-header-only
)

############################################################################################################################
# Build subdirectories
############################################################################################################################

add_subdirectory(tt_stl)
add_subdirectory(tt_metal)
add_subdirectory(ttnn)

if(TT_METAL_BUILD_TESTS OR TTNN_BUILD_TESTS)
    add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
endif()

############################################################################################################################
# Install targets for build artifacts and pybinds
#   If built with Tracy, cannot install 'all' since it will pick up install targets from Tracy
# For top level install: cmake --build build --target install  or  make/ninja install -C build
############################################################################################################################
# Install for build artifacts that will upload build/lib
install(
    TARGETS
        tt_metal
    ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT tar
)

# FIXME(17578): figure out what bits we actually need to ship and omit the rest
install(
    DIRECTORY
        runtime
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium
    USE_SOURCE_PERMISSIONS
    COMPONENT jit-build
)

# Custom clean target for `built` folder for when new kernel changes are pulled
add_custom_target(
    clean-built
    COMMAND
        ${CMAKE_COMMAND} -E remove_directory ${PROJECT_SOURCE_DIR}/built
    COMMENT "Cleaning `built` directory"
)

if(BUILD_TT_TRAIN)
    add_subdirectory(tt-train)
endif()

# If we've been asked to build static libs, packaging is left up to them to sort out.
if(TT_INSTALL)
    include(packaging)
endif()
