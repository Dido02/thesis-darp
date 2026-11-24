cmake_minimum_required(VERSION 3.14)
project(darp_solver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# ---------------------------------------------------------------------------
# ML selection — optional, controlled by -DENABLE_ML_SELECTION=ON/OFF
# ---------------------------------------------------------------------------
option(ENABLE_ML_SELECTION "Build with ML-guided operator selection (ONNX Runtime)" ON)

include(FetchContent)

# nlohmann/json (header-only) — always fetched; used by ML manifest + tests.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ONNX Runtime prebuilt (fetched only if ENABLE_ML_SELECTION)
set(ONNXRUNTIME_VERSION "1.19.2")
if(ENABLE_ML_SELECTION)
    if(APPLE)
        set(ONNXRT_URL
            "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-osx-universal2-${ONNXRUNTIME_VERSION}.tgz")
        set(ONNXRT_LIB_NAME "libonnxruntime.${ONNXRUNTIME_VERSION}.dylib")
    elseif(UNIX)
        set(ONNXRT_URL
            "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
        set(ONNXRT_LIB_NAME "libonnxruntime.so.${ONNXRUNTIME_VERSION}")
    else()
        message(FATAL_ERROR "ENABLE_ML_SELECTION: platform not supported. Disable with -DENABLE_ML_SELECTION=OFF.")
    endif()

    FetchContent_Declare(
        onnxruntime_prebuilt
        URL ${ONNXRT_URL}
    )
    FetchContent_MakeAvailable(onnxruntime_prebuilt)

    set(ONNXRT_ROOT ${onnxruntime_prebuilt_SOURCE_DIR})
    message(STATUS "ONNX Runtime: ${ONNXRT_ROOT}")

    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRT_ROOT}/lib/${ONNXRT_LIB_NAME}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRT_ROOT}/include"
    )
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(DARP_CORE_SOURCES
    src/parsers/cordeau.cpp
    src/parsers/fiedler.cpp
    src/parsers/li_lim.cpp
    src/solution.cpp
    src/initial_solution.cpp
    src/solver.cpp
    src/feasibility/slack_corrected.cpp
    src/feasibility/full_optimizer.cpp
    src/feasibility/o1_feasibility.cpp
    src/operators/operator_registry.cpp
    src/operators/destroy/random_removal.cpp
    src/operators/destroy/worst_removal.cpp
    src/operators/destroy/shaw_removal.cpp
    src/operators/destroy/ride_time_critical.cpp
    src/operators/destroy/low_slack_route.cpp
    src/operators/destroy/entire_route_removal.cpp
    src/operators/repair/greedy_insertion.cpp
    src/operators/repair/regret_insertion.cpp
    src/operators/repair/slack_insertion.cpp
    src/operators/repair/repair_utils.cpp
    src/dynamic_solver.cpp
    src/logging/alns_logger.cpp
    # Feature extractor is always built — it has no ONNX dependency and
    # is used both by MLSelector (when enabled) and by test_feature_parity.
    src/ml/feature_extractor.cpp
    src/ml/manifest.cpp
)

if(ENABLE_ML_SELECTION)
    list(APPEND DARP_CORE_SOURCES src/ml/ml_selector.cpp)
endif()

add_library(darp_core ${DARP_CORE_SOURCES})
target_include_directories(darp_core PUBLIC include)
target_link_libraries(darp_core PUBLIC nlohmann_json::nlohmann_json)

if(ENABLE_ML_SELECTION)
    target_link_libraries(darp_core PUBLIC onnxruntime)
    target_compile_definitions(darp_core PUBLIC DARP_ENABLE_ML)
endif()

# ---------------------------------------------------------------------------
# Main solver executable
# ---------------------------------------------------------------------------
add_executable(darp_solver main.cpp)
target_link_libraries(darp_solver darp_core)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_executable(test_parser tests/test_parser.cpp)
target_link_libraries(test_parser darp_core)

add_executable(test_feasibility tests/test_feasibility.cpp)
target_link_libraries(test_feasibility darp_core)

add_executable(test_initial tests/test_initial.cpp)
target_link_libraries(test_initial darp_core)

add_executable(test_solver tests/test_solver.cpp)
target_link_libraries(test_solver darp_core)

add_executable(test_o1_feasibility tests/test_o1_feasibility.cpp)
target_link_libraries(test_o1_feasibility darp_core)

add_executable(compare_feasibility tests/compare_feasibility.cpp)
target_link_libraries(compare_feasibility darp_core)

add_executable(test_o1_vs_full tests/test_o1_vs_full.cpp)
target_link_libraries(test_o1_vs_full darp_core)

add_executable(test_o1_diagnostic tests/test_o1_diagnostic.cpp)
target_link_libraries(test_o1_diagnostic darp_core)

# Feature parity test — uses feature_extractor (no ONNX needed).
add_executable(test_feature_parity tests/test_feature_parity.cpp)
target_link_libraries(test_feature_parity darp_core)
