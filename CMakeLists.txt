cmake_minimum_required(VERSION 3.15)

project(SimAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -g)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.3)
        message(FATAL_ERROR
                "g++ (GNU) version should be greater than 5.3, but found ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Default build type" FORCE)
endif()

set(NS3_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ns-3-alibabacloud/simulation")
set(ASTRA_SIM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud/astra-sim")
set(RESULT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/results")

file(MAKE_DIRECTORY "${RESULT_DIR}")

option(SYS_ASSERTS "Enable assert() in all profile with -UNDEBUG" OFF)

set(SIMAI_MODE "analytical" CACHE STRING "SimAI build mode")
set_property(CACHE SIMAI_MODE PROPERTY STRINGS analytical ns3 phy)

set(_valid_simai_modes analytical ns3 phy)
list(FIND _valid_simai_modes "${SIMAI_MODE}" _simai_mode_index)
if(_simai_mode_index EQUAL -1)
    message(FATAL_ERROR
            "Invalid SIMAI_MODE='${SIMAI_MODE}'. Expected one of: analytical, ns3, phy")
endif()

if(SYS_ASSERTS)
    add_compile_options(-UNDEBUG)
    message(STATUS "Asserts enabled for all profiles.")
endif()

# Select one SimCCL mock version for all backends.
# An explicit CMake cache value takes precedence over the environment.
set(_simai_nccl_default "v2.20")
if(DEFINED ENV{SIMAI_NCCL_VERSION}
        AND NOT "$ENV{SIMAI_NCCL_VERSION}" STREQUAL "")
    set(_simai_nccl_default "$ENV{SIMAI_NCCL_VERSION}")
endif()

set(SIMAI_NCCL_VERSION "${_simai_nccl_default}" CACHE STRING
        "SimCCL mock version")
set_property(CACHE SIMAI_NCCL_VERSION PROPERTY STRINGS v2.20 v2.30)

if(NOT SIMAI_NCCL_VERSION STREQUAL "v2.20"
        AND NOT SIMAI_NCCL_VERSION STREQUAL "v2.30")
    message(FATAL_ERROR
            "Invalid SIMAI_NCCL_VERSION='${SIMAI_NCCL_VERSION}'; expected v2.20 or v2.30")
endif()

set(SIMCCL_MOCK_DIR
        "${CMAKE_CURRENT_SOURCE_DIR}/SimCCL/src/mock/${SIMAI_NCCL_VERSION}")

if(NOT EXISTS "${SIMCCL_MOCK_DIR}/MockNcclGroup.cc")
    message(FATAL_ERROR
            "SimCCL mock sources are missing from '${SIMCCL_MOCK_DIR}'. Initialize the SimCCL submodule.")
endif()

# Shared by every backend: the ns3 frontend exports these into the ns-3 tree,
# while the AstraSim library consumes them directly.
file(GLOB SIMCCL_MOCK_SOURCES CONFIGURE_DEPENDS
        "${SIMCCL_MOCK_DIR}/*.cc")
file(GLOB SIMCCL_MOCK_HEADERS CONFIGURE_DEPENDS
        "${SIMCCL_MOCK_DIR}/*.hh"
        "${SIMCCL_MOCK_DIR}/*.h")

message(STATUS "SimCCL mock version: ${SIMAI_NCCL_VERSION}")

if(SIMAI_MODE STREQUAL "ns3")
    add_subdirectory(
            "${ASTRA_SIM_DIR}/network_frontend/ns3"
            simai_ns3
    )
    set(target_name "scratch_AstraSimNetwork")
    set(symlink_name "SimAI_simulator")
elseif(SIMAI_MODE STREQUAL "analytical")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud" AstraSim)
    add_subdirectory(
            "${ASTRA_SIM_DIR}/network_frontend/analytical"
            simai_analytical
    )
    set(target_name "SimAI_analytical")
    set(symlink_name "SimAI_analytical")
elseif(SIMAI_MODE STREQUAL "phy")
    set(USE_RDMA TRUE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud" AstraSim)
    add_subdirectory(
            "${ASTRA_SIM_DIR}/network_frontend/phynet"
            simai_phynet
    )
    set(target_name "SimAI_phynet")
    set(symlink_name "SimAI_phynet")
else()
    message(FATAL_ERROR "Unknown SIMAI_MODE=${SIMAI_MODE}")
endif()

# Link binaries
add_custom_target(create_symlink ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/bin"
        COMMAND ${CMAKE_COMMAND} -E create_symlink
        "$<TARGET_FILE:${target_name}>"
        "${CMAKE_SOURCE_DIR}/bin/${symlink_name}"
        DEPENDS ${target}
        COMMENT "Creating ${symlink_name} symlink..."
)

# Optional in-memory Python C-extension (pysimai)
find_package(pybind11 QUIET)
if(pybind11_FOUND AND SIMAI_MODE STREQUAL "ns3")
    message(STATUS "Found pybind11: configuring pysimai Python extension target")
    # Resolve libstdc++ against the active Python env's lib dir and record it in
    # the pysimai RUNPATH. The conda env's libstdc++ (>= GLIBCXX_3.4.32) is
    # required by pyarrow/libarrow, which is imported later in the same process;
    # without this the system libstdc++ is loaded first and the import crashes.
    # This replaces the old LD_LIBRARY_PATH workaround: no env var needed.
    if(DEFINED Python3_EXECUTABLE)
        set(_simai_python "${Python3_EXECUTABLE}")
    elseif(DEFINED PYTHON_EXECUTABLE)
        set(_simai_python "${PYTHON_EXECUTABLE}")
    else()
        set(_simai_python "python3")
    endif()
    execute_process(
        COMMAND "${_simai_python}"
                -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
        OUTPUT_VARIABLE PYTHON_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT PYTHON_LIB_DIR)
        message(WARNING "Could not determine Python LIBDIR; pysimai may load the system libstdc++")
    endif()
    # Mirror CMAKE_OUTPUT_DIRECTORY from the ns-3 subproject: its
    # include_directories() are scoped to the ns-3 subtree, but pysimai needs
    # the generated headers (e.g. ns3/mtp-interface.h) from that build dir.
    if("${NS3_OUTPUT_DIRECTORY}" STREQUAL "")
        set(NS3_BUILD_INCLUDE_DIR "${NS3_SRC_DIR}/build/include")
    elseif(IS_ABSOLUTE "${NS3_OUTPUT_DIRECTORY}")
        set(NS3_BUILD_INCLUDE_DIR "${NS3_OUTPUT_DIRECTORY}/include")
    else()
        set(NS3_BUILD_INCLUDE_DIR "${NS3_SRC_DIR}/${NS3_OUTPUT_DIRECTORY}/include")
    endif()
    pybind11_add_module(pysimai
        "${ASTRA_SIM_DIR}/runtime/pybind_simai.cc"
        "${ASTRA_SIM_DIR}/runtime/SimAIRuntime.cc"
        "${ASTRA_SIM_DIR}/network_frontend/ns3/AstraSimNetwork.cc"
    )
    target_include_directories(pysimai PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim-alibabacloud"
        "${ASTRA_SIM_DIR}"
        "${ASTRA_SIM_DIR}/network_frontend"
        "${ASTRA_SIM_DIR}/runtime"
        "${ASTRA_SIM_DIR}/network_frontend/ns3"
        "${NS3_SRC_DIR}/src/applications"
        "${NS3_BUILD_INCLUDE_DIR}"
    )
    # Enable the multithreaded ns-3 simulator (MultithreadedSimulatorImpl +
    # MtpInterface::Enable(num_threads)) for the in-memory Python backend so
    # that init_network(..., num_threads=N) actually partitions the topology
    # across N threads instead of falling back to the single-threaded
    # DefaultSimulatorImpl.
    target_compile_definitions(pysimai PRIVATE NS3_MTP)
    if(TARGET scratch_AstraSimNetwork)
        target_link_libraries(pysimai PRIVATE "${ns3-libs}")
    endif()
    if(PYTHON_LIB_DIR)
        target_link_options(pysimai PRIVATE "-Wl,-rpath,${PYTHON_LIB_DIR}")
    endif()
    set_target_properties(pysimai PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
    )
else()
    message(STATUS "pysimai Python extension skipped (pybind11 not found or SIMAI_MODE != ns3)")
endif()
