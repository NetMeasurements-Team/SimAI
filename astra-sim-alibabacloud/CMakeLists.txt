file(GLOB ASTRA_SIM_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/collective/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/fast-backend/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/memory/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/topology/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.cc"
)

file(GLOB ASTRA_SIM_HEADERS CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/collective/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/collective/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/fast-backend/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/fast-backend/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/memory/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/memory/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/scheduling/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/topology/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/topology/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.hh"
        "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/workload/*.h"
)

if(SIMAI_MODE STREQUAL "phy")
    file(GLOB PHY_COMMON_SOURCES CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/phy-common/*.cc"
    )
    file(GLOB PHY_COMMON_HEADERS CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/phy-common/*.hh"
            "${CMAKE_CURRENT_SOURCE_DIR}/astra-sim/system/phy-common/*.h"
    )
    list(APPEND ASTRA_SIM_SOURCES ${PHY_COMMON_SOURCES})
    list(APPEND ASTRA_SIM_HEADERS ${PHY_COMMON_HEADERS})
endif()

add_library(AstraSim ${ASTRA_SIM_SOURCES} ${ASTRA_SIM_HEADERS})
target_compile_features(AstraSim PUBLIC cxx_std_17)
target_include_directories(AstraSim PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT SIMAI_MODE STREQUAL "ns3")
    find_package(LibXml2 REQUIRED)
    include_directories(${LIBXML2_INCLUDE_DIR})
    target_link_libraries(AstraSim PRIVATE LibXml2::LibXml2)
endif()

if(USE_RDMA)
    target_compile_definitions(AstraSim PRIVATE PHY_RDMA PHY_MTP)

    find_package(MPI REQUIRED)
    target_link_libraries(AstraSim PUBLIC MPI::MPI_CXX)
endif()
# include_directories("${PROJECT_SOURCE_DIR}/../../")
# add_library(AstraSim ${astra_SRC})
# set_property(TARGET AstraSim PROPERTY CXX_STANDARD 17)
# set_property(TARGET AstraSim PROPERTY CXX_STANDARD_REQUIRED ON)
# # Suppress warnings about converting string literals to char* and allow GNU extensions
# target_compile_options(AstraSim PRIVATE -Wno-write-strings -std=gnu++17)

if(SIMAI_MODE STREQUAL "analytical")
    target_compile_definitions(AstraSim PRIVATE ANALYTI)
endif()

