#!/bin/bash
# set -e
# Absolue path to this script
SCRIPT_DIR=$(dirname "$(realpath $0)")
echo $SCRIPT_DIR

# Absolute paths to useful directories
GEM5_DIR="${SCRIPT_DIR:?}"/../../extern/network_backend/garnet/gem5_astra/
ASTRA_SIM_DIR="${SCRIPT_DIR:?}"/../../astra-sim
INPUT_DIR="${SCRIPT_DIR:?}"/../../inputs
NS3_DIR="${SCRIPT_DIR:?}"/../../extern/network_backend/ns3-interface
NS3_APPLICATION="${NS3_DIR:?}"/simulation/src/applications/
SIM_LOG_DIR=/etc/astra-sim
BUILD_DIR="${SCRIPT_DIR:?}"/build/
RESULT_DIR="${SCRIPT_DIR:?}"/result/
BINARY="${BUILD_DIR}"/gem5.opt
ASTRA_SIM_LIB_DIR="${SCRIPT_DIR:?}"/build/AstraSim

# Functions
function setup {
    mkdir -p "${BUILD_DIR}"
    mkdir -p "${RESULT_DIR}"
}

function cleanup {
    echo $BUILD_DIR
    rm -rf "${BUILD_DIR}"
    rm -rf "${NS3_DIR}"/simulation/build
    rm -rf "${NS3_DIR}"/simulation/cmake-cache
    rm -rf "${NS3_APPLICATION}"/astra-sim
    cd "${SCRIPT_DIR:?}"
}

function cleanup_result {
    rm -rf "${RESULT_DIR}"
}

function compile_astrasim {
    cd "${BUILD_DIR}" || exit
    cmake ..
    make
}

function compile {
    # Only compile & Run the AstraSimNetwork ns3program
    # if [ ! -f '"${INPUT_DIR}"/inputs/config/SimAI.conf' ]; then
    #     echo ""${INPUT_DIR}"/config/SimAI.conf is not exist"
    #     cp "${INPUT_DIR}"/config/SimAI.conf "${SIM_LOG_DIR}"/config/SimAI.conf
    # fi
    local profile="${1:-debug}"
    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/AstraSimNetwork.cc "${NS3_DIR}"/simulation/scratch/
    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/*.h "${NS3_DIR}"/simulation/scratch/
    rm -rf "${NS3_APPLICATION}"/astra-sim
    cp -r "${ASTRA_SIM_DIR}" "${NS3_APPLICATION}"/
    cd "${NS3_DIR}/simulation"
    CC='gcc' CXX='g++'
    max_threads=$(($(lscpu | grep '^CPU(s):' | awk '{print $2}') - 1))

    # Configure ns-3 with the selected build profile
    ./ns3 configure -d "${profile}" --enable-mtp
    ./ns3 build

    # Create a symbolic link to a known filename: ns3.36.1-AstraSimNetwork -> actual built binary
    target_dir="${NS3_DIR}/simulation/build/scratch"
    base_name="ns3.36.1-AstraSimNetwork"
    candidate="${target_dir}/${base_name}-${profile}"
    if [[ -e "${candidate}" ]]; then
        ln -sf "${candidate}" "${target_dir}/${base_name}"
        echo "Linked ${candidate} -> ${target_dir}/${base_name}"
    else
        echo "Warning: expected build output not found for profile '${profile}' at ${candidate}"
    fi
    cd "${SCRIPT_DIR:?}"
}

function debug {
    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/AstraSimNetwork.cc "${NS3_DIR}"/simulation/scratch/
    cp "${ASTRA_SIM_DIR}"/network_frontend/ns3/*.h "${NS3_DIR}"/simulation/scratch/
    cd "${NS3_DIR}/simulation"
    CC='gcc' CXX='g++'
    ./waf configure
    ./waf --run 'scratch/AstraSimNetwork' --command-template="gdb --args %s mix/config.txt"

    ./waf --run 'scratch/AstraSimNetwork mix/config.txt'

    cd "${SCRIPT_DIR:?}"
}

function parse_profile() {
    profile=
    while [[ $# -gt 0 ]]; do
        case "$1" in
            ns3) ;;
            -d|--build-profile)
                profile="$2"; shift ;;
            --build-profile=*)
                profile="${1#*=}" ;;
            debug|default|release|optimized)
                profile="$1" ;;
            *) ;;     # ignore unknown here
        esac
        shift
    done
    echo $profile
}

# Main Script
case "$1" in
-l|--clean)
    cleanup;;
-lr|--clean-result)
    cleanup
    cleanup_result;;
-d|--debug)
    setup
    debug;;
-c|--compile)
    setup
    shift
    profile=$(parse_profile "$@")
    compile_astrasim
    compile "${profile}";;
-r|--run)
    setup
    compile;;
-h|--help|*)
    printf "Prints help message";;
esac
