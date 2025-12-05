#!/bin/bash

BUILD_DIR_BASE_NAME="cmake-cache"
SCRIPT_DIR=$(dirname "$(realpath $0)")
ROOT_DIR=$(realpath "${SCRIPT_DIR:?}"/..)
SIMAI_DIR="${ROOT_DIR:?}/astra-sim-alibabacloud"
SOURCE_ANA_BIN_DIR="${SIMAI_DIR:?}/build/simai_analytical/build/simai_analytical/SimAI_analytical"
SOURCE_PHY_BIN_DIR="${SIMAI_DIR:?}/build/simai_phy/build/simai_phynet/SimAI_phynet"
SIM_LOG_DIR=/etc/astra-sim
NS3_SRC_DIR="${ROOT_DIR:?}/ns-3-alibabacloud/simulation"
TARGET_BIN_DIR="${SCRIPT_DIR:?}/../bin"

function get_build_dir_name {
  local profile="$1"
  if [[ $profile == "default" ]]; then
    echo ${BUILD_DIR_BASE_NAME}
  else
    echo ${BUILD_DIR_BASE_NAME}-${profile}
  fi
}

function compile {
    local mode="$1"
    local profile="$2"
    local native="$3"
    local sys_asserts="$4"
    local ns3_asserts="$5"

    mkdir -p "${SIM_LOG_DIR}"/inputs/system/
    mkdir -p "${SIM_LOG_DIR}"/inputs/workload/
    mkdir -p "${SIM_LOG_DIR}"/simulation/
    mkdir -p "${SIM_LOG_DIR}"/config/
    mkdir -p "${SIM_LOG_DIR}"/topo/
    mkdir -p "${SIM_LOG_DIR}"/results/

    build_dir=${ROOT_DIR}/$(get_build_dir_name $profile)
    case "$mode" in
    "ns3")
        mkdir -p "${build_dir}"
        cmake -DCMAKE_BUILD_TYPE="${profile}" -DNS3_NATIVE_OPTIMIZATIONS="${native}" -DSYS_ASSERTS="${sys_asserts}" \
          -DBUILD_SIM=ON -DNS3_MTP=ON ""${ns3_asserts:+-DNS3_ASSERT="${ns3_asserts}"} \
          -G "Unix Makefiles" -S "${ROOT_DIR}" -B "${build_dir}"
        cmake --build "${build_dir}" -j "$(($(lscpu | grep '^CPU(s):' | awk '{print $2}') - 1))"
        ;;
    "phy")
        mkdir -p "${TARGET_BIN_DIR:?}"
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_phynet" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_phynet
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr phy
        ./build.sh -c phy 
        ln -s "${SOURCE_PHY_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_phynet
        ;;
    "analytical")
        mkdir -p "${TARGET_BIN_DIR:?}"
        mkdir -p "${ROOT_DIR:?}"/results
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_analytical" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_analytical
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr analytical
        ./build.sh -c analytical 
        ln -s "${SOURCE_ANA_BIN_DIR:?}" "${TARGET_BIN_DIR:?}"/SimAI_analytical
        ;;
    esac
}

function cleanup_build {
    local mode="$1"
    local profile="$2"

    build_dir_name=$(get_build_dir_name $profile)
    build_dir=${ROOT_DIR}/${build_dir_name}
    case "$mode" in
    "ns3")
        set -x
        rm -rf "${build_dir}"
        rm -rf "${TARGET_BIN_DIR}"
        rm -rf "${NS3_SRC_DIR}/${build_dir_name}"
        rm -rf "${NS3_SRC_DIR}/build"
        rm -f "${NS3_SRC_DIR}/.lock-ns3_linux_build"
        mkdir -p "${build_dir}"
        set +x
        ;;
    "phy")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_phynet" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_phynet
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr phy
        ;;
    "analytical")
        if [ -L "${TARGET_BIN_DIR:?}/SimAI_analytical" ]; then
            rm -rf "${TARGET_BIN_DIR:?}"/SimAI_analytical
        fi
        cd "${SIMAI_DIR:?}"
        ./build.sh -lr analytical
        ;;
    esac
}

# Main Script
print_usage() {
    printf -- "Usage: $0 [options]\n"
    printf -- "-c|--compile <mode>  Compile for ns3|phy|analytical\n"
    printf -- "-l|--clean <mode>    Clean the build directory.\n"
    printf -- "-lc|-cl <mode>       Clean and then compile.\n"
    printf -- "-d|--build-profile   ns3 build profile debug|default|release|optimized.\n"
    printf -- "--sys-asserts        Enable asserts() in any build profile.\n"
    printf -- "--ns3-asserts        Enable NS3_ASSERT in any build profile.\n"
    printf -- "-h|--help            Show this help message.\n"
    printf -- "\n"
    printf -- "Example: $0 -lc ns3 -d optimized --sys-asserts\n"
}

profile=
native=OFF
mode=
clean=OFF
compile=OFF
sys_asserts=OFF
ns3_asserts=

# Expand -lc <mode> into -l <mode> -c <mode>
processed_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -lc|-cl)
      if [[ -n "$2" && ! "$2" =~ ^- ]]; then
        processed_args+=("-l" "$2" "-c" "$2"); shift 2
      else
        echo "Error: Option $1 requires a <mode> argument." >&2; print_usage; exit 1
      fi ;;
    *)
      processed_args+=("$1"); shift ;;
  esac
done

OPTS=$(getopt -o c:l:d:h --long compile:,clean:,help,build-profile:,sys-asserts,ns3-asserts -n "$0" -- "${processed_args[@]}")
if [ $? != 0 ]; then
  echo "Failed parsing options." >&2; print_usage; exit 1
fi
eval set -- "$OPTS"
while true; do
  case "$1" in
    -l|--clean)               clean=ON; mode="$2";                        shift 2 ;;
    -c|--compile)             compile=ON; mode="$2";                      shift 2 ;;
    -d|--build-profile)       case "$2" in
                                debug|default|release) profile="$2" ;;
                                optimized) profile=release; native=ON ;;
                                *) profile=default ;;
                              esac;                                       shift 2 ;;
    --sys-asserts)            sys_asserts=ON;                             shift ;;
    --ns3-asserts)            ns3_asserts=ON;                             shift ;;
    -h|--help)                print_usage;                                exit 0 ;;
    --)                       shift;                                      break ;;
    *)                        echo "Unexpected option: $1";
                              print_usage;                                exit 1 ;;

  esac
done
profile="${profile:-default}"

if [[ $clean == "ON" ]]; then
  cleanup_build "$mode" "$profile"
fi
if [[ $compile == "ON" ]]; then
  compile "$mode" "$profile" "$native" "$sys_asserts" "$ns3_asserts"
fi
