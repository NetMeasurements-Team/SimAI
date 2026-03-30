/*
* Copyright (c) 2024, Alibaba Group;
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#ifndef __NCCL_CUSTOM_HH__
#define __NCCL_CUSTOM_HH__

#include <assert.h>
#include <math.h>
#include<set>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>
#include<condition_variable>
#include "Algorithm.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MyPacket.hh"
#include "astra-sim/system/topology/RingTopology.hh"

#define COLLECTIVE_GRAPH 1


namespace AstraSim {

class NcclCustom : public Algorithm {
public:
    enum class FlowState { Sending, WritingBack };
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
    MemBus::Transmition transmition;
    int id;
    int nodes_in_ring; // Retained for compatibility, may not be directly used in core logic
    std::atomic<int> send_packets_in_flight;
    std::atomic<int> recv_packets_in_flight;
    int total_sends_initiated;
    int total_recvs_initiated;
    std::atomic<int> sends_completed;
    std::atomic<int> recvs_completed;
    int total_sends_to_initiate;
    std::map<int, int> flow_id_to_channel_id_map;

    std::map<int, int> indegree_mapping; //
    MockNccl::FlowModels _flow_models; //
    uint32_t m_channels; //
    std::map<int, FlowState> flow_states; // Map to track the state of each flow_id

    // Added for timed graph generation
    std::map<int, Tick> send_start_times;
    std::map<int, Tick> recv_end_times;
    
#if COLLECTIVE_GRAPH
    // Holds all timing and flow data for graph generation, shared across all NcclCustom instances.
    struct GlobalGraphData {
        std::mutex mtx;
        std::map<int, MockNccl::SingleFlow> flow_details;
        std::map<int, Tick> start_times;
        std::map<int, Tick> end_times;
    };
    static std::map<int, GlobalGraphData> g_graph_data_per_layer;
    // Atomic counter to track when all nodes have finished.
    static std::map<int, std::atomic<int>> g_finished_nodes_per_layer; // Layer-wise finished
#endif

    NcclCustom(
        ComType type,
        int id,
        int layer_num,
        RingTopology* ring_topology,
        uint64_t data_size,
        RingTopology::Direction direction,
        InjectionPolicy injection_policy,
        bool boost_mode,
        std::shared_ptr<MockNccl::FlowModels> ptr_flow_models,
        int treechannels);

    virtual ~NcclCustom();

    virtual void run(EventType event, CallData* data) override;
    void init_indegree_mapping(); //
    void send_flow(int channel_id, int flow_id);
    void check_completion();
    void send_flow_unlocked(int channel_id, int flow_id);
    void release_packet(int channel_id, int flow_id, bool NPU_to_MA);
    virtual void exit() override;

    #ifdef PHY_MTP
    bool phy_iteratable(int channel_id); //
    void waiting_to_exit(); //
    std::atomic<bool> judge_exit_flag;
    #endif
    class FlowCriticalSection {
    public:
      /**
       * @brief Acquires the lock upon construction.
       *
       * This constructor spins in a loop until it successfully sets the atomic
       * flag from 'false' to 'true', effectively waiting until the lock is free
       * and then taking it. The 'acquire' memory order ensures that subsequent
       * memory operations are not moved before this point.
       */
      FlowCriticalSection() {
        while (g_flow_inCriticalSection_ncclCustom.exchange(true, std::memory_order_acquire)) {
          // This is a spinlock; it will busy-wait.
        }
      }

      /**
       * @brief Releases the lock upon destruction.
       *
       * The destructor ensures the lock is released when the object goes out of
       * scope (RAII). The 'release' memory order ensures that preceding memory
       * operations are not moved after this point.
       */
      ~FlowCriticalSection() {
        this->ExitSection();
      }

      /**
       * @brief Explicitly releases the lock before the object is destroyed.
       */


      /**
       * @brief Checks if the lock is currently held by any thread.
       *
       * @return True if the lock is currently held, false otherwise.
       *
       * This function loads the current value of the atomic flag. It uses
       * 'relaxed' memory ordering as we are only interested in the atomic's state
       * at this moment and do not need to synchronize other memory operations.
       */
      static bool is_locked() {
        return g_flow_inCriticalSection_ncclCustom.load(std::memory_order_acquire);
      }
      // To make this class non-copyable and non-movable, which is good practice
      // for lock guard classes.
      FlowCriticalSection(const FlowCriticalSection&) = delete;
      FlowCriticalSection& operator=(const FlowCriticalSection&) = delete;
      FlowCriticalSection(FlowCriticalSection&&) = delete;
      FlowCriticalSection& operator=(FlowCriticalSection&&) = delete;

    private:
      static void ExitSection() {
        g_flow_inCriticalSection_ncclCustom.store(false, std::memory_order_release);
      }
    };
    static std::atomic<bool> g_flow_inCriticalSection_ncclCustom;
  };
};
#endif