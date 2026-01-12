/*
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#ifndef __NCCL_TREE_FLOW_MODEL_HH__
#define __NCCL_TREE_FLOW_MODEL_HH__

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <ctime>
#include <list>
#include <map>
#include "Algorithm.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MyPacket.hh"
#include "astra-sim/system/topology/RingTopology.hh"

namespace AstraSim {
class NcclTreeFlowModel : public Algorithm {
public:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_time;
  MemBus::Transmition transmition;
  int id;
  int nodes_in_ring;
  std::map<int, int> _stream_count;
  std::atomic<int> send_packets;
  std::atomic<int> recv_packets;
  int parallel_reduce;
  std::map<std::pair<int, int>, std::list<MyPacket>> packets;
  bool toggle;
  std::map<std::pair<int, int>, int> free_packets;
  bool processed;
  bool send_back;
  bool NPU_to_MA;

  std::map<int, int> indegree_mapping;
  std::map<int, int> inprocessing_indegree;
  std::map<int, int>* zero_latency_packets;
  std::map<int, int>* non_zero_latency_packets;
  MockNccl::FlowModels _flow_models;
  uint32_t m_channels;
  uint32_t len_channel;
#if PHY_RDMA
  // std::condition_variable judge_exit_cv;
  // std::mutex judge_exit_mutex;
  // std::mutex judge_mutex;
  std::atomic<bool> judge_exit_flag;
#endif

  NcclTreeFlowModel() {}
  ~NcclTreeFlowModel() override {}

  NcclTreeFlowModel(
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
  void run(EventType event, CallData* data) override;
  void process_stream_count(int channel_id);
  void release_packets(int channel_id, int flow_id, uint64_t message_size) const;
  void reduce(int channel_id, int flow_id);
  virtual int get_non_zero_latency_packets();
  void insert_packets(int channel_id, int flow_id);
  void init_indegree_mapping();
  bool ready(int channel_id, int flow_id);
  bool recv_ready(int channel_id, int flow_id);
  bool init_recv_ready();
  void exit() override;
#ifdef PHY_MTP
  bool phy_iteratable(int channel_id);
  bool phy_ready(int channel_id, int flow_id);
  void waiting_to_exit();
#endif
  class FlowCriticalSection {
  public:
    /**
     * @brief Acquires the lock upon construction.
     *
     * Spins in a loop until it successfully sets the atomic flag from 'false'
     * to 'true', effectively waiting until the lock is free and then taking it.
     * The 'acquire' memory order ensures that later memory operations are
     * not moved before this point.
     */
    FlowCriticalSection() {
      while (g_flow_inCriticalSection.exchange(true, std::memory_order_acquire))
        ;
    }

    /**
     * @brief Releases the lock upon destruction.
     *
     * The destructor ensures the lock is released when the object goes out of
     * scope (RAII). The 'release' memory order ensures that preceding memory
     * operations are not moved after this point.
     */
    ~FlowCriticalSection() { this->ExitSection(); }

    FlowCriticalSection(const FlowCriticalSection&) = delete;
    FlowCriticalSection& operator=(const FlowCriticalSection&) = delete;
    FlowCriticalSection(FlowCriticalSection&&) = delete;
    FlowCriticalSection& operator=(FlowCriticalSection&&) = delete;

  private:
    /**
     * @brief Releases the lock.
     *
     * This method is private as it should never be called outside the destructor
     * to prevent a thread from releasing the lock twice (possibly releasing it while
     * someone else was holding it).
     */
    static void ExitSection() { g_flow_inCriticalSection.store(false, std::memory_order_release); }
  };

  /**
   * Unlike FlowCriticalSection, this class does not automatically release the
   * lock on destruction. The caller must explicitly call ExitSection() to
   * release the lock once the critical section is complete.
   */
  class FlowExplicitCriticalSection {
  public:
    /**
     * @brief Acquires the lock upon construction.
     *
     * Spins in a loop until it successfully sets the atomic flag from 'false'
     * to 'true', effectively waiting until the lock is free and then taking it.
     * The 'acquire' memory order ensures that later memory operations are
     * not moved before this point.
     */
    FlowExplicitCriticalSection() {
      while (g_flow_inCriticalSection.exchange(true, std::memory_order_acquire))
        ;
    }

    /**
     * @brief Default destructor that does not release the lock.
     *
     * This destructor intentionally does not modify the lock state. This allows
     * the user to control precisely when the lock is released by calling
     * ExitSection() at the appropriate time. Failing to call ExitSection()
     * after entering the critical section results in the lock remaining held.
     */
    ~FlowExplicitCriticalSection() = default;

    /**
     * @brief Releases the lock explicitly.
     *
     * The user must call this static method to release the lock acquired in the
     * constructor.
     *
     * Calling ExitSection() without having successfully acquired the lock first
     * results in undefined behavior, as it would release a lock that the caller
     * does not hold.
     */
    static void ExitSection() { g_flow_inCriticalSection.store(false, std::memory_order_release); }
  };

  static std::atomic<bool> g_flow_inCriticalSection;
};
} // namespace AstraSim
#endif
