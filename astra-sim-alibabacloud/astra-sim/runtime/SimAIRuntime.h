/*
* SimAIRuntime.h
* In-memory simulation coordinator and resident Astra-Sim Sys manager.
*
* Bridges external serving frontends (Vidur, Frontier) with Astra-Sim
* and the ns-3 packet-level network frontend.
*/

#ifndef __SIMAI_RUNTIME_HH__
#define __SIMAI_RUNTIME_HH__

#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/DataSet.hh"
#include "astra-sim/network_frontend/ns3/AstraSimNetwork.h"

#ifdef NS3_MTP
#include <ns3/mtp-interface.h>
#endif

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace simai {

class SimAIRuntime {
public:
  static SimAIRuntime& get_instance();

  /**
   * One-time resident network and Astra-Sim systems initialization.
   *
   * @param topo_file Path to network topology file.
   * @param conf_file Path to network simulation configuration file.
   * @param sys_conf_file Path to Astra-Sim system configuration file.
   * @param num_threads Worker threads for parallel simulator execution.
   * @param min_lookahead_ns Minimum lookahead in nanoseconds for LP partitioning.
   */
  bool init_network(
      const std::string& topo_file,
      const std::string& conf_file,
      const std::string& sys_conf_file = "",
      int num_threads = 1,
      int min_lookahead_ns = 50);

  /**
   * Synchronous collective execution prediction.
   *
   * @param comm_type Collective operation ("ALLREDUCE", "ALLTOALL", "ALLGATHER", "REDUCESCATTER").
   * @param ranks List of participating GPU ranks.
   * @param bytes Total message payload in bytes.
   * @param compute_gap_us Simulated compute gap in microseconds inserted between
   *        consecutive collectives (steady-state recovery window for DCQCN).
   *        0 = stop the simulator immediately when idle (warm frozen state).
   * @return Simulated execution latency in microseconds.
   */
  double predict_collective(
      const std::string& comm_type,
      const std::vector<int>& ranks,
      uint64_t bytes,
      uint64_t compute_gap_us = 0);

  /**
   * Asynchronous non-blocking collective dispatch (for continuous co-simulation).
   *
   * @param batch_id Identifier of the batch/request.
   * @param comm_type Collective operation.
   * @param ranks List of participating GPU ranks.
   * @param bytes Total message payload in bytes.
   * @param compute_gap_us Simulated compute gap in microseconds inserted between
   *        consecutive collectives (steady-state recovery window for DCQCN).
   * @param completion_callback Callback invoked when the collective completes in simulation.
   */
  void post_collective(
      uint64_t batch_id,
      const std::string& comm_type,
      const std::vector<int>& ranks,
      uint64_t bytes,
      uint64_t compute_gap_us,
      std::function<void(double)> completion_callback);

  double current_time_ns() const;
  void inject_link_failure(int node_a, int node_b, double duration_us);

  bool is_initialized() const { return initialized_; }
  int get_total_gpus() const { return total_gpus_; }
  void shutdown();

  // Internal hooks for MTP public-LP collective injection and completion
  void drain_and_start_ready_collectives();
  void on_collective_finished(
      const std::vector<int>& ranks,
      std::function<void(double)> callback,
      double duration_us);

private:
  SimAIRuntime();
  ~SimAIRuntime();

  SimAIRuntime(const SimAIRuntime&) = delete;
  SimAIRuntime& operator=(const SimAIRuntime&) = delete;

  struct PendingCollective {
    std::string comm_type;
    std::vector<int> ranks;
    uint64_t bytes;
    std::function<void(double)> callback;
  };

  void submit_collective(
      const std::string& comm_type,
      std::vector<int> ranks,
      uint64_t bytes,
      std::function<void(double)> callback);
  void start_collective_on_sim_thread(PendingCollective req);
  void sim_worker_loop();
  void callback_worker_loop();

  // Runs on the public LP (or directly in non-MTP) when the simulator goes
  // idle. If a compute gap is configured, schedules a delayed stop so DCQCN
  // recovery events fire during the gap; otherwise stops immediately.
  void schedule_gap_stop();
  static void gap_stop_handler(void* arg);

  bool initialized_{false};
  std::vector<ASTRASimNetwork*> networks_;
  std::vector<AstraSim::Sys*> systems_;
  std::mutex runtime_mutex_;
  std::string dummy_workload_file_;

  int total_nodes_{0};
  int total_gpus_{0};
  int gpus_per_server_{8};
  int layer_counter_{0};

  // Concurrent collective coordination state (guarded by MtpInterface::CriticalSection
  // under NS3_MTP, or runtime_mutex_ otherwise).
  std::deque<PendingCollective> pending_collectives_;
  std::vector<bool> busy_ranks_;
  int active_collectives_{0};

  // Steady-state compute gap (ns) inserted after a collective completes while
  // the simulator would otherwise go idle. Written by the caller before
  // submit; read by the sim/worker threads at completion. Atomic because the
  // caller thread may issue the next predict while a gap event is pending.
  std::atomic<uint64_t> compute_gap_ns_{0};

  // Persistent simulation driver thread
  std::thread sim_thread_;
  std::mutex sim_cv_mutex_;
  std::condition_variable sim_cv_;
  bool sim_wakeup_{false};
  std::atomic<bool> shutting_down_{false};

  // Dedicated asynchronous callback dispatcher thread (prevents Python callbacks
  // from blocking MTP worker barriers)
  std::thread cb_thread_;
  std::mutex cb_mutex_;
  std::condition_variable cb_cv_;
  std::deque<std::function<void()>> completed_callbacks_;

  void ensure_dummy_workload_file(int total_gpus, int tp_size = 0);
};

} // namespace simai

#endif // __SIMAI_RUNTIME_HH__
