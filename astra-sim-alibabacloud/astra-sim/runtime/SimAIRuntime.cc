/*
* SimAIRuntime.cc
* In-memory simulation coordinator and resident Astra-Sim Sys manager.
*/

#include "SimAIRuntime.h"
#include "astra-sim/network_frontend/ns3/entry.h"
#include "astra-sim/system/IntData.hh"

#include <ns3/core-module.h>

#include <algorithm>
#include <fstream>
#include <future>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <thread>

#ifdef NS3_MTP
#include <ns3/mtp-interface.h>
#endif

namespace simai {

namespace {

bool is_supported_collective(const std::string& t) {
  return t == "ALLREDUCE" || t == "AllReduce" ||
         t == "ALLTOALL" || t == "AllToAll" ||
         t == "ALLGATHER" || t == "AllGather" ||
         t == "REDUCESCATTER" || t == "ReduceScatter";
}

class PerRankCompletionNotifier;

class CollectiveCompletionNotifier : public AstraSim::Callable {
public:
  SimAIRuntime* runtime;
  std::vector<int> ranks;
  std::atomic<int> pending_ranks;
  uint64_t start_tick{0};
  std::function<void(double)> callback;
  std::vector<PerRankCompletionNotifier*> per_rank_notifiers;

  CollectiveCompletionNotifier(
      SimAIRuntime* rt,
      std::vector<int> r,
      uint64_t start,
      std::function<void(double)> cb)
      : runtime(rt),
        ranks(std::move(r)),
        pending_ranks(static_cast<int>(ranks.size())),
        start_tick(start),
        callback(std::move(cb)) {}

  ~CollectiveCompletionNotifier() override;

  void call(AstraSim::EventType event, AstraSim::CallData* data) override;
};

// Mirrors the per-node accounting done by the Layer-driven path
// (Layer.cc: generator->increase_finished_streams(dataset_streams))
// so that streams_finished matches streams_injected when the runtime
// drives collectives directly instead of through a Workload/Layer.
class PerRankCompletionNotifier : public AstraSim::Callable {
public:
  AstraSim::Sys* sys;
  int dataset_streams;
  CollectiveCompletionNotifier* shared;

  PerRankCompletionNotifier(
      AstraSim::Sys* s,
      int streams,
      CollectiveCompletionNotifier* sh)
      : sys(s), dataset_streams(streams), shared(sh) {}

  void call(AstraSim::EventType event, AstraSim::CallData* data) override {
    if (sys != nullptr && dataset_streams > 0) {
      sys->streams_finished += dataset_streams;
    }
    shared->call(event, data);
  }
};

CollectiveCompletionNotifier::~CollectiveCompletionNotifier() {
  for (auto* p : per_rank_notifiers) {
    delete p;
  }
}

void CollectiveCompletionNotifier::call(AstraSim::EventType /*event*/, AstraSim::CallData* data) {
  if (data != nullptr) {
    delete data;
  }
  if (pending_ranks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    uint64_t end_tick = Simulator::Now().GetNanoSeconds();
    double duration_us = (end_tick >= start_tick)
        ? static_cast<double>(end_tick - start_tick) / 1000.0
        : 0.0;
    runtime->on_collective_finished(ranks, std::move(callback), duration_us);
    delete this;
  }
}

} // namespace

SimAIRuntime& SimAIRuntime::get_instance() {
  static SimAIRuntime instance;
  return instance;
}

SimAIRuntime::SimAIRuntime() = default;

void SimAIRuntime::shutdown() {
  shutting_down_.store(true, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lk(sim_cv_mutex_);
    sim_wakeup_ = true;
  }
  sim_cv_.notify_all();
  if (sim_thread_.joinable()) {
    sim_thread_.join();
  }

  cb_cv_.notify_all();
  if (cb_thread_.joinable()) {
    cb_thread_.join();
  }
}

SimAIRuntime::~SimAIRuntime() {
  shutdown();

  for (auto* sys : systems_) {
    delete sys;
  }
  for (auto* net : networks_) {
    delete net;
  }
}

void SimAIRuntime::ensure_dummy_workload_file(int total_gpus, int tp_size) {
  if (tp_size <= 0 || tp_size > total_gpus) {
    tp_size = total_gpus;
  }
  dummy_workload_file_ = "/tmp/simai_runtime_dummy_workload_" + std::to_string(total_gpus) + "g.txt";
  std::ofstream out(dummy_workload_file_, std::ios::trunc);
  out << "HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: " << tp_size
      << " ep: 1 pp: 1 vpp: 1 ga: 1 all_gpus: " << total_gpus
      << " checkpoints: 0 checkpoint_initiates: 0 pp_comm: 0\n";
  out << "1\n";
  out << "dummy_layer -1 1 ALLREDUCE 4096 0 NONE 0 0 NONE 0 100\n";
  out.close();
}

bool SimAIRuntime::init_network(
    const std::string& topo_file,
    const std::string& conf_file,
    const std::string& sys_conf_file,
    int num_threads,
    int min_lookahead_ns) {
  std::lock_guard<std::mutex> lock(runtime_mutex_);
  if (initialized_) {
    return true;
  }

  std::string actual_sys_conf = sys_conf_file.empty() ?
      "astra-sim-alibabacloud/inputs/system/default.txt" : sys_conf_file;

  enable_ns3_mtp(num_threads, min_lookahead_ns);
  if (setup_ns3_simulation(topo_file, conf_file, "runtime") != 0) {
    return false;
  }

  total_nodes_ = node_num - switch_num;
  total_gpus_ = node_num - nvswitch_num - switch_num;
  gpus_per_server_ = gpus_per_server;

  ensure_dummy_workload_file(total_gpus_);

  const SimAINetworkSetup setup =
      build_astra_sim_systems(actual_sys_conf, dummy_workload_file_, "runtime", GPUType::NONE);
  networks_ = setup.networks;
  systems_ = setup.systems;
  busy_ranks_.assign(total_nodes_, false);

  shutting_down_.store(false, std::memory_order_release);
  cb_thread_ = std::thread(&SimAIRuntime::callback_worker_loop, this);
  sim_thread_ = std::thread(&SimAIRuntime::sim_worker_loop, this);

  initialized_ = true;
  return true;
}

void SimAIRuntime::sim_worker_loop() {
  while (!shutting_down_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lk(sim_cv_mutex_);
      sim_cv_.wait(lk, [this]() {
        return sim_wakeup_ || shutting_down_.load(std::memory_order_acquire);
      });
      if (shutting_down_.load(std::memory_order_acquire)) {
        break;
      }
      sim_wakeup_ = false;
    }

    Simulator::Run();

    // Check if new collectives arrived just as Run() was exiting
    bool more_work = false;
    {
#ifdef NS3_MTP
      ns3::MtpInterface::CriticalSection cs;
#else
      std::lock_guard<std::mutex> lock(runtime_mutex_);
#endif
      more_work = !pending_collectives_.empty() || (active_collectives_ > 0);
    }
    if (more_work) {
#ifdef NS3_MTP
      ns3::MtpInterface::ScheduleGlobal(&SimAIRuntime::drain_and_start_ready_collectives, this);
#endif
      std::lock_guard<std::mutex> lk(sim_cv_mutex_);
      sim_wakeup_ = true;
    }
  }
}

void SimAIRuntime::callback_worker_loop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lk(cb_mutex_);
      cb_cv_.wait(lk, [this]() {
        return !completed_callbacks_.empty() || shutting_down_.load(std::memory_order_acquire);
      });
      if (completed_callbacks_.empty() && shutting_down_.load(std::memory_order_acquire)) {
        break;
      }
      task = std::move(completed_callbacks_.front());
      completed_callbacks_.pop_front();
    }
    if (task) {
      task();
    }
  }
}

void SimAIRuntime::start_collective_on_sim_thread(PendingCollective req) {
  uint64_t start_tick = Simulator::Now().GetNanoSeconds();
  auto* notifier = new CollectiveCompletionNotifier(
      this, req.ranks, start_tick, std::move(req.callback));

  int layer_id = layer_counter_++;

  for (int rank : req.ranks) {
    if (rank < 0 || rank >= total_nodes_ || systems_[rank] == nullptr) {
      continue;
    }
    AstraSim::Sys* sys = systems_[rank];
    std::vector<bool> involved_dims(sys->physical_dims.size(), true);

    AstraSim::DataSet* ds = nullptr;
    if (req.comm_type == "ALLREDUCE" || req.comm_type == "AllReduce") {
      ds = sys->generate_all_reduce(req.bytes, involved_dims, AstraSim::SchedulingPolicy::None, layer_id);
    } else if (req.comm_type == "ALLTOALL" || req.comm_type == "AllToAll") {
      ds = sys->generate_all_to_all(req.bytes, involved_dims, AstraSim::SchedulingPolicy::None, layer_id);
    } else if (req.comm_type == "ALLGATHER" || req.comm_type == "AllGather") {
      ds = sys->generate_all_gather(req.bytes, involved_dims, AstraSim::SchedulingPolicy::None, layer_id);
    } else if (req.comm_type == "REDUCESCATTER" || req.comm_type == "ReduceScatter") {
      ds = sys->generate_reduce_scatter(req.bytes, involved_dims, AstraSim::SchedulingPolicy::None, layer_id);
    }

    if (ds != nullptr) {
      auto* per_rank = new PerRankCompletionNotifier(sys, ds->total_streams, notifier);
      notifier->per_rank_notifiers.push_back(per_rank);
      ds->set_notifier(per_rank, AstraSim::EventType::General);
    }
  }
}

void SimAIRuntime::drain_and_start_ready_collectives() {
  std::vector<PendingCollective> ready;
  {
#ifdef NS3_MTP
    ns3::MtpInterface::CriticalSection cs;
#else
    std::lock_guard<std::mutex> lock(runtime_mutex_);
#endif
    for (auto it = pending_collectives_.begin(); it != pending_collectives_.end();) {
      bool conflict = false;
      for (int r : it->ranks) {
        if (r >= 0 && r < total_nodes_ && busy_ranks_[r]) {
          conflict = true;
          break;
        }
      }
      if (!conflict) {
        for (int r : it->ranks) {
          if (r >= 0 && r < total_nodes_) {
            busy_ranks_[r] = true;
          }
        }
        active_collectives_++;
        ready.push_back(std::move(*it));
        it = pending_collectives_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Important: Call start_collective_on_sim_thread OUTSIDE MtpInterface::CriticalSection
  // because NcclFlowModel::init_recv_ready() -> ASTRASimNetwork::sim_recv() acquires
  // MtpInterface::ExplicitCriticalSection internally.
  for (auto& req : ready) {
    start_collective_on_sim_thread(std::move(req));
  }
}

void SimAIRuntime::on_collective_finished(
    const std::vector<int>& ranks,
    std::function<void(double)> callback,
    double duration_us) {
  if (callback) {
    {
      std::lock_guard<std::mutex> lk(cb_mutex_);
      completed_callbacks_.emplace_back([cb = std::move(callback), duration_us]() {
        cb(duration_us);
      });
    }
    cb_cv_.notify_one();
  }

  bool has_pending = false;
  bool should_stop = false;
  {
#ifdef NS3_MTP
    ns3::MtpInterface::CriticalSection cs;
#else
    std::lock_guard<std::mutex> lock(runtime_mutex_);
#endif
    for (int r : ranks) {
      if (r >= 0 && r < total_nodes_) {
        busy_ranks_[r] = false;
      }
    }
    active_collectives_--;
    has_pending = !pending_collectives_.empty();
    should_stop = (active_collectives_ == 0 && !has_pending);
  }

  if (has_pending) {
#ifdef NS3_MTP
    ns3::MtpInterface::ScheduleGlobal(&SimAIRuntime::drain_and_start_ready_collectives, this);
#else
    drain_and_start_ready_collectives();
#endif
  } else if (should_stop) {
#ifdef NS3_MTP
    // on_collective_finished runs on a worker LP; hop to the public LP to
    // schedule the (possibly delayed) stop event.
    ns3::MtpInterface::ScheduleGlobal(&SimAIRuntime::schedule_gap_stop, this);
#else
    schedule_gap_stop();
#endif
  }
}

void SimAIRuntime::schedule_gap_stop() {
  uint64_t gap_ns = compute_gap_ns_.load(std::memory_order_relaxed);
  if (gap_ns > 0) {
    // Keep the simulator running for the compute gap so DCQCN recovery timers
    // (RateIncEventTimer / UpdateAlpha) fire and the network converges to
    // steady-state before the next collective. duration_us was already
    // captured at collective completion, so this does not pollute latency.
    Simulator::Schedule(NanoSeconds(gap_ns), &SimAIRuntime::gap_stop_handler, this);
  } else {
    Simulator::Stop();
  }
}

void SimAIRuntime::gap_stop_handler(void* /*arg*/) {
  Simulator::Stop();
}

void SimAIRuntime::submit_collective(
    const std::string& comm_type,
    std::vector<int> ranks,
    uint64_t bytes,
    std::function<void(double)> callback) {
  {
#ifdef NS3_MTP
    ns3::MtpInterface::CriticalSection cs;
#else
    std::lock_guard<std::mutex> lock(runtime_mutex_);
#endif
    pending_collectives_.push_back(PendingCollective{
        comm_type,
        std::move(ranks),
        bytes,
        std::move(callback)});
  }

#ifdef NS3_MTP
  ns3::MtpInterface::ScheduleGlobal(&SimAIRuntime::drain_and_start_ready_collectives, this);
#else
  drain_and_start_ready_collectives();
#endif

  {
    std::lock_guard<std::mutex> lk(sim_cv_mutex_);
    sim_wakeup_ = true;
  }
  sim_cv_.notify_one();
}

double SimAIRuntime::predict_collective(
    const std::string& comm_type,
    const std::vector<int>& ranks,
    uint64_t bytes,
    uint64_t compute_gap_us) {
  if (!initialized_) {
    throw std::runtime_error("[SimAIRuntime] predict_collective called before init_network");
  }

  compute_gap_ns_.store(compute_gap_us * 1000, std::memory_order_relaxed);

  std::vector<int> actual_ranks = ranks;
  if (actual_ranks.empty()) {
    actual_ranks.resize(total_gpus_);
    std::iota(actual_ranks.begin(), actual_ranks.end(), 0);
  }

  if (actual_ranks.size() <= 1 || bytes == 0) {
    return 0.0;
  }

  if (!is_supported_collective(comm_type)) {
    throw std::invalid_argument("[SimAIRuntime] Unsupported collective type: " + comm_type);
  }

  auto prom = std::make_shared<std::promise<double>>();
  std::future<double> fut = prom->get_future();

  submit_collective(
      comm_type,
      std::move(actual_ranks),
      bytes,
      [prom](double duration_us) {
        prom->set_value(duration_us);
      });

  return fut.get();
}

void SimAIRuntime::post_collective(
    [[maybe_unused]] uint64_t batch_id,
    const std::string& comm_type,
    const std::vector<int>& ranks,
    uint64_t bytes,
    uint64_t compute_gap_us,
    std::function<void(double)> completion_callback) {
  if (!initialized_) {
    throw std::runtime_error("[SimAIRuntime] post_collective called before init_network");
  }

  compute_gap_ns_.store(compute_gap_us * 1000, std::memory_order_relaxed);

  std::vector<int> actual_ranks = ranks;
  if (actual_ranks.empty()) {
    actual_ranks.resize(total_gpus_);
    std::iota(actual_ranks.begin(), actual_ranks.end(), 0);
  }

  if (actual_ranks.size() <= 1 || bytes == 0) {
    if (completion_callback) {
      completion_callback(0.0);
    }
    return;
  }

  if (!is_supported_collective(comm_type)) {
    throw std::invalid_argument("[SimAIRuntime] Unsupported collective type: " + comm_type);
  }

  submit_collective(
      comm_type,
      std::move(actual_ranks),
      bytes,
      std::move(completion_callback));
}

double SimAIRuntime::current_time_ns() const {
  if (!initialized_) {
    return 0.0;
  }
  return static_cast<double>(Simulator::Now().GetNanoSeconds());
}

void SimAIRuntime::inject_link_failure(int node_a, int node_b, double /*duration_us*/) {
  if (initialized_ && node_a >= 0 && node_b >= 0 &&
      static_cast<uint32_t>(node_a) < n.GetN() && static_cast<uint32_t>(node_b) < n.GetN()) {
    TakeDownLink(n, n.Get(node_a), n.Get(node_b));
  }
}

} // namespace simai
