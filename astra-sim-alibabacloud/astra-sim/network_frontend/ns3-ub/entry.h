/* astra-sim/network_frontend/ns3-ub/entry.h
 *
 * Entry-point helpers for the ns3-ub frontend:
 *   - setup_ns3_ub_simulation():  reads SimAI topo file, builds UB network
 *   - ub_send_flow():             dispatches a send via UbApp
 *   - ub_task_completed_callback(): invoked by UbApp WQE trace-source
 *   - check_sim_finish():         drains pending tasks before Simulator::Stop
 */
#ifndef __UB_ENTRY_H__
#define __UB_ENTRY_H__

#include "common.h"                         
#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "ns3/ub-app.h"
#include "ns3/ub-traffic-gen.h"
#include "ns3/ub-transport.h"
#include "ns3/ub-network-address.h"
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#include <map>
#include <atomic>
#include <mutex>

using namespace ns3;
using namespace utils;
using namespace std;

class MsgEvent {
public:
  int src;
  int dst;
  int type;
  uint64_t remaining_bytes;
  void (*msg_handler)(void *fun_arg);
  void *fun_arg;
  int tag;
  double schTime;

  MsgEvent(
      const int _src,
      const int _dst,
      const int _type,
      const uint64_t _remaining_msg_bytes,
      void (*_msg_handler)(void* fun_arg),
      void* _fun_arg,
      const int _tag = 0,
      const double _schTime = 0) :
      src(_src), dst(_dst), type(_type), remaining_bytes(_remaining_msg_bytes),
      msg_handler(_msg_handler), fun_arg(_fun_arg), tag(_tag), schTime(_schTime) {}

  MsgEvent() : src(0), dst(0), type(0), remaining_bytes(0), msg_handler(nullptr), fun_arg(nullptr), tag(0), schTime(0) {}

  void callHandler() const {
    if (msg_handler) msg_handler(fun_arg);
  }
};

/**
 * MsgEventKey is a key to uniquely identify each MsgEvent.
 *  - Pair <Tag, Pair <src_id, dst_id>>
 */
typedef pair<int, pair<int, int>> MsgEventKey;

// Pending UB send tasks: taskId → task context
inline std::map<uint32_t, MsgEvent> pending_ub_tasks;
inline uint32_t global_ub_task_id = 1;

inline std::map<MsgEventKey, MsgEvent>    expeRecvHash;    // waiting for data to arrive
inline std::map<MsgEventKey, uint64_t>    recvHash;        // data arrived before sim_recv
inline std::map<MsgEventKey, MsgEvent>    sentHash;        // outstanding sends


struct user_param {
    int thread;
    std::string workload;
    std::string network_topo;   
    std::string system_conf;
    std::string network_conf;
    std::string run_name;

    user_param() {
        thread       = 1;
        workload     = "";
        network_topo = "";
        system_conf  = "astra-sim-alibabacloud/inputs/system/default.txt";
        run_name     = "";
    }
    ~user_param() {}
};

static std::once_flag    sim_finished;
inline std::atomic<bool> waiting_sim_finish(false);

inline void ub_send_flow(int src, int dst, uint64_t count, int tag,
                          void (*msg_handler)(void*), void* fun_arg) {
    Ptr<Node> sourceNode = NodeList::GetNode(src);
    Ptr<UbApp> app = DynamicCast<UbApp>(sourceNode->GetApplication(0));
    if (!app) {
        std::cerr << "[UB] ERROR: no UbApp on node " << src << std::endl;
        return;
    }

    uint32_t taskId;
    {
#ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
#endif
        MsgEvent t(src, dst, 0, count, msg_handler, fun_arg, tag, ns3::Simulator::Now().GetNanoSeconds());
        taskId = global_ub_task_id++;
        pending_ub_tasks[taskId] = t;
    }

    utils::TrafficRecord record;
    record.taskId       = static_cast<int>(taskId);
    record.sourceNode   = src;
    record.destNode     = dst;
    record.dataSize     = static_cast<int>(count > INT32_MAX ? INT32_MAX : count);
    record.opType       = "URMA_WRITE";
    record.priority     = 1;
    record.delay        = "1ns";
    record.phaseId      = tag;
    // Register with UbTrafficGen singleton (required by UbApp::OnTaskCompleted cleanup)
    ns3::UbTrafficGen::Get()->AddTask(record);

    app->SendTraffic(record);
}

inline void ub_task_completed_callback(FILE* fout, uint32_t nodeId, uint32_t jettyNum, uint32_t taskId) {
    //TODO Check if pending_ub_tasks can be removed. In UB, nodes have the list of tasks inside them
    auto it = pending_ub_tasks.find(taskId);   
    if (it == pending_ub_tasks.end()) return;

    MsgEvent ctx = it->second;
    pending_ub_tasks.erase(it);

    uint32_t sip = NodeIdToIp(ctx.src).Get();
    uint32_t dip = NodeIdToIp(ctx.dst).Get();
    uint64_t size = ctx.remaining_bytes;
    uint64_t start_time = static_cast<uint64_t>(ctx.schTime);
    uint64_t fct = static_cast<uint64_t>(ns3::Simulator::Now().GetNanoSeconds() - start_time);
    uint64_t standalone_fct = 0;

    // sip, dip, sport, dport, size (B), start_time, fct (ns), standalone_fct (ns)
    fprintf(fout, "%08x %08x %u %u %lu %lu %lu %lu\n", sip, dip, 0, 0, size, start_time, fct, standalone_fct);
    fflush(fout);

    // Notify sender callback
    ctx.callHandler();

    // Notify receiver (if sim_recv already posted)
    MsgEventKey key = {ctx.tag, {ctx.src, ctx.dst}};
    {
#ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
#endif
        sentHash.erase(key);
        if (expeRecvHash.count(key)) {
            MsgEvent& r = expeRecvHash[key];
            if (ctx.remaining_bytes >= r.remaining_bytes) {
                if (ctx.remaining_bytes > r.remaining_bytes)
                    recvHash[key] += ctx.remaining_bytes - r.remaining_bytes;
                auto cb = r.msg_handler;
                auto arg = r.fun_arg;
                expeRecvHash.erase(key);
                if (cb) cb(arg);
            } else {
                r.remaining_bytes -= ctx.remaining_bytes;
            }
        } else {
            recvHash[key] += ctx.remaining_bytes;
        }
    }

    // Reschedule drain check
    Simulator::Schedule(Time(0), []() {
        if (waiting_sim_finish && pending_ub_tasks.empty() && expeRecvHash.empty()) {
            std::call_once(sim_finished, []() {
                std::cout << "[UB] All messages finished. Stopping simulation at "
                          << AstraSim::Sys::boostedTick() << " ns." << std::endl;
                Simulator::Stop();
            });
        }
    });
}

inline void check_sim_finish() {
    if (waiting_sim_finish && pending_ub_tasks.empty() && expeRecvHash.empty()) {
        std::call_once(sim_finished, []() {
            std::cout << "[UB] All messages finished. Stopping simulation at "
                      << AstraSim::Sys::boostedTick() << " ns." << std::endl;
            Simulator::Stop();
        });
    }
}

inline int setup_ns3_ub_simulation(const user_param& param) {
    RngSeedManager::SetSeed(10);

    if (!ReadConf(param.network_conf, param.run_name)) {
        std::cerr << "Unable to open configuration file: " << param.network_conf << std::endl;
        std::cerr << "This error is fatal." << std::endl;
        exit(1);
    }

    int gpu_count = SetupUBNetwork(param.network_topo, ub_task_completed_callback);
    if (gpu_count < 0) {
        std::cerr << "[UB] Topology setup failed." << std::endl;
        return -1;
    }

    ComputeAndInstallRouting();

    std::cout << "[UB] Simulation setup complete. GPU nodes: " << gpu_count << std::endl;
    return gpu_count;
}

#endif /* __UB_ENTRY_H__ */