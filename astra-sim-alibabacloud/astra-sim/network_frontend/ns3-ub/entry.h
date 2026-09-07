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
  int flow_id;
  double schTime;

  MsgEvent(
      const int _src,
      const int _dst,
      const int _type,
      const uint64_t _remaining_msg_bytes,
      void (*_msg_handler)(void* fun_arg),
      void* _fun_arg,
      const int _tag = 0,
      const int _flow_id = 0,
      const double _schTime = 0) :
      src(_src), dst(_dst), type(_type), remaining_bytes(_remaining_msg_bytes),
      msg_handler(_msg_handler), fun_arg(_fun_arg), tag(_tag), flow_id(_flow_id), schTime(_schTime) {}

  MsgEvent() : src(0), dst(0), type(0), remaining_bytes(0), msg_handler(nullptr), fun_arg(nullptr), tag(0), flow_id(0), schTime(0) {}

  void callHandler() const {
    if (msg_handler) msg_handler(fun_arg);
  }
};

/**
 * MsgEventKey is a key to uniquely identify each MsgEvent.
 *  - Pair <Tag, Pair <src_id, dst_id>>
 */
typedef pair<int, pair<int, int>> MsgEventKey;

/**
 * FlowIdKey is a key to uniquely identify a flow
 *  - Pair <flow_id, Pair <src_id, dst_id>>
 */
typedef pair<int, pair<int, int>> FlowIdKey;

// Pending UB send tasks: taskId → task context
inline std::map<uint32_t, MsgEvent> pending_ub_tasks;
inline uint32_t global_ub_task_id = 1;

inline std::map<MsgEventKey, MsgEvent>    expeRecvHash;    // waiting for data to arrive
inline std::map<MsgEventKey, uint64_t>    recvHash;        // data arrived before sim_recv
inline std::map<MsgEventKey, MsgEvent>    sentHash;        // outstanding sends

inline std::map<FlowIdKey, int> waiting_to_sent_callback;
inline std::map<FlowIdKey, int> waiting_to_notify_receiver;
inline std::map<FlowIdKey, int> waiting_to_message_finish;

inline std::map<FlowIdKey, uint64_t> received_chunksize;
inline std::map<FlowIdKey, uint64_t> sent_chunksize;
inline std::map<std::pair<int, bool>, uint64_t> nodeHash;

inline bool is_sending_finished(int src, int dst, int flow_id) {
  if (waiting_to_sent_callback.count(FlowIdKey{flow_id, {src, dst}})) {
    if (--waiting_to_sent_callback[FlowIdKey{flow_id, {src, dst}}] == 0) {
      waiting_to_sent_callback.erase(FlowIdKey{flow_id, {src, dst}});
      return true;
    }
  }
  return false;
}

inline bool is_receive_finished(int src, int dst, int flow_id) {
  if (waiting_to_notify_receiver.count(FlowIdKey{flow_id, {src, dst}})) {
    if (--waiting_to_notify_receiver[FlowIdKey{flow_id, {src, dst}}] == 0) {
      waiting_to_notify_receiver.erase(FlowIdKey{flow_id, {src, dst}});
      return true;
    }
  }
  return false;
}

inline bool is_message_finished(int src, int dst, int flow_id) {
  if (waiting_to_message_finish.count(FlowIdKey{flow_id, {src, dst}})) {
    if (--waiting_to_message_finish[FlowIdKey{flow_id, {src, dst}}] == 0) {
      waiting_to_message_finish.erase(FlowIdKey{flow_id, {src, dst}});
      return true;
    }
  }
  return false;
}


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

inline void ub_send_flow(MsgEvent send_event){
    Ptr<Node> sourceNode = NodeList::GetNode(send_event.src);
    Ptr<UbApp> app = DynamicCast<UbApp>(sourceNode->GetApplication(0));
    if (!app) {
        std::cerr << "[UB] ERROR: no UbApp on node " << send_event.src << std::endl;
        return;
    }

    uint32_t taskId;
    {
#ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
#endif
        MsgEvent t(send_event.src, send_event.dst, 0, send_event.remaining_bytes, send_event.msg_handler, send_event.fun_arg, send_event.tag, send_event.flow_id, ns3::Simulator::Now().GetNanoSeconds());
        taskId = global_ub_task_id++;
        pending_ub_tasks[taskId] = t;
    }

    utils::TrafficRecord record;
    record.taskId       = static_cast<int>(taskId);
    record.sourceNode   = send_event.src;
    record.destNode     = send_event.dst;
    record.dataSize     = static_cast<int>(send_event.remaining_bytes > INT32_MAX ? INT32_MAX : send_event.remaining_bytes);
    record.opType       = "URMA_WRITE";
    record.priority     = 1;
    record.delay        = "1ns";
    record.phaseId      = send_event.tag;
    // Register with UbTrafficGen singleton (required by UbApp::OnTaskCompleted cleanup)
    ns3::UbTrafficGen::Get()->AddTask(record);

    app->SendTraffic(record);
}

inline void check_sim_finish() {
    if (waiting_sim_finish && pending_ub_tasks.empty() && expeRecvHash.empty() && waiting_to_message_finish.empty()) {
        std::call_once(sim_finished, [] {
            {
                #ifdef NS3_MTP
                MtpInterface::CriticalSection cs;
                #endif
                std::cout   << "[UB] All messages finished. Stopping simulation at "
                            << AstraSim::Sys::boostedTick() << " ns." 
                            << std::endl;
            }
        });
        Simulator::Stop();
    }
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

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(
      NcclLogLevel::INFO,
      "task_completed, %d -> %d, flow_id %d, tag %u, total bytes %llu, at the tick %d",
      ctx.src,  ctx.dst, ctx.flow_id, ctx.tag, ctx.remaining_bytes, AstraSim::Sys::boostedTick());

    {
#ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
#endif
        nodeHash[make_pair((int)ctx.src, 0)] += size;
        nodeHash[make_pair((int)ctx.dst, 1)] += size;
    }

    // sip, dip, sport, dport, size (B), start_time, fct (ns), standalone_fct (ns)
    fprintf(fout, "%08x %08x %u %u %lu %lu %lu %lu\n", sip, dip, 0, 0, size, start_time, fct, standalone_fct);
    fflush(fout);

    uint64_t notify_size_send;
    {
#ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
#endif
        sent_chunksize[FlowIdKey{ctx.flow_id, {ctx.src, ctx.dst}}] += size;
        if (is_sending_finished(ctx.src, ctx.dst, ctx.flow_id)) {
            notify_size_send = sent_chunksize[FlowIdKey{ctx.flow_id, {ctx.src, ctx.dst}}];
            sent_chunksize.erase(FlowIdKey{ctx.flow_id, {ctx.src, ctx.dst}});
            MsgEventKey key = {ctx.tag, {ctx.src, ctx.dst}};
            if (sentHash.count(key)) {
                MsgEvent send_event = sentHash[key];
                if (send_event.remaining_bytes == notify_size_send) {
                    sentHash.erase(key);
                    send_event.callHandler();
                } else {
                    std::cerr << "Size mismatch in sentHash!" << std::endl;
                    exit(1);
                }
            }
        }
    }

    // NOTE For the moment use the task completed callback to signal the receiver it has finished 
    uint64_t notify_size_recv;
    {
#ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
#endif
        received_chunksize[FlowIdKey{ctx.flow_id, {ctx.src, ctx.dst}}] += size;
        NcclLog->writeLog(NcclLogLevel::DEBUG,
      "received, %d -> %d, flow_id %d, tag %u, total bytes %llu, at the tick %d",
      ctx.src,  ctx.dst, ctx.flow_id, ctx.tag, received_chunksize[FlowIdKey{ctx.flow_id, {ctx.src, ctx.dst}}], AstraSim::Sys::boostedTick());
        
        if (is_receive_finished(ctx.src, ctx.dst, ctx.flow_id)) {
            notify_size_recv = received_chunksize[FlowIdKey{ctx.flow_id, {ctx.src, ctx.dst}}];
            received_chunksize.erase(FlowIdKey{ctx.flow_id, {ctx.src, ctx.dst}});
            
            MsgEventKey key = {ctx.tag, {ctx.src, ctx.dst}};
            if (expeRecvHash.count(key)) {
                MsgEvent& r = expeRecvHash[key];
                if (notify_size_recv >= r.remaining_bytes) {
                    if (notify_size_recv > r.remaining_bytes)
                        recvHash[key] += notify_size_recv - r.remaining_bytes;
                    auto cb = r.msg_handler;
                    auto arg = r.fun_arg;
                    expeRecvHash.erase(key);
                    if (cb) {
                        Simulator::ScheduleWithContext(ctx.dst, NanoSeconds(0), cb, arg);
                    }
                } else {
                    r.remaining_bytes -= notify_size_recv;
                }
            } else {
                recvHash[key] += notify_size_recv;
            }
        }
    }

    {
#ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
#endif
        if (!is_message_finished(ctx.src, ctx.dst, ctx.flow_id)) {
            return;
        }
    }
    Simulator::Schedule(Time(0), &check_sim_finish);
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