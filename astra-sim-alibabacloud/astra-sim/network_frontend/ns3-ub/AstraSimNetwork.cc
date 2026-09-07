/* astra-sim/network_frontend/ns3-ub/AstraSimNetwork.cc
 *
 * Main entry point for SimAI-UB (Unified Bus) simulation mode.
 *
 * Topology input:  standard SimAI topology file (gen_Topo_Template.py output)
 * Config input:    astra-sim system config (same as RDMA mode)
 * Network backend: UbSwitch / UbPort / UbLink — NO UbUtils CSV pipeline\
 */
#include "AstraSimNetwork.h"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/SendPacketEventHandlerData.hh"
#include "entry.h"
#include "ns3/core-module.h"
#include <iomanip>
#include <iostream>
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif

#define RESULT_PATH "./csv_files/ncclFlowModel_"

using namespace std;
using namespace ns3;

// =============================================================================
// ASTRASimNetwork — AstraNetworkAPI implementation for the UB backend
// =============================================================================

int ASTRASimNetwork::sim_finish() {
    // Signal that all Sys workloads are done; drain remaining UB events.
    #ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
    #endif
    for (auto it = nodeHash.begin(); it != nodeHash.end(); ++it) {
        pair<int, int> p = it->first;
        if (p.second == 0) {
            std::cout << "sim_finish on sent, " << " Thread id: " << pthread_self() << std::endl;
            cout << "All data sent from node " << p.first << " is " << it->second
                << "\n";
        } else {
            std::cout << "sim_finish on received, " << " Thread id: " << pthread_self() << std::endl;
            cout << "All data received by node " << p.first << " is " << it->second
                << "\n";
        }
    }
    waiting_sim_finish = true;
    std::cout << "[UB] sim_finish: waiting for pending UB messages to drain..." << std::endl;
    Simulator::Schedule(Time(0), &check_sim_finish);
    return 0;
}

AstraSim::timespec_t ASTRASimNetwork::sim_get_time() {
    AstraSim::timespec_t ts;
    ts.time_val = Simulator::Now().GetNanoSeconds();
    ts.time_res = AstraSim::NS;
    return ts;
}

void ASTRASimNetwork::sim_schedule(AstraSim::timespec_t delta,
                                   void (*fun_ptr)(void* fun_arg),
                                   void* fun_arg) {
    Simulator::ScheduleWithContext(rank, NanoSeconds(delta.time_val), fun_ptr, fun_arg);
}

int ASTRASimNetwork::sim_send(void* /*buffer*/, uint64_t message_size, int /*type*/,
                               int dst, int tag,
                               AstraSim::sim_request* /*request*/,
                               void (*msg_handler)(void* fun_arg), void* fun_arg) {
    int actual_dst = dst + npu_offset;
    const auto ehd = static_cast<AstraSim::SendPacketEventHandlerData*>(fun_arg);
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, " sim_send to %d  from %d flow_id %d", dst, rank, ehd->flow_id);

    {
#ifdef NS3_MTP
        MtpInterface::CriticalSection cs;
#endif
        MsgEvent send_event(rank, actual_dst, 0, message_size, msg_handler, fun_arg, tag, ehd->flow_id);
        sentHash[MsgEventKey{tag, {send_event.src, send_event.dst}}] = send_event;
    }
    
    // Force the usage of multiple Jettys (equivalent of QP), based on the maximum segment size accepted by the UB's Tasks
    uint64_t max_chunk_size = 2000000000;
    int num_chunks = (message_size + max_chunk_size - 1) / max_chunk_size;
    
    for (int i = 0; i < num_chunks; i++) {
        uint64_t chunk_size = (i == num_chunks - 1) ? (message_size - i * max_chunk_size) : max_chunk_size;
        
        {
#ifdef NS3_MTP
            MtpInterface::CriticalSection cs;
#endif
            waiting_to_sent_callback[FlowIdKey{ehd->flow_id, {rank, actual_dst}}]++;
            waiting_to_notify_receiver[FlowIdKey{ehd->flow_id, {rank, actual_dst}}]++;
            waiting_to_message_finish[FlowIdKey{ehd->flow_id, {rank, actual_dst}}]++;
        }
        
        int send_lat = 6;
        if (const char* send_lat_env = std::getenv("AS_SEND_LAT")) {
            try {
                send_lat = std::stoi(send_lat_env);
            } catch (const std::invalid_argument&) {
                NcclLog->writeLog(NcclLogLevel::ERROR,"send_lat set error");
                exit(-1);
            }
        }
        send_lat *= 1000;
        MsgEvent se(rank, actual_dst, 0, chunk_size, msg_handler, fun_arg, tag, ehd->flow_id);
        Simulator::ScheduleWithContext(rank, Time(send_lat+1), ub_send_flow, se);

    }
    
    return 0;
}

int ASTRASimNetwork::sim_recv(void* /*buffer*/, uint64_t message_size, int /*type*/,
                               int src, int tag,
                               AstraSim::sim_request* /*request*/,
                               void (*msg_handler)(void* fun_arg), void* fun_arg) {
    int actual_src = src + npu_offset;

    const auto ehd = static_cast<AstraSim::RecvPacketEventHadndlerData*>(fun_arg);
    MsgEvent recv_event(actual_src, rank, 1, message_size, msg_handler, fun_arg, tag, ehd->flow_id);
    MsgEventKey key = {tag, {actual_src, rank}};

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(
        NcclLogLevel::DEBUG,
        "[Receive event registration] src %d sim_recv on rank %d tag %u channel id %d (flow_id %u)",
        actual_src, rank, tag, ehd->channel_id, ehd->flow_id);


#ifdef NS3_MTP
    MtpInterface::ExplicitCriticalSection ecs;
#endif
    if (recvHash.count(key)) {
        uint64_t arrived = recvHash[key];
        if (arrived == message_size) {
            recvHash.erase(key);
#ifdef NS3_MTP
            ecs.ExitSection();
#endif
            NcclLog->writeLog(
            NcclLogLevel::DEBUG,
                " [Message arrived early, skip registering] recvHash already had the expected bytes for src %d, dst %d,"
                " tag %u; directly invoke handler: t.message_size %llu, tag %u, flow_id %d",
                recv_event.src, recv_event.dst, tag, message_size, ehd->flow_id);
            recv_event.callHandler();
            return 0;
        } else if (arrived > message_size) {
            recvHash[key] = arrived - message_size;
#ifdef NS3_MTP
            ecs.ExitSection();
#endif
            NcclLog->writeLog(
            NcclLogLevel::DEBUG,
                " [Message arrived early (more left), skip registering] recvHash had more bytes (%u) than expected for "
                "src %d, dst %d, tag %u, directly invoke handler for them: t.message_size %llu, tag %u, flow_id %d",
                arrived, recv_event.src, recv_event.dst, tag, message_size, ehd->flow_id);
            recv_event.callHandler();
            return 0;
        } else {
            // Partial arrival — consume and keep waiting for remainder
            recvHash.erase(key);
            recv_event.remaining_bytes -= arrived;
            expeRecvHash[key] = recv_event;
        }
    } else if (!expeRecvHash.count(key)){ 
        expeRecvHash[key] = recv_event;
        NcclLog->writeLog(
            NcclLogLevel::DEBUG,
            " [Message not arrived yet, registering] recvHash had no entry for src %d, dst %d, tag %u; register the"
            " message in expeRecvHash: t.message_size %llu, flow_id %d",
            recv_event.src, recv_event.dst, tag, message_size, ehd->flow_id);
    }
    else 
            expeRecvHash[key].remaining_bytes += message_size;

#ifdef NS3_MTP
    ecs.ExitSection();
#endif
    return 0;
}

static int parse_user_params(int argc, char* argv[], user_param* p) {
    int opt;
    while ((opt = getopt(argc, argv, "ht:w:n:s:r:c:")) != -1) {
        switch (opt) {
        case 'h':
            std::cout << "SimAI-UB: SimAI with Unified Bus network backend\n"
                      << "  -t  threads (default 1)\n"
                      << "  -w  workload file\n"
                      << "  -n  SimAI topology file (same format as RDMA mode)\n"
                      << "  -c  network conf\n"
                      << "  -s  system config file\n"
                      << "  -r  run name\n";
            return 1;
        case 't': p->thread       = std::stoi(optarg); break;
        case 'w': p->workload     = optarg; break;
        case 'n': p->network_topo = optarg; break;
        case 's': p->system_conf  = optarg; break;
        case 'r': p->run_name     = optarg; break;
        case 'c': p->network_conf     = optarg; break;
        default:
            std::cerr << "Unknown option. Use -h for help.\n";
            return 1;
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
    user_param param;
    if (parse_user_params(argc, argv, &param)) return 0;

    MockNcclLog::set_log_name("SimAI" + (param.run_name.empty() ? "" : "." + param.run_name) + ".log");
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::INFO, "init SimAI-UB log");

#ifdef NS3_MTP
    MtpInterface::Enable(param.thread);
    if (param.thread == 1) 
        GlobalValue::Bind("PartitionSchedulingMethod", StringValue ("ByPendingEventCount"));
#endif

    // --- Build UB network from SimAI topology file (no UbUtils) ---
    int gpu_count = setup_ns3_ub_simulation(param);
    if (gpu_count <= 0) {
        std::cerr << "[UB] ERROR: Network setup failed or zero GPU nodes found.\n";
        return 1;
    }

    LogComponentEnable("SIMAI_UB_SIMULATION", LOG_LEVEL_INFO);
    LogComponentEnable("UbApp", LOG_LEVEL_INFO);
    LogComponentEnable("UbTrafficGen", LOG_LEVEL_INFO);
    LogComponentEnable("UbTransportChannel", LOG_LEVEL_INFO);

    // --- Derive Sys parameters from the parsed topology ---
    //
    // In the UB model, ALL inter-accelerator connectivity (both intra-host
    // scale-up and inter-host scale-out) goes over Unified Bus links.
    // There are NO NVSwitches and NO NVLink in the UB topology:
    //   - node.csv / SimAI topology only have DEVICE and SWITCH node types
    //   - the unified-bus module has zero references to nvswitch/nvlink
    //
    // However, MockNcclGroup::MockNcclGroup() indexes _NVSwitch[node_idx]
    // for every server node_idx (0..nNodes-1) when building TP/DP/EP groups.
    // It uses the NVSwitch ID to set GroupIndex membership.  When there are
    // no physical NVSwitches we use a self-mapping:
    //     server_i  ──►  server_i   (each GPU server acts as its own "switch")
    // This is equivalent to what the RDMA frontend does for flat (no-NVSwitch)
    // topologies.
    int nodes_num      = gpu_count;                 // one Sys per GPU host
    GPUType gpu_type   = ub_gpu_type;
    int gpus_per_server = static_cast<int>(ub_gpus_per_server);
    int n_servers      = nodes_num / gpus_per_server;  // physical server count

    // Build NVSwitch offset-mapping: nv_map[server_idx] = nodes_num + server_idx
    // MockNcclGroup uses _NVSwitch[node_idx] to populate GroupIndex entries
    // for NVSwitch nodes (lines 83, 104, 151, 178). These IDs MUST NOT overlap
    // with GPU rank IDs (0..nodes_num-1), otherwise GroupIndex entries for GPU
    // ranks get overwritten, corrupting ring generation and causing the
    // simulation to hang. Placing them at nodes_num+ matches the RDMA
    // frontend convention (NVSwitch IDs = gpu_num + server_idx).
    std::vector<int> nv_map;       // length = n_servers, indexed by server idx
    for (int s = 0; s < n_servers; s++)
        nv_map.push_back(nodes_num + s);   // offset: IDs after all GPU ranks

    std::cout << "--- UB Topology Summary ---\n"
              << "  Total nodes   : " << ub_node_num      << "\n"
              << "  GPU nodes     : " << nodes_num         << "\n"
              << "  GPUs/server   : " << gpus_per_server   << "\n"
              << "  Servers       : " << n_servers          << "\n"
              << "  Switches      : " << ub_switch_num     << "\n"
              << "  NVSwitches    : none (UB uses Unified Bus for all links)\n"
              << "  GPU type      : " << static_cast<int>(gpu_type) << "\n"
              << "---------------------------\n";

    // --- Instantiate ASTRASimNetwork + Sys for each GPU node ---
    std::vector<ASTRASimNetwork*>  networks(nodes_num, nullptr);
    std::vector<AstraSim::Sys*>    systems(nodes_num,  nullptr);

    // Map GPU host index → ns-3 node index
    // Host nodes are the ones with type==0; collect them in order.
    std::vector<uint32_t> host_node_ids;
    for (uint32_t i = 0; i < ub_node_num; i++)
        if (ub_node_type_vec[i] == 0) host_node_ids.push_back(i);

    // NOTE: Create one Sys more to handle last timestamp reading. 
    for (int j = 0; j < nodes_num+1; j++) {
        networks[j] = new ASTRASimNetwork(j, /*npu_offset=*/0);
        systems[j] = new AstraSim::Sys(
            networks[j],         // network
            nullptr,             // memory (unused)
            j,                   // id
            0,                   // npu_offset
            1,                   // num_passes
            {nodes_num},         // dims   — flat 1-D collective domain
            {1},                 // queues_per_dim
            "",                  // sys_string
            param.workload,      // workload
            1,                   // comm_scale
            1,                   // compute_scale
            1,                   // injection_scale
            1,                   // total_stat_rows
            0,                   // stat_row
            RESULT_PATH,         // path
            param.run_name,      // run_name
            true,                // separate_log
            false,               // rendezvous_enabled
            gpu_type,            // _gpu_type
            {gpu_count},         // _all_gpus
            nv_map,              // _NVSwitchs — self-mapping (no physical NVSwitches)
            gpus_per_server      // _ngpus_per_node
        );
        // These must be set after construction (RDMA frontend does the same).
        // nvswitch_id: MockNcclGroup uses this for per-GPU group membership.
        //   With UB self-mapping, GPU j maps to "switch" j.
        systems[j]->nvswitch_id = nv_map[j / gpus_per_server];
        // num_gpus: Sys::~Sys() iterates all_generators[0..num_gpus-1] to decide
        //   when to exit the simulation loop. Must equal total GPU count.
        systems[j]->num_gpus = nodes_num;
    }
    
    // --- Uncomment the following for a Manual connectivity test 
//     ns3::Simulator::Schedule(ns3::NanoSeconds(100), []() {
       

//         auto completion_cb = [](void* arg) {
//             std::cout << "[DEBUG] Transmission SUCCESS! Received at "
//                       << ns3::Simulator::Now().GetNanoSeconds() << " ns." << std::endl;
//             ns3::Simulator::Stop();
//         };
        
//         MsgEvent se(/*src=*/0, /*dst=*/8, 0, /*chunk_size*/4096, completion_cb, nullptr, /*tag*/0, /*flow_id*/0);
//         std::cout << "[DEBUG] Injecting a 4096-byte flow from GPU " << se.src << " to GPU " << se.dst << "..." << std::endl;
//         {
// #ifdef NS3_MTP
//             MtpInterface::CriticalSection cs;
// #endif
//             sentHash[MsgEventKey{se.tag, {se.src, se.dst}}] = se;
//             waiting_to_sent_callback[FlowIdKey{se.flow_id, {se.src, se.dst}}]++;
//             waiting_to_notify_receiver[FlowIdKey{se.flow_id, {se.src, se.dst}}]++;
//             waiting_to_message_finish[FlowIdKey{se.flow_id, {se.src, se.dst}}]++;
//         }
        
//         int send_lat = 2;
//         ub_send_flow(se);
//     });

//     ns3::Simulator::Schedule(ns3::MilliSeconds(100), []() {
//         std::cout << "[DEBUG] Timeout reached. Transmission hung or routing failed!" << std::endl;
//         ns3::Simulator::Stop();
//     });


    // --- Fire workloads ---
    for (int i = 0; i < nodes_num; i++)
        systems[i]->workload->fire();

    std::cout << "[UB] Running simulation..." << std::endl;
    Simulator::Stop(Seconds(simulator_stop_time));
    Simulator::Run();
    Simulator::Destroy();

    #ifdef NS3_MPI
    MpiInterface::Disable();
    #endif
    return 0;
}