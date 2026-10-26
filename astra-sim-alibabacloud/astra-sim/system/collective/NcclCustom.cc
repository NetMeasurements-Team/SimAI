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

#include "NcclCustom.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "SimCCL/mock/MockNcclLog.h"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/StreamBaseline.hh"
#include "astra-sim/system/PacketBundle.hh"
#include <climits>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#ifdef PHY_MTP
#include <mpi.h>
#include "astra-sim/system/phy-common/PhyMultiThread.hh"
#endif

#ifdef PHY_RDMA
#include "astra-sim/system/phy-common/SimAiFlowModelRdma.hh"
extern FlowPhyRdma flow_rdma;
#endif

namespace AstraSim {

std::atomic<bool> NcclCustom::g_flow_inCriticalSection_ncclCustom(false);

// Define the static members for graph generation
std::map<int, NcclCustom::GlobalGraphData> NcclCustom::g_graph_data_per_layer;
std::map<int, std::atomic<int>> NcclCustom::g_finished_nodes_per_layer = {};

NcclCustom::NcclCustom(
    ComType type,
    int id,
    int layer_num,
    RingTopology *ring_topology,
    uint64_t data_size,
    RingTopology::Direction direction,
    InjectionPolicy injection_policy,
    bool boost_mode,
    bool collective_graph,
    std::shared_ptr<MockNccl::FlowModels> ptr_flow_models,
    int treechannels,
    int tot_gpus,
    int algorithm,
    int protocol)
    : Algorithm(layer_num) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: NcclCustom constructor ENTRY. Layer: %d, Channels: %d", id, layer_num, treechannels);

    this->id = id;
    this->comType = type;
    this->logicalTopology = ring_topology;
    this->data_size = data_size;
    this->nodes_in_ring = ring_topology->get_nodes_in_ring();
    this->m_channels = treechannels;
    this->tot_gpus = tot_gpus;
    this->collective_graph = collective_graph;
    this->m_algorithm = algorithm;
    this->m_protocol = protocol;

    // MemBus injection state, initialized exactly as in NcclFlowModel.
    this->toggle = false;
    for (uint32_t channel_id = 0; channel_id < m_channels; channel_id++) {
        zero_latency_packets[channel_id] = 0;
        non_zero_latency_packets[channel_id] = 0;
    }

    if (ptr_flow_models) {
        this->_flow_models = *ptr_flow_models;
    }

    this->send_packets_in_flight = 0;
    this->recv_packets_in_flight = 0;
    this->sends_completed = 0;
    this->recvs_completed = 0;
    this->total_sends_to_initiate = 0;
    this->total_recvs_initiated = 0;
    
    for (const auto& flow_pair : _flow_models) {
        if (flow_pair.second.dest == this->id) {
            this->total_recvs_initiated++;
        }
        if (flow_pair.second.src == this->id) {
            this->total_sends_to_initiate++;
        }
    }
    
    NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Initialized with total_sends_to_initiate: %d, total_recvs_initiated: %d", id, total_sends_to_initiate, total_recvs_initiated);
    if (collective_graph == true){
        static std::map<int, bool> layer_reset;
        static std::mutex init_mutex;

        // Part 1: Reset global state ONCE per layer.
        // A lock ensures this section runs only for the first node that gets here for a new layer.
        {
            std::lock_guard<std::mutex> lg(init_mutex);
            if (layer_reset.find(layer_num) == layer_reset.end()) {
                g_finished_nodes_per_layer[layer_num].store(0);
                g_graph_data_per_layer[layer_num].start_times.clear();
                g_graph_data_per_layer[layer_num].end_times.clear();
                g_graph_data_per_layer[layer_num].flow_details.clear();
                layer_reset[layer_num] = true;
                NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Initialized and reset global data for layer %d.", id, layer_num);
            }
        }

        // Part 2: Aggregate flow models from EVERY node.
        // This part runs for each node to contribute its flows to the shared map.
        {
            std::lock_guard<std::mutex> lg(g_graph_data_per_layer[layer_num].mtx);
            for (const auto& flow_pair : _flow_models) {
                // Use try_emplace to add this node's flows. This correctly builds the complete map
                // from all nodes' perspectives.
                g_graph_data_per_layer[layer_num].flow_details.try_emplace(flow_pair.second.flow_id, flow_pair.second);
            }
        }
    }

    init_indegree_mapping();

    switch (type) {
        case ComType::All_Reduce: this->final_data_size = data_size; break;
        case ComType::All_Gather: this->final_data_size = data_size * nodes_in_ring; break;
        case ComType::Reduce_Scatter: this->final_data_size = data_size / nodes_in_ring; break;
        case ComType::All_to_All: this->final_data_size = data_size; break;
        default: this->final_data_size = data_size; break;
    }

    if (id == 0) {
        NcclLog->writeLog(NcclLogLevel::DEBUG, "NcclCustom algorithm initialized.");
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: NcclCustom constructor EXIT.", id);
}

NcclCustom::~NcclCustom() {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: NcclCustom destructor ENTRY.", id);
}

void NcclCustom::init_indegree_mapping() {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Taking critical section to initialize indegree mapping.", id);
    NcclCustom::FlowCriticalSection cs;
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: init_indegree_mapping ENTRY.", id);
    for (const auto& flow_pair : _flow_models) {
        const auto& flow = flow_pair.second;
        // create the init only if the flow is relevant to this node (the one I have to send to track dependencies)
        if(flow.src == this->id) {
            indegree_mapping[flow.flow_id] = flow.parent_flow_id.size();
            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Flow %d (src: %d, dest: %d) has initial indegree: %zu", id, flow.flow_id, flow.src, flow.dest, flow.parent_flow_id.size());
        }
        if(flow.src == this->id || flow.dest == this->id) {       
            // i need to retrieve the channel if I'm involved in any way     
            flow_id_to_channel_id_map[flow.flow_id] = flow.channel_id;
        }
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: init_indegree_mapping EXIT.", id);
}

// Tag used to match a send with its receive in the network frontend (keyed by tag, src, dst).
// Same layout as NcclFlowModel: layer_num * chunk_count * m_channels + channel_id * chunk_count + chunk_id.
// The layer term keeps cached flow models reused by different layers from colliding.
// Sender and receiver must compute the same value, so every term comes either from the flow
// itself (identical on both ranks) or from m_channels, which is the same on every rank.
int NcclCustom::flow_tag(const MockNccl::SingleFlow& flow) const {
    assert(flow.channel_id >= 0 && flow.channel_id < static_cast<int>(m_channels));
    assert(flow.chunk_id >= 0 && flow.chunk_id < flow.chunk_count);
    const int64_t tag =
        static_cast<int64_t>(layer_num) * flow.chunk_count * m_channels
      + static_cast<int64_t>(flow.channel_id) * flow.chunk_count
      + flow.chunk_id;
    assert(tag >= 0 && tag <= INT_MAX); // the tag is an int down to the frontend
    return static_cast<int>(tag);
}

// Hands one flow to the MemBus, mirroring NcclFlowModel::insert_packets() + release_packets().
// Per channel, packets are released in rounds: the first PARALLEL_REDUCE packet goes NPU->MA
// with no processing, the following (nodes_in_ring - 1) * PARALLEL_REDUCE go MA->NPU and,
// for reducing collectives, pay the reduction cost (needs_processing). As in NcclFlowModel,
// `toggle` is shared by all channels and flips at the start of every round, which marks the
// reduce-scatter half of an AllReduce.
// Must be called with the FlowCriticalSection held (the counters are shared state).
void NcclCustom::release_packet(int channel_id, int flow_id) {
    assert(NcclCustom::FlowCriticalSection::is_locked());
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    const auto& flow = _flow_models.at({channel_id, flow_id});

    int& zero_latency = zero_latency_packets[channel_id];
    int& non_zero_latency = non_zero_latency_packets[channel_id];
    if (zero_latency == 0 && non_zero_latency == 0) {
        zero_latency = PARALLEL_REDUCE;
        non_zero_latency = (nodes_in_ring - 1) * PARALLEL_REDUCE;
        toggle = !toggle;
    }

    bool processed;
    bool send_back;
    bool npu_to_ma;
    if (zero_latency > 0) {
        processed = false;
        send_back = false;
        npu_to_ma = true;
        zero_latency--;
    } else {
        processed = comType == ComType::Reduce_Scatter || (comType == ComType::All_Reduce && toggle);
        send_back = non_zero_latency > PARALLEL_REDUCE;
        npu_to_ma = false;
        non_zero_latency--;
    }

    // Every released flow comes back as a send (General -> front_end_sim_send -> PacketSentFinshed),
    // whichever MemBus direction it takes, so it is always counted as in flight.
    send_packets_in_flight++;
    NcclLog->writeLog(NcclLogLevel::DEBUG,
        "ID %d: Releasing flow %d on channel %d (%s, processed %d, send_back %d), send_packets_in_flight %d.",
        id, flow_id, channel_id, npu_to_ma ? "NPU->MA" : "MA->NPU", processed, send_back,
        send_packets_in_flight.load());

    PacketBundle* pb = new PacketBundle(
        stream->owner,
        stream,
        {},
        processed,
        send_back,
        flow.flow_size,
        MemBus::Transmition::Usual,
        channel_id,
        flow_id);

    if (npu_to_ma) {
        pb->send_to_MA();
    } else {
        pb->send_to_NPU();
    }
}

void NcclCustom::send_flow_unlocked(int channel_id, int flow_id) {
    assert(NcclCustom::FlowCriticalSection::is_locked());

    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: send_flow_unlocked ENTRY for flow %d on channel %d.", id, flow_id, channel_id);

    if (flow_states.count(flow_id)) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: send_flow_unlocked called for already-initiated flow %d. Ignoring.", id, flow_id);
        return;
    }

    flow_states[flow_id] = FlowState::Sending;

    if (_flow_models.find({channel_id, flow_id}) == _flow_models.end()) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: Attempted to send non-existent flow %d on channel %d.", id, flow_id, channel_id);
        return;
    }

    auto& flow= _flow_models.at({channel_id, flow_id});
    Tick current_time = Sys::boostedTick();
    
    if (collective_graph == true){
        // Record the start time for this flow under a thread-safe lock.
        {
            std::lock_guard<std::mutex> lg(g_graph_data_per_layer[layer_num].mtx);
            g_graph_data_per_layer[layer_num].start_times[flow.flow_id] = current_time;
        }
    }

    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Packet sent for flow %d with chunk_id/source/dest (%d/%d/%d) on channel %d at tick %llu.",
            id, flow.flow_id, flow.chunk_id, flow.src, flow.dest, flow.channel_id, current_time);

    release_packet(channel_id, flow_id);

    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: send_flow_unlocked EXIT.", id);
}

void NcclCustom::run(EventType event, CallData* data) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "NcclCustom::run ENTRY: ID %d, Event %s", id, getEventTypeString(event).c_str());

    switch (event) {
        case EventType::StreamInit: {
            // Basic setup and logging
            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: StreamInit beginning", id);
            start_time = std::chrono::high_resolution_clock::now();

            // Local vectors to prepare work without holding a global lock
            std::vector<std::pair<int, int>> sends_to_initiate;
            std::vector<decltype(_flow_models)::value_type::second_type> receives_to_post;

            // --- PHASE 1: STAGING ---
            // First, iterate through all flow models without any locks to figure out
            // which flows this node needs to send and which it needs to receive.
            // This phase only reads data and writes to local variables, so it's safe.
            for (const auto& flow_pair : _flow_models) {
                const auto& flow = flow_pair.second;
                if (flow.src == this->id) {
                    // Stage sends that have no outstanding dependencies.
                    if (indegree_mapping.at(flow.flow_id) == 0) {
                        sends_to_initiate.emplace_back(flow.channel_id, flow.flow_id);
                    }
                } else if (flow.dest == this->id) {
                    // Stage all flows that are destined for this node.
                    receives_to_post.push_back(flow);
                }
            }
            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Staged %zu sends and %zu receives.", id, sends_to_initiate.size(), receives_to_post.size());

            // --- PHASE 2: SEND EXECUTION ---
            // Execute all staged sends within a single, short-lived critical section.
            // The `send_flow_unlocked` function asserts that a lock is held.
            {
                NcclCustom::FlowCriticalSection cs;
                NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Acquired lock to execute %zu sends.", id, sends_to_initiate.size());
                for (const auto& send_op : sends_to_initiate) {
                    // Call the unlocked version since we are inside the critical section.
                    send_flow_unlocked(send_op.first /* channel_id */, send_op.second /* flow_id */);
                }
            } // The lock is automatically released here.

            // --- PHASE 3: RECEIVE POSTING ---
            // Post all staged receives. The blocking call `front_end_sim_recv` will
            // now happen safely outside of any lock, preventing deadlocks.
            for (const auto& flow : receives_to_post) {
                // Create the event handler data for this receive operation.
                auto* rcv_ehd = new RecvPacketEventHadndlerData(
                    stream,
                    flow.src,
                    id,
                    flow_tag(flow),
                    EventType::PacketReceived,
                    stream->current_queue_id,
                    1);
                rcv_ehd->flow_id = flow.flow_id;
                rcv_ehd->channel_id = flow.channel_id;

                // FIXME this is never used
                sim_request rcv_req;

                // Count the receive as in flight before posting it: if it completes synchronously,
                // PacketReceived decrements the counter back instead of driving it negative.
                ++recv_packets_in_flight;
                NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Increasing recv_packets_in_flight to %d for flow %d.", id, recv_packets_in_flight.load(), flow.flow_id);

                // Make the potentially blocking call. This is now safe.
                NcclLog->writeLog(NcclLogLevel::DEBUG, "[RECV] ID %d: Preparing to receive flow %d on channel %d.", id, flow.flow_id, flow.channel_id);
                stream->owner->front_end_sim_recv(
                    0,
                    Sys::dummy_data,
                    flow.flow_size,
                    UINT8,
                    flow.src,
                    rcv_ehd->tag,
                    &rcv_req,
                    &Sys::handleEvent,
                    rcv_ehd);
            }

            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: StreamInit EXIT.", id);
            break;
        }
        case EventType::PacketReceived: {
            auto rcv_ehd = static_cast<RecvPacketEventHadndlerData*>(data);
            NcclLog->writeLog(
                NcclLogLevel::DEBUG,
                "ID %d: Taking critical section for PacketReceived for flow %d.",
                id, rcv_ehd->flow_id);
            if (_flow_models.find({rcv_ehd->channel_id, rcv_ehd->flow_id}) == _flow_models.end()) {
                 NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: Received packet for non-existent flow %d.", id, rcv_ehd->flow_id);
                 break;
            }
            const auto& received_flow = _flow_models[std::make_pair(rcv_ehd->channel_id, rcv_ehd->flow_id)];
              
            Tick current_time = Sys::boostedTick();

            NcclLog->writeLog(
                NcclLogLevel::DEBUG,
                "ID %d: Packet received for flow %d with chunk_id/source/dest (%d/%d/%d) with channel %d at tick %llu with vnet %d and tag_id %d.",
                id, received_flow.flow_id, received_flow.chunk_id, received_flow.src, received_flow.dest,
                received_flow.channel_id, current_time, rcv_ehd->vnet, rcv_ehd->tag);

            // Process dependencies immediately upon packet reception (like NcclTreeFlowModel).
            recvs_completed++;
            recv_packets_in_flight--;
            NcclLog->writeLog(
                NcclLogLevel::DEBUG,
                "ID %d: Decreasing recv_packets_in_flight to %d for flow %d on channel %d, recvs_completed now %d.",
                id, recv_packets_in_flight.load(), received_flow.flow_id, received_flow.channel_id, recvs_completed.load());
            NcclLog->writeLog(
                NcclLogLevel::DEBUG,
                "ID %d: Packet received and processed for flow %d. Updating dependencies immediately.",
                id, received_flow.flow_id);
            for (int child_flow_id : received_flow.child_flow_id) {
                if (received_flow.dest == this->id) {
                    if (flow_id_to_channel_id_map.count(child_flow_id) == 0) continue;
                    int child_channel_id = flow_id_to_channel_id_map.at(child_flow_id);
                    if (_flow_models.find({child_channel_id, child_flow_id}) == _flow_models.end()) continue;

                    const auto& child_flow = _flow_models.at({child_channel_id, child_flow_id});
                    if (child_flow.src == this->id) {
                        NcclCustom::FlowCriticalSection cs;
                        if (indegree_mapping.count(child_flow_id) != 0) {
                            int old_indegree = indegree_mapping[child_flow_id];
                            indegree_mapping[child_flow_id]--;
                            NcclLog->writeLog(
                                NcclLogLevel::DEBUG,
                                "ID %d: Flow %d completed at time %llu, decreasing indegree for child flow %d. New indegree: %d (old %d).",
                                id, received_flow.flow_id, Sys::boostedTick(), child_flow_id, indegree_mapping[child_flow_id], old_indegree);
                            if (indegree_mapping[child_flow_id] == 0) {
                                NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Dependencies met for flow %d. Initiating send.", id, child_flow_id);
                                send_flow_unlocked(child_flow.channel_id, child_flow.flow_id);
                            }
                        }
                    }
                }
            }
            if (check_completion()) {
                return; // this object has been deleted
            }
            break;
        }
        case EventType::PacketSentFinshed: {
            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Taking critical section for PacketSentFinshed.", id);
            {
                // The lock must be released before check_completion(): exit() synchronously starts the
                // next waiting stream (proceed_to_next_vnet_baseline -> SchedulerUnit -> StreamBaseline::init),
                // whose NcclCustom::run(StreamInit) takes this same global spinlock -> self-deadlock.
                NcclCustom::FlowCriticalSection cs;
                auto snd_ehd = static_cast<SendPacketEventHandlerData*>(data);
                Tick current_time = Sys::boostedTick();
                NcclLog->writeLog(
                    NcclLogLevel::DEBUG,
                    "ID %d: Packet sent finished for flow %d, with source/dest (%d/%d) with channel %d at tick %llu.",
                    id, snd_ehd->flow_id, snd_ehd->senderNodeId,
                    snd_ehd->receiverNodeId, snd_ehd->channel_id, current_time);

                // Check if flow exists to avoid crash
                if (_flow_models.find({snd_ehd->channel_id, snd_ehd->flow_id}) == _flow_models.end()) {
                     NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: Sent packet for non-existent flow %d.", id, snd_ehd->flow_id);
                     break; // releases the lock
                }
                const auto& sent_flow = _flow_models.at({snd_ehd->channel_id, snd_ehd->flow_id});
            
                if (collective_graph == true){
                    // Record the end time for this flow under a thread-safe lock.
                    {
                        std::lock_guard<std::mutex> lg(g_graph_data_per_layer[layer_num].mtx);
                        g_graph_data_per_layer[layer_num].end_times[sent_flow.flow_id] = current_time;
                    }
                }

                for(auto& child_flow: sent_flow.child_flow_id){
                    // we can have child which are not directed to us, so we need to check if the child flow is directed to us
                    if(flow_id_to_channel_id_map.count(child_flow) == 0) {
                        //not ours (might be a child of another thread/node)
                        continue;
                    }
                    if (_flow_models.find({flow_id_to_channel_id_map.at(child_flow), child_flow}) == _flow_models.end()) {
                        NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: FATAL: Could not find flow model for child flow %d on channel %d!", id, child_flow, flow_id_to_channel_id_map.at(child_flow));
                    }
                    const auto& child_flow_model = _flow_models.at({flow_id_to_channel_id_map.at(child_flow), child_flow});
                    if (child_flow_model.src != this->id) {
                        NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: Child flow %d is not sent by this node (src: %d). Skipping indegree decrease.", id, child_flow, child_flow_model.src);
                        continue;
                    }
                    // if the child flow has been sent by us we can decrease its indegree
                    if(indegree_mapping.count(child_flow) > 0) {
                        int old_indegree = indegree_mapping[child_flow];
                        indegree_mapping[child_flow]--;
                        NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Flow %d sent (immediate), decreasing indegree for child flow %d. New indegree: %d (old %d).", id, sent_flow.flow_id, child_flow, indegree_mapping[child_flow], old_indegree);
                        if (indegree_mapping[child_flow] == 0) {
                            if (flow_id_to_channel_id_map.count(child_flow)) {
                                int child_channel_id = flow_id_to_channel_id_map.at(child_flow);
                                NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Dependencies met for flow %d on its correct channel %d. Initiating send.", id, child_flow, child_channel_id);
                                send_flow_unlocked(child_channel_id, child_flow);
                            } else {
                                NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: FATAL: Could not find channel for child flow %d in helper map!", id, child_flow);
                            }
                        }
                    } else {
                        NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: FATAL: Could not find indegree mapping for child flow %d!", id, child_flow);
                    }
                }

                sends_completed++;
                send_packets_in_flight--;
                NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Decreasing send_packets_in_flight to %d for flow %d on channel %d. Total sends completed: %d/%d.",
                    id, send_packets_in_flight.load(), sent_flow.flow_id, sent_flow.channel_id, sends_completed.load(), total_sends_to_initiate);
            } // lock released here
            if (check_completion()) {
                return; // this object has been deleted
            }
            break;
        }
        case EventType::General: {
            BasicEventHandlerData* ehd = static_cast<BasicEventHandlerData*>(data);
            int channel_id = ehd->channel_id;
            int flow_id = ehd->flow_id;
            if (collective_graph == true){
                // Record the start time for this flow under a thread-safe lock.
                {
                    std::lock_guard<std::mutex> lg(g_graph_data_per_layer[layer_num].mtx);
                    auto current_time = Sys::boostedTick();
                    g_graph_data_per_layer[layer_num].start_times[flow_id] = current_time;
                }
            }
            {
                NcclLog->writeLog(NcclLogLevel::DEBUG, "[SEND] ID %d: Sending packet for flow %d on channel %d.", id, flow_id, channel_id);
                const auto& flow = _flow_models.at({channel_id, flow_id});

                // Create the event handler data for this send operation.
                auto* snd_ehd = new SendPacketEventHandlerData(
                    stream,
                    id,
                    flow.dest,
                    flow_tag(flow),
                    EventType::PacketSentFinshed);
                snd_ehd->flow_id = flow.flow_id;
                snd_ehd->channel_id = flow.channel_id;
                snd_ehd->chunk_id = flow.chunk_id;
                snd_ehd->nvls_on = comType == ComType::All_Reduce_NVLS;
                // Selection used by the frontend's send-latency model (as in NcclFlowModel).
                snd_ehd->algorithm = m_algorithm;
                snd_ehd->protocol = m_protocol;

                // TODO is this needed? It is used only by the physical frontend
                sim_request snd_req;
                snd_req.srcRank = id;
                snd_req.dstRank = flow.dest;
                snd_req.tag = snd_ehd->tag;
                snd_req.reqType = UINT8;
                snd_req.vnet = stream->current_queue_id;
                snd_req.layerNum = layer_num;
                snd_req.reqCount = flow.flow_size;

                NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Preparing to send flow %d with chunk_id/source/dest (%d/%d/%d) on channel %d with tag_id %d = %d * %d * %u + %d * %d + %d.",
                    id, flow_id, flow.chunk_id, flow.src, flow.dest, channel_id, snd_ehd->tag,
                    layer_num, flow.chunk_count, m_channels,
                    flow.channel_id, flow.chunk_count,
                    flow.chunk_id);

                stream->owner->front_end_sim_send(
                    0,
                    Sys::dummy_data,
                    snd_req.reqCount,
                    UINT8,
                    flow.dest,
                    snd_ehd->tag,
                    &snd_req,
                    &Sys::handleEvent,
                    snd_ehd);
            }
            break;
        }
        default:
            NcclLog->writeLog(NcclLogLevel::WARNING, "ID %d: Received unknown event type: %d.", id, static_cast<int>(event));
            break;
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: run() finished.", id);
}

// Returns true when the collective finished: exit() then deletes this object (via
// proceed_to_next_vnet_baseline), so the caller must return without touching any member.
// Must be called without holding the FlowCriticalSection (see PacketSentFinshed).
bool NcclCustom::check_completion() {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: check_completion ENTRY. Sends completed: %d/%d, Recvs completed: %d/%d, sendInAir: %d, recvInAir: %d",
                        id, sends_completed.load(), total_sends_to_initiate,    
                        recvs_completed.load(), total_recvs_initiated,
                        send_packets_in_flight.load(), recv_packets_in_flight.load());
    if (sends_completed.load() == total_sends_to_initiate && recvs_completed.load() == total_recvs_initiated &&
        send_packets_in_flight.load() == 0 && recv_packets_in_flight.load() == 0) {
        NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: All tasks completed. Sends: %d/%d, Receives: %d/%d, sendInAir: %d, recvInAir: %d",
                        id, sends_completed.load(), total_sends_to_initiate,
                        recvs_completed.load(), total_recvs_initiated,
                        send_packets_in_flight.load(), recv_packets_in_flight.load());
        exit();
        return true;
    }
    return false;
}

void NcclCustom::exit() {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: exit() at tick %llu", id, Sys::boostedTick());
    end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Communication Latency: %lld us", id, duration.count());

    if (collective_graph == true){
        // Atomically increment the finished nodes counter. If this is the last node, generate the graph.
        int total_nodes = this->nodes_in_ring;
        if (g_finished_nodes_per_layer[layer_num].fetch_add(1, std::memory_order_relaxed) + 1 == total_nodes) {
            NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Last node finished. Generating collective graph.", id);

            std::string filename = "collective_graph_layer_" + std::to_string(layer_num) + ".dot";
            std::ofstream dot_file(filename);
            if (!dot_file.is_open()) {
                NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: Failed to open graph file '%s' for writing.", id, filename.c_str());
            } else {
                dot_file << "digraph CollectiveGraph {\n";
                dot_file << "  rankdir=LR;\n";
                dot_file << "  label=\"Collective Communication Graph - Layer " << layer_num << "\";\n";
                dot_file << "  node [shape=box, style=rounded];\n\n";

                std::map<int, std::stringstream> chunk_subgraphs;
                
                // Lock the global data to safely read it for graph generation.
                std::lock_guard<std::mutex> lg(g_graph_data_per_layer[layer_num].mtx);

                // Group all flows into subgraphs by their chunk ID.
                for (const auto& flow_pair : g_graph_data_per_layer[layer_num].flow_details) {
                    int flow_id = flow_pair.first;
                    const auto& flow = flow_pair.second;

                    if (g_graph_data_per_layer[layer_num].start_times.count(flow_id) && g_graph_data_per_layer[layer_num].end_times.count(flow_id)) {
                        Tick start_tick = g_graph_data_per_layer[layer_num].start_times.at(flow_id);
                        Tick end_tick = g_graph_data_per_layer[layer_num].end_times.at(flow_id);
                        chunk_subgraphs[flow.chunk_id] << "    " << flow.src << " -> " << flow.dest
                                                    << " [label=\"" << start_tick << " / " << end_tick <<" on ch: " << flow.channel_id << " with size: " << flow.flow_size << "\"];\n";
                    }
                }

                // Write the clustered subgraphs to the DOT file.
                for (auto const& [chunk_id, ss] : chunk_subgraphs) {
                    dot_file << "  subgraph cluster_chunk_" << chunk_id << " {\n";
                    dot_file << "    label = \"Chunk " << chunk_id << "\";\n";
                    dot_file << "    style=filled;\n";
                    dot_file << "    color=lightgrey;\n";
                    dot_file << ss.str();
                    dot_file << "  }\n\n";
                }

                dot_file << "}\n";
                dot_file.close();
                NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Collective graph saved to %s", id, filename.c_str());
            }
        }
    }

    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: exit() EXIT.", id);
    // proceed_to_next_vnet_baseline deletes this algorithm: it must be the last statement.
    if (stream && stream->owner) {
        stream->owner->proceed_to_next_vnet_baseline(static_cast<StreamBaseline*>(stream));
    }
}

#ifdef PHY_MTP
bool NcclCustom::phy_iteratable(int channel_id) {
    if (send_packets_in_flight == 0 && recv_packets_in_flight == 0) {
        judge_exit_flag.store(true);
        return false;
    }
    return true;
}

void NcclCustom::waiting_to_exit() {
    while (!judge_exit_flag) {}
    exit();
}
#endif

} // namespace AstraSim