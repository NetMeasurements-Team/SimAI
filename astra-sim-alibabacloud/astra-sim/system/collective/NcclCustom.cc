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
#include "astra-sim/system/MockNcclLog.h"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/StreamBaseline.hh"
#include "astra-sim/system/PacketBundle.hh"
#include <fstream>
#include <sstream>
#include <string>

#ifdef PHY_MTP
#include <mpi.h>
#include "astra-sim/system/PhyMultiThread.hh"
#endif

#ifdef PHY_RDMA
#include "astra-sim/system/SimAiFlowModelRdma.hh"
extern FlowPhyRdma flow_rdma;
#endif

// If AFTER is 1, dependency updates happen AFTER the MA->NPU write-back (slower, more realistic).
// If AFTER is 0, dependency updates happen IMMEDIATELY upon packet reception (faster, like NcclTreeFlowModel).
#define AFTER 0
#define NO_MA_NPU 0
namespace AstraSim {

std::atomic<bool> NcclCustom::g_flow_inCriticalSection_ncclCustom(false);

#if COLLECTIVE_GRAPH
// Define the static members for graph generation
std::map<int, NcclCustom::GlobalGraphData> NcclCustom::g_graph_data_per_layer;
std::map<int, std::atomic<int>> NcclCustom::g_finished_nodes_per_layer = {};
#endif

NcclCustom::NcclCustom(
    ComType type,
    int id,
    int layer_num,
    RingTopology *ring_topology,
    uint64_t data_size,
    RingTopology::Direction direction,
    InjectionPolicy injection_policy,
    bool boost_mode,
    std::shared_ptr<MockNccl::FlowModels> ptr_flow_models,
    int treechannels)
    : Algorithm(layer_num) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: NcclCustom constructor ENTRY. Layer: %d, Channels: %d", id, layer_num, treechannels);

    this->id = id;
    this->comType = type;
    this->logicalTopology = ring_topology;
    this->data_size = data_size;
    this->nodes_in_ring = ring_topology->get_nodes_in_ring();
    this->m_channels = treechannels;

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
#if COLLECTIVE_GRAPH
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
#endif

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

void NcclCustom::release_packet(int channel_id, int flow_id, bool NPU_to_MA) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    if(!NPU_to_MA){
        NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Increasing send_packets_in_flight to %d for flow %d on channel %d.", id, send_packets_in_flight.load() + 1, flow_id, channel_id);
        send_packets_in_flight++;
    }
    const auto& flow = _flow_models.at({channel_id, flow_id});
    
    bool needs_processing = false;

    PacketBundle* pb = new PacketBundle(
        stream->owner,
        stream,
        {},
        needs_processing,
        false, 
        flow.flow_size,
        MemBus::Transmition::Usual,
        channel_id,
        flow_id);

    if (NPU_to_MA) {
        NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Releasing NPU->MA packet for flow %d.", id, flow_id);
        pb->send_to_MA();
    } else {
        NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Releasing MA->NPU packet for flow %d.", id, flow_id);
        pb->send_to_NPU();
    }
}

void NcclCustom::send_flow(int channel_id, int flow_id) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: send_flow ENTRY for flow %d on channel %d.", id, flow_id, channel_id);
    if (_flow_models.find({channel_id, flow_id}) == _flow_models.end()) {
        NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: Attempted to send non-existent flow %d on channel %d.", id, flow_id, channel_id);
        return;
    }

    auto& flow= _flow_models.at({channel_id, flow_id});
    Tick current_time = Sys::boostedTick();

#if COLLECTIVE_GRAPH
    // Record the start time for this flow under a thread-safe lock.
    {
        std::lock_guard<std::mutex> lg(g_graph_data_per_layer[layer_num].mtx);
        g_graph_data_per_layer[layer_num].start_times[flow.flow_id] = current_time;
    }
#endif

    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Packet sent for flow %d with chunk_id/source/dest (%d/%d/%d) on channel %d at tick %llu.",
                id, flow.flow_id, flow.chunk_id, flow.src, flow.dest, flow.channel_id, current_time);

    flow_states[flow_id] = FlowState::Sending;
    release_packet(channel_id, flow_id, false);
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: send_flow EXIT.", id);
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
    
#if COLLECTIVE_GRAPH
    // Record the start time for this flow under a thread-safe lock.
    {
        std::lock_guard<std::mutex> lg(g_graph_data_per_layer[layer_num].mtx);
        g_graph_data_per_layer[layer_num].start_times[flow.flow_id] = current_time;
    }
#endif

    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Packet sent for flow %d with chunk_id/source/dest (%d/%d/%d) on channel %d at tick %llu.",
            id, flow.flow_id, flow.chunk_id, flow.src, flow.dest, flow.channel_id, current_time);

    #if NO_MA_NPU
         send_packets_in_flight++;

                sim_request snd_req;
                snd_req.srcRank = id;
                snd_req.dstRank = flow.dest;
                snd_req.tag = channel_id;
                snd_req.reqType = UINT8;
                snd_req.vnet = this->stream->current_queue_id;
                snd_req.layerNum = layer_num;
                snd_req.reqCount = flow.flow_size;
                snd_req.flowTag.current_flow_id = flow_id;
                snd_req.flowTag.chunk_id = flow.chunk_id;
                snd_req.flowTag.channel_id = channel_id;
                snd_req.flowTag.flow_size = flow.flow_size;
                snd_req.flowTag.sender_node = id;
                snd_req.flowTag.receiver_node = flow.dest;
                snd_req.flowTag.tag_id = snd_req.flowTag.tag_id = layer_num * flow.chunk_count * m_channels * nodes_in_ring * nodes_in_ring +
                                        flow.chunk_count * flow.channel_id * nodes_in_ring * nodes_in_ring +
                                        flow.chunk_id * nodes_in_ring * nodes_in_ring +
                                        flow.src * nodes_in_ring +
                                        flow.dest;
                NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Preparing to send flow %d with chunk_id/source/dest (%d/%d/%d) on channel %d with tag_id %d = %d * %d * %d * %d * %d + %d * %d * %d * %d + %d * %d * %d + %d * %d + %d.",
                    id, flow_id, flow.chunk_id, flow.src, flow.dest, channel_id,
                    snd_req.flowTag.tag_id,
                    // Term 1: layer_num * flow.chunk_count * m_channels * nodes_in_ring * nodes_in_ring
                    layer_num, flow.chunk_count, m_channels, nodes_in_ring, nodes_in_ring,
                    // Term 2: flow.channel_id * flow.chunk_count * nodes_in_ring * nodes_in_ring
                    flow.channel_id, flow.chunk_count, nodes_in_ring, nodes_in_ring,
                    // Term 3: flow.chunk_id * nodes_in_ring * nodes_in_ring
                    flow.chunk_id, nodes_in_ring, nodes_in_ring,
                    // Term 4: flow.src * nodes_in_ring
                    flow.src, nodes_in_ring,
                    // Term 5: flow.dest
                    flow.dest);
                NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Preparing to send flow %d with chunk_id/source/dest (%d/%d/%d) on channel %d with tag_id %d = %d * %d * %d * %d + %d * %d * %d + %d * %d + %d.",
                    id, flow_id, flow.chunk_id, flow.src, flow.dest, channel_id,
                    snd_req.flowTag.tag_id, layer_num, flow.chunk_count, m_channels,
                    nodes_in_ring, flow.channel_id, flow.chunk_count, nodes_in_ring, flow.chunk_id, nodes_in_ring,
                    flow.dest);
                SendPacketEventHandlerData* send_ehd = new SendPacketEventHandlerData(
                    stream, id, flow.dest, channel_id, EventType::PacketSentFinshed);
                send_ehd->flowTag = snd_req.flowTag;

                stream->owner->front_end_sim_send(
                    0, Sys::dummy_data, snd_req.reqCount, UINT8, flow.dest,
                    snd_req.tag, &snd_req, &Sys::handleEvent, send_ehd);


    #else
        release_packet(channel_id, flow_id, false);
    #endif

    // Update dependencies immediately after initiating the send
    for(auto& child_flow: flow.child_flow_id){
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
            NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Flow %d sent (immediate), decreasing indegree for child flow %d. New indegree: %d (old %d).", id, flow_id, child_flow, indegree_mapping[child_flow], old_indegree);
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
            static std::map<int, std::tuple<int, int, int, int>> tag_id_tracker;
            for (const auto& flow : receives_to_post) {
                // Create the event handler data for this receive operation.
                RecvPacketEventHadndlerData* recv_ehd = new RecvPacketEventHadndlerData(
                    stream, stream->owner->id, EventType::PacketReceived, this->stream->current_queue_id, 1);
                recv_ehd->flowTag.channel_id = flow.channel_id;
                recv_ehd->flowTag.current_flow_id = -1; // Will be filled in when the packet arrives.
                recv_ehd->flowTag.tag_id = layer_num * flow.chunk_count * m_channels * nodes_in_ring * nodes_in_ring +
                                        flow.chunk_count * flow.channel_id * nodes_in_ring * nodes_in_ring +
                                        flow.chunk_id * nodes_in_ring * nodes_in_ring +
                                        flow.src * nodes_in_ring +
                                        flow.dest;

                // Use a fine-grained lock ONLY to protect the shared `tag_id_tracker`.
                {
                    NcclCustom::FlowCriticalSection cs;
                    auto current_tuple = std::make_tuple(flow.src, flow.dest, flow.chunk_id, flow.channel_id);
                    if (tag_id_tracker.count(recv_ehd->flowTag.tag_id)) {
                        auto& original_tuple = tag_id_tracker.at(recv_ehd->flowTag.tag_id);
                        if (original_tuple != current_tuple) {
                            NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: FATAL: tag_id collision detected! Tag %d is repeated.", id, recv_ehd->flowTag.tag_id);
                            NcclLog->writeLog(NcclLogLevel::ERROR, " -> Original: src=%d, dst=%d, chunk=%d, chan=%d", std::get<0>(original_tuple), std::get<1>(original_tuple), std::get<2>(original_tuple), std::get<3>(original_tuple));
                            NcclLog->writeLog(NcclLogLevel::ERROR, " -> Collision: src=%d, dst=%d, chunk=%d, chan=%d", flow.src, flow.dest, flow.chunk_id, flow.channel_id);
                        }
                    } else {
                        tag_id_tracker[recv_ehd->flowTag.tag_id] = current_tuple;
                    }
                } // Fine-grained lock is released here.

                // Prepare the simulation request for the receive call.
                sim_request rcv_req;
                rcv_req.vnet = this->stream->current_queue_id;
                rcv_req.layerNum = layer_num;
                rcv_req.flowTag = recv_ehd->flowTag;
                rcv_req.tag = flow.channel_id;

                // Make the potentially blocking call. This is now safe.
                NcclLog->writeLog(NcclLogLevel::DEBUG, "[RECV] ID %d: Preparing to receive flow %d on channel %d.", id, flow.flow_id, flow.channel_id);
                stream->owner->front_end_sim_recv(
                    0, Sys::dummy_data, flow.flow_size, UINT8, flow.src,
                    rcv_req.tag, &rcv_req, &Sys::handleEvent, recv_ehd);

                // Increment the atomic counter for in-flight packets.
                recv_packets_in_flight++;
                NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Increasing recv_packets_in_flight to %d for flow %d.", id, recv_packets_in_flight.load(), flow.flow_id);
            }

            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: StreamInit EXIT.", id);
            break;
        }
        case EventType::PacketReceived: {
            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Taking critical section for PacketReceived for flow %d.", id, static_cast<RecvPacketEventHadndlerData*>(data)->flowTag.current_flow_id);
            RecvPacketEventHadndlerData* rcehd = static_cast<RecvPacketEventHadndlerData*>(data);
            int completed_flow_id = rcehd->flowTag.current_flow_id;
            int channel_id = rcehd->flowTag.channel_id;
          
            if (_flow_models.find({channel_id, completed_flow_id}) == _flow_models.end()) {
                 NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: Received packet for non-existent flow %d.", id, completed_flow_id);
                 break;
            }
            const auto& completed_flow = _flow_models.at({channel_id, completed_flow_id});
              
            Tick current_time = Sys::boostedTick();

#if COLLECTIVE_GRAPH
            // Record the end time for this flow under a thread-safe lock.
            {
                std::lock_guard<std::mutex> lg(g_graph_data_per_layer[layer_num].mtx);
                g_graph_data_per_layer[layer_num].end_times[completed_flow_id] = current_time;
            }
#endif

            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Packet received for flow %d with chunk_id/source/dest (%d/%d/%d) with channel %d at tick %llu with vnet %d and tag_id %d.",
                        id, completed_flow_id, completed_flow.chunk_id, completed_flow.src, completed_flow.dest,completed_flow.channel_id, current_time, rcehd->vnet, rcehd->flowTag.tag_id);
            
#if AFTER
            // Slower, more realistic model: trigger a separate write-back event
            NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Network packet received for flow %d. Starting writeback.", id, completed_flow_id);
            flow_states[completed_flow_id] = FlowState::WritingBack;
            release_packet(channel_id, completed_flow_id, true);
#else
            // Faster, simplified model: process dependencies immediately
            recvs_completed++;
            recv_packets_in_flight--;
            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Decreasing recv_packets_in_flight to %d for flow %d on channel %d, recvs_completed now %d.",
                        id, recv_packets_in_flight.load(), completed_flow_id, channel_id, recvs_completed.load());
            NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Packet received and processed for flow %d. Updating dependencies immediately.", id, completed_flow_id);
            { 
                for (int child_flow_id : completed_flow.child_flow_id) {
                    if(completed_flow.dest == this->id) {
                        if (flow_id_to_channel_id_map.count(child_flow_id) == 0) continue;
                        int child_channel_id = flow_id_to_channel_id_map.at(child_flow_id);
                        if (_flow_models.find({child_channel_id, child_flow_id}) == _flow_models.end()) continue;
                        
                        const auto& child_flow = _flow_models.at({child_channel_id, child_flow_id});
                        if(child_flow.src == this->id){
                            NcclCustom::FlowCriticalSection cs;
                            if(indegree_mapping.count(child_flow_id) != 0){
                                int old_indegree = indegree_mapping[child_flow_id];
                                indegree_mapping[child_flow_id]--;
                                NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Flow %d completed at time %d, decreasing indegree for child flow %d. New indegree: %d (old %d).", id, completed_flow_id, Sys::boostedTick(),child_flow_id, indegree_mapping[child_flow_id], old_indegree);
                                if (indegree_mapping[child_flow_id] == 0) {
                                    NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Dependencies met for flow %d. Initiating send.", id, child_flow_id);
                                    send_flow_unlocked(child_flow.channel_id, child_flow.flow_id);
                                }
                            }
                        }
                    }
                }
            }
            check_completion();
#endif
            break;
        }
        case EventType::PacketSentFinshed: {
            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Taking critical section for PacketSentFinshed.", id);
            NcclCustom::FlowCriticalSection cs;
            SendPacketEventHandlerData* sehd = static_cast<SendPacketEventHandlerData*>(data);
            Tick current_time = Sys::boostedTick();
            NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: Packet sent finished for flow %d, with chunk_id/source/dest (%d/%d/%d) with channel %d at tick %llu.",
                        id, sehd->flowTag.current_flow_id, sehd->flowTag.channel_id, sehd->flowTag.chunk_id, sehd->senderNodeId, sehd->receiverNodeId, current_time);
            
            // Dependencies are now updated immediately in send_flow_unlocked(), 
            // so we don't process them here anymore.
            
            sends_completed++;
            send_packets_in_flight--;
            NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Decreasing send_packets_in_flight to %d for flow %d on channel %d. Total sends completed: %d/%d.",
                id, send_packets_in_flight.load(), sehd->flowTag.current_flow_id, sehd->flowTag.channel_id, sends_completed.load(), total_sends_to_initiate);
            check_completion();
            break;
        }
        case EventType::General: {
            BasicEventHandlerData* ehd = static_cast<BasicEventHandlerData*>(data);
            int channel_id = ehd->channel_id;
            int flow_id = ehd->flow_id;
            Tick current_time = Sys::boostedTick();
            #if COLLECTIVE_GRAPH
                // Record the start time for this flow under a thread-safe lock.
                {
                    std::lock_guard<std::mutex> lg(g_graph_data_per_layer[layer_num].mtx);
                    g_graph_data_per_layer[layer_num].start_times[flow_id] = current_time;
                }
            #endif
#if AFTER
            // This block handles the completion of the MA->NPU write-back event.
            // It only executes if we are using the slower, more realistic model.
            if (flow_states.count(flow_id) && flow_states[flow_id] == FlowState::WritingBack) {
                recvs_completed++;
                recv_packets_in_flight--;
                
                NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Writeback finished for flow %d. Progress: recvs_completed %d/%d.", id, flow_id, recvs_completed.load(), total_recvs_initiated);
                
                if(_flow_models.find({channel_id, flow_id}) == _flow_models.end()) {
                    NcclLog->writeLog(NcclLogLevel::ERROR, "ID %d: FATAL: Could not find flow model for channel %d and flow %d!", id, channel_id, flow_id);
                }
                const auto& completed_flow = _flow_models.at({channel_id, flow_id});
                for (int child_flow_id : completed_flow.child_flow_id) {
                    if(completed_flow.dest == this->id) {
                        if (flow_id_to_channel_id_map.count(child_flow_id) == 0) continue;
                        
                        int child_channel_id = flow_id_to_channel_id_map.at(child_flow_id);
                         if (_flow_models.find({child_channel_id, child_flow_id}) == _flow_models.end()) continue;

                        const auto& child_flow = _flow_models.at({child_channel_id, child_flow_id});
                        if(child_flow.src == this->id){
                            NcclCustom::FlowCriticalSection cs;
                            if(indegree_mapping.count(child_flow_id) != 0){
                                indegree_mapping[child_flow_id]--;
                                NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Flow %d completed, decreasing indegree for child flow %d. New indegree: %d.", id, flow_id, child_flow_id, indegree_mapping[child_flow_id]);
                                if (indegree_mapping[child_flow_id] == 0) {
                                    NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Dependencies met for flow %d. Initiating send.", id, child_flow_id);
                                    send_flow_unlocked(child_flow.channel_id, child_flow.flow_id);
                                }
                            }
                        }
                    }
                }
            } else
#endif
             {
                NcclLog->writeLog(NcclLogLevel::DEBUG, "[SEND] ID %d: Sending packet for flow %d on channel %d.", id, flow_id, channel_id);
                const auto& flow = _flow_models.at({channel_id, flow_id});
                sim_request snd_req;
                snd_req.srcRank = id;
                snd_req.dstRank = flow.dest;
                snd_req.tag = channel_id;
                snd_req.reqType = UINT8;
                snd_req.vnet = this->stream->current_queue_id;
                snd_req.layerNum = layer_num;
                snd_req.reqCount = flow.flow_size;
                snd_req.flowTag.current_flow_id = flow_id;
                snd_req.flowTag.chunk_id = flow.chunk_id;
                snd_req.flowTag.channel_id = channel_id;
                snd_req.flowTag.flow_size = flow.flow_size;
                snd_req.flowTag.sender_node = id;
                snd_req.flowTag.receiver_node = flow.dest;
                snd_req.flowTag.tag_id = layer_num * flow.chunk_count * m_channels * nodes_in_ring * nodes_in_ring +
                                        flow.chunk_count * flow.channel_id * nodes_in_ring * nodes_in_ring +
                                        flow.chunk_id * nodes_in_ring * nodes_in_ring +
                                        flow.src * nodes_in_ring +
                                        flow.dest;
                NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Preparing to send flow %d with chunk_id/source/dest (%d/%d/%d) on channel %d with tag_id %d = %d * %d * %d * %d * %d + %d * %d * %d * %d + %d * %d * %d + %d * %d + %d.",
                    id, flow_id, flow.chunk_id, flow.src, flow.dest, channel_id,
                    snd_req.flowTag.tag_id,
                    // Term 1: layer_num * flow.chunk_count * m_channels * nodes_in_ring * nodes_in_ring
                    layer_num, flow.chunk_count, m_channels, nodes_in_ring, nodes_in_ring,
                    // Term 2: flow.channel_id * flow.chunk_count * nodes_in_ring * nodes_in_ring
                    flow.channel_id, flow.chunk_count, nodes_in_ring, nodes_in_ring,
                    // Term 3: flow.chunk_id * nodes_in_ring * nodes_in_ring
                    flow.chunk_id, nodes_in_ring, nodes_in_ring,
                    // Term 4: flow.src * nodes_in_ring
                    flow.src, nodes_in_ring,
                    // Term 5: flow.dest
                    flow.dest);
                SendPacketEventHandlerData* send_ehd = new SendPacketEventHandlerData(
                    stream, id, flow.dest, channel_id, EventType::PacketSentFinshed);
                send_ehd->flowTag = snd_req.flowTag;

                stream->owner->front_end_sim_send(
                    0, Sys::dummy_data, snd_req.reqCount, UINT8, flow.dest,
                    snd_req.tag, &snd_req, &Sys::handleEvent, send_ehd);
            }
            break;
        }
        default:
            NcclLog->writeLog(NcclLogLevel::WARNING, "ID %d: Received unknown event type: %d.", id, static_cast<int>(event));
            break;
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: run() finished.", id);
}

void NcclCustom::check_completion() {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: check_completion ENTRY. Sends completed: %d/%d, Recvs completed: %d/%d, sendInAir: %d, recvInAir: %d",
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
    }
}

void NcclCustom::exit() {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: exit() at tick %llu", id, Sys::boostedTick());
    end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    NcclLog->writeLog(NcclLogLevel::INFO, "ID %d: Communication Latency: %lld us", id, duration.count());

#if COLLECTIVE_GRAPH
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
#endif

    if (stream && stream->owner) {
        stream->owner->proceed_to_next_vnet_baseline(static_cast<StreamBaseline*>(stream));
    }
     NcclLog->writeLog(NcclLogLevel::DEBUG, "ID %d: exit() EXIT.", id);
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