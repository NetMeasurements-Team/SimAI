/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "RecvPacketEventHadndlerData.hh"
namespace AstraSim {
RecvPacketEventHadndlerData::RecvPacketEventHadndlerData(
    BaseStream* owner,
    int senderNodeId,
    int receiverNodeId,
    int tag,
    EventType event,
    int vnet,
    int stream_num)
    : BasicEventHandlerData(owner->owner, event) {
  this->owner = owner;
  this->vnet = vnet;
  this->stream_num = stream_num;
  this->message_end = true;
  this->senderNodeId = senderNodeId;
  this->receiverNodeId = receiverNodeId;
  this->tag = tag;
  ready_time = Sys::boostedTick();
  flow_id = -2;
  child_flow_id = -1;
}
RecvPacketEventHadndlerData::RecvPacketEventHadndlerData(
    BaseStream* owner,
    int senderNodeId,
    int receiverNodeId,
    int tag,
    int vnet,
    int stream_num)
    : BasicEventHandlerData(owner->owner, EventType::PacketReceived) {
  this->owner = owner;
  this->vnet = vnet;
  this->stream_num = stream_num;
  this->message_end = true;
  this->senderNodeId = senderNodeId;
  this->receiverNodeId = receiverNodeId;
  this->tag = tag;
  ready_time = Sys::boostedTick();
  flow_id = -2;
  child_flow_id = -1;
}

} // namespace AstraSim
