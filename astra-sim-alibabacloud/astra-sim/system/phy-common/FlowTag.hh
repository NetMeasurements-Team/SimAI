#ifndef __SIMAI_FLOWTAG_HH__
#define __SIMAI_FLOWTAG_HH__

#include <cstddef>
#include <cstdint>
#include <vector>

namespace AstraSim {
/**
 * TODO: maybe there is no need for this structure at all, as the PacketEventHandler data is enough.
 *  Rather use request?
 */
struct ncclFlowTag {
  int channel_id;
  int chunk_id;
  int current_flow_id;
  [[deprecated("children are read directly from _flow_models, no need to set this")]]
  int child_flow_id;
  int sender_node;
  int receiver_node;
  uint64_t flow_size;
  size_t tag_id;
  [[deprecated("children are read directly from _flow_models, no need to set this")]]
  std::vector<int> tree_flow_list;
  bool nvls_on;
  ncclFlowTag():
  channel_id(-1),
  chunk_id(-1),
  current_flow_id(-1),
  child_flow_id(-1),
  sender_node(-1),
  receiver_node(-1),
  flow_size(-1),
  tag_id(-1),
  nvls_on(false){};
  ncclFlowTag(
      int _channel_id,
      int _chunk_id,
      int _current_flow_id,
      int _child_flow_id,
      int _sender_node,
      int _receiver_node,
      uint64_t _flow_size,
      int _tag_id,
      bool _nvls_on)
      : channel_id(_channel_id),
        chunk_id(_chunk_id),
        current_flow_id(_current_flow_id),
        child_flow_id(_child_flow_id),
        sender_node(_sender_node),
        receiver_node(_receiver_node),
        flow_size(_flow_size),
        tag_id(_tag_id),
        nvls_on(_nvls_on) {};
  ~ncclFlowTag() {};
};
}

#endif // __SIMAI_FLOWTAG_HH__
