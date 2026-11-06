/*
* pybind_simai.cc
* Python In-Memory Bindings for SimAI Inference and Co-Simulation.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "SimAIRuntime.h"

namespace py = pybind11;

namespace simai {

class PySimAIBridge {
public:
  static PySimAIBridge& get_instance() {
    static PySimAIBridge bridge;
    return bridge;
  }

  void init_network(
      const std::string& topo_file,
      const std::string& conf_file,
      const std::string& sys_conf_file = "",
      int num_threads = 1,
      int min_lookahead_ns = 50) {
    py::gil_scoped_release release;
    SimAIRuntime::get_instance().init_network(
        topo_file,
        conf_file,
        sys_conf_file,
        num_threads,
        min_lookahead_ns);
  }

  double predict_collective(
      const std::string& comm_type,
      const std::vector<int>& ranks,
      uint64_t bytes,
      uint64_t compute_gap_us = 0) {
    py::gil_scoped_release release;
    return SimAIRuntime::get_instance().predict_collective(
        comm_type,
        ranks,
        bytes,
        compute_gap_us);
  }

  void post_collective(
      uint64_t batch_id,
      const std::string& comm_type,
      const std::vector<int>& ranks,
      uint64_t bytes,
      uint64_t compute_gap_us,
      py::function completion_callback) {
    // Use Python's native PyGILState_Ensure / PyGILState_Release rather than
    // py::gil_scoped_acquire on C++ worker threads so temporary PyThreadStates
    // are cleaned up immediately while holding the GIL instead of inside a
    // pthread TLS destructor during interpreter finalization.
    struct PyGilGuard {
      PyGILState_STATE state;
      PyGilGuard() : state(PyGILState_Ensure()) {}
      ~PyGilGuard() { PyGILState_Release(state); }
    };
    struct PyCallbackWrapper {
      py::function fn;
      explicit PyCallbackWrapper(py::function f) : fn(std::move(f)) {}
      ~PyCallbackWrapper() {
        if (fn && Py_IsInitialized()) {
          PyGilGuard gil;
          fn = py::function();
        } else {
          fn.release();
        }
      }
    };
    auto wrapper = std::make_shared<PyCallbackWrapper>(std::move(completion_callback));
    auto safe_callback = [wrapper](double duration_us) {
      if (!Py_IsInitialized()) {
        return;
      }
      PyGilGuard gil;
      if (wrapper->fn) {
        wrapper->fn(duration_us);
        wrapper->fn = py::function();
      }
    };

    py::gil_scoped_release release;
    SimAIRuntime::get_instance().post_collective(
        batch_id,
        comm_type,
        ranks,
        bytes,
        compute_gap_us,
        safe_callback);
  }

  double current_time_ns() {
    return SimAIRuntime::get_instance().current_time_ns();
  }

  void inject_link_failure(int node_a, int node_b, double duration_us) {
    py::gil_scoped_release release;
    SimAIRuntime::get_instance().inject_link_failure(
        node_a,
        node_b,
        duration_us);
  }

  bool is_initialized() {
    return SimAIRuntime::get_instance().is_initialized();
  }

  int get_total_gpus() {
    return SimAIRuntime::get_instance().get_total_gpus();
  }

private:
  PySimAIBridge() = default;
};

} // namespace simai

PYBIND11_MODULE(pysimai, m) {
  m.doc() = "SimAI In-Memory Inference Simulator & Co-Simulation Bindings";

  py::class_<simai::PySimAIBridge>(m, "PySimAIBridge")
      .def_static("get_instance", &simai::PySimAIBridge::get_instance,
                  py::return_value_policy::reference)
      .def("init_network", &simai::PySimAIBridge::init_network,
           py::arg("topo_file"),
           py::arg("conf_file"),
           py::arg("sys_conf_file") = "",
           py::arg("num_threads") = 1,
           py::arg("min_lookahead_ns") = 50,
           "Initialize resident network topology and Astra-Sim systems")
      .def("predict_collective", &simai::PySimAIBridge::predict_collective,
           py::arg("comm_type"),
           py::arg("ranks"),
           py::arg("bytes"),
           py::arg("compute_gap_us") = 0,
           "Synchronously simulate and predict collective communication latency in microseconds")
      .def("post_collective", &simai::PySimAIBridge::post_collective,
           py::arg("batch_id"),
           py::arg("comm_type"),
           py::arg("ranks"),
           py::arg("bytes"),
           py::arg("compute_gap_us") = 0,
           py::arg("completion_callback"),
           "Asynchronously dispatch a collective and trigger a callback upon completion")
      .def("current_time_ns", &simai::PySimAIBridge::current_time_ns,
           "Query current simulation time in nanoseconds")
      .def("inject_link_failure", &simai::PySimAIBridge::inject_link_failure,
           py::arg("node_a"),
           py::arg("node_b"),
           py::arg("duration_us"),
           "Inject a dynamic link outage between node_a and node_b")
      .def("is_initialized", &simai::PySimAIBridge::is_initialized,
           "Check if the simulation backend has been initialized")
      .def("get_total_gpus", &simai::PySimAIBridge::get_total_gpus,
           "Get total simulated GPU ranks");

  auto atexit = py::module_::import("atexit");
  atexit.attr("register")(py::cpp_function([]() {
    py::gil_scoped_release release;
    simai::SimAIRuntime::get_instance().shutdown();
  }));
}
