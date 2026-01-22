# Tutorial: Creating a Custom Offload Target in sdfglib

This tutorial guides you through implementing a custom offload target (e.g., GPU, FPGA, custom accelerator) in sdfglib, using the CUDA implementation as a reference example.
This tutorial assumes an offloading target with dedicated memory between host and device, requiring explicit data transfers.
A target device with shared memory is also possible, but using the offloading infrastructure will introduce restrictions regarding data movement that may not be necessary.
The transformation explicitly targets maps, i.e., for loops with proven parallelism.
If you want to offload other sdfg nodes a fully custom transformations may be more appropriate.


## Overview

An offload target in sdfglib requires implementing four main components:

1. **Schedule Type** - Defines how loops are scheduled on the target
2. **Data Offloading Node** - Handles memory transfers and allocation
3. **Map Dispatcher** - Generates target-specific kernel code
4. **Transform** - Orchestrates the offloading transformation
5. **Plugin Regisration** - Ensures that the dispatchers etc, are integrated

## Component 1: Schedule Type

The schedule type defines target-specific scheduling properties and metadata for a single map.

### Example: CUDA Schedule Type

```cpp
// cuda.h
enum CUDADimension { X = 0, Y = 1, Z = 2 };

class ScheduleType_CUDA {
public:
    // Property accessors
    static void dimension(ScheduleType& schedule, const CUDADimension& dimension);
    static CUDADimension dimension(const ScheduleType& schedule);
    static void block_size(ScheduleType& schedule, const symbolic::Expression block_size);
    static symbolic::Integer block_size(const ScheduleType& schedule);
    static bool nested_sync(const ScheduleType& schedule);
    static void nested_sync(ScheduleType& schedule, const bool nested_sync);

    // Required methods
    static const std::string value() { return "CUDA"; }
    static ScheduleType create() {
        auto schedule_type = ScheduleType(value());
        dimension(schedule_type, CUDADimension::X);
        return schedule_type;
    }
};

// Define the target type
inline codegen::TargetType TargetType_CUDA{ScheduleType_CUDA::value()};
```

### Key Points
- Store target-specific properties (e.g., block size, dimension) using `set_property()` and retrieve with `properties().at()`
- Implement `value()` to return a unique string identifier for your target
- Implement `create()` to construct a default schedule with sensible defaults
- Define a `TargetType` constant using the schedule type's value

## Component 2: Data Offloading Node

Handles memory allocation, deallocation, and transfers between host and device.
Extending the base class `DataOffloadingNode` is recommended to enable reuse of common functionality and data transfer minimization logic.

### Example: CUDA Data Offloading Node

```cpp
// cuda_data_offloading_node.h
class CUDADataOffloadingNode : public offloading::DataOffloadingNode {
private:
    symbolic::Expression device_id_;

public:
    CUDADataOffloadingNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        symbolic::Expression size,
        symbolic::Expression device_id,
        offloading::DataTransferDirection transfer_direction,
        offloading::BufferLifecycle buffer_lifecycle
    );

    // Override base class methods
    void validate(const Function& function) const override; // Test semantic correctness of node
    std::unique_ptr<data_flow::DataFlowNode> clone(...) const override; // Deep copy
    symbolic::SymbolSet symbols() const override; // Return used symbols
    void replace(...) override; // Replace symbols with new expressions
    bool blocking() const override; // Indicate if node is a blocking of asynchroneous operation
    bool redundant_with(const offloading::DataOffloadingNode& other) const override; // if this and other cover identical data
    bool equal_with(const offloading::DataOffloadingNode& other) const override; // if this and other are identical in data movement and buffer lifecycle
};
```

### Constructor Implementation

Populates parameter based on the operation type.
Make sure that the input and output connectors match the attached memlets during creation.

```cpp
CUDADataOffloadingNode::CUDADataOffloadingNode(
    size_t element_id,                                      // populated by builder
    const DebugInfo& debug_info,                            // populated by builder
    const graph::Vertex vertex,                             // populated by builder
    data_flow::DataFlowGraph& parent,                       // populated by builder
    symbolic::Expression size,                  // size of data to transfer/allocate
    symbolic::Expression device_id,                          // device identifier
    offloading::DataTransferDirection transfer_direction,   // H2D, D2H, or NONE
    offloading::BufferLifecycle buffer_lifecycle              // ALLOC, FREE, NO_CHANGE
)
    : offloading::DataOffloadingNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_CUDA_Offloading,
          {},
          {},
          transfer_direction,
          buffer_lifecycle,
          size
      ),
      device_id_(device_id) {

    // Configure inputs/outputs based on operation type
    if (!is_NONE(transfer_direction)) {
        this->inputs_.push_back("_src");
        this->outputs_.push_back("_dst");
    } else if (is_ALLOC(buffer_lifecycle)) {
        this->outputs_.push_back("_ret");
    } else if (is_FREE(buffer_lifecycle)) {
        this->inputs_.push_back("_ptr");
        this->outputs_.push_back("_ptr");
    }
}
```

### Dispatcher for Code Generation

Ensures that the appropriate code is generated for each offloading operation.
Since DataOffloadingNodes are library nodes, implement a `LibraryNodeDispatcher`.

```cpp
class CUDADataOffloadingNodeDispatcher : public codegen::LibraryNodeDispatcher {
public:
    void dispatch_code(
        codegen::PrettyPrinter& stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override {
        auto& offloading_node = static_cast<const CUDADataOffloadingNode&>(this->node_);

        // Generate allocation code
        if (offloading_node.is_alloc()) {
            stream << "cudaMalloc(&" << offloading_node.output(0) << ", "
                   << this->language_extension_.expression(offloading_node.size())
                   << ");" << std::endl;
        }

        // Generate H2D transfer code
        if (offloading_node.is_h2d()) {
            stream << "cudaMemcpy(" << offloading_node.output(0) << ", "
                   << offloading_node.input(0) << ", "
                   << this->language_extension_.expression(offloading_node.size())
                   << ", cudaMemcpyHostToDevice);" << std::endl;
        }

        // Generate D2H transfer code
        if (offloading_node.is_d2h()) {
            stream << "cudaMemcpy(" << offloading_node.output(0) << ", "
                   << offloading_node.input(0) << ", "
                   << this->language_extension_.expression(offloading_node.size())
                   << ", cudaMemcpyDeviceToHost);" << std::endl;
        }

        // Generate deallocation code
        if (offloading_node.is_free()) {
            stream << "cudaFree(" << offloading_node.input(0) << ");" << std::endl;
        }
    }
};
```

### Serialization Support

In order to save/load SDFGs with your offloading nodes, implement a `LibraryNodeSerializer`.
Non-library node properties can be copied.

```cpp
nlohmann::json CUDADataOffloadingNodeSerializer::serialize(const sdfg::data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const CUDADataOffloadingNode&>(library_node);
    nlohmann::json j;

    // Library node
    j["type"] = "library_node";
    j["element_id"] = library_node.element_id();

    // Debug info
    auto& debug_info = library_node.debug_info();
    j["has"] = debug_info.has();
    j["filename"] = debug_info.filename();
    j["start_line"] = debug_info.start_line();
    j["start_column"] = debug_info.start_column();
    j["end_line"] = debug_info.end_line();
    j["end_column"] = debug_info.end_column();

    // Library node properties
    j["code"] = std::string(library_node.code().value());

    // Offloading node properties
    sdfg::serializer::JSONSerializer serializer;
    if (node.size().is_null()) {
        j["size"] = nlohmann::json::value_t::null;
    } else {
        j["size"] = serializer.expression(node.size());
    }
    j["device_id"] = serializer.expression(node.device_id());
    j["transfer_direction"] = static_cast<int8_t>(node.transfer_direction());
    j["buffer_lifecycle"] = static_cast<int8_t>(node.buffer_lifecycle());

    return j;
}

data_flow::LibraryNode& CUDADataOffloadingNodeSerializer::deserialize(
    const nlohmann::json& j, sdfg::builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_CUDA_Offloading.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    symbolic::Expression size;
    if (!j.contains("size") || j.at("size").is_null()) {
        size = SymEngine::null;
    } else {
        size = symbolic::parse(j.at("size"));
    }
    SymEngine::Expression device_id(j.at("device_id"));
    auto transfer_direction = static_cast<offloading::DataTransferDirection>(j["transfer_direction"].get<int8_t>());
    auto buffer_lifecycle = static_cast<offloading::BufferLifecycle>(j["buffer_lifecycle"].get<int8_t>());
Ds
    return builder.add_library_node<
        CUDADataOffloadingNode>(parent, debug_info, size, device_id, transfer_direction, buffer_lifecycle);
}
```

## Component 3: Map Dispatcher

Generates kernel code for parallel loops scheduled on the target.

### Example: CUDA Map Dispatcher

```cpp
class CUDAMapDispatcher : public codegen::NodeDispatcher {
private:
    structured_control_flow::Map& node_;

    // Helper methods
    void dispatch_kernel_body(...);  // Generates the function logic of the kernel
    void dispatch_header(...);  // Generates the kernel declaration for the header file
    void dispatch_kernel_call(...);  // Generates the kernel call in place of the original map
    void dispatch_kernel_preamble(...);  // Generates kernel preamble (e.g., thread/block indices, local variables etc.)

    // Analysis helpers
    symbolic::Expression find_nested_cuda_blocksize(...);  // find the block sizes of nested cuda maps, required for boundary checks and the kernel launch configuration
    symbolic::Expression find_nested_cuda_iterations(...);  // find the iteration counts of nested cuda maps, required for boundary checks and the kernel launch configuration
    bool is_outermost_cuda_map(...);  // check if this is the outermost cuda map in a nested structure, if only the kernel body is emitted
    symbolic::SymbolSet get_indvars(...); // get all induction variables replaced by nested cuda maps, required for boundary checks

public:
    void dispatch_node(
        codegen::PrettyPrinter& main_stream,
        codegen::PrettyPrinter& globals_stream,
        codegen::CodeSnippetFactory& library_snippet_factory
    ) override;

    codegen::InstrumentationInfo instrumentation_info() const override;
};
```

### Key Implementation Steps

1. **Analyze the loop structure** - Determine parallelism dimensions, block sizes, iteration counts
2. **Prepare arguments** - Separate device vs. host arguments, handle scalars
3. **Generate kernel if outermost** - For nested GPU loops, only the outermost generates a kernel
4. **Emit kernel call** - Launch configuration with grid/block dimensions
5. **Emit kernel body** - Include synchronization, boundary checks, and actual computation

### Example dispatch_node() Structure

```cpp
void CUDAMapDispatcher::dispatch_node(...) {
    // 1. Analyze arguments and locals
    auto& used_arguments = arguments_analysis.arguments(...);
    auto& locals = arguments_analysis.locals(...);

    // 2. Calculate dimensions
    auto block_size_x = find_nested_cuda_blocksize(..., CUDADimension::X);
    auto num_iters_x = find_nested_cuda_iterations(..., CUDADimension::X);
    auto num_blocks_x = ceiling(num_iters_x / block_size_x);

    // 3. Generate code
    if (is_outermost_cuda_map(...)) {
        // Emit kernel call in main stream
        dispatch_kernel_call(main_stream, kernel_name, ...);

        // Emit kernel declaration in globals
        dispatch_header(globals_stream, kernel_name, ...);

        // Emit kernel implementation in library file
        dispatch_kernel_preamble(library_stream, ...);
        dispatch_kernel_body(library_stream, ...);
    } else {
        // Nested: just emit body
        dispatch_kernel_body(main_stream, ...);
    }
}
```

## Component 4: Transform

Orchestrates the complete offloading transformation.

### Example: CUDA Transform

```cpp
class CUDATransform : public transformations::OffloadTransform {
public:
    explicit CUDATransform(
        structured_control_flow::Map& map, // Map to offload
        int block_size = 32,             // CUDA block size
        bool allow_dynamic_sizes = false  // Allow dynamic sizes, i.e., interpret data transfer sizes with malloc_usable_size() (may lead to an overestimation of required memory)
    ) : OffloadTransform(map, allow_dynamic_sizes), block_size_(block_size) {}

    std::string name() const override { return "CUDATransform"; }

protected:
    // Define storage types
    types::StorageType local_device_storage_type() override {
        return types::StorageType("NV_Generic", ...);
    }

    types::StorageType global_device_storage_type(symbolic::Expression arg_size) override {
        return types::StorageType("NV_Generic", arg_size, ...);
    }

    // Define schedule type
    ScheduleType transformed_schedule_type() override {
        auto schedule = ScheduleType_CUDA::create();
        ScheduleType_CUDA::block_size(schedule, symbolic::integer(block_size_));
        return schedule;
    }

    // Prefix for device copies
    std::string copy_prefix() override { return CUDA_DEVICE_PREFIX; }  // prefix to differentiate device copies of arguments

    // Memory operations - delegate to DataOffloadingNode
    void allocate_device_arg(...) override;   // create an allocation on the device, perferably using your extended DataOffloadingNode
    void deallocate_device_arg(...) override;  // create a deallocation on the device, perferably using your extended DataOffloadingNode
    void copy_to_device(...) override;     // create a copy to device operation, perferably using your extended DataOffloadingNode
    void copy_from_device(...) override;  // create a copy from device operation, perferably using your extended DataOffloadingNode

    // Optional device setup/teardown
    void setup_device(...) override {}   // e.g., set device context, not required for CUDA (done implicitly on first contact with device)
    void teardown_device(...) override {}   // e.g., reset device context, not required for CUDA

private:
    int block_size_;
};
```

### Memory Operation Implementation

Each method creates the appropriate `DataOffloadingNode`:
Remember that a `DataOffloadingNode` is a library node, so you need to integrate it with the appropriate access/constant nodes and memlets.
```cpp
void CUDATransform::copy_to_device(...) {
    auto& access_node_host = builder.add_access(copy_block, host_arg_name);
    auto& access_node_device = builder.add_access(copy_block, device_arg_name);

    auto& memcpy_node = builder.add_library_node<CUDADataOffloadingNode>(
        copy_block,
        this->map_.debug_info(),
        size,
        symbolic::integer(0),  // device_id
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE
    );

    builder.add_computational_memlet(copy_block, access_node_host,
                                     memcpy_node, "_src", {}, in_type);
    builder.add_computational_memlet(copy_block, memcpy_node, "_dst",
                                     access_node_device, {}, out_type);
}
```

## Component 5: Plugin Registration

The plugin system provides a clean way to register all target components with the sdfglib runtime.
The registration of serializers, map dispatchers, and library node dispatchers is required for proper integration.
It ensures that external extensions, like your plugin, can be used seamlessly with sdfglib's code generation and serialization infrastructure.

Each registry has a singleton instance that manages the registered components.
The components are registered using factory functions (lambdas) that create instances of the respective classes when needed.
To find the correct dispatchers and serializers, they are mapped using the schedule type, library node type, and (implementation type, in case of the library node dispatcher).

If your plugin is not correctly registered, sdfglib will raise errors when trying to generate code or serialize SDFGs that use your target.

### Location
- Header: `include/sdfg/targets/<target>/plugin.h`

### Example: CUDA Plugin

```cpp
// plugin.h
#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/cuda/cuda_map_dispatcher.h"

namespace sdfg {
namespace cuda {

inline void register_cuda_plugin() {
    // 1. Register Map Dispatcher
    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        ScheduleType_CUDA::value(),
        [](codegen::LanguageExtension& language_extension,
           StructuredSDFG& sdfg,
           analysis::AnalysisManager& analysis_manager,
           structured_control_flow::Map& node,
           codegen::InstrumentationPlan& instrumentation_plan,
           codegen::ArgCapturePlan& arg_capture_plan) {
            return std::make_unique<CUDAMapDispatcher>(
                language_extension, sdfg, analysis_manager, node,
                instrumentation_plan, arg_capture_plan
            );
        }
    );

    // 2. Register Library Node Dispatcher
    codegen::LibraryNodeDispatcherRegistry::instance()
        .register_library_node_dispatcher(
            LibraryNodeType_CUDA_Offloading.value() + "::" +
            data_flow::ImplementationType_NONE.value(),
            [](codegen::LanguageExtension& language_extension,
               const Function& function,
               const data_flow::DataFlowGraph& data_flow_graph,
               const data_flow::LibraryNode& node) {
                return std::make_unique<CUDADataOffloadingNodeDispatcher>(
                    language_extension, function, data_flow_graph, node
                );
            }
        );

    // 3. Register Serializer
    serializer::LibraryNodeSerializerRegistry::instance()
        .register_library_node_serializer(
            LibraryNodeType_CUDA_Offloading.value(),
            []() {
                return std::make_unique<CUDADataOffloadingNodeSerializer>();
            }
        );
}

} // namespace cuda
} // namespace sdfg
```

### Key Components

#### 1. Map Dispatcher Registration

```cpp
codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
    ScheduleType_CUDA::value(),  // Schedule type identifier
    [](/* parameters */) {        // Factory lambda
        return std::make_unique<CUDAMapDispatcher>(...);
    }
);
```

**Purpose**: Associates your schedule type with the dispatcher that generates code for loops scheduled on your target.

**Parameters**:
- `schedule_type_id`: The unique string identifier from `ScheduleType_<Target>::value()`
- `factory`: Lambda that creates your map dispatcher instance

#### 2. Library Node Dispatcher Registration

```cpp
codegen::LibraryNodeDispatcherRegistry::instance()
    .register_library_node_dispatcher(
        LibraryNodeType_CUDA_Offloading.value() + "::" +
        data_flow::ImplementationType_NONE.value(),
        [](/* parameters */) {
            return std::make_unique<CUDADataOffloadingNodeDispatcher>(...);
        }
    );
```

**Purpose**: Associates your library node type with the dispatcher that generates code for data offloading operations.

**Key**: Combination of library node type and implementation type (typically `NONE` for offloading nodes)

**Parameters**:
- `language_extension`: Code generation utilities
- `function`: The SDFG function being compiled
- `data_flow_graph`: Parent graph containing the node
- `node`: The library node to dispatch

#### 3. Serializer Registration

```cpp
serializer::LibraryNodeSerializerRegistry::instance()
    .register_library_node_serializer(
        LibraryNodeType_CUDA_Offloading.value(),
        []() {
            return std::make_unique<CUDADataOffloadingNodeSerializer>();
        }
    );
```

**Purpose**: Enables saving/loading SDFGs with your offloading nodes to/from JSON.

**Important**: Only the library node serializer needs registration. Map serialization is handled automatically.

#### 3. Header, Linker, and Compiler Flag Registration (WIP)

```cpp

```

**Purpose**: Enables inclusion of target-specific headers, linker flags, and compiler flags during code generation.

**Important**: WIP

### Using Your Plugin

To use your plugin, call the registration function early in your pipeline and tests.
Registering plugins multiple times will raise errors.

```cpp
#include <sdfg/targets/cuda/plugin.h>

int main() {
    // Register plugin before using any CUDA features
    sdfg::cuda::register_cuda_plugin();

    // Now you can create SDFGs with CUDA scheduling
    // ...
}
```

### Plugin Best Practices

1. **Single Registration Function**: Encapsulate all registrations in one function
2. **Inline Header**: Use `inline` to allow header-only registration
3. **Namespace Organization**: Keep plugin in your target's namespace
4. **Early Registration**: Call registration before creating any SDFGs

## Integration Steps

### 1. Create Plugin Header

Create `include/sdfg/targets/<target>/plugin.h` with registration function as shown above.

### 2. Call Registration

In your application or test setup:

```cpp
#include <sdfg/targets/<target>/plugin.h>

sdfg::<target>::register_<target>_plugin();
```

### 3. Verify Registration

Test that your components are properly registered:

```cpp
// Check map dispatcher
auto& map_registry = codegen::MapDispatcherRegistry::instance();
ASSERT_TRUE(map_registry.has_dispatcher(ScheduleType_<Target>::value()));

// Check library node dispatcher
auto& lib_registry = codegen::LibraryNodeDispatcherRegistry::instance();
ASSERT_TRUE(lib_registry.has_dispatcher(
    LibraryNodeType_<Target>_Offloading.value() + "::" +
    data_flow::ImplementationType_NONE.value()
));
```

### 4. Language Extensions (Optional)

Create target-specific language extensions for code generation:

```cpp
class CUDALanguageExtension : public codegen::LanguageExtension {
public:
    std::string expression(const symbolic::Expression& expr) const override {
        // Handle CUDA-specific expressions (e.g., threadIdx, blockIdx)
        if (expr is threadIdx_x) return "threadIdx.x";
        if (expr is blockIdx_x) return "blockIdx.x";
        // ... delegate to base for others
    }
};
```

## Testing

Create comprehensive tests for your target:

```cpp
TEST(TargetTests, BasicOffload) {
    // 1. Create SDFG with a map
    // 2. Apply your transform
    // 3. Verify the transformed SDFG structure
    // 4. Generate code and compile
    // 5. Run and verify correctness
}
```

## Summary Checklist

- [ ] Implement `ScheduleType_<Target>` class
- [ ] Define `TargetType_<Target>` constant
- [ ] Define `LibraryNodeType_<Target>_Offloading` constant
- [ ] Implement `<Target>DataOffloadingNode` class
- [ ] Implement `<Target>DataOffloadingNodeDispatcher` class
- [ ] Implement `<Target>DataOffloadingNodeSerializer` class
- [ ] Implement `<Target>MapDispatcher` class
- [ ] Implement `<Target>Transform` class
- [ ] Create `plugin.h` with registration function
- [ ] Register all dispatchers and serializers in plugin
- [ ] Call plugin registration in application/tests
- [ ] Create language extension (if needed)
- [ ] Write comprehensive tests
- [ ] Update build system (CMakeLists.txt)

## Best Practices

1. **Reuse base classes**: Extend `DataOffloadingNode` and `OffloadTransform` rather than reimplementing
2. **Handle nested parallelism**: Check if you're the outermost map before generating kernel code
3. **Validate early**: Implement `validate()` methods to catch errors during SDFG construction
4. **Support serialization**: Ensure your nodes can be saved/loaded for testing and persistence
5. **Use symbolic expressions**: Support symbolic sizes and parameters throughout
6. **Instrument for profiling**: Implement `instrumentation_info()` for performance analysis

## Additional Resources

- Base classes: `sdfg/targets/offloading/data_offloading_node.h`
- Transform base: `sdfg/transformations/offloading/offload_transform.h`
- Code generation: `sdfg/codegen/dispatchers/`
- Examples: The CUDA implementation serves as a complete reference

---

*This tutorial is based on the CUDA implementation in sdfglib. Adapt the patterns to your specific target's requirements.*
