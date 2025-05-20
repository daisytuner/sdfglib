#include "sdfg/visualizer/dot_visualizer.h"

#include <cstddef>
#include <string>
#include <utility>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/schedule.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace visualizer {

void DotVisualizer::visualizeBlock(Schedule& schedule, structured_control_flow::Block& block) {
    this->stream_ << "subgraph cluster_" << block.name() << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl;
    this->last_comp_name_cluster_ = "cluster_" + block.name();
    if (block.dataflow().nodes().empty()) {
        this->stream_ << block.name() << " [shape=point,style=invis,label=\"\"];" << std::endl;
        this->stream_.setIndent(this->stream_.indent() - 4);
        this->stream_ << "}" << std::endl;
        this->last_comp_name_ = block.name();
        return;
    }
    this->last_comp_name_.clear();
    std::list<data_flow::DataFlowNode*> nodes = block.dataflow().topological_sort();
    for (data_flow::DataFlowNode* node : nodes) {
        if (const data_flow::Tasklet* tasklet = dynamic_cast<data_flow::Tasklet*>(node)) {
            this->stream_ << tasklet->name() << " [shape=octagon,label=\""
                          << tasklet->output(0).first << " = ";
            this->visualizeTasklet(*tasklet);
            this->stream_ << "\"];" << std::endl;
            for (data_flow::Memlet& iedge : block.dataflow().in_edges(*tasklet)) {
                data_flow::AccessNode const& src =
                    dynamic_cast<data_flow::AccessNode const&>(iedge.src());
                this->stream_ << src.name() << " -> " << tasklet->name() << " [label=\"   "
                              << iedge.dst_conn() << " = " << src.data();
                if (!symbolic::is_nvptx(symbolic::symbol(src.data()))) {
                    types::IType const& type = schedule.sdfg().type(src.data());
                    this->visualizeSubset(schedule.sdfg(), type, iedge.subset());
                }
                this->stream_ << "   \"];" << std::endl;
            }
            for (data_flow::Memlet& oedge : block.dataflow().out_edges(*tasklet)) {
                data_flow::AccessNode const& dst =
                    dynamic_cast<data_flow::AccessNode const&>(oedge.dst());
                types::IType const& type = schedule.sdfg().type(dst.data());
                this->stream_ << tasklet->name() << " -> " << dst.name() << " [label=\"   "
                              << dst.data();
                this->visualizeSubset(schedule.sdfg(), type, oedge.subset());
                this->stream_ << " = " << oedge.src_conn() << "   \"];" << std::endl;
            }
            if (this->last_comp_name_.empty()) this->last_comp_name_ = tasklet->name();
        } else if (const data_flow::AccessNode* access_node =
                       dynamic_cast<data_flow::AccessNode*>(node)) {
            bool source = false, sink = false;
            for (data_flow::Memlet& edge : block.dataflow().out_edges(*access_node)) {
                if ((source = (edge.src_conn() == "void"))) break;
            }
            for (data_flow::Memlet& edge : block.dataflow().in_edges(*access_node)) {
                if ((sink = (edge.dst_conn() == "void"))) break;
            }
            if (!source && !sink) continue;
            this->stream_ << access_node->name() << " [";
            if (!schedule.sdfg().is_internal(access_node->data())) this->stream_ << "penwidth=3.0,";
            if (schedule.sdfg().is_transient(access_node->data()))
                this->stream_ << "style=\"dashed,filled\",";
            this->stream_ << "label=\"" << access_node->data() << "\"];" << std::endl;
        } else if (const data_flow::LibraryNode* libnode =
                       dynamic_cast<data_flow::LibraryNode*>(node)) {
            this->stream_ << libnode->name() << " [shape=doubleoctagon,label=\"";
            this->visualizeLibraryNode(libnode->call());
            this->stream_ << "\"];" << std::endl;
            if (this->last_comp_name_.empty()) this->last_comp_name_ = libnode->name();
        }
    }
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
}

void DotVisualizer::visualizeSequence(Schedule& schedule,
                                      structured_control_flow::Sequence& sequence) {
    std::string last_comp_name_tmp, last_comp_name_cluster_tmp;
    for (size_t i = 0; i < sequence.size(); ++i) {
        std::pair<ControlFlowNode&, Transition&> child = sequence.at(i);
        this->visualizeNode(schedule, child.first);
        if ((i > 0) && !last_comp_name_tmp.empty() && !this->last_comp_name_.empty()) {
            this->stream_ << last_comp_name_tmp << " -> " << this->last_comp_name_ << " [";
            if (!last_comp_name_cluster_tmp.empty())
                this->stream_ << "ltail=\"" << last_comp_name_cluster_tmp << "\",";
            if (!this->last_comp_name_cluster_.empty())
                this->stream_ << "lhead=\"" << this->last_comp_name_cluster_ << "\",";
            this->stream_ << "minlen=3]"
                          << ";" << std::endl;
        }
        last_comp_name_tmp = this->last_comp_name_;
        this->last_comp_name_.clear();
        last_comp_name_cluster_tmp = this->last_comp_name_cluster_;
        this->last_comp_name_cluster_.clear();
    }
}

void DotVisualizer::visualizeIfElse(Schedule& schedule, structured_control_flow::IfElse& if_else) {
    this->stream_ << "subgraph cluster_" << if_else.name() << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";"
                  << std::endl
                  << if_else.name() << " [shape=point,style=invis,label=\"\"];" << std::endl;
    for (size_t i = 0; i < if_else.size(); ++i) {
        this->stream_ << "subgraph cluster_" << if_else.name() << "_" << std::to_string(i) << " {"
                      << std::endl;
        this->stream_.setIndent(this->stream_.indent() + 4);
        this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\""
                      << this->expression(if_else.at(i).second->__str__()) << "\";" << std::endl;
        this->visualizeSequence(schedule, if_else.at(i).first);
        this->stream_.setIndent(this->stream_.indent() - 4);
        this->stream_ << "}" << std::endl;
    }
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = if_else.name();
    this->last_comp_name_cluster_ = "cluster_" + if_else.name();
}

void DotVisualizer::visualizeWhile(Schedule& schedule, structured_control_flow::While& while_loop) {
    this->stream_ << "subgraph cluster_" << while_loop.name() << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"while:\";"
                  << std::endl
                  << while_loop.name() << " [shape=point,style=invis,label=\"\"];" << std::endl;
    this->visualizeSequence(schedule, while_loop.root());
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = while_loop.name();
    this->last_comp_name_cluster_ = "cluster_" + while_loop.name();
}

void DotVisualizer::visualizeFor(Schedule& schedule, structured_control_flow::For& loop) {
    this->stream_ << "subgraph cluster_" << loop.name() << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: ";
    this->visualizeForBounds(loop.indvar(), loop.init(), loop.condition(), loop.update());
    LoopSchedule loop_schedule = schedule.loop_schedule(&loop);
    if (loop_schedule == LoopSchedule::VECTORIZATION) this->stream_ << " (vectorized)";
    if (loop_schedule == LoopSchedule::MULTICORE) this->stream_ << " (parallelized)";
    this->stream_ << "\";" << std::endl
                  << loop.name() << " [shape=point,style=invis,label=\"\"];" << std::endl;
    this->visualizeSequence(schedule, loop.root());
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = loop.name();
    this->last_comp_name_cluster_ = "cluster_" + loop.name();
}

void DotVisualizer::visualizeReturn(Schedule& schedule,
                                    structured_control_flow::Return& return_node) {
    this->stream_ << return_node.name() << " [shape=cds,label=\" return  \"];" << std::endl;
    this->last_comp_name_ = return_node.name();
    this->last_comp_name_cluster_.clear();
}
void DotVisualizer::visualizeBreak(Schedule& schedule, structured_control_flow::Break& break_node) {
    this->stream_ << break_node.name() << " [shape=cds,label=\" break  \"];" << std::endl;
    this->last_comp_name_ = break_node.name();
    this->last_comp_name_cluster_.clear();
}

void DotVisualizer::visualizeContinue(Schedule& schedule,
                                      structured_control_flow::Continue& continue_node) {
    this->stream_ << continue_node.name() << " [shape=cds,label=\" continue  \"];" << std::endl;
    this->last_comp_name_ = continue_node.name();
    this->last_comp_name_cluster_.clear();
}

void DotVisualizer::visualizeKernel(Schedule& schedule,
                                    structured_control_flow::Kernel& kernel_node) {
    bool gridDim_x = false;
    bool gridDim_y = false;
    bool gridDim_z = false;
    bool blockDim_x = false;
    bool blockDim_y = false;
    bool blockDim_z = false;
    bool blockIdx_x = false;
    bool blockIdx_y = false;
    bool blockIdx_z = false;
    bool threadIdx_x = false;
    bool threadIdx_y = false;
    bool threadIdx_z = false;

    for (std::string const& container : schedule.sdfg().containers()) {
        gridDim_x = gridDim_x || (container == kernel_node.gridDim_x()->get_name());
        gridDim_y = gridDim_y || (container == kernel_node.gridDim_y()->get_name());
        gridDim_z = gridDim_z || (container == kernel_node.gridDim_z()->get_name());
        blockDim_x = blockDim_x || (container == kernel_node.blockDim_x()->get_name());
        blockDim_y = blockDim_y || (container == kernel_node.blockDim_y()->get_name());
        blockDim_z = blockDim_z || (container == kernel_node.blockDim_z()->get_name());
        blockIdx_x = blockIdx_x || (container == kernel_node.blockIdx_x()->get_name());
        blockIdx_y = blockIdx_y || (container == kernel_node.blockIdx_y()->get_name());
        blockIdx_z = blockIdx_z || (container == kernel_node.blockIdx_z()->get_name());
        threadIdx_x = threadIdx_x || (container == kernel_node.threadIdx_x()->get_name());
        threadIdx_y = threadIdx_y || (container == kernel_node.threadIdx_y()->get_name());
        threadIdx_z = threadIdx_z || (container == kernel_node.threadIdx_z()->get_name());
    }

    size_t replacements_size_before = this->replacements_.size();

    if (gridDim_x)
        this->replacements_.push_back(
            {kernel_node.gridDim_x()->get_name(), kernel_node.gridDim_x_init()->__str__()});
    if (gridDim_y)
        this->replacements_.push_back(
            {kernel_node.gridDim_y()->get_name(), kernel_node.gridDim_y_init()->__str__()});
    if (gridDim_z)
        this->replacements_.push_back(
            {kernel_node.gridDim_z()->get_name(), kernel_node.gridDim_z_init()->__str__()});
    if (blockDim_x)
        this->replacements_.push_back(
            {kernel_node.blockDim_x()->get_name(), kernel_node.blockDim_x_init()->__str__()});
    if (blockDim_y)
        this->replacements_.push_back(
            {kernel_node.blockDim_y()->get_name(), kernel_node.blockDim_y_init()->__str__()});
    if (blockDim_z)
        this->replacements_.push_back(
            {kernel_node.blockDim_z()->get_name(), kernel_node.blockDim_z_init()->__str__()});
    if (blockIdx_x)
        this->replacements_.push_back(
            {kernel_node.blockIdx_x()->get_name(), kernel_node.blockIdx_x_init()->__str__()});
    if (blockIdx_y)
        this->replacements_.push_back(
            {kernel_node.blockIdx_y()->get_name(), kernel_node.blockIdx_y_init()->__str__()});
    if (blockIdx_z)
        this->replacements_.push_back(
            {kernel_node.blockIdx_z()->get_name(), kernel_node.blockIdx_z_init()->__str__()});
    if (threadIdx_x)
        this->replacements_.push_back(
            {kernel_node.threadIdx_x()->get_name(), kernel_node.threadIdx_x_init()->__str__()});
    if (threadIdx_y)
        this->replacements_.push_back(
            {kernel_node.threadIdx_y()->get_name(), kernel_node.threadIdx_y_init()->__str__()});
    if (threadIdx_z)
        this->replacements_.push_back(
            {kernel_node.threadIdx_z()->get_name(), kernel_node.threadIdx_z_init()->__str__()});

    this->visualizeSequence(schedule, kernel_node.root());

    this->replacements_.resize(replacements_size_before);
}

void DotVisualizer::visualizeMap(Schedule& schedule, structured_control_flow::Map& map_node) {
    // TODO: Implement map visualization @Adrian
}

void DotVisualizer::visualize() {
    this->stream_.clear();
    this->stream_ << "digraph " << this->schedule_.name() << " {" << std::endl;
    this->stream_.setIndent(4);
    this->stream_ << "graph [compound=true];" << std::endl;
    for (size_t i = 0; i < schedule_.size(); ++i) {
        StructuredSDFG const& sdfg = this->schedule_.schedule(i).sdfg();
        StructuredSDFG& function = this->schedule_.schedule(i).builder().subject();
        this->stream_ << "subgraph cluster_" << sdfg.name() << " {" << std::endl;
        this->stream_.setIndent(8);
        this->stream_ << "node [style=filled,fillcolor=white];" << std::endl
                      << "style=filled;color=lightblue;label=\"";
        std::string condition = this->expression(this->schedule_.condition(i)->__str__());
        if (condition != "True") this->stream_ << condition;
        this->stream_ << "\";" << std::endl;
        this->visualizeSequence(this->schedule_.schedule(i), function.root());
        this->stream_.setIndent(4);
        this->stream_ << "}" << std::endl;
    }
    this->stream_.setIndent(0);
    this->stream_ << "}" << std::endl;
}

}  // namespace visualizer
}  // namespace sdfg
