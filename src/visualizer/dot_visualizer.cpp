#include "sdfg/visualizer/dot_visualizer.h"

#include <cstddef>
#include <string>
#include <utility>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_sdfg.h"
#include <regex>

namespace sdfg {
namespace visualizer {

static std::regex dotIdBadChars("[^a-zA-Z0-9_]+");

static std::string escapeDotId(size_t id, const std::string& prefix = "") {
    return prefix + std::to_string(id);
}

static std::string escapeDotId(const std::string& id, const std::string& prefix = "") {
    return prefix + std::regex_replace(id, dotIdBadChars, "_");
}

void DotVisualizer::visualizeBlock(StructuredSDFG& sdfg, structured_control_flow::Block& block) {
    auto id = escapeDotId(block.element_id(), "block_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"\";" << std::endl;
    this->last_comp_name_cluster_ = "cluster_" + id;
    if (block.dataflow().nodes().empty()) {
        this->stream_ << id << " [shape=point,style=invis,label=\"\"];"
                      << std::endl;
        this->stream_.setIndent(this->stream_.indent() - 4);
        this->stream_ << "}" << std::endl;
        this->last_comp_name_ = id;
        return;
    }
    this->last_comp_name_.clear();
    std::list<data_flow::DataFlowNode*> nodes = block.dataflow().topological_sort();
    for (data_flow::DataFlowNode* node : nodes) {
        if (const data_flow::Tasklet* tasklet = dynamic_cast<data_flow::Tasklet*>(node)) {
            auto taskletId = escapeDotId(tasklet->element_id(), "tasklet_");
            this->stream_ << taskletId << " [shape=octagon,label=\""
                          << tasklet->output().first << " = ";
            this->visualizeTasklet(*tasklet);
            this->stream_ << "\"];" << std::endl;
            for (data_flow::Memlet& iedge : block.dataflow().in_edges(*tasklet)) {
                data_flow::AccessNode const& src =
                    dynamic_cast<data_flow::AccessNode const&>(iedge.src());
                this->stream_ << escapeDotId(src.element_id(), "n_") << " -> " << taskletId
                              << " [label=\"   " << iedge.dst_conn() << " = " << src.data();
                if (!symbolic::is_nv(symbolic::symbol(src.data()))) {
                    types::IType const& type = sdfg.type(src.data());
                    this->visualizeSubset(sdfg, type, iedge.subset());
                }
                this->stream_ << "   \"];" << std::endl;
            }
            for (data_flow::Memlet& oedge : block.dataflow().out_edges(*tasklet)) {
                data_flow::AccessNode const& dst =
                    dynamic_cast<data_flow::AccessNode const&>(oedge.dst());
                types::IType const& type = sdfg.type(dst.data());
                this->stream_ << taskletId << " -> " << escapeDotId(dst.element_id(), "n_")
                              << " [label=\"   " << dst.data();
                this->visualizeSubset(sdfg, type, oedge.subset());
                this->stream_ << " = " << oedge.src_conn() << "   \"];" << std::endl;
            }
            if (this->last_comp_name_.empty()) this->last_comp_name_ = taskletId;
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
            this->stream_ << escapeDotId(access_node->element_id(), "n_") << " [";
            if (!sdfg.is_internal(access_node->data())) this->stream_ << "penwidth=3.0,";
            if (sdfg.is_transient(access_node->data())) this->stream_ << "style=\"dashed,filled\",";
            this->stream_ << "label=\"" << access_node->data() << "\"];" << std::endl;
        } else if (const data_flow::LibraryNode* libnode =
                       dynamic_cast<data_flow::LibraryNode*>(node)) {
            auto id = escapeDotId(libnode->element_id(), "lib_");
            this->stream_ << id << " [shape=doubleoctagon,label=\""
                          << libnode->toStr() << "\"];" << std::endl;
            if (this->last_comp_name_.empty()) this->last_comp_name_ = id;
        }
    }
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
}

void DotVisualizer::visualizeSequence(StructuredSDFG& sdfg,
                                      structured_control_flow::Sequence& sequence) {
    std::string last_comp_name_tmp, last_comp_name_cluster_tmp;
    for (size_t i = 0; i < sequence.size(); ++i) {
        std::pair<structured_control_flow::ControlFlowNode&, structured_control_flow::Transition&>
            child = sequence.at(i);
        this->visualizeNode(sdfg, child.first);
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

void DotVisualizer::visualizeIfElse(StructuredSDFG& sdfg,
                                    structured_control_flow::IfElse& if_else) {
    auto id = escapeDotId(if_else.element_id(), "if_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"if:\";"
                  << std::endl
                  << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    for (size_t i = 0; i < if_else.size(); ++i) {
        this->stream_ << "subgraph cluster_" << id << "_" << std::to_string(i)
                      << " {" << std::endl;
        this->stream_.setIndent(this->stream_.indent() + 4);
        this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\""
                      << this->expression(if_else.at(i).second->__str__()) << "\";" << std::endl;
        this->visualizeSequence(sdfg, if_else.at(i).first);
        this->stream_.setIndent(this->stream_.indent() - 4);
        this->stream_ << "}" << std::endl;
    }
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_ = "cluster_" + id;
}

void DotVisualizer::visualizeWhile(StructuredSDFG& sdfg,
                                   structured_control_flow::While& while_loop) {

    auto id = escapeDotId(while_loop.element_id(), "while_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"while:\";"
                  << std::endl
                  << id << " [shape=point,style=invis,label=\"\"];"
                  << std::endl;
    this->visualizeSequence(sdfg, while_loop.root());
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_ = "cluster_" + id;
}

void DotVisualizer::visualizeFor(StructuredSDFG& sdfg, structured_control_flow::For& loop) {
    auto id = escapeDotId(loop.element_id(), "loop_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"for: ";
    this->visualizeForBounds(loop.indvar(), loop.init(), loop.condition(), loop.update());
    this->stream_ << "\";" << std::endl
                  << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    this->visualizeSequence(sdfg, loop.root());
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_ = "cluster_" + id;
}

void DotVisualizer::visualizeReturn(StructuredSDFG& sdfg,
                                    structured_control_flow::Return& return_node) {
    auto id = escapeDotId(return_node.element_id(), "return_");
    this->stream_ << id << " [shape=cds,label=\" return  \"];" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_.clear();
}
void DotVisualizer::visualizeBreak(StructuredSDFG& sdfg,
                                   structured_control_flow::Break& break_node) {
    auto id = escapeDotId(break_node.element_id(), "break_");
    this->stream_ << id << " [shape=cds,label=\" break  \"];" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_.clear();
}

void DotVisualizer::visualizeContinue(StructuredSDFG& sdfg,
                                      structured_control_flow::Continue& continue_node) {
    auto id = escapeDotId(continue_node.element_id(), "cont_");
    this->stream_ << id << " [shape=cds,label=\" continue  \"];"
                  << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_.clear();
}

void DotVisualizer::visualizeMap(StructuredSDFG& sdfg, structured_control_flow::Map& map_node) {
    auto id = escapeDotId(map_node.element_id(), "map_");
    this->stream_ << "subgraph cluster_" << id << " {" << std::endl;
    this->stream_.setIndent(this->stream_.indent() + 4);
    this->stream_ << "style=filled;shape=box;fillcolor=white;color=black;label=\"map: ";
    this->visualizeForBounds(map_node.indvar(), map_node.init(), map_node.condition(),
                             map_node.update());
    this->stream_ << "\";" << std::endl
                  << id << " [shape=point,style=invis,label=\"\"];" << std::endl;
    this->visualizeSequence(sdfg, map_node.root());
    this->stream_.setIndent(this->stream_.indent() - 4);
    this->stream_ << "}" << std::endl;
    this->last_comp_name_ = id;
    this->last_comp_name_cluster_ = "cluster_" + id;
}

void DotVisualizer::visualize() {
    this->stream_.clear();
    this->stream_ << "digraph " << escapeDotId(this->sdfg_.name()) << " {" << std::endl;
    this->stream_.setIndent(4);
    this->stream_ << "graph [compound=true];" << std::endl;
    this->stream_ << "subgraph cluster_" << escapeDotId(this->sdfg_.name()) << " {" << std::endl;
    this->stream_.setIndent(8);
    this->stream_ << "node [style=filled,fillcolor=white];" << std::endl
                  << "style=filled;color=lightblue;label=\"\";" << std::endl;
    this->visualizeSequence(this->sdfg_, this->sdfg_.root());
    this->stream_.setIndent(4);
    this->stream_ << "}" << std::endl;
    this->stream_.setIndent(0);
    this->stream_ << "}" << std::endl;
}

}  // namespace visualizer
}  // namespace sdfg
