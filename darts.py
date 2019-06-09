import torch
from torch import nn
from ops import *
from utils import *
from operator import *
import networkx as nx
import graphviz as gv


class DARTS(nn.Module):
    """Differentiable architecture search module.

    Based on the following papers.
    1. [DARTS: Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055.pdf)
    2. ...

    """

    def __init__(self, operations, num_nodes, num_input_nodes, num_cells, reduction_cells,
                 discrete_mode, num_top_operations, num_channels, num_classes):
        """Build DARTS with the given operations.

        Args:
            operations (dict): Dict with name as keys and nn.Module initializer
                that takes in_channels, out_channels, stride as arguments as values.
            num_nodes (int): Number of nodes in each cell.
            num_input_nodes (int): Number of input nodes in each cell.
            num_cells (int): Number of cells in the network.
            reduction_cells (list): List of cell index that performs spatial reduction.
            discrete_mode (bool): Whether use discrete architecture.
            num_top_operations (int): Number of top strongest operations retained in discrete architecture.
            num_channels (int): Number of channels of the first cell.
            num_classes (int): Number of classes for classification.

        """
        super().__init__()

        self.operations = operations
        self.num_nodes = num_nodes
        self.num_input_nodes = num_input_nodes
        self.num_cells = num_cells
        self.reduction_cells = reduction_cells
        self.discrete_mode = discrete_mode
        self.num_top_operations = num_top_operations
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.dag = Dict()
        self.dag.normal = nx.DiGraph()
        self.dag.reduction = nx.DiGraph()
        self.architecture = nn.ParameterDict()
        self.architecture.normal = nn.ParameterDict()
        self.architecture.reduction = nn.ParameterDict()
        self.network = nn.ModuleDict()

        if self.discrete_mode:
            self.build_dag()
            self.build_architecture()
            self.build_discrete_dag()
            self.build_discrete_network()
        else:
            self.build_dag()
            self.build_architecture()
            self.build_network()

    def build_dag(self, reduction=None):
        """Build Directed Acyclic Graph that represents each cell.
        """
        if reduction is None:
            self.build_dag(reduction=True)
            self.build_dag(reduction=False)

        dag = self.dag.reduction if reduction else self.dag.normal
        architecture = self.architecture.reduction if reduction else self.architecture.normal

        for m in range(self.num_nodes):
            for n in range(self.num_input_nodes, self.num_nodes):
                if m < n:
                    dag.add_edge(m, n)

    def build_discrete_dag(self, reduction=None):
        """Build Directed Acyclic Graph that represents each cell.
        """
        if reduction is None:
            self.build_discrete_dag(reduction=True)
            self.build_discrete_dag(reduction=False)

        dag = self.dag.reduction if reduction else self.dag.normal
        architecture = self.architecture.reduction if reduction else self.architecture.normal

        for n in dag.nodes():
            operations = sorted(((*max(zip(
                nn.functional.softmax(architecture[str((m, n))], dim=0),
                self.operations.keys()
            ), key=itemgetter(0)), m) for m in dag.predecessors(n)), key=itemgetter(0))
            for weight, operation, m in operations[:-self.num_top_operations]:
                dag.remove_edge(m, n)

    def build_architecture(self, reduction=None):
        """Build parameters that represent the cell architectures (normal and reduction).
        """
        if reduction is None:
            self.build_architecture(reduction=True)
            self.build_architecture(reduction=False)

        dag = self.dag.reduction if reduction else self.dag.normal
        architecture = self.architecture.reduction if reduction else self.architecture.normal

        architecture.update({str((m, n)): nn.Parameter(torch.zeros(len(self.operations))) for m, n in dag.edges()})

    def build_network(self):
        """Build modules that represent the whole network.
        """
        # NOTE: Why multiplier is 3?
        num_channels = self.num_channels
        out_channels = num_channels * 3

        self.network.conv = Conv2d(
            in_channels=3,
            out_channels=out_channels,
            stride=1,
            kernel_size=3,
            padding=1,
            affine=True,
            preactivation=False
        )

        out_channels = [out_channels] * self.num_input_nodes
        self.network.cells = nn.ModuleList()

        for i in range(self.num_cells):

            reduction = False
            if i in self.reduction_cells:
                reduction = True
                num_channels <<= 1

            dag = self.dag.reduction if reduction else self.dag.normal
            architecture = self.architecture.reduction if reduction else self.architecture.normal

            cell = nn.ModuleDict({
                # NOTE: Should be factorized reduce?
                **{
                    str((n - self.num_input_nodes, n)): Conv2d(
                        in_channels=out_channels[n - self.num_input_nodes],
                        out_channels=num_channels,
                        stride=1 << len([j for j in self.reduction_cells if k < j < i]),
                        kernel_size=1,
                        padding=0,
                        affine=False
                    ) for n, k in zip(range(self.num_input_nodes), range(i - self.num_input_nodes, i))
                },
                **{
                    str((m, n)): nn.ModuleList([
                        operation(
                            in_channels=num_channels,
                            out_channels=num_channels,
                            stride=2 if reduction and m in range(self.num_input_nodes) else 1
                        ) for operation in self.operations.values()
                    ]) for m, n in dag.edges()
                }
            })

            out_channels.append(num_channels * (self.num_nodes - self.num_input_nodes))
            self.network.cells.append(cell)

        self.network.global_avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.network.linear = nn.Linear(out_channels[-1], self.num_classes, bias=True)

    def build_discrete_network(self):
        """Build modules that represent the whole network.
        """
        # NOTE: Why multiplier is 3?
        num_channels = self.num_channels
        out_channels = num_channels * 3

        self.network.conv = Conv2d(
            in_channels=3,
            out_channels=out_channels,
            stride=1,
            kernel_size=3,
            padding=1,
            affine=True,
            preactivation=False
        )

        out_channels = [out_channels] * self.num_input_nodes
        self.network.cells = nn.ModuleList()

        for i in range(self.num_cells):

            reduction = False
            if i in self.reduction_cells:
                reduction = True
                num_channels <<= 1

            dag = self.dag.reduction if reduction else self.dag.normal
            architecture = self.architecture.reduction if reduction else self.architecture.normal

            cell = nn.ModuleDict({
                # NOTE: Should be factorized reduce?
                **{
                    str((n - self.num_input_nodes, n)): Conv2d(
                        in_channels=out_channels[n - self.num_input_nodes],
                        out_channels=num_channels,
                        stride=1 << len([j for j in self.reduction_cells if k < j < i]),
                        kernel_size=1,
                        padding=0,
                        affine=False
                    ) for n, k in zip(range(self.num_input_nodes), range(i - self.num_input_nodes, i))
                },
                **{
                    str((m, n)): max(((weight, operation) for weight, (name, operation) in zip(
                        nn.functional.softmax(architecture[str((m, n))], dim=0),
                        self.operations.items()
                    ) if 'zero' not in name), key=itemgetter(0))[1](
                        in_channels=num_channels,
                        out_channels=num_channels,
                        stride=2 if reduction and m in range(self.num_input_nodes) else 1
                    ) for m, n in dag.edges()
                }
            })

            out_channels.append(num_channels * (self.num_nodes - self.num_input_nodes))
            self.network.cells.append(cell)

        self.network.global_avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.network.linear = nn.Linear(out_channels[-1], self.num_classes, bias=True)

    def forward_cell(self, cell, reduction, cell_outputs, node_outputs, n):
        """forward in the given cell.

        Args:
            cell (dict): A dict with edges as keys and operations as values.
            reduction (bool): Whether the cell performs spatial reduction.
            node_outputs (dict): A dict with node as keys and its outputs as values.
                This is to avoid duplicate calculation in recursion.
            n (int): The output node in the cell.

        """
        dag = self.dag.reduction if reduction else self.dag.normal
        architecture = self.architecture.reduction if reduction else self.architecture.normal

        if n not in node_outputs:
            if n in range(self.num_input_nodes):
                node_outputs[n] = cell[str((n - self.num_input_nodes, n))](cell_outputs[n - self.num_input_nodes])
            else:
                if self.discrete_mode:
                    node_outputs[n] = sum(
                        cell[str((m, n))](self.forward_cell(cell, reduction, cell_outputs, node_outputs, m))
                        for m in dag.predecessors(n)
                    )
                else:
                    node_outputs[n] = sum(sum(
                        weight * operation(self.forward_cell(cell, reduction, cell_outputs, node_outputs, m))
                        for weight, operation in zip(nn.functional.softmax(architecture[str((m, n))], dim=0), cell[str((m, n))])
                    ) for m in dag.predecessors(n))
        return node_outputs[n]

    def forward(self, input):
        output = self.network.conv(input)
        cell_outputs = [output] * self.num_input_nodes
        for i, cell in enumerate(self.network.cells):
            node_outputs = {}
            cell_outputs.append(torch.cat([
                self.forward_cell(cell, i in self.reduction_cells, cell_outputs, node_outputs, n)
                for n in range(self.num_input_nodes, self.num_nodes)
            ], dim=1))
        output = cell_outputs[-1]
        output = self.network.global_avg_pool2d(output).squeeze()
        output = self.network.linear(output)
        return output

    def render_discrete_architecture(self, reduction, name, directory):
        """Render the given architecture.

        Args:
            architecture (dict): A dict with edges as keys and parameters as values.
            name (str): Name of the given architecture for saving.
            directory (str): Directory for saving.

        """
        dag = self.dag.reduction if reduction else self.dag.normal
        architecture = self.architecture.reduction if reduction else self.architecture.normal

        discrete_dag = gv.Digraph(name)
        for n in dag.nodes():
            operations = sorted(((*max(zip(
                nn.functional.softmax(architecture[str((m, n))], dim=0),
                self.operations.keys()
            ), key=itemgetter(0)), m) for m in dag.predecessors(n)), key=itemgetter(0))
            for weight, operation, m in operations[:-self.num_top_operations]:
                discrete_dag.edge(str(m), str(n), label='', color='white')
            for weight, operation, m in operations[-self.num_top_operations:]:
                discrete_dag.edge(str(m), str(n), label=operation, color='black')
        return discrete_dag.render(directory=directory, format='png')
