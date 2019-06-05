import torch
from torch import nn
from ops import *
from operator import *
import networkx as nx
import graphviz as gv
import matplotlib.pyplot as plt


class DARTS(nn.Module):
    """Differentiable architecture search module.

    Based on the following papers.
    1. [DARTS: Differentiable Architecture Search](https://arxiv.org/pdf/1806.09055.pdf)
    2. ...

    """

    def __init__(self, operations, num_nodes, num_input_nodes, num_cells, reduction_cells, num_channels, num_classes):
        """Build DARTS with the given operations.

        Args:
            operations (list): List of nn.Module initializer that takes in_channels, out_channels, stride as arguments.
            num_nodes (int): Number of nodes in each cell.
            num_input_nodes (int): Number of input nodes in each cell.
            num_cells (int): Number of cells in the network.
            reduction_cells (list): List of cell index that performs spatial reduction.
            num_channels (int): Number of channels of the first cell.
            num_classes (int): Number of classes for classification.

        """
        super().__init__()

        self.operations = operations
        self.num_nodes = num_nodes
        self.num_input_nodes = num_input_nodes
        self.num_cells = num_cells
        self.reduction_cells = reduction_cells
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.build_dag()
        self.build_architecture()
        self.build_network()

    def build_dag(self):
        """Build Directed Acyclic Graph that represents each cell.
        """
        self.dag = nx.DiGraph()
        for parent in range(self.num_nodes):
            for child in range(self.num_input_nodes, self.num_nodes):
                if parent < child:
                    self.dag.add_edge(parent, child)

    def build_architecture(self):
        """Build parameters that represent the cell architectures (normal and reduction).
        """
        self.architecture = nn.ParameterDict()
        self.architecture.normal = nn.ParameterDict({
            str((parent, child)): nn.Parameter(torch.zeros(len(self.operations)))
            for parent, child in self.dag.edges()
        })
        self.architecture.reduction = nn.ParameterDict({
            str((parent, child)): nn.Parameter(torch.zeros(len(self.operations)))
            for parent, child in self.dag.edges()
        })

    def build_network(self):
        """Build modules that represent the whole network.
        """
        self.network = nn.ModuleDict()

        # NOTE: Why multiplier is 3?
        multiplier = 3

        self.network.conv = Conv2d(
            in_channels=3,
            out_channels=self.num_channels * multiplier,
            stride=1,
            kernel_size=3,
            padding=1,
            affine=True,
            preactivation=False
        )

        self.network.cells = nn.ModuleList()

        out_channels = [self.num_channels * multiplier] * self.num_input_nodes
        num_channels = self.num_channels

        for cell in range(self.num_cells):

            reduction = False
            if cell in self.reduction_cells:
                reduction = True
                num_channels <<= 1

            self.network.cells.append(nn.ModuleDict({
                **{
                    str((parent, child)): nn.ModuleList([
                        operation(
                            in_channels=num_channels,
                            out_channels=num_channels,
                            stride=2 if reduction and parent in [0, 1] else 1
                        ) for operation in self.operations
                    ]) for parent, child in self.dag.edges()
                },
                # NOTE: Should be factorized reduce?
                **{
                    str((input_node - self.num_input_nodes, input_node)): Conv2d(
                        in_channels=out_channels[input_node - self.num_input_nodes],
                        out_channels=num_channels,
                        stride=1 << len([
                            reduction_cell for reduction_cell in self.reduction_cells
                            if (cell - input_node) < reduction_cell < cell
                        ]),
                        kernel_size=1,
                        padding=0,
                        affine=False
                    ) for input_node in range(self.num_input_nodes)
                }
            }))

            out_channels.append(num_channels * (self.num_nodes - self.num_input_nodes))

        self.network.global_avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.network.linear = nn.Linear(out_channels[-1], self.num_classes, bias=True)

    def forward_cell(self, cell, reduction, child, node_outputs):
        """forward in the given cell.

        Args:
            cell (dict): A dict with edges as keys and operations as values.
            reduction (bool): Whether the cell performs spatial reduction.
            child (int): The output node in the cell.
            node_outputs (dict): A dict with node as keys and its outputs as values.
                This is to avoid duplicate calculation in recursion.

        """
        architecture = self.architecture.reduction if reduction else self.architecture.normal
        if self.dag.predecessors(child):
            if child not in node_outputs:
                node_outputs[child] = sum(sum(
                    operation(self.forward_cell(cell, reduction, parent, node_outputs)) * weight
                    for operation, weight in zip(cell[str((parent, child))], nn.functional.softmax(architecture[str((parent, child))], dim=0))
                ) for parent in self.dag.predecessors(child))
        return node_outputs[child]

    def forward(self, input):
        output = self.network.conv(input)
        cell_outputs = [output] * self.num_input_nodes
        for i, cell in enumerate(self.network.cells):
            node_outputs = {
                input_node: cell[str((input_node - self.num_input_nodes, input_node))](cell_outputs[input_node - self.num_input_nodes])
                for input_node in range(self.num_input_nodes)
            }
            self.forward_cell(cell, i in self.reduction_cells, self.num_nodes - 1, node_outputs)
            _, node_outputs = zip(*sorted(node_outputs.items()))
            cell_outputs.append(torch.cat(node_outputs[self.num_input_nodes:], dim=1))
        output = cell_outputs[-1]
        output = self.network.global_avg_pool2d(output).squeeze()
        output = self.network.linear(output)
        return output

    def draw_architecture(self, archirecture, num_operations, name, directory):
        """Render the given architecture.

        Args: 
            architecture (dict): A dict with edges as keys and parameters as values.
            num_operations (int): Retain the top-k strongest operations from distinct nodes.
            name (str): Name of the given architecture for saving.

        """
        dag = gv.Digraph(name)
        for child in self.dag.nodes():
            edges = []
            for parent in self.dag.predecessors(child):
                operations = map(str, self.operations)
                weights = nn.functional.softmax(archirecture[str((parent, child))], dim=0)
                weight, operation = max((weight, operation) for weight, operation in zip(weights, operations) if 'zero' not in operation)
                edges.append((map(str, (parent, child)), (weight, operation)))
            if edges:
                edges = sorted(edges, key=itemgetter(1))[-num_operations:]
                for (parent, child), (weight, operation) in edges:
                    dag.edge(parent, child, label=operation)
        return dag.render(directory=directory, format='png')

    def draw_normal_architecture(self, num_operations, name, directory):
        return self.draw_architecture(self.architecture.normal, num_operations, name, directory)

    def draw_reduction_architecture(self, num_operations, name, directory):
        return self.draw_architecture(self.architecture.reduction, num_operations, name, directory)
