class ParseTree:

    def __init__(self, root):
        self.root = root
        self.children = []

    @property
    def terminal(self):
        return not self.children

    def __repr__(self):
        return ParseTree2string(self, 0)

def ParseTree2string(tree, indent):

    strings = []
    for child in tree.children:
        strings.append(ParseTree2string(child, indent + 1))

    return "| " * indent + tree.root + "\n" + ''.join(strings)

def Graphviz(tree):

    graphviz_tree = []
    layers = [[tree]]
    terminal = []
    arcs = []

    while True:

        new_layer = []

        for id, node in enumerate(layers[-1]):

            node_label = 'N' + str(len(layers) - 1) + '_' + str(id)

            for child in node.children:

                label = 'N' + str(len(layers)) + '_' + str(len(new_layer))

                arcs.append((node_label, label))

                new_layer.append(child)

        if not new_layer:
            break

        layers.append(new_layer)

    graphviz_tree.append('digraph G {')

    leaf_nodes = []

    for i, layer in enumerate(layers):

        layer_nodes = set()

        graphviz_tree.append('  subgraph depth_%d {' % i)
        # print('    node [style=filled];')
        graphviz_tree.append('    rank = same;')

        for j, node in enumerate(layer):

            node_label = 'N' + str(i) + '_' + str(j)

            if node.terminal:
                leaf_nodes.append((node, node_label))
                continue

            graphviz_tree.append('      %s [label="%s"];' % (node_label, node.root))

        graphviz_tree.append('  }')

    graphviz_tree.append('  subgraph leafs {')
    graphviz_tree.append('    node [style=filled shape="box" color="#090030" fontcolor="#ffffff"];')
    graphviz_tree.append('    rank = same;')

    for node, label in leaf_nodes:
        graphviz_tree.append('      %s [label="%s"];' % (label, node.root))
    graphviz_tree.append('  }')


    for left, right in arcs:
        graphviz_tree.append('  %s -> %s' % (left, right))

    graphviz_tree.append('}')

    return '\n'.join(graphviz_tree)
