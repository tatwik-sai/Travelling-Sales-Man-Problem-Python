class Tree:
    """A Data Structure for implementation of tree."""

    def __init__(self, root_nodes=None, auto_correct=False):
        """
        :param root_nodes: The list of root nodes that are on the top level.
        :param auto_correct: if True: Will not raise 'Multiple parents pointing to the same node.' Exception.
        """
        if root_nodes is None:
            root_nodes = []
        self.auto_correct = auto_correct
        self.tree = {'nil': (None, root_nodes)}
        for node in root_nodes:
            self.tree[str(node)] = ('nil', [])

    def add_node(self, node: tuple):
        """
        Adds the node to the tree.
        :param node: Tuple with parent and child.
        :exception 1)if already the child exists:- Multiple parents pointing to the same node.
                   2)if the parent do not exists:- No such parent exists.
        """
        try:
            _ = self.tree[str(node[0])]
        except KeyError:
            raise Exception('No such parent exists.')
        try:
            _ = self.tree[str(node[1])]
            if not self.auto_correct:
                raise Exception('Multiple parents pointing to the same node.')
            else:
                pass
        except KeyError:
            parent, child = str(node[0]), str(node[1])
            self.tree[child] = (node[0], [])
            self.tree[parent][1].append(node[1])

    def add_children(self, node, children: list):
        """
        Adds multiple children to the node.
        :param node: The name of the node.
        :param children: List of children of the node.
        """
        for child in children:
            self.add_node((node, child))

    def get_parent(self, node):
        """
        Returns the parent of the given node.
        :param node: The node name.
        :exception Raises exception if node do not exists:- No such node exists.
        :return: the parent of the node.
        """
        try:
            return self.tree[str(node)][0]
        except KeyError:
            raise Exception(f"No such node with name '{node}' exists in the tree.")

    def get_children(self, node) -> list:
        """
        Returns the list of children of the given node.
        :param node: The name of the node.
        :return: the list of children for the node.
        """
        try:
            return self.tree[str(node)][1]
        except KeyError:
            raise Exception(f"No such node with name '{node}' exists in the tree.")

    def get_path(self, node) -> list:
        """
        Returns The path from root node to the given node.
        :param node: The name of the node.
        :return: The list of path from root node to the given node.
        """
        path = [node]
        parent = self.get_parent(node)
        while parent != 'nil':
            path.append(parent)
            parent = self.get_parent(parent)
        path.reverse()
        return path

    def delete(self, node):
        """
        :param node: The name of the node.
        """
        children = self.get_children(node)
        self.tree[str(self.get_parent(node))][1].remove(node)
        self.tree.pop(str(node))
        while len(children) != 0:
            new_children = []
            for child in children:
                new_children.extend(self.get_children(child))
                self.tree.pop(str(child))
            children = new_children

    def get_depth(self, node) -> int:
        """
        Returns the depth at where the node is located in the tree.
        :param node: The name of the node.
        :return: The depth at which the node is located in the tree.
        """
        return len(self.get_path(node))

    def reset(self, root_nodes):
        """
        Resets the complete tree with given nodes root nodes.
        :param root_nodes: The root nodes of the tree.
        """
        self.tree = {'nil': (None, root_nodes)}
        for node in root_nodes:
            self.tree[str(node)] = ('nil', [])


# Example
if __name__ == '__main__':
    a, b, c, d, e, f, g, h, i, j = 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'
    tree = Tree()
    tree.add_children('nil', [a])
    tree.add_children(a, [b, c, d])
    tree.add_children(b, [e, f])
    tree.add_children(d, [g, h])
    tree.add_children(h, [i, j])

    print(tree.tree)
    print(tree.get_parent(c))
    print(tree.get_children(e))
    print(tree.get_path(f))
    tree.delete(b)
    tree.delete(c)
    print(tree.tree)
