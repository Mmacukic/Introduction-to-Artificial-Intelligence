import queue


class Node:
    def __init__(self, parent, state, cost, heuristic=0.0):
        self.parent = parent
        self. state = state
        self.cost = cost
        self.heuristic = heuristic

    def expand(self, graph):
        q = queue.Queue()

        for l in graph[self.state]:
            q.put(Node(self, l, float(graph[self.state][l]) + float(self.cost)))
        return q

    def get_path(self):
        path = []
        while self.parent:
            path.append(self.state)
            self = self.parent

        path.append(self.state)
        return path

    def funcc(self):
        sum = float(self.heuristic) + float(self.cost)
        return sum

    def __lt__(self, other):
        return self.funcc() < other.funcc()

    def __eq__(self, other):
        return self.funcc() == other.funcc()
