import numbers
import os
import sys
import queue
from node import Node

check_consistent = False
check_optimistic = False
heuristic_path = ""
file_name = ""
algorithm = ""

for i in range(1, len(sys.argv)):
    # Check if there is a corresponding value for the current argument
    if i + 1 < len(sys.argv):
        arg = sys.argv[i]
        value = sys.argv[i + 1]

        # Check the argument and assign its value accordingly
        if arg == "--ss":
            file_name = value
        elif arg == "--alg":
            algorithm = str(value)
        elif arg == "--h":
            heuristic_path = value
        elif value == "--check-optimistic":
            check_optimistic = True
        elif arg == "--check-consistent":
            check_consistent = True

FILE = open(file_name)

if heuristic_path:
    h = open(heuristic_path)
    heuristic = {}
    h_lines = h.readlines()
    for line in h_lines:
        line = line.rstrip("\n")
        if "#" in line:
            continue
        heuristic[line.split(":")[0]] = line.split(":")[1]

# h = open("istra_heuristic.txt")
lines = FILE.readlines()
goal_states = []
initial_state = None
count = 0
transitions = {}

for line in lines:
    line = line.rstrip("\n")
    if "#" in line:
        continue
    if initial_state is None:
        initial_state = line
        continue
    if not goal_states:
        goal_states = line.strip().split()
        continue

    stanje = line.split(":")[0]
    transitions[stanje] = line.split(":")[1][0:].split()
for transition in transitions:
    count = 0
    for i in transitions[transition]:
        new = i.split(",")
        transitions[transition].remove(i)
        transitions[transition].insert(count, new)
        count += 1

new_transitions = {}


for key, value in transitions.items():
    temp_dict = {item[0]: item[1] for item in value}
    new_transitions[key] = temp_dict

for key, value in new_transitions.items():
    new_transitions[key] = {k: value[k] for k in sorted(value)}


def breadth_first_search(s0, goal, graph):
    n: Node
    n = Node(None, s0, 0.0)
    q = queue.Queue()
    q.put(n)
    visited = set()
    while q:
        if n.state in goal:
            visited.add(n.state)
            return n, visited, n.get_path()
        if n.state not in visited:
            new_q = q.get().expand(graph)
            while not new_q.empty():
                q.put(new_q.get())
            visited.add(n.state)
        else:
            q.get()
        n = q.queue[0]
    return None, visited, None


def uniform_cost_search(s0, goal, graph):
    n: Node
    n = Node(None, s0, 0)
    q = queue.PriorityQueue()
    q.put(n)
    visited = set()
    while q:
        if n.state in goal:
            visited.add(n.state)
            return n, visited, n.get_path()
        if n.state not in visited:
            new_q = q.get().expand(graph)
            while not new_q.empty():
                q.put(new_q.get())
            visited.add(n.state)
        else:
            q.get()
        n = q.queue[0]
    return None, visited, None


def checkoptimistic():
    print("# HEURISTIC-OPTIMISTIC ", heuristic_path)
    isoptimistic = True
    for each in heuristic:
        check = float(heuristic[each]) <= float(uniform_cost_search(each, goal_states, new_transitions)[0].cost)
        if check:
            print(f"[CONDITION]: [OK] h({each}) <= h*: {float(heuristic[each])} <= {float(uniform_cost_search(each, goal_states, new_transitions)[0].cost)}")
        else:
            print(f"[CONDITION]: [ERR] h({each}) <= h*: {float(heuristic[each])} <= {float(uniform_cost_search(each, goal_states, new_transitions)[0].cost)}")
            isoptimistic = False
    if isoptimistic:
        print("[CONCLUSION]: Heuristic is optimistic.")
    else:
        print("[CONCLUSION]: Heuristic is not optimistic.")


def checkconsistent():
    print("# HEURISTIC-CONSISTENT ", heuristic_path)
    isconsistent = True
    # heuristic for current node + cost until this node <= the cost + heuristic for next node
    for each1 in heuristic:
        for each2 in new_transitions[each1]:
            f1 = float(heuristic[each1])
            f2 = float(new_transitions[each1][each2]) + float(heuristic[each2])
            check = f1 <= f2
            if check:
                print(f"[CONDITION]: [OK] h({each1}) <= h({each2}) + c: {f1} <= {float(heuristic[each2])} + {float(new_transitions[each1][each2])}")
            else:
                isconsistent = False
                print(f"[CONDITION]: [ERR] h({each1}) <= h({each2}) + c: {f1} <= {float(heuristic[each2])} + {float(new_transitions[each1][each2])}")
    if isconsistent:
        print("[CONCLUSION]: Heuristic is consistent.")
    else:
        print("[CONCLUSION]: Heuristic is not consistent.")


def astar_search(s0, goal, graph):
    n: Node
    n = Node(None, s0, 0, heuristic[s0])
    q = queue.PriorityQueue()
    q.put(n)
    visited = set()
    while q:
        if n.state in goal:
            visited.add(n.state)
            return n, visited, n.get_path()
        if n.state not in visited:
            new_q = q.get().expand(graph)
            while not new_q.empty():
                temp = new_q.get()
                temp.heuristic = heuristic[temp.state]
                q.put(temp)
            visited.add(n.state)
        else:
            q.get()
        n = q.queue[0]
    return None, visited, None


if algorithm and not check_optimistic and not check_consistent:
    if algorithm == "bfs":
        n = breadth_first_search(initial_state, goal_states, new_transitions)
        print("# BFS ")
        print("[FOUND_SOLUTION]: yes")
        print(f"[STATES_VISITED]: {len(n[1])}")
        print(f"[PATH_LENGTH]: {len(n[2])}")
        print(f"[TOTAL_COST]: {float(n[0].cost)}")
        print(f"[PATH]: {n[2][::-1]}")
    elif algorithm == "ucs":
        n = uniform_cost_search(initial_state, goal_states, new_transitions)
        print("# UCS ")
        print("[FOUND_SOLUTION]: yes")
        print(f"[STATES_VISITED]: {len(n[1])}")
        print(f"[PATH_LENGTH]: {len(n[2])}")
        print(f"[TOTAL_COST]: {float(n[0].cost)}")
        print(f"[PATH]: {n[2][::-1]}")
    elif algorithm == "astar":
        n = astar_search(initial_state, goal_states, new_transitions)
        print("# A-STAR ")
        print("[FOUND_SOLUTION]: yes")
        print(f"[STATES_VISITED]: {len(n[1])}")
        print(f"[PATH_LENGTH]: {len(n[2])}")
        print(f"[TOTAL_COST]: {float(n[0].cost)}")
        print(f"[PATH]: {n[2][::-1]}")
else:
    if check_optimistic:
        checkoptimistic()
    else:
        checkconsistent()





