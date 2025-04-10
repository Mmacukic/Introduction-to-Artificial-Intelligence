import sys

cooking = False
resolution = False
File = []
File1 = []
File2 = []
count = 0
refutation = set()
sos = set()
goal = ''


class Unit:
    def __init__(self, name, parents, negated):
        self.negated = negated
        self.name = name
        self.parents = parents

    def __eq__(self, other):
        return self.name == other.name and self.negated == other.negated

    def __hash__(self):
        return hash(self.name)


def printcurrent():
    for each in refutation:
        for a in each:
            print(a.name, " ", a.negated)
        print()
    print("------------")
    for each in sos:
        for a in each:
            print(a.name, " ", a.negated)
        print()
    print("XXXXXXXXX")


def remove_tautology():
    global refutation
    temprefutation = set(refutation)
    for claus in refutation:
        if len(claus) == 1:
            continue
        for each1 in claus:
            for each2 in claus:
                if each1.name == each2.name and each1.negated and not each2.negated:
                    remove = {each2, each1}
                    new_frozen_set = frozenset(element for element in claus if element not in remove)
                    temprefutation.remove(claus)
                    temprefutation.add(new_frozen_set)
    refutation = temprefutation


def remove_redundant():
    global refutation
    temp = set(refutation)
    for claus1 in refutation:
        for claus2 in refutation:
            if claus1.__eq__(claus2):
                continue
            if claus1.issubset(claus2) and claus2 in temp:
                temp.remove(claus2)
    refutation = temp


def resolve(claus1, claus2):
    new_clauses = set()
    for each1 in claus1:
        for each2 in claus2:
            if each1.name == each2.name and ((not each1.negated and each2.negated)
                                             or (each1.negated and not each2.negated)):
                remove = {each2}
                new_frozen_set = frozenset(element for element in claus2 if element not in remove)
                if len(new_frozen_set) == 0:
                    new_clauses.add("NIL")
                else:
                    new_clauses.add(new_frozen_set)
    return new_clauses


def refutation_resolution():
    remove_tautology()
    remove_redundant()
    global sos
    checked = set()
    while True:
        resolution_occurred = False
        for each1 in sos:
            for each2 in refutation:
                temp = resolve(each1, each2)
                if "NIL" in temp:
                    print("NIL")
                    return True
                if temp:
                    sos = sos.union(temp)
                    resolution_occurred = True
                    printcurrent()
        if not resolution_occurred:
            break
    return False


for i in sys.argv:
    if i == "resolution":
        resolution = True
    elif i == "cooking":
        cooking = True
    elif ".txt" in i and resolution:
        File = open(f".venv/files/{i}")
    elif cooking and count == 2:
        File1 = open(f".venv/files/{i}")
    elif cooking:
        File2 = open(f".venv/files/{i}")
    count += 1
isfirsti = True
if cooking:
    recipe = File2.readlines()
    ingredients = File1.readlines()
if resolution:
    lines = File.readlines()
    for line in lines:
        temp = set()
        line = line.lower().strip()
        if "#" in line:
            continue
        if line == lines[-1].strip().lower() and not isfirsti:
            goal = line
            for unit in line.split():
                if "v" in unit:
                    continue
                if "~" in unit:
                    sos.add(frozenset({Unit(unit.strip("~"), None, False)}))
                else:
                    sos.add(frozenset({Unit(unit.strip("~"), None, True)}))
            continue
        for unit in line.split():
            if "v" in unit:
                continue
            if "~" in unit:
                temp.add(Unit(unit.strip("~"), None, True))
                isfirsti = False
            else:
                temp.add(Unit(unit, None, False))
                isfirsti = False

        refutation.add(frozenset(temp))
        temp.clear()

    if refutation_resolution():
        print("[CONCLUSION]: ", goal, " is true")
    else:
        print("[CONCLUSION]: ", goal, " is unknown")