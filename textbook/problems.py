from collections import defaultdict

from .csp import CSP

# ______________________________________________________________________________
# Map Coloring CSP Problems


class UniversalDict:
    """A universal dict maps any key to the same value. We use it here
    as the domains dict for CSPs in which all variables have the same domain.
    >>> d = UniversalDict(42)
    >>> d['life']
    42
    """

    def __init__(self, value): self.value = value

    def __getitem__(self, key): return self.value

    def __repr__(self): return '{{Any: {0!r}}}'.format(self.value)


def different_values_constraint(A, a, B, b):
    """A constraint saying two neighboring variables must differ in value."""
    return a != b


def MapColoringCSP(colors, neighbors):
    """Make a CSP for the problem of coloring a map with different colors
    for any two adjacent regions. Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries. This dict may also be
    specified as a string of the form defined by parse_neighbors."""
    if isinstance(neighbors, str):
        neighbors = parse_neighbors(neighbors)
    return CSP(list(neighbors.keys()), UniversalDict(colors), neighbors, different_values_constraint)


def parse_neighbors(neighbors):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors. The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name. If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    True
    """
    dic = defaultdict(list)
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (A, Aneighbors) in specs:
        A = A.strip()
        for B in Aneighbors.split():
            dic[A].append(B)
            dic[B].append(A)
    return dic


def australia_csp():
    return MapColoringCSP(
        list("RGB"), """SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: """
    )

def usa_csp():
    return MapColoringCSP(
        list("RGBY"),
        """WA: OR ID; OR: ID NV CA; CA: NV AZ; NV: ID UT AZ; ID: MT WY UT;
                         UT: WY CO AZ; MT: ND SD WY; WY: SD NE CO; CO: NE KA OK NM; NM: OK TX AZ;
                         ND: MN SD; SD: MN IA NE; NE: IA MO KA; KA: MO OK; OK: MO AR TX;
                         TX: AR LA; MN: WI IA; IA: WI IL MO; MO: IL KY TN AR; AR: MS TN LA;
                         LA: MS; WI: MI IL; IL: IN KY; IN: OH KY; MS: TN AL; AL: TN GA FL;
                         MI: OH IN; OH: PA WV KY; KY: WV VA TN; TN: VA NC GA; GA: NC SC FL;
                         PA: NY NJ DE MD WV; WV: MD VA; VA: MD DC NC; NC: SC; NY: VT MA CT NJ;
                         NJ: DE; DE: MD; MD: DC; VT: NH MA; MA: NH RI CT; CT: RI; ME: NH;
                         HI: ; AK: """,
    )

def france_csp():
    return MapColoringCSP(
        list("RGBY"),
        """AL: LO FC; AQ: MP LI PC; AU: LI CE BO RA LR MP; BO: CE IF CA FC RA
                            AU; BR: NB PL; CA: IF PI LO FC BO; CE: PL NB NH IF BO AU LI PC; FC: BO
                            CA LO AL RA; IF: NH PI CA BO CE; LI: PC CE AU MP AQ; LO: CA AL FC; LR:
                            MP AU RA PA; MP: AQ LI AU LR; NB: NH CE PL BR; NH: PI IF CE NB; NO:
                            PI; PA: LR RA; PC: PL CE LI AQ; PI: NH NO CA IF; PL: BR NB CE PC; RA:
                            AU BO FC PA LR""",
    )


# ______________________________________________________________________________
# n-Queens Problem


def queen_constraint(A, a, B, b):
    """Constraint is satisfied (true) if A, B are really the same variable,
    or if they are not in the same row, down diagonal, or up diagonal."""
    return A == B or (a != b and A + a != B + b and A - a != B - b)


class NQueensCSP(CSP):
    """
    Make a CSP for the nQueens problem for search with min_conflicts.
    Suitable for large n, it uses only data structures of size O(n).
    Think of placing queens one per column, from left to right.
    That means position (x, y) represents (var, val) in the CSP.
    The main structures are three arrays to count queens that could conflict:
        rows[i]      Number of queens in the ith row (i.e. val == i)
        downs[i]     Number of queens in the \ diagonal
                     such that their (x, y) coordinates sum to i
        ups[i]       Number of queens in the / diagonal
                     such that their (x, y) coordinates have x-y+n-1 = i
    We increment/decrement these counts each time a queen is placed/moved from
    a row/diagonal. So moving is O(1), as is nconflicts.  But choosing
    a variable, and a best value for the variable, are each O(n).
    If you want, you can keep track of conflicted variables, then variable
    selection will also be O(1).
    >>> len(backtracking_search(NQueensCSP(8)))
    8
    """

    def __init__(self, n):
        """Initialize data structures for n Queens."""
        CSP.__init__(self, list(range(n)), UniversalDict(list(range(n))),
                     UniversalDict(list(range(n))), queen_constraint)

        self.rows = [0] * n
        self.ups = [0] * (2 * n - 1)
        self.downs = [0] * (2 * n - 1)

    def nconflicts(self, var, val, assignment):
        """The number of conflicts, as recorded with each assignment.
        Count conflicts in row and in up, down diagonals. If there
        is a queen there, it can't conflict with itself, so subtract 3."""
        n = len(self.variables)
        c = self.rows[val] + self.downs[var + val] + self.ups[var - val + n - 1]
        if assignment.get(var, None) == val:
            c -= 3
        return c

    def assign(self, var, val, assignment):
        """Assign var, and keep track of conflicts."""
        old_val = assignment.get(var, None)
        if val != old_val:
            if old_val is not None:  # Remove old val if there was one
                self.record_conflict(assignment, var, old_val, -1)
            self.record_conflict(assignment, var, val, +1)
            CSP.assign(self, var, val, assignment)

    def unassign(self, var, assignment):
        """Remove var from assignment (if it is there) and track conflicts."""
        if var in assignment:
            self.record_conflict(assignment, var, assignment[var], -1)
        CSP.unassign(self, var, assignment)

    def record_conflict(self, assignment, var, val, delta):
        """Record conflicts caused by addition or deletion of a Queen."""
        n = len(self.variables)
        self.rows[val] += delta
        self.downs[var + val] += delta
        self.ups[var - val + n - 1] += delta

    def display(self, assignment):
        """Print the queens and the nconflicts values (for debugging)."""
        n = len(self.variables)
        for val in range(n):
            for var in range(n):
                if assignment.get(var, '') == val:
                    ch = 'Q'
                elif (var + val) % 2 == 0:
                    ch = '.'
                else:
                    ch = '-'
                print(ch, end=' ')
            print('    ', end=' ')
            for var in range(n):
                if assignment.get(var, '') == val:
                    ch = '*'
                else:
                    ch = ' '
                print(str(self.nconflicts(var, val, assignment)) + ch, end=' ')
            print()

def Zebra():
    """Return an instance of the Zebra Puzzle."""
    Colors = 'Red Yellow Blue Green Ivory'.split()
    Pets = 'Dog Fox Snails Horse Zebra'.split()
    Drinks = 'OJ Tea Coffee Milk Water'.split()
    Countries = 'Englishman Spaniard Norwegian Ukranian Japanese'.split()
    Smokes = 'Kools Chesterfields Winston LuckyStrike Parliaments'.split()
    variables = Colors + Pets + Drinks + Countries + Smokes
    domains = {}
    for var in variables:
        domains[var] = list(range(1, 6))
    domains['Norwegian'] = [1]
    domains['Milk'] = [3]
    neighbors = parse_neighbors("""Englishman: Red;
                Spaniard: Dog; Kools: Yellow; Chesterfields: Fox;
                Norwegian: Blue; Winston: Snails; LuckyStrike: OJ;
                Ukranian: Tea; Japanese: Parliaments; Kools: Horse;
                Coffee: Green; Green: Ivory""")
    for type in [Colors, Pets, Drinks, Countries, Smokes]:
        for A in type:
            for B in type:
                if A != B:
                    if B not in neighbors[A]:
                        neighbors[A].append(B)
                    if A not in neighbors[B]:
                        neighbors[B].append(A)

    def zebra_constraint(A, a, B, b, recurse=0):
        same = (a == b)
        next_to = abs(a - b) == 1
        if A == 'Englishman' and B == 'Red':
            return same
        if A == 'Spaniard' and B == 'Dog':
            return same
        if A == 'Chesterfields' and B == 'Fox':
            return next_to
        if A == 'Norwegian' and B == 'Blue':
            return next_to
        if A == 'Kools' and B == 'Yellow':
            return same
        if A == 'Winston' and B == 'Snails':
            return same
        if A == 'LuckyStrike' and B == 'OJ':
            return same
        if A == 'Ukranian' and B == 'Tea':
            return same
        if A == 'Japanese' and B == 'Parliaments':
            return same
        if A == 'Kools' and B == 'Horse':
            return next_to
        if A == 'Coffee' and B == 'Green':
            return same
        if A == 'Green' and B == 'Ivory':
            return a - 1 == b
        if recurse == 0:
            return zebra_constraint(B, b, A, a, 1)
        if ((A in Colors and B in Colors) or
                (A in Pets and B in Pets) or
                (A in Drinks and B in Drinks) or
                (A in Countries and B in Countries) or
                (A in Smokes and B in Smokes)):
            return not same
        raise Exception('error')

    return CSP(variables, domains, neighbors, zebra_constraint)
