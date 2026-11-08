#!/usr/bin/env python3
"""5022CMD Advanced Algorithms - Question 2: Directed graph and course prerequisites.

Layers (each with one responsibility)
-------------------------------------
1. DirectedGraph[V]   - generic, unweighted, directed graph (adjacency lists).
                        Knows nothing about courses.
2. Course, CourseType - domain entity and its validation / visibility rules.
3. CourseCatalogue    - service: owns the graph (vertices = course codes) and the
                        code -> Course lookup; enforces business rules.
4. CourseCLI          - menu interface (input/output only).

Edge direction (important)
--------------------------
    prerequisite Y  --->  dependent course X          (edge Y -> X)
* incoming neighbours of X = the prerequisites of X
* outgoing neighbours of Y = the courses unlocked by Y

Run:
    python question2_graph.py          # interactive menu
    python question2_graph.py --demo   # deterministic demonstration (no input)

Complexity (V vertices, E edges; adjacency lists for BOTH directions)
---------------------------------------------------------------------
    addVertex                    O(1) average
    addEdge / removeEdge         O(deg) - duplicate check / removal scans one list
    listIncoming/OutgoingAdjacentVertex  O(deg) to copy the neighbour list
    containsVertex               O(1) average
    hasPath / isAcyclic          O(V + E)
    Space                        O(V + E) (each edge is stored twice)
Storing incoming lists as well as outgoing lists doubles edge storage but makes
"list prerequisites" O(in-degree) instead of an O(V + E) scan of every list.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import (Callable, Deque, Dict, Generic, Hashable, List, Optional,
                    Sequence, Tuple, TypeVar)

V = TypeVar("V", bound=Hashable)


# =========================================================================== #
# 1. GENERIC DIRECTED GRAPH (no Course-specific code)
# =========================================================================== #
class GraphError(Exception):
    """Base class for graph errors."""


class VertexNotFoundError(GraphError):
    pass


class DuplicateVertexError(GraphError):
    pass


class DuplicateEdgeError(GraphError):
    pass


class EdgeNotFoundError(GraphError):
    pass


class SelfLoopError(GraphError):
    pass


class DirectedGraph(Generic[V]):
    """Unweighted directed graph using adjacency lists.

    Vertices can be any hashable value. Method names follow the assignment
    (camelCase); snake_case aliases are provided for Pythonic use.
    Self-loops are rejected unless ``allow_self_loops=True``.
    Public methods return COPIES of internal lists so callers cannot corrupt the graph.
    """

    def __init__(self, allow_self_loops: bool = False) -> None:
        self._outgoing: Dict[V, List[V]] = {}   # vertex -> vertices it points to
        self._incoming: Dict[V, List[V]] = {}   # vertex -> vertices pointing to it
        self._allow_self_loops = allow_self_loops

    # ------------------------------------------------------------- vertices
    def addVertex(self, vertex: V) -> None:
        """Add a vertex. Raises DuplicateVertexError if it already exists."""
        if vertex in self._outgoing:
            raise DuplicateVertexError(f"vertex {vertex!r} already exists")
        self._outgoing[vertex] = []
        self._incoming[vertex] = []

    def containsVertex(self, vertex: V) -> bool:
        return vertex in self._outgoing

    def listVertices(self) -> List[V]:
        """All vertices in insertion order."""
        return list(self._outgoing)

    # ---------------------------------------------------------------- edges
    def addEdge(self, source: V, target: V) -> None:
        """Add the directed edge source -> target.

        Raises VertexNotFoundError, SelfLoopError or DuplicateEdgeError.
        """
        self._require(source)
        self._require(target)
        if source == target and not self._allow_self_loops:
            raise SelfLoopError(f"self-loop on {source!r} is not allowed")
        if target in self._outgoing[source]:
            raise DuplicateEdgeError(f"edge {source!r} -> {target!r} already exists")
        self._outgoing[source].append(target)
        self._incoming[target].append(source)

    def removeEdge(self, source: V, target: V) -> None:
        """Remove source -> target. Raises VertexNotFoundError / EdgeNotFoundError."""
        self._require(source)
        self._require(target)
        if target not in self._outgoing[source]:
            raise EdgeNotFoundError(f"edge {source!r} -> {target!r} does not exist")
        self._outgoing[source].remove(target)
        self._incoming[target].remove(source)

    def hasEdge(self, source: V, target: V) -> bool:
        self._require(source)
        self._require(target)
        return target in self._outgoing[source]

    # ------------------------------------------------------------ neighbours
    def listIncomingAdjacentVertex(self, vertex: V) -> List[V]:
        """Vertices u with an edge u -> vertex."""
        self._require(vertex)
        return list(self._incoming[vertex])

    def listOutgoingAdjacentVertex(self, vertex: V) -> List[V]:
        """Vertices w with an edge vertex -> w."""
        self._require(vertex)
        return list(self._outgoing[vertex])

    # ------------------------------------------------------------ algorithms
    def hasPath(self, start: V, goal: V) -> bool:
        """True if `goal` is reachable from `start` (breadth-first search, O(V+E))."""
        self._require(start)
        self._require(goal)
        if start == goal:
            return True
        seen = {start}
        queue: Deque[V] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbour in self._outgoing[current]:
                if neighbour == goal:
                    return True
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        return False

    def isAcyclic(self) -> bool:
        """True if the graph has no directed cycle (Kahn's algorithm, O(V+E))."""
        in_degree = {v: len(sources) for v, sources in self._incoming.items()}
        ready: Deque[V] = deque(v for v, d in in_degree.items() if d == 0)
        visited = 0
        while ready:
            current = ready.popleft()
            visited += 1
            for neighbour in self._outgoing[current]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    ready.append(neighbour)
        return visited == len(in_degree)

    # ---------------------------------------------------------------- misc
    def _require(self, vertex: V) -> None:
        if vertex not in self._outgoing:
            raise VertexNotFoundError(f"vertex {vertex!r} does not exist")

    def __len__(self) -> int:
        return len(self._outgoing)

    def __contains__(self, vertex: object) -> bool:
        return vertex in self._outgoing

    def edgeCount(self) -> int:
        return sum(len(targets) for targets in self._outgoing.values())

    # Pythonic (PEP 8) aliases - same behaviour.
    add_vertex = addVertex
    add_edge = addEdge
    remove_edge = removeEdge
    has_edge = hasEdge
    contains_vertex = containsVertex
    list_vertices = listVertices
    list_incoming_adjacent_vertex = listIncomingAdjacentVertex
    list_outgoing_adjacent_vertex = listOutgoingAdjacentVertex
    has_path = hasPath
    is_acyclic = isAcyclic
    edge_count = edgeCount


# =========================================================================== #
# 2. DOMAIN MODEL
# =========================================================================== #
class CourseType(Enum):
    CORE = "core"
    ELECTIVE = "elective"

    @classmethod
    def parse(cls, text: str) -> "CourseType":
        """Case-insensitive; raises ValueError listing the valid choices."""
        try:
            return cls(text.strip().lower())
        except ValueError:
            valid = ", ".join(member.value for member in cls)
            raise ValueError(f"course type must be one of: {valid}") from None


COURSE_CODE_PATTERN = re.compile(r"^[A-Z0-9]{4,10}$")
MIN_CREDIT_HOURS, MAX_CREDIT_HOURS = 1, 12


def normalise_code(text: str) -> str:
    """Canonical form used for lookup: trimmed, upper-case."""
    return text.strip().upper()


@dataclass(frozen=True)
class Course:
    """A university course. Unique key: ``course_code`` (also the graph vertex)."""

    course_code: str
    course_name: str
    credit_hours: int
    course_type: CourseType
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.course_code, str) or not COURSE_CODE_PATTERN.match(self.course_code):
            raise ValueError("course code must be 4-10 upper-case letters/digits (e.g. CS101)")
        if not isinstance(self.course_name, str) or not self.course_name.strip():
            raise ValueError("course name must not be empty")
        if isinstance(self.credit_hours, bool) or not isinstance(self.credit_hours, int):
            raise TypeError("credit hours must be an integer")
        if not MIN_CREDIT_HOURS <= self.credit_hours <= MAX_CREDIT_HOURS:
            raise ValueError(f"credit hours must be between {MIN_CREDIT_HOURS} and {MAX_CREDIT_HOURS}")
        if not isinstance(self.course_type, CourseType):
            raise TypeError("course_type must be a CourseType")
        if not isinstance(self.description, str):
            raise TypeError("description must be text")

    def full_details(self) -> List[Tuple[str, str]]:
        """Every field, ignoring visibility (used by menu option 2)."""
        return [("Course code", self.course_code), ("Course name", self.course_name),
                ("Credit hours", str(self.credit_hours)),
                ("Course type", self.course_type.value), ("Description", self.description)]

    def public_profile(self) -> List[Tuple[str, str]]:
        """Visibility rule: core -> all details; elective -> name only."""
        if self.course_type is CourseType.CORE:
            return self.full_details()
        return [("Course name", self.course_name)]


# =========================================================================== #
# 3. SERVICE LAYER
# =========================================================================== #
class CatalogueError(Exception):
    """Base class for course-catalogue rule violations."""


class DuplicateCourseError(CatalogueError):
    pass


class CourseNotFoundError(CatalogueError):
    pass


class InvalidPrerequisiteError(CatalogueError):
    pass


class CourseCatalogue:
    """Courses plus prerequisite relationships.

    The graph's vertices are course CODES (stable, hashable, cheap). Course
    objects are kept in a separate code -> Course lookup. Edge Y -> X means
    "Y is a prerequisite of X".
    """

    def __init__(self) -> None:
        self._graph: DirectedGraph[str] = DirectedGraph()
        self._courses: Dict[str, Course] = {}

    # -------------------------------------------------------------- courses
    def add_course(self, course: Course) -> None:
        if course.course_code in self._courses:
            raise DuplicateCourseError(f"course {course.course_code} already exists")
        self._graph.addVertex(course.course_code)
        self._courses[course.course_code] = course

    def get_course(self, code: str) -> Course:
        key = normalise_code(code)
        if key not in self._courses:
            raise CourseNotFoundError(f"unknown course code '{key}'")
        return self._courses[key]

    def list_courses(self) -> List[Course]:
        return [self._courses[code] for code in self._graph.listVertices()]

    def codes(self) -> List[str]:
        return self._graph.listVertices()

    # -------------------------------------------------------- prerequisites
    def prerequisites_of(self, code: str) -> List[Course]:
        """Incoming edges of the course vertex."""
        course = self.get_course(code)
        return [self._courses[c] for c in self._graph.listIncomingAdjacentVertex(course.course_code)]

    def unlocked_by(self, code: str) -> List[Course]:
        """Outgoing edges of the course vertex."""
        course = self.get_course(code)
        return [self._courses[c] for c in self._graph.listOutgoingAdjacentVertex(course.course_code)]

    def add_prerequisite(self, course_code: str, prerequisite_code: str) -> None:
        """Make `prerequisite_code` a prerequisite of `course_code` (edge Y -> X).

        Rejects: unknown codes, self-prerequisite, duplicates, and any edge that
        would create a cycle (X could then never be taken). A cycle appears
        exactly when X can already reach Y, so we test hasPath(X, Y) first.
        """
        course = self.get_course(course_code).course_code
        prerequisite = self.get_course(prerequisite_code).course_code
        if course == prerequisite:
            raise InvalidPrerequisiteError(f"{course} cannot be a prerequisite of itself")
        if self._graph.hasEdge(prerequisite, course):
            raise InvalidPrerequisiteError(f"{prerequisite} is already a prerequisite of {course}")
        if self._graph.hasPath(course, prerequisite):
            raise InvalidPrerequisiteError(
                f"adding {prerequisite} as a prerequisite of {course} would create a cycle "
                f"({course} already leads to {prerequisite})")
        self._graph.addEdge(prerequisite, course)

    def remove_prerequisite(self, course_code: str, prerequisite_code: str) -> None:
        course = self.get_course(course_code).course_code
        prerequisite = self.get_course(prerequisite_code).course_code
        if not self._graph.hasEdge(prerequisite, course):
            raise InvalidPrerequisiteError(f"{prerequisite} is not a prerequisite of {course}")
        self._graph.removeEdge(prerequisite, course)

    def verify_acyclic(self) -> bool:
        return self._graph.isAcyclic()

    def statistics(self) -> str:
        return f"{len(self._graph)} courses, {self._graph.edgeCount()} prerequisite links"


# Pre-defined data (9 courses; limit for predefined courses is 10).
SAMPLE_COURSES: Tuple[Course, ...] = (
    Course("CS101", "Programming Fundamentals", 4, CourseType.CORE, "Introduction to programming and problem solving."),
    Course("CS102", "Discrete Mathematics", 3, CourseType.CORE, "Logic, sets, relations, graphs and proofs."),
    Course("CS201", "Data Structures", 4, CourseType.CORE, "Lists, trees, hash tables and graphs."),
    Course("CS202", "Object-Oriented Programming", 3, CourseType.CORE, "Classes, inheritance and design principles."),
    Course("CS301", "Advanced Algorithms", 4, CourseType.CORE, "Complexity analysis, graph and hashing algorithms."),
    Course("CS302", "Database Systems", 3, CourseType.CORE, "Relational model, SQL and transactions."),
    Course("MA201", "Linear Algebra", 3, CourseType.ELECTIVE, "Vectors, matrices and linear transformations."),
    Course("CS310", "Machine Learning", 3, CourseType.ELECTIVE, "Supervised and unsupervised learning."),
    Course("CS320", "Cloud Computing", 3, CourseType.ELECTIVE, "Distributed systems and cloud services."),
)

# course -> its prerequisites (each pair becomes the edge prerequisite -> course)
SAMPLE_PREREQUISITES: Dict[str, Tuple[str, ...]] = {
    "CS201": ("CS101",),            # one prerequisite
    "CS202": ("CS101",),            # CS101 therefore unlocks several courses
    "CS301": ("CS201", "CS102"),    # multiple prerequisites
    "CS302": ("CS201",),
    "CS310": ("CS301", "MA201"),    # multiple prerequisites, elective
    "CS320": ("CS302",),
}                                   # CS101, CS102, MA201 have no prerequisite


def build_sample_catalogue() -> CourseCatalogue:
    catalogue = CourseCatalogue()
    for course in SAMPLE_COURSES:
        catalogue.add_course(course)
    for course_code, prerequisites in SAMPLE_PREREQUISITES.items():
        for prerequisite in prerequisites:
            catalogue.add_prerequisite(course_code, prerequisite)
    if not catalogue.verify_acyclic():           # defensive check of the data itself
        raise RuntimeError("sample prerequisite data contains a cycle")
    return catalogue


# =========================================================================== #
# 4. PRESENTATION HELPERS (pure functions returning text)
# =========================================================================== #
def render_course_table(courses: Sequence[Course], title: str) -> str:
    lines = [title, "-" * 72, f"{'Code':<8} {'Course name':<32} {'Credits':>7}  {'Type':<9}", "-" * 72]
    if not courses:
        lines.append("(none)")
    for c in courses:
        lines.append(f"{c.course_code:<8} {c.course_name:<32} {c.credit_hours:>7}  {c.course_type.value:<9}")
    return "\n".join(lines)


def render_fields(fields: Sequence[Tuple[str, str]], title: str) -> str:
    width = max(len(label) for label, _ in fields)
    lines = [title, "-" * 72]
    lines += [f"{label:<{width}} : {value}" for label, value in fields]
    return "\n".join(lines)


# =========================================================================== #
# 5. COMMAND-LINE INTERFACE
# =========================================================================== #
class ActionCancelled(Exception):
    """User typed 'cancel'."""


class CourseCLI:
    MENU = ("1. Display all course names",
            "2. View full details of one course (ignores visibility)",
            "3. View prerequisites of a course (incoming edges)",
            "4. View courses unlocked by a course (outgoing edges)",
            "5. Add a course",
            "6. Add a prerequisite",
            "7. Remove a prerequisite",
            "8. Display public course profile (core vs elective)",
            "9. Exit")

    def __init__(self, catalogue: CourseCatalogue,
                 input_fn: Callable[[str], str] = input,
                 output_fn: Callable[[str], None] = print) -> None:
        self._catalogue = catalogue
        self._input = input_fn
        self._print = output_fn

    # -------------------------------------------------------------- helpers
    def _ask(self, prompt: str) -> str:
        while True:
            text = self._input(f"{prompt}: ").strip()
            if text.lower() == "cancel":
                raise ActionCancelled
            if text:
                return text
            self._print("  Input cannot be empty (type 'cancel' to abort).")

    def _ask_existing_code(self, prompt: str) -> str:
        while True:
            try:
                return self._catalogue.get_course(self._ask(prompt)).course_code
            except CatalogueError as error:
                self._print(f"  Error: {error}. Known codes: {', '.join(self._catalogue.codes())}")

    def _ask_new_course(self) -> Course:
        """Prompt field by field, re-asking only the field that was invalid."""
        while True:
            code = normalise_code(self._ask("Course code (e.g. CS330)"))
            try:
                Course(code, "x", MIN_CREDIT_HOURS, CourseType.CORE, "")   # validates code only
                break
            except (TypeError, ValueError) as error:
                self._print(f"  Invalid: {error}")
        name = self._ask("Course name")
        while True:
            try:
                credits = int(self._ask(f"Credit hours ({MIN_CREDIT_HOURS}-{MAX_CREDIT_HOURS})"))
                Course(code, name, credits, CourseType.CORE, "")
                break
            except (TypeError, ValueError) as error:
                self._print(f"  Invalid credit hours: {error}")
        while True:
            try:
                course_type = CourseType.parse(self._ask("Course type (core/elective)"))
                break
            except ValueError as error:
                self._print(f"  Invalid: {error}")
        description = self._ask("Description")
        return Course(code, name, credits, course_type, description)

    # -------------------------------------------------------------- actions
    def _list_all(self) -> None:
        self._print(render_course_table(self._catalogue.list_courses(), "ALL COURSES"))

    def _details(self) -> None:
        code = self._ask_existing_code("Course code")
        self._print(render_fields(self._catalogue.get_course(code).full_details(), f"FULL DETAILS: {code}"))

    def _prerequisites(self) -> None:
        code = self._ask_existing_code("Course code")
        self._print(render_course_table(self._catalogue.prerequisites_of(code), f"PREREQUISITES OF {code}"))

    def _unlocked(self) -> None:
        code = self._ask_existing_code("Course code")
        self._print(render_course_table(self._catalogue.unlocked_by(code), f"COURSES UNLOCKED BY {code}"))

    def _add_course(self) -> None:
        course = self._ask_new_course()
        try:
            self._catalogue.add_course(course)
        except CatalogueError as error:
            self._print(f"  Error: {error}")
            return
        self._print(f"  Added {course.course_code} - {course.course_name} ({course.course_type.value}). "
                    f"{self._catalogue.statistics()}.")

    def _add_prerequisite(self) -> None:
        course = self._ask_existing_code("Course X (the course that will REQUIRE a prerequisite)")
        prerequisite = self._ask_existing_code("Course Y (the prerequisite)")
        try:
            self._catalogue.add_prerequisite(course, prerequisite)
        except CatalogueError as error:
            self._print(f"  Rejected: {error}")
            return
        self._print(f"  OK: {prerequisite} is now a prerequisite of {course} (edge {prerequisite} -> {course}).")

    def _remove_prerequisite(self) -> None:
        course = self._ask_existing_code("Course X (the course that currently requires Y)")
        prerequisite = self._ask_existing_code("Course Y (the prerequisite to remove)")
        try:
            self._catalogue.remove_prerequisite(course, prerequisite)
        except CatalogueError as error:
            self._print(f"  Rejected: {error}")
            return
        self._print(f"  OK: {prerequisite} is no longer a prerequisite of {course}.")

    def _profile(self) -> None:
        code = self._ask_existing_code("Course code")
        course = self._catalogue.get_course(code)
        self._print(render_fields(course.public_profile(), f"PUBLIC PROFILE ({course.course_type.value} course)"))

    # ----------------------------------------------------------------- loop
    def run(self) -> None:
        actions = {"1": self._list_all, "2": self._details, "3": self._prerequisites,
                   "4": self._unlocked, "5": self._add_course, "6": self._add_prerequisite,
                   "7": self._remove_prerequisite, "8": self._profile}
        self._print("\n" + "=" * 72)
        self._print("  UNIVERSITY COURSE PREREQUISITE SYSTEM  (directed graph)")
        self._print("=" * 72)
        self._print(f"  {self._catalogue.statistics()}. Type 'cancel' at any prompt to abort an action.")
        try:
            while True:
                self._print("\n" + "\n".join(f"  {item}" for item in self.MENU))
                choice = self._input("Choose an option (1-9): ").strip()
                if choice == "9":
                    break
                action = actions.get(choice)
                if action is None:
                    self._print("  Invalid choice. Please enter a number from 1 to 9.")
                    continue
                try:
                    action()
                except ActionCancelled:
                    self._print("  Action cancelled.")
        except (EOFError, KeyboardInterrupt):
            self._print("\n  Input closed.")
        self._print("  Goodbye.")


# =========================================================================== #
# Deterministic demonstration (for screenshots)
# =========================================================================== #
def run_demo() -> None:
    """Scripted walkthrough; every step prints what it does and what happened."""
    step = [0]

    def heading(text: str) -> None:
        step[0] += 1
        print(f"\n=== Demo {step[0]}: {text} ===")

    def attempt(label: str, action: Callable[[], object]) -> None:
        try:
            action()
            print(f"  ACCEPTED: {label}")
        except (CatalogueError, GraphError, TypeError, ValueError) as error:
            print(f"  REJECTED: {label} -> {error}")

    heading("Generic graph used on plain strings (no Course code involved)")
    graph: DirectedGraph[str] = DirectedGraph()
    for vertex in ("A", "B", "C"):
        graph.addVertex(vertex)
    graph.addEdge("A", "C")
    graph.addEdge("B", "C")
    print(f"  vertices={graph.listVertices()}  edges={graph.edgeCount()}")
    print(f"  listIncomingAdjacentVertex('C') = {graph.listIncomingAdjacentVertex('C')}")
    print(f"  listOutgoingAdjacentVertex('A') = {graph.listOutgoingAdjacentVertex('A')}")
    try:
        graph.addEdge("A", "C")
    except GraphError as error:
        print(f"  duplicate edge rejected: {error}")
    try:
        graph.addEdge("A", "Z")
    except GraphError as error:
        print(f"  missing vertex rejected: {error}")

    catalogue = build_sample_catalogue()
    print(f"\nSample catalogue loaded: {catalogue.statistics()}; acyclic = {catalogue.verify_acyclic()}")

    heading("List all courses")
    print(render_course_table(catalogue.list_courses(), "ALL COURSES"))

    heading("Full details of one course (CS301)")
    print(render_fields(catalogue.get_course("CS301").full_details(), "FULL DETAILS: CS301"))

    heading("Course with MULTIPLE incoming edges (prerequisites of CS301)")
    print(render_course_table(catalogue.prerequisites_of("CS301"), "PREREQUISITES OF CS301"))

    heading("Course with NO prerequisite (CS101)")
    print(render_course_table(catalogue.prerequisites_of("CS101"), "PREREQUISITES OF CS101"))

    heading("Course that unlocks MULTIPLE courses (outgoing edges of CS101)")
    print(render_course_table(catalogue.unlocked_by("CS101"), "COURSES UNLOCKED BY CS101"))

    heading("Add a course on demand (CS330, elective) and a core course (CS340)")
    attempt("add CS330", lambda: catalogue.add_course(
        Course("CS330", "Mobile App Development", 3, CourseType.ELECTIVE, "Building apps for phones.")))
    attempt("add CS340", lambda: catalogue.add_course(
        Course("CS340", "Cyber Security", 3, CourseType.CORE, "Threats, defences and secure design.")))
    print(f"  {catalogue.statistics()}")

    heading("Add prerequisites on demand")
    attempt("CS202 is a prerequisite of CS330", lambda: catalogue.add_prerequisite("CS330", "CS202"))
    print(render_course_table(catalogue.prerequisites_of("CS330"), "PREREQUISITES OF CS330"))
    print(render_course_table(catalogue.unlocked_by("CS202"), "COURSES UNLOCKED BY CS202"))

    heading("Reject invalid or duplicate requests")
    attempt("duplicate edge: CS202 -> CS330 again", lambda: catalogue.add_prerequisite("CS330", "CS202"))
    attempt("self-prerequisite: CS101 requires CS101", lambda: catalogue.add_prerequisite("CS101", "CS101"))
    attempt("cycle: CS301 as prerequisite of CS101", lambda: catalogue.add_prerequisite("CS101", "CS301"))
    attempt("unknown course code ZZ999", lambda: catalogue.get_course("ZZ999"))
    attempt("duplicate course code CS101", lambda: catalogue.add_course(SAMPLE_COURSES[0]))
    attempt("invalid credit hours (0)", lambda: Course("CS350", "Bad Credits", 0, CourseType.CORE, ""))
    attempt("invalid course type 'optional'", lambda: CourseType.parse("optional"))
    attempt("remove a prerequisite that does not exist", lambda: catalogue.remove_prerequisite("CS101", "CS302"))

    heading("Remove a prerequisite")
    attempt("remove CS202 as a prerequisite of CS330", lambda: catalogue.remove_prerequisite("CS330", "CS202"))
    print(render_course_table(catalogue.prerequisites_of("CS330"), "PREREQUISITES OF CS330"))

    heading("Public profile: core course versus elective course")
    print(render_fields(catalogue.get_course("CS301").public_profile(), "PUBLIC PROFILE (core): CS301"))
    print(render_fields(catalogue.get_course("CS310").public_profile(), "PUBLIC PROFILE (elective): CS310"))
    print(render_fields(catalogue.get_course("CS310").full_details(), "FULL DETAILS (elective): CS310"))


# =========================================================================== #
# Entry point
# =========================================================================== #
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="5022CMD Q2 - course prerequisite graph")
    parser.add_argument("--demo", action="store_true", help="run the scripted demonstration and exit")
    args = parser.parse_args(argv)
    if args.demo:
        run_demo()
        return 0
    CourseCLI(build_sample_catalogue()).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
