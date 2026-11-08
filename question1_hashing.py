from __future__ import annotations

import argparse
import csv
import functools
import gc
import random
import statistics
import sys
import time
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import (Callable, Generic, Hashable, Iterator, List, Optional,
                    Sequence, Tuple, TypeVar, Union)

# --------------------------------------------------------------------------- #
# Configuration constants (change here or via command-line arguments)
# --------------------------------------------------------------------------- #
CURRENCY = "RM"
MAX_PRICE = Decimal("1000000.00")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_CSV_PATH = RESULTS_DIR / "question1_search_results.csv"

INVENTORY_INITIAL_CAPACITY = 13   # small on purpose so collisions are visible
INVENTORY_MAX_LOAD_FACTOR = 0.6

DEFAULT_SIZES = (1_000, 5_000, 10_000)   # dataset sizes for the experiment
DEFAULT_QUERIES = 200                    # distinct IDs searched in one timed batch
DEFAULT_REPEATS = 20                     # timed batches per structure/scenario
DEFAULT_WARMUP = 2                       # untimed warm-up batches
DEFAULT_SEED = 5022                      # recorded so the dataset is reproducible
EXPERIMENT_MAX_LOAD_FACTOR = 0.5

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


# =========================================================================== #
# 1. DOMAIN MODEL
# =========================================================================== #
@dataclass(frozen=True)
class PharmacyProduct:
    """A product stocked by the pharmacy.

    Unique key: ``product_id`` (used as the hash-table key).

    Why ``Decimal`` for price: binary floats cannot represent most decimal
    fractions exactly (0.1 + 0.2 != 0.3), so money arithmetic can drift.
    ``Decimal`` stores base-10 values exactly, which is what currency needs.

    The dataclass is frozen (immutable): the same object can safely be shared by
    the hash table and the list. "Editing" creates a validated copy.
    """

    product_id: int
    name: str
    price: Decimal
    stock_quantity: int
    category: str

    def __post_init__(self) -> None:
        if isinstance(self.product_id, bool) or not isinstance(self.product_id, int):
            raise TypeError("product_id must be an integer")
        if self.product_id <= 0:
            raise ValueError("product_id must be a positive integer")
        for label, text in (("name", self.name), ("category", self.category)):
            if not isinstance(text, str):
                raise TypeError(f"{label} must be text")
            if not text.strip():
                raise ValueError(f"{label} must not be empty")
        if not isinstance(self.price, Decimal):
            raise TypeError("price must be a Decimal")
        if not self.price.is_finite() or self.price <= 0:
            raise ValueError("price must be a positive, finite amount")
        if self.price > MAX_PRICE:
            raise ValueError(f"price must not exceed {MAX_PRICE}")
        if self.price != self.price.quantize(Decimal("0.01")):
            raise ValueError("price must have at most 2 decimal places")
        if isinstance(self.stock_quantity, bool) or not isinstance(self.stock_quantity, int):
            raise TypeError("stock_quantity must be an integer")
        if self.stock_quantity < 0:
            raise ValueError("stock_quantity must not be negative")

    @property
    def key(self) -> int:
        """The unique key used by the hash table."""
        return self.product_id

    def __str__(self) -> str:
        return (f"#{self.product_id:<5} {self.name:<32} {self.category:<16} "
                f"{CURRENCY} {self.price:>8.2f}  stock={self.stock_quantity}")


def parse_price(text: str) -> Decimal:
    """Convert user text to a Decimal price; raise ValueError if not a number."""
    try:
        value = Decimal(text.strip())
    except InvalidOperation:
        raise ValueError(f"'{text}' is not a valid price") from None
    if not value.is_finite():
        raise ValueError(f"'{text}' is not a valid price")
    return value


def sample_products() -> List[PharmacyProduct]:
    """Illustrative pre-defined records (not clinical advice, not real prices).

    IDs 1001, 1014 and 1027 share the same home slot when capacity is 13,
    which deliberately produces collisions for demonstration.
    """
    rows = [
        (1001, "Paracetamol 500mg Tablets", "4.50", 120, "Analgesic"),
        (1002, "Amoxicillin 250mg Capsules", "12.80", 45, "Antibiotic"),
        (1014, "Ibuprofen 400mg Tablets", "7.90", 80, "Analgesic"),
        (2005, "Cetirizine 10mg Tablets", "6.30", 60, "Antihistamine"),
        (3010, "Omeprazole 20mg Capsules", "15.40", 35, "Gastrointestinal"),
        (1027, "Loratadine 10mg Tablets", "8.20", 50, "Antihistamine"),
        (4120, "Salbutamol 100mcg Inhaler", "18.90", 25, "Respiratory"),
    ]
    return [PharmacyProduct(i, n, Decimal(p), q, c) for i, n, p, q, c in rows]


# =========================================================================== #
# 2. GENERIC HASH TABLE (open addressing, linear probing)
# =========================================================================== #
class HashTableError(Exception):
    """Base class for hash-table errors."""


class DuplicateKeyError(HashTableError):
    """Raised by insert() when the key already exists."""


class KeyNotFoundError(HashTableError):
    """Raised by update()/delete() when the key does not exist."""


class HashTableFullError(HashTableError):
    """Raised when a fixed-capacity table has no free slot."""


class _Tombstone:
    """Marker for a deleted slot: 'occupied once, keep probing past me'."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<TOMBSTONE>"


_TOMBSTONE = _Tombstone()   # single shared instance, compared with `is`


class _Entry(Generic[K, V]):
    """One occupied slot: an explicit key/value pair."""

    __slots__ = ("key", "value")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value


@dataclass(frozen=True)
class PlacementInfo:
    """Where insert() stored an entry."""

    index: int          # final slot
    home_index: int     # slot the hash function pointed at
    displacement: int   # extra slots walked past (0 = no collision)


@dataclass(frozen=True)
class LookupResult(Generic[V]):
    """Outcome of locate(): value plus probe information."""

    found: bool
    value: Optional[V]
    index: Optional[int]   # slot of the key, None if absent
    home_index: int
    probes: int            # slots examined


# Internal probe result: (found_index, free_index, probes_examined)
_Probe = Tuple[Optional[int], Optional[int], int]
_Slot = Union[None, _Entry, _Tombstone]


def _next_prime(n: int) -> int:
    """Smallest prime >= n (trial division; fine for table sizes)."""
    candidate = max(2, n)
    while True:
        if all(candidate % d for d in range(2, int(candidate ** 0.5) + 1)):
            return candidate
        candidate += 1


class LinearProbingHashTable(Generic[K, V]):
    """Generic hash table: open addressing + linear probing.

    Slot states:  None (never used) | _Entry (key/value) | _TOMBSTONE (deleted).

    Probe sequence for a key:  (home + step) % capacity, step = 0, 1, 2, ...
    Collisions are resolved ONLY by this linear probing.

    Design choices
    --------------
    * insert() rejects duplicates (DuplicateKeyError); update() changes an
      existing key (KeyNotFoundError if absent). Explicit and safe for inventory.
    * Deletion writes a tombstone so later keys in the same probe chain stay
      reachable. Insert may reuse the first tombstone seen on its chain.
    * With ``auto_resize=True`` (default) the table grows to the next prime
      >= 2*capacity+1 when (active + tombstones + 1) / capacity would exceed
      ``max_load_factor``; every active entry is rehashed. If the cause is
      mostly tombstones the table is rehashed at the same capacity to purge them.
    * With ``auto_resize=False`` the capacity is fixed and HashTableFullError
      is raised when no slot is free. All loops are bounded by capacity, so a
      full table can never cause an infinite loop.
    * The default hash function is Python's built-in ``hash``; any callable
      mapping a key to an int can be supplied. The slot index is
      ``hash_function(key) % capacity`` (non-negative for a positive modulus).
      Python's built-in dict is NOT used for storage.
    """

    DEFAULT_CAPACITY = 11
    DEFAULT_MAX_LOAD_FACTOR = 0.6

    def __init__(self,
                 capacity: int = DEFAULT_CAPACITY,
                 *,
                 max_load_factor: float = DEFAULT_MAX_LOAD_FACTOR,
                 auto_resize: bool = True,
                 hash_function: Callable[[K], int] = hash) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if not 0 < max_load_factor <= 1:
            raise ValueError("max_load_factor must satisfy 0 < value <= 1")
        self._slots: List[_Slot] = [None] * capacity
        self._size = 0           # active entries
        self._tombstones = 0     # deleted markers still occupying slots
        self._max_load_factor = max_load_factor
        self._auto_resize = auto_resize
        self._hash_function = hash_function
        self.resize_count = 0    # how many times the table was rehashed

    # ------------------------------------------------------------- properties
    @property
    def capacity(self) -> int:
        return len(self._slots)

    @property
    def tombstone_count(self) -> int:
        return self._tombstones

    def __len__(self) -> int:
        return self._size

    def __contains__(self, key: K) -> bool:
        return self._probe(key)[0] is not None

    def load_factor(self) -> float:
        """Active entries / capacity (tombstones are reported separately)."""
        return self._size / len(self._slots)

    # ---------------------------------------------------------------- hashing
    def _home_index(self, key: K) -> int:
        return self._hash_function(key) % len(self._slots)

    def _probe(self, key: K) -> _Probe:
        """Walk the probe sequence once.

        Returns (found_index, free_index, probes):
          found_index - slot holding `key`, else None
          free_index  - best slot to insert (first tombstone seen, else the
                        first empty slot), else None if the table is full
          probes      - number of slots examined
        Stops at the first never-used slot (key cannot be further along) and
        after at most `capacity` steps (so it always terminates).
        """
        capacity = len(self._slots)
        home = self._hash_function(key) % capacity
        first_tombstone: Optional[int] = None
        for step in range(capacity):
            index = (home + step) % capacity
            slot = self._slots[index]
            if slot is None:
                return None, (index if first_tombstone is None else first_tombstone), step + 1
            if slot is _TOMBSTONE:
                if first_tombstone is None:
                    first_tombstone = index
            elif slot.key == key:           # type: ignore[union-attr]
                return index, None, step + 1
        return None, first_tombstone, capacity

    # -------------------------------------------------------------- operations
    def insert(self, key: K, value: V) -> PlacementInfo:
        """Add a new key. Raises DuplicateKeyError / HashTableFullError."""
        found, free, _ = self._probe(key)
        if found is not None:
            raise DuplicateKeyError(f"key {key!r} already exists")
        if self._auto_resize and self._would_exceed_load():
            self._rehash(self._grown_capacity())
            _, free, _ = self._probe(key)
        if free is None:
            raise HashTableFullError(
                f"table is full (capacity {self.capacity}, auto_resize disabled)")
        if self._slots[free] is _TOMBSTONE:
            self._tombstones -= 1               # tombstone slot is reused
        self._slots[free] = _Entry(key, value)
        self._size += 1
        home = self._home_index(key)
        return PlacementInfo(free, home, (free - home) % self.capacity)

    def locate(self, key: K) -> LookupResult[V]:
        """Search with probe information (slot index, probes examined)."""
        found, _, probes = self._probe(key)
        home = self._home_index(key)
        if found is None:
            return LookupResult(False, None, None, home, probes)
        entry = self._slots[found]
        return LookupResult(True, entry.value, found, home, probes)  # type: ignore[union-attr]

    def search(self, key: K) -> Optional[V]:
        """Return the value for `key`, or None if absent.

        Use locate() or `key in table` if None could be a legitimate value.
        """
        found = self._probe(key)[0]
        return None if found is None else self._slots[found].value  # type: ignore[union-attr]

    def update(self, key: K, value: V) -> int:
        """Replace the value of an existing key; return its slot index."""
        found = self._probe(key)[0]
        if found is None:
            raise KeyNotFoundError(f"key {key!r} not found")
        self._slots[found].value = value        # type: ignore[union-attr]
        return found

    def delete(self, key: K) -> V:
        """Remove `key` (leaves a tombstone) and return its value."""
        found = self._probe(key)[0]
        if found is None:
            raise KeyNotFoundError(f"key {key!r} not found")
        entry = self._slots[found]
        self._slots[found] = _TOMBSTONE
        self._size -= 1
        self._tombstones += 1
        return entry.value                       # type: ignore[union-attr]

    def items(self) -> Iterator[Tuple[K, V]]:
        """Iterate over active (key, value) pairs in slot order."""
        for slot in self._slots:
            if isinstance(slot, _Entry):
                yield slot.key, slot.value

    # ----------------------------------------------------------------- resize
    def _would_exceed_load(self) -> bool:
        used_after = self._size + self._tombstones + 1
        return used_after > self._max_load_factor * len(self._slots)

    def _grown_capacity(self) -> int:
        """Grow if live entries need it; otherwise keep size and purge tombstones."""
        if (self._size + 1) > self._max_load_factor * len(self._slots):
            return _next_prime(2 * len(self._slots) + 1)
        return len(self._slots)

    def _rehash(self, new_capacity: int) -> None:
        """Rebuild the slot list; tombstones disappear, entries are re-placed."""
        active = [slot for slot in self._slots if isinstance(slot, _Entry)]
        self._slots = [None] * new_capacity
        self._tombstones = 0
        for entry in active:
            index = self._home_index(entry.key)
            while self._slots[index] is not None:       # table has free slots
                index = (index + 1) % new_capacity
            self._slots[index] = entry
        self.resize_count += 1

    # ---------------------------------------------------------------- display
    def display_table(self, value_formatter: Callable[[V], str] = str) -> str:
        """Return a readable dump of every slot (home slot and displacement shown)."""
        lines = [
            f"Hash table: capacity={self.capacity}, active={self._size}, "
            f"tombstones={self._tombstones}, load factor={self.load_factor():.3f}, "
            f"resizes={self.resize_count}",
            "-" * 96,
        ]
        for index, slot in enumerate(self._slots):
            if slot is None:
                text = "<EMPTY>"
            elif slot is _TOMBSTONE:
                text = "<TOMBSTONE> (deleted; probe chain continues)"
            else:
                home = self._home_index(slot.key)           # type: ignore[union-attr]
                shift = (index - home) % self.capacity
                note = "" if shift == 0 else f"   [home={home}, +{shift} collision]"
                text = f"key={slot.key!r} -> {value_formatter(slot.value)}{note}"  # type: ignore[union-attr]
            lines.append(f"[{index:>3}] {text}")
        return "\n".join(lines)


# =========================================================================== #
# 3. SERVICE LAYER (pharmacy-specific behaviour lives here, not in the table)
# =========================================================================== #
class PharmacyInventorySystem:
    """Pharmacy operations built on LinearProbingHashTable[int, PharmacyProduct]."""

    def __init__(self,
                 capacity: int = INVENTORY_INITIAL_CAPACITY,
                 max_load_factor: float = INVENTORY_MAX_LOAD_FACTOR,
                 auto_resize: bool = True) -> None:
        self._table: LinearProbingHashTable[int, PharmacyProduct] = LinearProbingHashTable(
            capacity, max_load_factor=max_load_factor, auto_resize=auto_resize)

    @property
    def capacity(self) -> int:
        return self._table.capacity

    def load_sample_data(self) -> None:
        for product in sample_products():
            self.add_product(product)

    def add_product(self, product: PharmacyProduct) -> PlacementInfo:
        return self._table.insert(product.key, product)

    def find_product(self, product_id: int) -> LookupResult[PharmacyProduct]:
        return self._table.locate(product_id)

    def edit_product(self, product_id: int, *, name: Optional[str] = None,
                     price: Optional[Decimal] = None,
                     stock_quantity: Optional[int] = None,
                     category: Optional[str] = None) -> PharmacyProduct:
        """Apply the given changes to a validated copy and store it.

        product_id (the key) cannot be edited; delete and re-insert instead.
        """
        current = self._table.locate(product_id)
        if not current.found or current.value is None:
            raise KeyNotFoundError(f"product {product_id} not found")
        changes = {field: value for field, value in
                   (("name", name), ("price", price),
                    ("stock_quantity", stock_quantity), ("category", category))
                   if value is not None}
        updated = replace(current.value, **changes)   # re-runs validation
        self._table.update(product_id, updated)
        return updated

    def remove_product(self, product_id: int) -> PharmacyProduct:
        return self._table.delete(product_id)

    def products(self) -> List[PharmacyProduct]:
        return sorted((p for _, p in self._table.items()), key=lambda p: p.product_id)

    def render_table(self) -> str:
        return self._table.display_table(
            lambda p: f"{p.name} ({CURRENCY} {p.price:.2f}, qty {p.stock_quantity})")

    def summary_line(self) -> str:
        t = self._table
        return (f"capacity={t.capacity}, active={len(t)}, tombstones={t.tombstone_count}, "
                f"load factor={t.load_factor():.3f}")


# =========================================================================== #
# 4. EXPERIMENT: hash table vs one-dimensional list
# =========================================================================== #
@dataclass(frozen=True)
class ExperimentConfig:
    sizes: Tuple[int, ...] = DEFAULT_SIZES
    queries: int = DEFAULT_QUERIES
    repeats: int = DEFAULT_REPEATS
    warmup: int = DEFAULT_WARMUP
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if not self.sizes or any(s < 1 for s in self.sizes):
            raise ValueError("every dataset size must be >= 1")
        if self.queries < 1 or self.repeats < 1 or self.warmup < 0:
            raise ValueError("queries and repeats must be >= 1; warmup must be >= 0")


@dataclass
class TimingSeries:
    """Raw timing data for one (dataset size, scenario, structure)."""

    dataset_size: int
    scenario: str                 # "successful" | "unsuccessful"
    structure: str                # "hash_table" | "list"
    queries_per_trial: int
    trial_ns: List[int]           # RAW elapsed ns of each timed batch

    @property
    def per_search_ns(self) -> List[float]:
        return [t / self.queries_per_trial for t in self.trial_ns]

    @property
    def repeats(self) -> int:
        return len(self.trial_ns)

    @property
    def total_ns(self) -> int:
        return sum(self.trial_ns)

    @property
    def mean_ns(self) -> float:
        return self.total_ns / (self.repeats * self.queries_per_trial)

    @property
    def median_ns(self) -> float:
        return statistics.median(self.per_search_ns)

    @property
    def min_ns(self) -> float:
        return min(self.per_search_ns)

    @property
    def max_ns(self) -> float:
        return max(self.per_search_ns)


@dataclass(frozen=True)
class StructureStats:
    """Hash-table shape for one dataset size (computed, not timed)."""

    dataset_size: int
    capacity: int
    load_factor: float
    mean_probes_hit: float
    mean_probes_miss: float


def linear_search(products: Sequence[PharmacyProduct], product_id: int) -> Optional[PharmacyProduct]:
    """Sequential scan of the one-dimensional array (list)."""
    for product in products:
        if product.product_id == product_id:
            return product
    return None


_NAME_STEMS = ("Paracetamol", "Ibuprofen", "Amoxicillin", "Cetirizine", "Omeprazole",
               "Loratadine", "Metformin", "Atorvastatin", "Salbutamol", "Vitamin C")
_CATEGORIES = ("Analgesic", "Antibiotic", "Antihistamine", "Gastrointestinal",
               "Respiratory", "Supplement", "Cardiovascular", "Diabetes")


def generate_products(count: int, rng: random.Random) -> List[PharmacyProduct]:
    """Synthetic dataset with unique, shuffled integer IDs (seeded RNG)."""
    ids = rng.sample(range(10_000, 10_000 + 20 * count), count)
    return [PharmacyProduct(
        product_id=pid,
        name=f"{rng.choice(_NAME_STEMS)} {rng.choice((50, 100, 250, 500))}mg batch {i}",
        price=Decimal(rng.randint(100, 9999)) / 100,
        stock_quantity=rng.randint(0, 500),
        category=rng.choice(_CATEGORIES)) for i, pid in enumerate(ids)]


def build_structures(products: Sequence[PharmacyProduct]
                     ) -> Tuple[LinearProbingHashTable[int, PharmacyProduct], List[PharmacyProduct]]:
    """Insert references to the SAME records into both structures."""
    table: LinearProbingHashTable[int, PharmacyProduct] = LinearProbingHashTable(
        capacity=11, max_load_factor=EXPERIMENT_MAX_LOAD_FACTOR)
    array: List[PharmacyProduct] = []
    for product in products:
        table.insert(product.key, product)
        array.append(product)
    return table, array


def verify_equivalence(table: LinearProbingHashTable[int, PharmacyProduct],
                       array: Sequence[PharmacyProduct], query_ids: Sequence[int]) -> None:
    """Both structures must return the identical record (or None) for every query."""
    for pid in query_ids:
        if table.search(pid) is not linear_search(array, pid):
            raise RuntimeError(f"structures disagree for product id {pid}")


def _run_batch(search_fn: Callable[[int], object], query_ids: Sequence[int]) -> None:
    for pid in query_ids:
        search_fn(pid)


def _timed_batch(search_fn: Callable[[int], object], query_ids: Sequence[int]) -> int:
    """Time one batch. Only the search loop lies between the two timer reads."""
    start = time.perf_counter_ns()
    for pid in query_ids:
        search_fn(pid)
    return time.perf_counter_ns() - start


def measure_series(search_fn: Callable[[int], object], query_ids: Sequence[int],
                   repeats: int, warmup: int) -> List[int]:
    """Warm up (untimed), then record `repeats` raw batch durations in ns.

    The cyclic garbage collector is paused during timing so a collection does
    not land inside a measurement; it is restored afterwards.
    """
    for _ in range(warmup):
        _run_batch(search_fn, query_ids)
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        return [_timed_batch(search_fn, query_ids) for _ in range(repeats)]
    finally:
        if gc_was_enabled:
            gc.enable()


def _mean_probes(table: LinearProbingHashTable[int, PharmacyProduct], ids: Sequence[int]) -> float:
    return sum(table.locate(pid).probes for pid in ids) / len(ids)


def run_experiment(config: ExperimentConfig) -> Tuple[List[TimingSeries], List[StructureStats]]:
    """Run every dataset size; return raw series and structure statistics."""
    all_series: List[TimingSeries] = []
    all_stats: List[StructureStats] = []
    for size in config.sizes:
        rng = random.Random(config.seed + size)
        products = generate_products(size, rng)
        table, array = build_structures(products)                 # NOT timed
        ids = [p.product_id for p in products]
        hit_ids = rng.sample(ids, min(config.queries, size))
        top = 10_000 + 20 * size
        miss_ids = [top + i for i in range(len(hit_ids))]          # guaranteed absent

        verify_equivalence(table, array, hit_ids + miss_ids)       # correctness first
        sample = hit_ids[0]
        info = table.locate(sample)
        print(f"\n[size={size}] equivalence verified for {len(hit_ids) + len(miss_ids)} queries. "
              f"Sample id {sample}: hash table slot {info.index} ({info.probes} probe(s)); "
              f"list index {array.index(table.search(sample))}; same record: "
              f"{table.search(sample) is linear_search(array, sample)}")
        all_stats.append(StructureStats(size, table.capacity, table.load_factor(),
                                        _mean_probes(table, hit_ids), _mean_probes(table, miss_ids)))

        searchers = {"hash_table": table.search,
                     "list": functools.partial(linear_search, array)}
        for scenario, queries in (("successful", hit_ids), ("unsuccessful", miss_ids)):
            for structure, fn in searchers.items():
                print(f"  timing {scenario:<12} {structure:<10} ...", flush=True)
                trials = measure_series(fn, queries, config.repeats, config.warmup)
                all_series.append(TimingSeries(size, scenario, structure, len(queries), trials))
    return all_series, all_stats


CSV_FIELDS = ["record_type", "dataset_size", "scenario", "structure", "trial",
              "queries_per_trial", "elapsed_ns", "ns_per_search", "repeats",
              "total_elapsed_ns", "mean_ns_per_search", "median_ns_per_search",
              "min_ns_per_search", "max_ns_per_search", "seed", "warmup_batches",
              "ht_capacity", "ht_load_factor", "ht_mean_probes_hit", "ht_mean_probes_miss"]


def write_csv(path: Path, config: ExperimentConfig, series: Sequence[TimingSeries],
              stats: Sequence[StructureStats]) -> None:
    """Write raw trial rows, then summary rows, to one CSV (record_type column)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    by_size = {s.dataset_size: s for s in stats}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for s in series:
            for trial, ns in enumerate(s.trial_ns, start=1):
                writer.writerow({
                    "record_type": "raw", "dataset_size": s.dataset_size,
                    "scenario": s.scenario, "structure": s.structure, "trial": trial,
                    "queries_per_trial": s.queries_per_trial, "elapsed_ns": ns,
                    "ns_per_search": f"{ns / s.queries_per_trial:.3f}"})
        for s in series:
            st = by_size[s.dataset_size]
            writer.writerow({
                "record_type": "summary", "dataset_size": s.dataset_size,
                "scenario": s.scenario, "structure": s.structure,
                "queries_per_trial": s.queries_per_trial, "repeats": s.repeats,
                "total_elapsed_ns": s.total_ns,
                "mean_ns_per_search": f"{s.mean_ns:.3f}",
                "median_ns_per_search": f"{s.median_ns:.3f}",
                "min_ns_per_search": f"{s.min_ns:.3f}",
                "max_ns_per_search": f"{s.max_ns:.3f}",
                "seed": config.seed, "warmup_batches": config.warmup,
                "ht_capacity": st.capacity, "ht_load_factor": f"{st.load_factor:.4f}",
                "ht_mean_probes_hit": f"{st.mean_probes_hit:.3f}",
                "ht_mean_probes_miss": f"{st.mean_probes_miss:.3f}"})


def format_summary(series: Sequence[TimingSeries], stats: Sequence[StructureStats]) -> str:
    """Compact printable summary of the measured data (no conclusions drawn)."""
    header = (f"{'size':>7} {'scenario':<12} {'structure':<10} {'repeats':>7} "
              f"{'total ns':>16} {'mean ns/search':>15} {'median ns/search':>17} "
              f"{'min':>12} {'max':>12}")
    lines = [header, "-" * len(header)]
    for s in series:
        lines.append(f"{s.dataset_size:>7} {s.scenario:<12} {s.structure:<10} {s.repeats:>7} "
                     f"{s.total_ns:>16,} {s.mean_ns:>15,.1f} {s.median_ns:>17,.1f} "
                     f"{s.min_ns:>12,.1f} {s.max_ns:>12,.1f}")
    lines.append("\nHash-table shape (computed, not timed):")
    for st in stats:
        lines.append(f"  n={st.dataset_size}: capacity={st.capacity}, load factor={st.load_factor:.3f}, "
                     f"mean probes hit={st.mean_probes_hit:.2f}, miss={st.mean_probes_miss:.2f}")
    lines.append("\nRatio of mean ns/search (list / hash_table), a measured number only:")
    index = {(s.dataset_size, s.scenario, s.structure): s for s in series}
    for s in series:
        if s.structure == "hash_table":
            other = index[(s.dataset_size, s.scenario, "list")]
            lines.append(f"  n={s.dataset_size:<7} {s.scenario:<12} ratio = {other.mean_ns / s.mean_ns:,.2f}")
    return "\n".join(lines)


def run_and_report(config: ExperimentConfig, csv_path: Path) -> None:
    print("Search-performance experiment")
    print(f"  sizes={list(config.sizes)} queries/batch={config.queries} repeats={config.repeats} "
          f"warm-up batches={config.warmup} seed={config.seed}")
    print(f"  Python {sys.version.split()[0]} ({sys.implementation.name})")
    series, stats = run_experiment(config)
    write_csv(csv_path, config, series, stats)
    print("\n" + format_summary(series, stats))
    print(f"\nRaw and summary data written to: {csv_path}")


# =========================================================================== #
# 5. COMMAND-LINE INTERFACE
# =========================================================================== #
class ActionCancelled(Exception):
    """User typed 'cancel' at a prompt."""


class InventoryCLI:
    """Menu UI. Reads/prints only; all rules live in the service and the table."""

    MENU = ("1. Insert product", "2. Search product", "3. Edit product",
            "4. Delete product", "5. Display inventory / hash-table slots",
            "6. Run search-performance experiment", "7. Exit")

    def __init__(self, system: PharmacyInventorySystem,
                 input_fn: Callable[[str], str] = input,
                 output_fn: Callable[[str], None] = print) -> None:
        self._system = system
        self._input = input_fn
        self._print = output_fn

    # ------------------------------------------------------------ input helpers
    def _ask(self, prompt: str, allow_blank: bool = False) -> str:
        while True:
            text = self._input(f"{prompt}: ").strip()
            if text.lower() == "cancel":
                raise ActionCancelled
            if text or allow_blank:
                return text
            self._print("  Input cannot be empty (type 'cancel' to abort).")

    def _ask_int(self, prompt: str, minimum: int, allow_blank: bool = False) -> Optional[int]:
        while True:
            text = self._ask(prompt, allow_blank)
            if not text:
                return None
            try:
                value = int(text)
            except ValueError:
                self._print("  Please enter a whole number.")
                continue
            if value < minimum:
                self._print(f"  Value must be at least {minimum}.")
                continue
            return value

    def _ask_price(self, prompt: str, allow_blank: bool = False) -> Optional[Decimal]:
        while True:
            text = self._ask(prompt, allow_blank)
            if not text:
                return None
            try:
                price = parse_price(text)
                # Validate through the entity so rules live in one place.
                PharmacyProduct(1, "x", price, 0, "x")
                return price
            except (TypeError, ValueError) as error:
                self._print(f"  Invalid price: {error}")

    def _show_placement(self, info: PlacementInfo) -> None:
        if info.displacement:
            self._print(f"  Stored in slot {info.index} (home slot {info.home_index}; "
                        f"collision resolved by linear probing, +{info.displacement}).")
        else:
            self._print(f"  Stored in slot {info.index} (home slot, no collision).")

    # ------------------------------------------------------------------ actions
    def _insert(self) -> None:
        product_id = self._ask_int("Product ID (positive integer)", 1)
        assert product_id is not None
        if self._system.find_product(product_id).found:
            self._print(f"  Error: product ID {product_id} already exists. Use Edit instead.")
            return
        name = self._ask("Name")
        category = self._ask("Category")
        price = self._ask_price(f"Price ({CURRENCY})")
        quantity = self._ask_int("Stock quantity (0 or more)", 0)
        assert price is not None and quantity is not None
        try:
            product = PharmacyProduct(product_id, name, price, quantity, category)
            before = self._system.capacity
            info = self._system.add_product(product)
        except (TypeError, ValueError) as error:
            self._print(f"  Invalid product: {error}")
            return
        except HashTableError as error:
            self._print(f"  Cannot insert: {error}")
            return
        self._print(f"  Inserted: {product}")
        self._show_placement(info)
        if self._system.capacity != before:
            self._print(f"  Table resized {before} -> {self._system.capacity} and all entries rehashed.")

    def _search(self) -> None:
        product_id = self._ask_int("Product ID to search", 1)
        assert product_id is not None
        result = self._system.find_product(product_id)
        if not result.found:
            self._print(f"  Product {product_id} not found "
                        f"({result.probes} slot(s) examined, home slot {result.home_index}).")
            return
        self._print(f"  Found: {result.value}")
        self._print(f"  Located in slot {result.index} after {result.probes} probe(s) "
                    f"(home slot {result.home_index}).")

    def _edit(self) -> None:
        product_id = self._ask_int("Product ID to edit", 1)
        assert product_id is not None
        result = self._system.find_product(product_id)
        if not result.found:
            self._print(f"  Product {product_id} not found.")
            return
        self._print(f"  Current: {result.value}")
        self._print("  Press Enter to keep a field unchanged.")
        name = self._ask("New name", True) or None
        category = self._ask("New category", True) or None
        price = self._ask_price("New price", True)
        quantity = self._ask_int("New stock quantity", 0, True)
        try:
            updated = self._system.edit_product(
                product_id, name=name, price=price, stock_quantity=quantity, category=category)
        except (TypeError, ValueError, HashTableError) as error:
            self._print(f"  Edit rejected: {error}")
            return
        self._print(f"  Updated: {updated}")

    def _delete(self) -> None:
        product_id = self._ask_int("Product ID to delete", 1)
        assert product_id is not None
        result = self._system.find_product(product_id)
        if not result.found:
            self._print(f"  Product {product_id} not found.")
            return
        self._print(f"  About to delete: {result.value}")
        if self._ask("Type 'yes' to confirm").lower() != "yes":
            self._print("  Deletion cancelled.")
            return
        self._system.remove_product(product_id)
        self._print(f"  Deleted. Slot {result.index} now holds a tombstone.")

    def _display(self) -> None:
        self._print(self._system.render_table())

    def _experiment(self) -> None:
        text = self._ask(f"Dataset sizes, comma-separated [default {','.join(map(str, DEFAULT_SIZES))}]", True)
        try:
            sizes = tuple(int(p) for p in text.split(",")) if text else DEFAULT_SIZES
            config = ExperimentConfig(sizes=sizes)
        except ValueError as error:
            self._print(f"  Invalid configuration: {error}")
            return
        run_and_report(config, DEFAULT_CSV_PATH)

    # --------------------------------------------------------------------- loop
    def run(self) -> None:
        actions = {"1": self._insert, "2": self._search, "3": self._edit,
                   "4": self._delete, "5": self._display, "6": self._experiment}
        self._print("\n" + "=" * 62)
        self._print("  PHARMACY INVENTORY SYSTEM  (hash table, linear probing)")
        self._print("=" * 62)
        self._print(f"  {self._system.summary_line()}")
        self._print("  Tip: type 'cancel' at any prompt to abort the current action.")
        try:
            while True:
                self._print("\n" + "\n".join(f"  {item}" for item in self.MENU))
                choice = self._input("Choose an option (1-7): ").strip()
                if choice == "7":
                    break
                action = actions.get(choice)
                if action is None:
                    self._print("  Invalid choice. Please enter a number from 1 to 7.")
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
    """Collision, tombstone, slot reuse and resize demonstration (no user input).

    A looser load-factor threshold (0.8) is used so the tombstone-reuse step is
    not pre-empted by a rehash; tombstones count towards the resize trigger.
    """
    system = PharmacyInventorySystem(max_load_factor=0.8)
    system.load_sample_data()
    print("1) Pre-loaded table (IDs 1001, 1014, 1027 share home slot 0 when capacity=13):")
    print(system.render_table())

    print("\n2) Search for a product that was displaced by a collision (ID 1014):")
    result = system.find_product(1014)
    print(f"   found={result.found} slot={result.index} home={result.home_index} probes={result.probes}")

    print("\n3) Delete ID 1002 (sits inside the probe chain of 1014) -> tombstone:")
    system.remove_product(1002)
    print(system.render_table())

    print("\n4) Search ID 1014 again - still found because the tombstone keeps the chain intact:")
    result = system.find_product(1014)
    print(f"   found={result.found} slot={result.index} probes={result.probes}")

    print("\n5) Insert ID 1040 (home slot 0): the first tombstone on its chain is reused:")
    info = system.add_product(PharmacyProduct(1040, "Aspirin 100mg Tablets", Decimal("5.10"), 70, "Analgesic"))
    print(f"   placed in slot {info.index} (home {info.home_index}, +{info.displacement})")

    print("\n6) Edit ID 1001 (price and stock) and show the stored record:")
    print("   ", system.edit_product(1001, price=Decimal("4.80"), stock_quantity=100))

    print("\n7) Insert products until the load-factor threshold forces a resize:")
    next_id = 5001
    start_capacity = system.capacity
    while system.capacity == start_capacity:
        system.add_product(PharmacyProduct(next_id, f"Demo item {next_id}", Decimal("1.00"), 1, "Demo"))
        next_id += 1
    print(f"   capacity {start_capacity} -> {system.capacity} after inserting ID {next_id - 1}")
    print(system.render_table())


# =========================================================================== #
# Entry point
# =========================================================================== #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="5022CMD Q1 - hash table pharmacy inventory")
    parser.add_argument("--demo", action="store_true", help="run the deterministic demo and exit")
    parser.add_argument("--experiment", action="store_true", help="run the search experiment and exit")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES), help="dataset sizes")
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES, help="IDs searched per timed batch")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="timed batches per series")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="untimed warm-up batches")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed for the dataset")
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV_PATH, help="CSV output path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.demo:
        run_demo()
        return 0
    if args.experiment:
        try:
            config = ExperimentConfig(tuple(args.sizes), args.queries, args.repeats, args.warmup, args.seed)
        except ValueError as error:
            print(f"Invalid experiment settings: {error}", file=sys.stderr)
            return 2
        run_and_report(config, args.output)
        return 0
    system = PharmacyInventorySystem()
    system.load_sample_data()
    InventoryCLI(system).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
