
from __future__ import annotations
import argparse
import copy
import csv
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SPEED_KMPH = 30.0
CAPACITY_M3 = 7.2
FIXED_COST = 90.0
TRAVEL_COST_PER_KM = 7.5
LATENESS_COST_PER_HOUR = 10.0
EPOCH_HOURS = 1.0
SEED = 20261015

ORDER_ID = 'Order number'
CALL = 'Call-in time'
CX, CY = 'Horizontal coordinate of customer node', 'Vertical coordinate of customer node'
MX, MY = 'Horizontal coordinate of merchant node', 'Vertical coordinate of merchant node'
PE, PL = 'Pickup time window (early)', 'Pickup time window (late)'
DE, DL = 'Delivery time window (early)', 'Delivery time window (late)'
DIST = 'Distance between customer and merchant'
TRAVEL_H = 'Time consuming (h)'
VOL = 'Volume of goods (m3)'
SERVICE = 'Loading and unloading time (h)'
VID, VEPOCH = 'Vehicle number', 'Updating Epoch'
VX, VY = 'Horizontal coordinate of vehicle node', 'Vertical coordinate of vehicle node'


def elapsed_fmt(hours: Optional[float]) -> str:
    """Format elapsed hours as HH:MM, never round minutes to 60 without carry."""
    if hours is None or not math.isfinite(float(hours)):
        return '-'
    total_minutes = max(0, int(round(float(hours) * 60)))
    return f'{total_minutes // 60:02d}:{total_minutes % 60:02d}'


def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Plan:
    departure: float
    pickup_eta: float
    delivery_eta: float
    travel_distance: float
    pickup_late_h: float
    delivery_late_h: float


@dataclass
class Vehicle:
    vid: int
    epoch: int
    start_pos: Tuple[float, float]
    current_pos: Tuple[float, float]
    route: List[int]
    activated: bool = False


class CFDRPSimulator:
    def __init__(self, orders_file: str, vehicles_file: str, report_file: str,
                 history_file: str, population: int = 80, generations: int = 80,
                 sa_iterations: int = 1500, seed: int = SEED):
        import pandas as pd
        self.orders_file = Path(orders_file)
        self.vehicles_file = Path(vehicles_file)
        self.orders_df = pd.read_csv(self.orders_file)
        self.vehicles_df = pd.read_csv(self.vehicles_file)
        required_orders = {ORDER_ID, CALL, CX, CY, MX, MY, PE, PL, DE, DL, DIST, TRAVEL_H, VOL, SERVICE}
        required_vehicles = {VID, VEPOCH, VX, VY}
        missing_orders = sorted(required_orders - set(self.orders_df.columns))
        missing_vehicles = sorted(required_vehicles - set(self.vehicles_df.columns))
        if missing_orders:
            raise ValueError(f'Orders CSV is missing required columns: {missing_orders}')
        if missing_vehicles:
            raise ValueError(f'Vehicles CSV is missing required columns: {missing_vehicles}')
        if self.orders_df.empty:
            raise ValueError('Orders CSV contains no orders.')
        if self.vehicles_df.empty:
            raise ValueError('Vehicles CSV contains no vehicle update rows.')
        if self.orders_df[ORDER_ID].isna().any() or self.orders_df[ORDER_ID].duplicated().any():
            raise ValueError(f'Orders CSV column {ORDER_ID!r} must contain unique, non-empty IDs.')
        if self.vehicles_df[[VID, VEPOCH]].isna().any().any():
            raise ValueError('Vehicles CSV must contain non-empty vehicle IDs and updating epochs.')
        self.orders_df = self.orders_df.sort_values([CALL, ORDER_ID]).reset_index(drop=True)
        self.vehicles_df = self.vehicles_df.sort_values([VEPOCH, VID]).reset_index(drop=True)
        self.report_file = Path(report_file)
        self.history_file = Path(history_file)
        self.algorithm_file = self.report_file.with_name(f'{self.report_file.stem}_algorithm_comparison.csv')
        self.population_size = max(8, int(population))
        self.generations = max(1, int(generations))
        self.sa_iterations = max(100, int(sa_iterations))
        self.random = random.Random(seed)
        self.seed = seed
        self.orders: Dict[int, dict] = {int(r[ORDER_ID]): r.to_dict() for _, r in self.orders_df.iterrows()}
        # Keep every update record grouped by epoch. A vehicle ID may occur in
        # more than one epoch in replacement datasets, so do not collapse rows
        # into a single dictionary keyed only by vehicle ID.
        self.vehicle_updates_by_epoch: Dict[int, List[dict]] = {}
        for _, row in self.vehicles_df.iterrows():
            record = row.to_dict()
            self.vehicle_updates_by_epoch.setdefault(int(record[VEPOCH]), []).append(record)
        self.vehicles: Dict[int, Vehicle] = {}
        self.plans: Dict[int, Plan] = {}
        self.assignment: Dict[int, int] = {}
        self.assignment_epoch: Dict[int, int] = {}
        self.delivered: Dict[int, float] = {}
        self.locked: Dict[int, List[int]] = {}
        self.history: List[dict] = []
        self.algorithm_rows: List[dict] = []
        self.last_epoch = int(self.vehicles_df[VEPOCH].max())
        self.max_order_id = max(self.orders)
        self.epoch_delivery_counts: Dict[int, int] = {}
        self._distance_between = {}
        self._initialize_distance_cache()
        oversized = sorted((oid, vol) for oid, vol in self.volume.items() if vol > CAPACITY_M3 + 1e-9)
        if oversized:
            detail = ', '.join(f'order {oid}: {vol:.3f} m3' for oid, vol in oversized[:10])
            raise ValueError(
                f'Order volume exceeds vehicle capacity {CAPACITY_M3:.1f} m3 ({detail}). '
                'This vehicle cannot serve those orders without a split/other vehicle model.'
            )
        self._initialize_objective_weights()

    def _initialize_distance_cache(self):
        # The input supplies the merchant-customer distance; all cross-order legs
        # are Euclidean because no road-network matrix is present in the CSV.
        self.merchant_xy = {oid: (float(o[MX]), float(o[MY])) for oid, o in self.orders.items()}
        self.customer_xy = {oid: (float(o[CX]), float(o[CY])) for oid, o in self.orders.items()}
        self.intra_distance = {oid: float(o[DIST]) for oid, o in self.orders.items()}
        self.intra_time = {oid: float(o[TRAVEL_H]) for oid, o in self.orders.items()}
        self.volume = {oid: float(o[VOL]) for oid, o in self.orders.items()}
        self.call_time = {oid: float(o[CALL]) for oid, o in self.orders.items()}
        self.pick_early = {oid: float(o[PE]) for oid, o in self.orders.items()}
        self.pick_late = {oid: float(o[PL]) for oid, o in self.orders.items()}
        self.del_early = {oid: float(o[DE]) for oid, o in self.orders.items()}
        self.del_late = {oid: float(o[DL]) for oid, o in self.orders.items()}
        self.service_half = {oid: float(o[SERVICE]) / 2.0 for oid, o in self.orders.items()}
        for a in self.orders:
            for b in self.orders:
                self._distance_between[(a, b)] = euclidean(self.customer_xy[a], self.merchant_xy[b])

    def _initialize_objective_weights(self):
        """Build bounded lexicographic weights for window feasibility first.

        Priority: (1) fewer orders with any pickup/delivery window violation,
        (2) lower total lateness in 0.1-minute ticks, (3) lower monetary cost.
        This makes the search try to save more orders from being late before
        minimizing the size of lateness on orders that still miss a window.
        """
        n = max(1, len(self.orders))
        n_vehicles = max(1, int(self.vehicles_df[VID].nunique()))
        time_cols = [CALL, PE, PL, DE, DL]
        max_abs_time = 0.0
        for col in time_cols:
            if col in self.orders_df:
                values = [abs(float(v)) for v in self.orders_df[col].dropna().tolist()]
                if values:
                    max_abs_time = max(max_abs_time, max(values))
        max_abs_time = max(max_abs_time, float(self.last_epoch + 1), 1.0)

        start_positions = list(self.customer_xy.values())
        start_positions.extend(
            (float(row[VX]), float(row[VY]))
            for _, row in self.vehicles_df.iterrows()
        )
        merchant_positions = list(self.merchant_xy.values())
        max_leg_km = max(
            (euclidean(a, b) for a in start_positions for b in merchant_positions),
            default=0.0,
        )
        max_travel_h = max((max(0.0, t) for t in self.intra_time.values()), default=0.0)
        max_service_h = max((max(0.0, float(o[SERVICE])) for o in self.orders.values()), default=0.0)

        # Conservative bounds used only to separate the three priorities.
        max_step_h = 4.0 * max_abs_time + max_leg_km / SPEED_KMPH + max_travel_h + max_service_h + 1.0
        max_eta_h = max_abs_time + n * max_step_h
        max_lateness_hours = 2.0 * n * (max_eta_h + max_abs_time)
        max_lateness_ticks = int(math.ceil(max_lateness_hours * 600.0)) + 1
        max_lateness_cost = max_lateness_hours * LATENESS_COST_PER_HOUR
        distance_bound = n * max_leg_km + sum(max(0.0, d) for d in self.intra_distance.values())
        max_money_cost = (n_vehicles * FIXED_COST
                          + distance_bound * TRAVEL_COST_PER_KM
                          + max_lateness_cost + 1.0)

        self.money_cost_upper_bound = max(1.0, max_money_cost)
        # One 0.1-minute lateness tick dominates every possible money-cost change.
        self.lateness_tick_weight = self.money_cost_upper_bound + 1.0
        # One additional violating order dominates all possible lateness ticks and money.
        self.order_count_weight = (
            max_lateness_ticks * self.lateness_tick_weight
            + self.money_cost_upper_bound + 1.0
        )

    def _objective_score(self, monetary_cost: float, total_lateness_hours: float,
                         violating_orders: int) -> float:
        lateness_ticks = int(math.ceil(max(0.0, total_lateness_hours) * 600.0 - 1e-9))
        return (
            max(0, int(violating_orders)) * self.order_count_weight
            + lateness_ticks * self.lateness_tick_weight
            + monetary_cost
        )

    def _add_vehicles_for_epoch(self, epoch: int, now: float):
        # Add vehicles on first appearance. If a replacement CSV contains a
        # later position update for an existing vehicle ID, refresh its current
        # position while preserving its route and any already-locked delivery.
        for row in self.vehicle_updates_by_epoch.get(epoch, []):
            vid = int(row[VID])
            pos = (float(row[VX]), float(row[VY]))
            if vid not in self.vehicles:
                self.vehicles[vid] = Vehicle(vid, epoch, pos, pos, [])
            else:
                self.vehicles[vid].current_pos = pos

    def _plan_one(self, oid: int, pos: Tuple[float, float], start_time: float) -> Plan:
        o = self.orders[oid]
        leg1 = euclidean(pos, self.merchant_xy[oid])
        depart = start_time
        pickup_arrival = start_time + leg1 / SPEED_KMPH
        pickup_start = max(pickup_arrival, self.call_time[oid], self.pick_early[oid])
        pickup_finish = pickup_start + self.service_half[oid]
        pickup_late = max(0.0, pickup_finish - self.pick_late[oid])
        customer_arrival = pickup_finish + self.intra_time[oid]
        delivery_start = max(customer_arrival, self.del_early[oid])
        delivery_finish = delivery_start + self.service_half[oid]
        delivery_late = max(0.0, delivery_finish - self.del_late[oid])
        return Plan(departure=depart, pickup_eta=pickup_finish,
                    delivery_eta=delivery_finish,
                    travel_distance=leg1 + self.intra_distance[oid],
                    pickup_late_h=pickup_late, delivery_late_h=delivery_late)

    def _process_completed_at(self, now: float):
        """Commit completed deliveries and lock each vehicle's active route prefix."""
        self.locked = {}
        newly = 0
        for vid, vehicle in sorted(self.vehicles.items()):
            while vehicle.route:
                oid = vehicle.route[0]
                p = self.plans.get(oid)
                if p is None or p.delivery_eta > now + 1e-9:
                    break
                self.delivered.setdefault(oid, p.delivery_eta)
                vehicle.current_pos = self.customer_xy[oid]
                vehicle.activated = True
                vehicle.route.pop(0)
                newly += 1

            # Preserve the full started prefix. In a pickup batch, several orders
            # share the dispatch time and cannot be reassigned once the route starts.
            locked_prefix = []
            for oid in vehicle.route:
                p = self.plans.get(oid)
                if p is not None and p.departure <= now + 1e-9 and p.delivery_eta > now + 1e-9:
                    locked_prefix.append(oid)
                else:
                    break
            if locked_prefix:
                self.locked[vid] = locked_prefix
        self.epoch_delivery_counts[len(self.epoch_delivery_counts) + 1] = newly

    def _pending_ids(self, now: float) -> List[int]:
        return sorted([oid for oid, o in self.orders.items()
                       if self.call_time[oid] <= now + 1e-9 and oid not in self.delivered])

    def _revealed_ids(self, at_time: float) -> List[int]:
        return sorted([oid for oid in self.orders if self.call_time[oid] <= at_time + 1e-9])

    def _routes_from_genome(self, ids: List[int], assignments: Dict[int, int], perm: List[int]) -> Dict[int, List[int]]:
        routes = {vid: [] for vid in self.vehicles}
        for oid in perm:
            vid = assignments[oid]
            routes[vid].append(oid)
        # Started jobs, including outstanding items in a started batch, stay
        # at the head of their original vehicle route in committed order.
        for vid, locked_ids in self.locked.items():
            if vid not in routes:
                continue
            for oid in locked_ids:
                if oid in routes[vid]:
                    routes[vid].remove(oid)
            routes[vid][0:0] = locked_ids
        return routes

    def _evaluate(self, ids: List[int], assignments: Dict[int, int], perm: List[int], now: float,
                  return_plans: bool = False):
        """Evaluate routes with same-merchant pickup batching subject to capacity.

        The CSV lacks a road-network matrix. Vehicle-to-merchant and customer-to-
        customer legs therefore use Euclidean distance at SPEED_KMPH; each batch's
        first merchant-to-customer leg uses the input distance/time.
        """
        routes = self._routes_from_genome(ids, assignments, perm)
        monetary_cost = 0.0
        total_lateness_hours = 0.0
        violating_orders = 0
        plans = {}

        for vid, route in routes.items():
            if not route:
                continue
            vehicle = self.vehicles[vid]
            if not vehicle.activated:
                monetary_cost += FIXED_COST

            locked_ids = self.locked.get(vid, [])
            locked_n = len(locked_ids)
            if locked_n and route[:locked_n] == locked_ids:
                for locked_oid in locked_ids:
                    plans[locked_oid] = self.plans[locked_oid]
                last_locked = locked_ids[-1]
                pos = self.customer_xy[last_locked]
                t = self.plans[last_locked].delivery_eta
                remaining = route[locked_n:]
            else:
                pos = vehicle.current_pos
                t = now
                remaining = route

            i = 0
            while i < len(remaining):
                first_oid = remaining[i]
                first_merchant = self.merchant_xy[first_oid]
                batch = [first_oid]
                batch_volume = self.volume[first_oid]
                j = i + 1
                while j < len(remaining):
                    candidate = remaining[j]
                    same_merchant = euclidean(first_merchant, self.merchant_xy[candidate]) <= 1e-6
                    new_volume = batch_volume + self.volume[candidate]
                    if not same_merchant or new_volume > CAPACITY_M3 + 1e-9:
                        break
                    batch.append(candidate)
                    batch_volume = new_volume
                    j += 1

                dispatch_time = t
                to_merchant_km = euclidean(pos, first_merchant)
                merchant_arrival = dispatch_time + to_merchant_km / SPEED_KMPH

                # Pickup all orders in the batch at the same merchant in route order.
                pickup_times = {}
                pickup_lateness = {}
                pickup_clock = merchant_arrival
                for oid in batch:
                    pickup_start = max(pickup_clock, self.call_time[oid], self.pick_early[oid])
                    pickup_finish = pickup_start + self.service_half[oid]
                    pickup_times[oid] = pickup_finish
                    pickup_lateness[oid] = max(0.0, pickup_finish - self.pick_late[oid])
                    pickup_clock = pickup_finish

                # Deliver each pickup in route order. The first merchant-customer
                # leg uses source-supplied time/distance; later customer legs use
                # Euclidean travel because no customer-customer matrix was supplied.
                delivery_clock = pickup_clock
                delivery_pos = first_merchant
                start_leg_unallocated = to_merchant_km
                for batch_index, oid in enumerate(batch):
                    if batch_index == 0:
                        delivery_leg_km = self.intra_distance[oid]
                        delivery_leg_h = self.intra_time[oid]
                    else:
                        delivery_leg_km = euclidean(delivery_pos, self.customer_xy[oid])
                        delivery_leg_h = delivery_leg_km / SPEED_KMPH

                    customer_arrival = delivery_clock + delivery_leg_h
                    delivery_start = max(customer_arrival, self.del_early[oid])
                    delivery_finish = delivery_start + self.service_half[oid]
                    delivery_lateness = max(0.0, delivery_finish - self.del_late[oid])

                    p = Plan(
                        departure=dispatch_time,
                        pickup_eta=pickup_times[oid],
                        delivery_eta=delivery_finish,
                        travel_distance=start_leg_unallocated + delivery_leg_km,
                        pickup_late_h=pickup_lateness[oid],
                        delivery_late_h=delivery_lateness,
                    )
                    plans[oid] = p

                    movement_km = start_leg_unallocated + delivery_leg_km
                    monetary_cost += movement_km * TRAVEL_COST_PER_KM
                    monetary_cost += (p.pickup_late_h + p.delivery_late_h) * LATENESS_COST_PER_HOUR
                    order_lateness = p.pickup_late_h + p.delivery_late_h
                    total_lateness_hours += order_lateness
                    if order_lateness > 1e-9:
                        violating_orders += 1

                    delivery_clock = delivery_finish
                    delivery_pos = self.customer_xy[oid]
                    start_leg_unallocated = 0.0

                t = delivery_clock
                pos = delivery_pos
                i = j

        score = self._objective_score(monetary_cost, total_lateness_hours, violating_orders)
        if return_plans:
            return score, routes, plans
        return score

    def _flatten_routes(self, routes: Dict[int, List[int]]) -> List[int]:
        return [oid for vid in sorted(routes) for oid in routes[vid]]

    def _make_greedy_seed(self, ids: List[int], old_assign: Dict[int, int], new_ids: List[int], now: float):
        assignments = {oid: old_assign[oid] for oid in ids if oid in old_assign}
        routes = {vid: [oid for oid in self.vehicles[vid].route if oid in assignments]
                  for vid in self.vehicles}
        for vid, locked_ids in self.locked.items():
            if vid in routes:
                for oid in locked_ids:
                    if oid in routes[vid]:
                        routes[vid].remove(oid)
                routes[vid][0:0] = locked_ids

        # Insert by urgency and score all legal positions. Adjacent orders at the
        # same merchant can then form a capacity-feasible pickup batch.
        def urgency(oid):
            return (self.del_late[oid], self.pick_late[oid], self.call_time[oid], oid)

        for oid in sorted(new_ids, key=urgency):
            best = None
            for vid in sorted(self.vehicles):
                base_route = routes[vid]
                locked_prefix = self.locked.get(vid, [])
                first_pos = len(locked_prefix) if base_route[:len(locked_prefix)] == locked_prefix else 0
                for posi in range(first_pos, len(base_route) + 1):
                    trial_routes = {v: list(r) for v, r in routes.items()}
                    trial_routes[vid].insert(posi, oid)
                    trial_assign = dict(assignments)
                    trial_assign[oid] = vid
                    trial_ids = list(trial_assign)
                    perm = self._flatten_routes(trial_routes)
                    cost = self._evaluate(trial_ids, trial_assign, perm, now)
                    key = (cost, vid, posi)
                    if best is None or key < best[0]:
                        best = (key, vid, trial_routes, trial_assign)
            if best is None:
                raise RuntimeError(f'No vehicle candidate is available for order {oid}.')
            _, _, routes, assignments = best

        perm = self._flatten_routes(routes)
        for oid in ids:
            if oid not in assignments:
                raise RuntimeError(f'Greedy seed failed to assign order {oid}.')
        return assignments, perm

    def _normalize_genome(self, ids, assignments, perm):
        # Keep exactly one occurrence of every pending order ID.
        valid = set(ids)
        p = [o for o in perm if o in valid]
        seen = set(p)
        p.extend(o for o in ids if o not in seen)
        a = {o: assignments[o] for o in ids}
        for oid in ids:
            if a[oid] not in self.vehicles:
                a[oid] = min(self.vehicles)
        return a, p

    def _route_lists(self, ids, assignments, perm):
        routes = self._routes_from_genome(ids, assignments, perm)
        return routes

    def _apply_operator(self, assignments: Dict[int, int], perm: List[int], ids: List[int],
                        new_ids: List[int], op: Optional[str] = None):
        a, p = dict(assignments), list(perm)
        if not ids or not self.vehicles:
            return a, p
        if op is None:
            op = self.random.choice(['PDDV', 'PDSV', 'EUV', 'IPDV'])
        routes = self._routes_from_genome(ids, a, p)
        movable = [o for o in new_ids if o in a]
        if op == 'PDDV' and len(movable) >= 2:
            x, y = self.random.sample(movable, 2)
            a[x], a[y] = a[y], a[x]
        elif op == 'PDSV':
            candidates = []
            for v, route in routes.items():
                locked_prefix = self.locked.get(v, [])
                eligible = route[len(locked_prefix):] if route[:len(locked_prefix)] == locked_prefix else route
                if len(eligible) >= 2:
                    candidates.append((route, eligible))
            if candidates:
                r, eligible = self.random.choice(candidates)
                x, y = self.random.sample(eligible, 2)
                r[r.index(x)], r[r.index(y)] = y, x
                p = self._flatten_routes(routes)
        elif op == 'EUV' and movable:
            used = set(a.values())
            empty = [v for v in self.vehicles if v not in used]
            x = self.random.choice(movable)
            if empty:
                a[x] = self.random.choice(empty)
            else:
                # No empty vehicle: move an order to the current least-loaded route.
                counts = {v: sum(1 for z in a.values() if z == v) for v in self.vehicles}
                target = min(counts, key=lambda v: (counts[v], v))
                a[x] = target
        elif op == 'IPDV' and movable:
            x = self.random.choice(movable)
            target_options = [v for v in self.vehicles if v != a[x]]
            if target_options:
                target = self.random.choice(target_options)
                source = a[x]
                src_route = routes[source]
                if x in src_route:
                    src_route.remove(x)
                target_route = routes[target]
                target_locked = self.locked.get(target, [])
                locked_count = len(target_locked) if target_route[:len(target_locked)] == target_locked else 0
                insert_at = self.random.randint(locked_count, len(target_route))
                target_route.insert(insert_at, x)
                a[x] = target
                p = self._flatten_routes(routes)
        # Mutation/feasibility-repair: move a flexible (new) order toward an
        # empty or least-loaded vehicle; previous order-to-vehicle matches stay fixed.
        if self.random.random() < 0.10 and movable:
            candidate = self.random.choice(movable)
            used = set(a.values())
            empty = [v for v in self.vehicles if v not in used]
            counts = {v: sum(1 for z in a.values() if z == v) for v in self.vehicles}
            target = self.random.choice(empty) if empty else min(counts, key=lambda v: (counts[v], v))
            a[candidate] = target
        a, p = self._normalize_genome(ids, a, p)
        return a, p

    def _ox(self, p1: List[int], p2: List[int]) -> List[int]:
        n = len(p1)
        if n < 2:
            return list(p1)
        i, j = sorted(self.random.sample(range(n), 2))
        child = [None] * n
        child[i:j+1] = p1[i:j+1]
        seg = set(p1[i:j+1])
        fill = [x for x in p2 if x not in seg]
        slots = list(range(j+1, n)) + list(range(0, i))
        for slot, val in zip(slots, fill):
            child[slot] = val
        # If i=0 and j=n-1 the slots list is empty and the copied segment is full.
        return [int(x) for x in child]

    def _ipga(self, ids: List[int], old_assign: Dict[int, int], new_ids: List[int], now: float):
        seed_assign, seed_perm = self._make_greedy_seed(ids, old_assign, new_ids, now)
        base_cost = self._evaluate(ids, seed_assign, seed_perm, now)
        population = []
        population.append((base_cost, seed_assign, seed_perm))
        # Seed a mix of route-preserving and exploratory individuals.
        for i in range(self.population_size - 1):
            a, p = dict(seed_assign), list(seed_perm)
            if i % 3 == 0:
                # Randomly reroute only new orders, keeping old assignments pinned.
                for oid in new_ids:
                    if self.random.random() < 0.35:
                        a[oid] = self.random.choice(list(self.vehicles))
            elif i % 3 == 1:
                for oid in new_ids:
                    a[oid] = self.random.choice(list(self.vehicles))
            # Permutation variation: random shuffle affects route order within each vehicle.
            if len(p) > 1 and i % 4 != 0:
                p = list(p)
                self.random.shuffle(p)
            a, p = self._normalize_genome(ids, a, p)
            population.append((self._evaluate(ids, a, p, now), a, p))
        population.sort(key=lambda x: x[0])
        for generation in range(self.generations):
            population.sort(key=lambda x: x[0])
            next_pop = population[:2]  # elitism: never lose the best two solutions.
            # Rank-based selection avoids near-uniform probabilities when the
            # lexicographic time-window objective contains very large weights.
            population.sort(key=lambda x: x[0])
            weights = [1.0 / (rank + 1) for rank in range(len(population))]
            while len(next_pop) < self.population_size:
                p1 = self.random.choices(population, weights=weights, k=1)[0]
                p2 = self.random.choices(population, weights=weights, k=1)[0]
                child_perm = self._ox(p1[2], p2[2])
                child_assign = {oid: (p1[1][oid] if self.random.random() < 0.5 else p2[1][oid])
                                for oid in ids}
                # Pin assignments from prior epochs; they can never be transferred.
                for oid, vid in old_assign.items():
                    child_assign[oid] = vid
                child_assign, child_perm = self._apply_operator(child_assign, child_perm, ids, new_ids)
                child_cost = self._evaluate(ids, child_assign, child_perm, now)
                next_pop.append((child_cost, child_assign, child_perm))
            population = next_pop
        population.sort(key=lambda x: x[0])
        return population[0]

    def _sa(self, ids: List[int], old_assign: Dict[int, int], new_ids: List[int], now: float,
            starting_genome):
        current = (starting_genome[0], dict(starting_genome[1]), list(starting_genome[2]))
        current_cost = self._evaluate(ids, current[1], current[2], now)
        best = (current_cost, dict(current[1]), list(current[2]))
        temperature = 100.0
        t_min = 0.1
        cooling_rate = 0.9  # Q = 0.9 as reported in the paper.
        iterations_per_temperature = 100
        for iteration in range(self.sa_iterations):
            cand_assign, cand_perm = self._apply_operator(current[1], current[2], ids, new_ids)
            for oid, vid in old_assign.items():
                cand_assign[oid] = vid
            cand_cost = self._evaluate(ids, cand_assign, cand_perm, now)
            delta = cand_cost - current_cost
            if delta < 0 or self.random.random() < math.exp(-min(700.0, max(0.0, delta) / max(temperature, 1e-9))):
                current = (cand_cost, cand_assign, cand_perm)
                current_cost = cand_cost
            if cand_cost < best[0]:
                best = (cand_cost, dict(cand_assign), list(cand_perm))
            if (iteration + 1) % iterations_per_temperature == 0:
                temperature *= cooling_rate
            if temperature <= t_min and iteration > 0:
                break
        return best

    def _snapshot_status(self, oid: int, snapshot_time: float) -> str:
        if oid in self.delivered and self.delivered[oid] <= snapshot_time + 1e-9:
            return 'Delivered'
        p = self.plans.get(oid)
        if p is not None and p.delivery_eta <= snapshot_time + 1e-9:
            return 'Delivered'
        return 'In transit'

    def _snapshot_rows(self, epoch: int, label: str, snapshot_time: float, phase: str):
        # Keep an order in this epoch's snapshots even if it is delivered during
        # this same epoch, so its Delivered transition can be audited at END.
        # Once its event has been committed at an epoch boundary, remove it from
        # every snapshot in all later epochs (including the next START).
        epoch_start = float(epoch) * EPOCH_HOURS
        for oid in self._revealed_ids(snapshot_time):
            completed_at = self.delivered.get(oid)
            if completed_at is not None and completed_at <= epoch_start + 1e-9:
                continue
            o = self.orders[oid]
            vid = self.assignment.get(oid)
            p = self.plans.get(oid)
            status = self._snapshot_status(oid, snapshot_time)
            call_time = self.call_time[oid]
            pickup_eta = p.pickup_eta if p else None
            delivery_eta = p.delivery_eta if p else None
            elapsed = (delivery_eta - call_time) * 60.0 if delivery_eta is not None else None
            pickup_late = max(0.0, p.pickup_late_h) * 60.0 if p is not None else None
            delivery_late = max(0.0, p.delivery_late_h) * 60.0 if p is not None else None
            total_window_late = (pickup_late + delivery_late) if p is not None else None
            if p is None:
                window_status = 'NO_ETA'
            elif total_window_late <= 1e-6:
                window_status = 'FEASIBLE'
            else:
                window_status = 'INFEASIBLE'
            late = delivery_late
            rec = {
                'Epoch': epoch,
                'Snapshot': label,
                'Snapshot time (elapsed HH:MM)': elapsed_fmt(snapshot_time),
                'Snapshot hour': round(snapshot_time, 6),
                'Order number': oid,
                'Pickup point (merchant x,y)': f'({float(o[MX]):.3f},{float(o[MY]):.3f})',
                'Delivery point (customer x,y)': f'({float(o[CX]):.3f},{float(o[CY]):.3f})',
                'Order arrival time': elapsed_fmt(call_time),
                'Assigned driver': f'V{vid:02d}' if vid is not None else '-',
                'Assigned epoch': self.assignment_epoch.get(oid, '-'),
                'Pickup ETA': elapsed_fmt(pickup_eta),
                'Delivery ETA': elapsed_fmt(delivery_eta),
                'ETA elapsed from call-in (min)': round(elapsed, 1) if elapsed is not None else '-',
                'Pickup lateness (min)': round(pickup_late, 1) if pickup_late is not None else '-',
                'Delivery lateness (min)': round(delivery_late, 1) if delivery_late is not None else '-',
                'Total window violation (min)': round(total_window_late, 1) if total_window_late is not None else '-',
                'Time-window status': window_status,
                'Status': status,
                'Model phase': phase,
            }
            self.history.append(rec)

    def _apply_solution(self, ids: List[int], solution, now: float, epoch: int):
        _, assignments, perm = solution
        assignments, perm = self._normalize_genome(ids, assignments, perm)
        cost, routes, plans = self._evaluate(ids, assignments, perm, now, return_plans=True)

        # Update every pending order's selected vehicle. Only an order already
        # in progress is pinned; unstarted orders may move when new vehicles
        # become available in later update epochs.
        for oid in ids:
            selected_vid = assignments[oid]
            if oid not in self.assignment or self.assignment[oid] != selected_vid:
                self.assignment[oid] = selected_vid
                self.assignment_epoch[oid] = epoch

        # Commit route order and schedule for all still-pending jobs.
        for vid, vehicle in self.vehicles.items():
            vehicle.route = list(routes.get(vid, []))
        for oid, p in plans.items():
            self.plans[oid] = p

        # Do not allow a currently executing order to change drivers.
        for vid, locked_ids in self.locked.items():
            for oid in locked_ids:
                if oid in ids and self.assignment.get(oid) != vid:
                    raise RuntimeError(
                        f'Locked order {oid} moved from vehicle {vid}; refusing inconsistent plan.'
                    )
        return cost

    def _run_epoch(self, epoch: int):
        print(f'\n[Epoch {epoch}/{self.last_epoch}] starting optimization...', flush=True)
        now = float(epoch) * EPOCH_HOURS
        self._process_completed_at(now)
        self._add_vehicles_for_epoch(epoch, now)
        self._snapshot_rows(epoch, 'START', now, 'before epoch optimization')
        ids = self._pending_ids(now)

        # Rolling-horizon correction:
        # pin only orders currently in progress. A previously assigned order
        # that has NOT started may be reassigned to a newly available vehicle.
        locked_assign = {oid: vid for vid, locked_ids in self.locked.items() for oid in locked_ids if oid in ids}
        mutable_ids = [oid for oid in ids if oid not in locked_assign]
        newly_revealed = [oid for oid in ids if oid not in self.assignment]
        previous_assign = {oid: self.assignment[oid] for oid in ids if oid in self.assignment}

        t0 = time.perf_counter()
        ipga = self._ipga(ids, locked_assign, mutable_ids, now)
        ipga_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        sa_seed_assign, sa_seed_perm = self._make_greedy_seed(ids, locked_assign, mutable_ids, now)
        sa_seed = (self._evaluate(ids, sa_seed_assign, sa_seed_perm, now),
                   sa_seed_assign, sa_seed_perm)
        sa = self._sa(ids, locked_assign, mutable_ids, now, sa_seed)
        sa_time = time.perf_counter() - t1

        # Apply the better measured candidate under the same time-window-first
        # objective. Do not force IPGA if SA found a lower score.
        if sa[0] < ipga[0]:
            chosen, applied_solver = sa, 'SA'
        else:
            chosen, applied_solver = ipga, 'IPGA'
        chosen_cost = self._apply_solution(ids, chosen, now, epoch)
        reassigned = sum(
            1 for oid in ids
            if oid in previous_assign and self.assignment.get(oid) != previous_assign[oid]
        )

        self.algorithm_rows.append({
            'Epoch': epoch, 'Decision time': elapsed_fmt(now), 'Visible pending orders': len(ids),
            'Newly revealed orders assigned': len(newly_revealed),
            'Pending orders reassigned': reassigned,
            'Active vehicles available': len(self.vehicles),
            'IPGA window-priority objective score': round(ipga[0], 3),
            'SA window-priority objective score': round(sa[0], 3),
            'Applied solver': applied_solver, 'Applied window-priority objective score': round(chosen_cost, 3),
            'IPGA runtime seconds': round(ipga_time, 3), 'SA runtime seconds': round(sa_time, 3),
        })
        self._snapshot_rows(epoch, 'MIDDLE', now + 0.5, 'route plan after optimization; events executed to snapshot time')
        self._snapshot_rows(epoch, 'END', now + 1.0, 'route plan after optimization; events executed to snapshot time')
        print(
            f'[Epoch {epoch}/{self.last_epoch}] done | pending={len(ids)} | '
            f'newly_revealed={len(newly_revealed)} | reassigned={reassigned} | '
            f'vehicles={len(self.vehicles)} | applied={applied_solver} | '
            f'IPGA={ipga_time:.2f}s | SA={sa_time:.2f}s',
            flush=True
        )

    def _run_continuation(self):
        # No more vehicle rows or new orders after the last supplied update.
        # Keep the last committed plan unchanged and only execute its genuine ETAs.
        start = float(self.last_epoch + 1) * EPOCH_HOURS
        end_latest = max((p.delivery_eta for p in self.plans.values()), default=start)
        # Epoch k spans [k, k+1]; a delivery at 14:07 belongs to EPOCH 14, not 15.
        final_epoch = max(self.last_epoch, int(math.floor(end_latest / EPOCH_HOURS)))
        for epoch in range(self.last_epoch + 1, final_epoch + 1):
            now = float(epoch) * EPOCH_HOURS
            self._process_completed_at(now)
            self._snapshot_rows(epoch, 'START', now, 'execution-only continuation; no new vehicle/order input')
            self._snapshot_rows(epoch, 'MIDDLE', now + 0.5, 'execution-only continuation; no re-optimization')
            self._snapshot_rows(epoch, 'END', now + 1.0, 'execution-only continuation; no re-optimization')
        # Commit the remaining events at the end of the final recorded interval.
        self._process_completed_at(float(final_epoch + 1) * EPOCH_HOURS)

    def _format_report(self):
        lines = []
        lines += [
            '=' * 170,
            'CFDRP — CSV INPUT | ROLLING-HORIZON TERMINAL REPORT',
            '=' * 170,
            f'Orders source       : {self.orders_file.name}',
            f'Input orders file   : {self.orders_df.shape[0]} rows ({self.orders_df[ORDER_ID].min()}–{self.orders_df[ORDER_ID].max()})',
            f'Vehicles source     : {self.vehicles_file.name}',
            f'Input vehicles file : {self.vehicles_df.shape[0]} update rows; {self.vehicles_df[VID].nunique()} unique vehicle IDs',
            f'Update epochs       : {self.last_epoch}; epoch interval = {EPOCH_HOURS:.0f} hour',
            f'Algorithms          : IPGA-style population search (applied) + Simulated Annealing (comparison)',
            f'Run parameters      : population={self.population_size}, IPGA generations={self.generations}, SA max iterations={self.sa_iterations}, seed={self.seed}',
            'Paper algorithm settings: IPGA population=500, crossover=100%, mutation=0.1, max iterations=500; SA cooling Q=0.9.',
            'SA settings not recoverable from the attached PDF text layer are declared run settings: T0=100, Tmin=0.1, M=100 iterations/temperature, cap R=10000.',
            f'Paper cost settings : fixed vehicle={FIXED_COST:.2f} RMB, travel={TRAVEL_COST_PER_KM:.2f} RMB/km, lateness={LATENESS_COST_PER_HOUR:.2f} RMB/hour, speed={SPEED_KMPH:.1f} km/h, capacity={CAPACITY_M3:.1f} m3',
            'Time interpretation : Vehicle updating epoch k is a decision point at elapsed hour k; each interval is 1 hour.',
            'Status vocabulary   : Status column contains only "In transit" and "Delivered".',
            'No future knowledge  : At START, only revealed, not-yet-completed orders whose call-in time is <= snapshot time are displayed.',
            'Optimization epochs : one optimizer run per supplied vehicle update epoch; later epochs only execute the final plan.',
            'Continuation        : After the last vehicle update, no new vehicles are fabricated; the last planned routes simply execute until finish.',
            'Travel approximation: Euclidean vehicle-to-merchant/customer-to-customer legs; source distance/time for each batch first delivery leg.',
            'Capacity rule       : adjacent orders at the same merchant may be picked together up to 7.2 m3; deliveries then follow route order.',
            'Method scope        : restricted rolling-horizon IPGA/SA heuristic with same-merchant capacity batching; not a bit-for-bit reproduction of full article MATLAB/MILP.',
            '=' * 170,
        ]
        hist_by_key = {}
        for r in self.history:
            hist_by_key.setdefault((r['Epoch'], r['Snapshot']), []).append(r)
        for epoch in sorted({r['Epoch'] for r in self.history}):
            for label in ['START', 'MIDDLE', 'END']:
                rows = hist_by_key.get((epoch, label), [])
                if not rows:
                    continue
                snapshot_time = rows[0]['Snapshot time (elapsed HH:MM)']
                lines += ['', '#' * 170, f'EPOCH {epoch} | {label} | elapsed time {snapshot_time}', '#' * 170]
                header = (f"{'Order':>5} | {'Pickup merchant (x,y)':<23} | {'Customer (x,y)':<23} | {'Call-in':>7} | {'Driver':>6} | {'AsgnEp':>6} | {'Pickup ETA':>10} | {'Delivery ETA':>12} | {'Elapsed min':>11} | {'Pick late':>9} | {'Del late':>8} | {'TW status':<11} | {'Status':<10}")
                lines.append(header)
                lines.append('-' * len(header))
                for r in sorted(rows, key=lambda x: int(x['Order number'])):
                    lines.append(
                        f"{int(r['Order number']):5d} | {r['Pickup point (merchant x,y)']:<23} | {r['Delivery point (customer x,y)']:<23} | "
                        f"{r['Order arrival time']:>7} | {r['Assigned driver']:>6} | {str(r['Assigned epoch']):>6} | {r['Pickup ETA']:>10} | {r['Delivery ETA']:>12} | "
                        f"{str(r['ETA elapsed from call-in (min)']):>11} | {str(r['Pickup lateness (min)']):>9} | {str(r['Delivery lateness (min)']):>8} | "
                        f"{r['Time-window status']:<11} | {r['Status']:<10}"
                    )
                delivered_n = sum(1 for r in rows if r['Status'] == 'Delivered')
                transit_n = sum(1 for r in rows if r['Status'] == 'In transit')
                lines.append('-' * len(header))
                lines.append(f'Snapshot totals | orders shown={len(rows)} | Delivered={delivered_n} | In transit={transit_n}')
                if label == 'END':
                    # Because previously committed deliveries are filtered from
                    # this epoch, Delivered rows at END represent the deliveries
                    # completed within this epoch's displayed interval.
                    newly_delivered = [r for r in rows if r['Status'] == 'Delivered']
                    lines += [
                        '',
                        f'DELIVERY SUMMARY — EPOCH {epoch} ({elapsed_fmt(epoch * EPOCH_HOURS)} to {elapsed_fmt((epoch + 1) * EPOCH_HOURS)})',
                        f'Orders delivered in this epoch: {len(newly_delivered)}',
                    ]
                    if newly_delivered:
                        detail = ', '.join(
                            f"#{int(r['Order number'])} @ {r['Delivery ETA']}"
                            for r in sorted(newly_delivered, key=lambda x: (str(x['Delivery ETA']), int(x['Order number'])))
                        )
                        lines.append(f'Delivered orders (ID @ ETA): {detail}')
                    else:
                        lines.append('Delivered orders (ID @ ETA): none')
                    lines.append('These completed orders are excluded from the next epoch and are not re-optimized.')
        lines += ['', '=' * 170, 'ALGORITHM COMPARISON BY DECISION EPOCH', '=' * 170]
        if self.algorithm_rows:
            hdr = 'Epoch | Time  | Pending | New revealed | Reassigned | Vehicles | IPGA window score | SA window score | Applied'
            lines.extend([hdr, '-' * len(hdr)])
            for r in self.algorithm_rows:
                lines.append(f"{r['Epoch']:5d} | {r['Decision time']:>5} | {r['Visible pending orders']:7d} | {r['Newly revealed orders assigned']:12d} | {r.get('Pending orders reassigned', 0):10d} | {r['Active vehicles available']:8d} | {r['IPGA window-priority objective score']:15.2f} | {r['SA window-priority objective score']:13.2f} | {r['Applied solver']}")
        final_time = max((p.delivery_eta for p in self.plans.values()), default=0.0)
        total_delivered = sum(1 for oid in self.orders if oid in self.plans and self.plans[oid].delivery_eta <= final_time + 1e-6)
        missing_assigned = [oid for oid in self.orders if oid not in self.assignment]
        unplanned = [oid for oid in self.orders if oid not in self.plans]
        max_vol = max(self.volume.values())
        planned_pairs = [(oid, p) for oid, p in self.plans.items()]
        window_infeasible = [(oid, p) for oid, p in planned_pairs
                             if (p.pickup_late_h + p.delivery_late_h) > 1e-9]
        window_feasible_n = len(planned_pairs) - len(window_infeasible)
        total_pickup_late_min = sum(p.pickup_late_h for _, p in planned_pairs) * 60.0
        total_delivery_late_min = sum(p.delivery_late_h for _, p in planned_pairs) * 60.0
        max_delivery_late_min = max((p.delivery_late_h * 60.0 for _, p in planned_pairs), default=0.0)
        window_infeasible_ids = [oid for oid, _ in sorted(window_infeasible)]
        lines += [
            '', '=' * 170, 'FINAL DELIVERY AUDIT (actual scheduled event times; not forced at epoch end)', '=' * 170,
            f'Original orders in CSV       : {len(self.orders)}',
            f'Orders assigned to a driver  : {len(self.assignment)}',
            f'Orders with a calculated ETA: {len(self.plans)}',
            f'Orders with scheduled delivery events: {total_delivered}',
            f'Orders still without a scheduled ETA : {len(self.orders) - total_delivered}',
            f'Time-window feasible planned orders   : {window_feasible_n}',
            f'Time-window violating planned orders  : {len(window_infeasible)}',
            f'Total pickup-window lateness (min)    : {total_pickup_late_min:.1f}',
            f'Total delivery-window lateness (min)  : {total_delivery_late_min:.1f}',
            f'Max delivery-window lateness (min)    : {max_delivery_late_min:.1f}',
            f'Window-violating order IDs             : {window_infeasible_ids[:80] if window_infeasible_ids else "none"}',
            f'Last scheduled delivery time           : {elapsed_fmt(final_time)} (elapsed since start)',
            f'Maximum order volume in CSV  : {max_vol:.3f} m3 (vehicle capacity {CAPACITY_M3:.1f} m3)',
            f'Unassigned order IDs         : {missing_assigned[:40] if missing_assigned else "none"}',
            f'Orders without ETA            : {unplanned[:40] if unplanned else "none"}',
            'Objective priority: minimize the number of orders violating either time window, then total pickup+delivery lateness, then monetary cost.',
            'INFEASIBLE means the selected plan violates at least one supplied pickup/delivery late-window; it is not a mathematical proof that no alternative global plan exists.',
            'Delivery rule: status becomes Delivered only when the calculated delivery ETA is <= the snapshot time.',
            'No order is force-marked as Delivered just to make the final total equal the input row count.',
            '=' * 170,
            'END OF REPORT',
        ]
        self.report_file.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    def run(self):
        print('CFDRP run started.', flush=True)
        print(f'Orders CSV: {self.orders_file.resolve()} ({len(self.orders)} orders)', flush=True)
        print(f'Vehicles CSV: {self.vehicles_file.resolve()} ({self.vehicles_df[VID].nunique()} unique vehicles; {len(self.vehicles_df)} update rows)', flush=True)
        print(f'Output TXT: {self.report_file.resolve()}', flush=True)
        print(f'Parameters: population={self.population_size}, generations={self.generations}, SA iterations={self.sa_iterations}', flush=True)
        for epoch in range(1, self.last_epoch + 1):
            self._run_epoch(epoch)
        print('Vehicle updates finished; executing remaining scheduled deliveries...', flush=True)
        self._run_continuation()
        self._format_report()
        # write detailed snapshot history for audit/reproducibility
        import pandas as pd
        pd.DataFrame(self.history).to_csv(self.history_file, index=False)
        alg_path = self.algorithm_file
        pd.DataFrame(self.algorithm_rows).to_csv(alg_path, index=False)
        final_delivery_by_order = {}
        for oid, p in self.plans.items():
            final_delivery_by_order[oid] = p.delivery_eta
        assigned_n = len(self.assignment)
        eta_n = len(self.plans)
        # All jobs after the last decision time execute their current committed ETA.
        final_t = max(final_delivery_by_order.values(), default=0.0)
        delivered_n = sum(1 for oid in self.orders if oid in final_delivery_by_order and final_delivery_by_order[oid] <= final_t + 1e-9)
        status_counts = {'Delivered': delivered_n, 'In transit': len(self.orders) - delivered_n}
        print(f'ORDERS_SOURCE={len(self.orders)}')
        print(f'VEHICLE_UPDATE_ROWS_SOURCE={len(self.vehicles_df)}')
        print(f'UNIQUE_VEHICLES={self.vehicles_df[VID].nunique()}')
        print(f'ASSIGNED={assigned_n}')
        print(f'WITH_ETA={eta_n}')
        print(f'FINAL_DELIVERED_BY_SCHEDULED_ETA={delivered_n}')
        print(f'FINAL_IN_TRANSIT={status_counts["In transit"]}')
        tw_bad = sum(1 for p in self.plans.values() if p.pickup_late_h + p.delivery_late_h > 1e-9)
        tw_good = len(self.plans) - tw_bad
        print(f'TIME_WINDOW_FEASIBLE={tw_good}')
        print(f'TIME_WINDOW_INFEASIBLE_SELECTED_PLAN={tw_bad}')
        print(f'TOTAL_DELIVERY_LATENESS_MIN={sum(p.delivery_late_h for p in self.plans.values()) * 60.0:.1f}')
        print(f'LAST_DELIVERY={elapsed_fmt(final_t)}')
        print(f'IPGA_POPULATION={self.population_size}; IPGA_GENERATIONS={self.generations}; SA_ITERATIONS={self.sa_iterations}')
        print(f'TXT={self.report_file}')
        print(f'HISTORY_CSV={self.history_file}')
        print(f'ALGORITHM_CSV={alg_path}')


def main():
    # Print before argument validation or CSV loading. This makes it obvious
    # that the saved script on disk is the one the terminal actually executes.
    print(f'[BOOT] Running script: {Path(__file__).resolve()}', flush=True)
    print('[BOOT] Preparing command-line arguments...', flush=True)
    parser = argparse.ArgumentParser(description='CFDRP rolling-horizon report from supplied orders and vehicles CSV files. Unstarted pending orders may be reassigned when new vehicles appear.')
    parser.add_argument('--orders', required=True, help='Path to orders CSV (any new CSV with the same required column names).')
    parser.add_argument('--vehicles', required=True, help='Path to vehicle updates CSV (any new CSV with the same required column names).')
    parser.add_argument('--report', default='day_02_epoch_output_timewindow_priority.txt', help='Output terminal TXT path.')
    parser.add_argument('--history', default='day_02_epoch_history_timewindow_priority.csv', help='Output snapshot history CSV path.')
    parser.add_argument('--population', type=int, default=500, help='IPGA population; paper uses 500 in its numerical experiments')
    parser.add_argument('--generations', type=int, default=500, help='IPGA iterations; paper uses 500 in its numerical experiments')
    parser.add_argument('--sa-iterations', type=int, default=10000, help='SA iteration cap; article extraction does not expose all numeric temperature settings')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    for f in [args.orders, args.vehicles]:
        path = Path(f).expanduser()
        if not path.is_file():
            parser.error(f'Input CSV not found: {path.resolve()}')
    if args.population < 8:
        parser.error('--population must be at least 8')
    if args.generations < 1:
        parser.error('--generations must be at least 1')
    if args.sa_iterations < 100:
        parser.error('--sa-iterations must be at least 100')
    print(f'[BOOT] Orders CSV: {Path(args.orders).expanduser().resolve()}', flush=True)
    print(f'[BOOT] Vehicles CSV: {Path(args.vehicles).expanduser().resolve()}', flush=True)
    print(f'[BOOT] Parameters: population={args.population}, generations={args.generations}, SA iterations={args.sa_iterations}', flush=True)
    print('[BOOT] Loading CSV files and preparing optimizer...', flush=True)
    sim = CFDRPSimulator(args.orders, args.vehicles, args.report, args.history,
                         args.population, args.generations, args.sa_iterations, args.seed)
    print('[BOOT] Optimizer initialized. Starting run now...', flush=True)
    sim.run()

if __name__ == '__main__':
    main()
