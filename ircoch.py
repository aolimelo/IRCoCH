#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from collections import defaultdict as ddict
from copy import deepcopy
from multiprocessing import Process, Manager
from statistics import mean

import math
import numpy as np
import rdflib
import time
from rdflib.namespace import RDFS, SKOS, RDF, OWL
from scipy.stats import entropy


nt_label_regex = re.compile("<(.+?)> <(.+?)> \"(.+)\"[\@[a-zA-Z\-]+]? \.")
nt_regex = re.compile("<(.+)> <(.+)> <(.+)> \.")

ancestors_cache = {}
descendants_cache = {}

shorten = lambda x: x[max(x.rfind("/"), x.rfind("#")) + 1:]

logger = logging.getLogger("IRCoCH")
logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)


class IdDict(dict):
    def __getitem__(self, key):
        if not self.__contains__(key):
            super(IdDict, self).__setitem__(key, len(self))
        return dict.__getitem__(self, key)

    def __setitem__(self, key, value):
        raise ("Not allowed in IdDict, just get the item and if it does not exists it sets it to an incremented id")


class DAGNode():
    def __init__(self, iri):
        self.iri = iri
        self.parents = list()
        self.children = list()


def is_covered_by(i, nodes, hier):
    """
    Checks whether the node i is covered by a set of nodes
    :param i:
    :param nodes:
    :param hier:
    :return: True if any of the nodes is ancestor of i or is i itself
    """
    return (get_all_ancestors(i, hier) | set([i])) & nodes


def get_subsumed_entities(r_id, r_entities, hier):
    q = [r_id]
    entities = set()
    traversed_r = set()
    while len(q):
        nd, q = q[0], q[1:]
        if nd in hier and nd not in traversed_r:
            traversed_r.add(nd)
            for c in hier[r_id].children:
                q.append(c)
                if r_id in r_entities:
                    entities.update(r_entities[r_id])
    return entities


def get_all_ancestors(n, hier):
    if n in ancestors_cache:
        return ancestors_cache[n]
    ancestors = set()
    if n in hier:
        q = [hier[n]]
        while len(q):
            nd, q = q[0], q[1:]
            for p in nd.parents:
                if p not in ancestors:
                    ancestors.add(p)
                    if p in ancestors_cache:
                        ancestors.update(ancestors_cache[p])
                    else:
                        q.append(hier[p])
    ancestors_cache[n] = ancestors
    return ancestors


def get_all_descendants(n, hier):
    if n in descendants_cache:
        return descendants_cache[n]
    descendants = set()
    if n in hier:
        for p in hier[n].children:
            if p not in descendants:
                descendants.add(p)
                if p in descendants_cache:
                    descendants.update(descendants_cache[p])
                else:
                    p_descendants = get_all_descendants(p, hier)
                    descendants.update(p_descendants)
    descendants_cache[n] = descendants
    return descendants


def get_all_descendants_with_diff(n, hier, descendants=set()):
    q = [n]
    diff = set()
    while len(q):
        nd, q = q[0], q[1:]
        if nd in hier and nd not in descendants:
            for c in hier[nd].children:
                if c not in descendants:
                    descendants.add(c)
                    diff.add(c)
                    q.append(c)
    return descendants, diff


def get_roots_and_descendants(entities, hier):
    descendants = set()
    roots = set()

    for e in entities:
        if e not in descendants:
            if e not in roots:
                roots.add(e)
                if e in hier:
                    descendants, diff = get_all_descendants_with_diff(e, hier, descendants)
                    roots = roots - diff

    return roots, descendants


def get_roots(hier):
    roots = set()
    for i, n in hier.items():
        if not n.parents:
            roots.add(i)
    return roots


def level_hierarchy(hier, roots):
    remaining = set(hier.keys())
    level = roots
    levels = []
    while level:
        next_level = []
        for n in level:
            for c in hier[n].children:
                if c in remaining:
                    next_level.append(c)
                    remaining.remove(c)
        levels.append(level)
        level = next_level
    return levels


def get_coverage(roots, entities, hier):
    root_coverage = Counter()
    for e in entities:
        root_coverage.update(roots & (get_all_ancestors(e, hier) | set([e])))
    return root_coverage


def get_number_of_descendants(levels, hier):
    descendants = {}
    for i, level in enumerate(reversed(levels)):
        for e in level:
            descendants[e] = set()
            if e in hier and hier[e].children:
                descendants[e].update(hier[e].children)
                for d in hier[e].children:
                    if d in descendants:
                        descendants[e].update(descendants[d])
    descendants = {e: len(desc) for e, desc in descendants.items()}
    return descendants


def load_data(path):
    # print("Load concept hierarchy to DAG and relation assertions")
    ent_hier = {}
    prop_hier = {}
    labels = {}
    rd = IdDict()
    ed = IdDict()
    r_subjects = ddict(lambda: [])
    r_objects = ddict(lambda: [])

    domains, ranges = ddict(lambda: []), ddict(lambda: [])
    with open(path, "r") as f:
        for row in f:
            m = nt_regex.match(row)
            if m is not None:
                s, p, o = [m.group(i) for i in range(1, 4)]
                p = rdflib.URIRef(p)
                s = rdflib.URIRef(s)
                o = rdflib.URIRef(o)

                if p == RDFS.subClassOf or p == RDF.type:
                    s_id = ed[s]
                    o_id = ed[o]

                    if s_id not in ent_hier:
                        ent_hier[s_id] = DAGNode(s)
                    if o_id not in ent_hier:
                        ent_hier[o_id] = DAGNode(o)
                    if s != o:
                        ent_hier[s_id].parents.append(o_id)
                        ent_hier[o_id].children.append(s_id)

                elif p == RDFS.subPropertyOf:
                    s_id = rd[s]
                    o_id = rd[o]

                    if s_id not in prop_hier:
                        prop_hier[s_id] = DAGNode(s)
                    if o_id not in prop_hier:
                        prop_hier[o_id] = DAGNode(o)
                    if s != o:
                        prop_hier[s_id].parents.append(o_id)
                        prop_hier[o_id].children.append(s_id)

                elif p in [RDFS.domain, RDFS.range]:
                    s_id = rd[s]
                    o_id = ed[o]

                    if p == RDFS.domain:
                        domains[s_id].append(o_id)
                    elif p == RDFS.range:
                        ranges[s_id].append(o_id)

                elif all([(not i.startswith(str(n))) for i in [s,p,o] for n in [RDF,RDFS,SKOS,OWL]]):
                    s_id = ed[s]
                    o_id = ed[o]
                    p_id = rd[p]

                    r_subjects[p_id].append(s_id)
                    r_objects[p_id].append(o_id)


            else:
                m = nt_label_regex.match(row)
                if m is not None:
                    s, p, o = [m.group(i) for i in range(1, 4)]
                    p = rdflib.URIRef(p)
                    s = rdflib.URIRef(s)

                    if p == SKOS.prefLabel:
                        labels[s] = o

    return ent_hier, prop_hier, dict(rd), dict(r_subjects), dict(r_objects), dict(ed), labels, dict(domains), dict(
        ranges)


def naive_dag_dist(n1, n2, ent_hier, entity_level, descendants_per_node=None):
    """
    Level based distance computation based on the number of hops
    :param n1:
    :param n2:
    :param ent_hier:
    :param entity_level:
    :param fan_weight: If false a hop counts as 1, else it depends on the fanout
    :return:
    """
    if n1 == n2 or n1 not in entity_level or n2 not in entity_level:
        return 0
    l1 = entity_level[n1]
    l2 = entity_level[n2]
    ancs1 = get_all_ancestors(n1, ent_hier)
    ancs2 = get_all_ancestors(n2, ent_hier)

    # Find lowest common ancestor
    if n1 in ancs2:
        lca = n1
    elif n2 in ancs1:
        lca = n2
    else:
        ca = list(ancs1 & ancs2)
        ca.sort(key=lambda x: entity_level[x], reverse=True)
        lca = ca[0]
    llca = entity_level[lca]

    if descendants_per_node is None:
        return (l1 - llca) + (l2 - llca)
    else:
        ndesc1 = descendants_per_node[n1] + 1
        ndesc2 = descendants_per_node[n2] + 1
        ndesclca = descendants_per_node[lca] + 1
        return (l1 - llca) * math.log10(ndesclca / ndesc1) + (l2 - llca) * math.log10(ndesclca / ndesc2)


def mm(roots, root_coverage, confidence, specificity, hier, entity_level):
    # TODO 1: add distance in the hierarchy (maybe avg number of steps up the merged nodes need to take)
    # TODO 2: relations with few assertions end up having low values. Adjust for that please :)
    measures = []

    roots = list(roots)
    avg_dist = 1.0  # TODO
    outlier_factor = 1.0  #  TODO

    measures.append(confidence)
    measures.append(confidence ** 2 * specificity * 1 / math.log1p(len(roots)))
    measures.append(confidence ** 2 * specificity * outlier_factor * 1 / math.log1p(len(roots)))
    measures.append(confidence ** 2 * 1 / (1 + avg_dist) * specificity * outlier_factor * 1 / math.log1p(len(roots)))

    return measures

def get_roots_efficiently(entities, hier, add_parent_threshold=1000):
    roots = set()
    ignore = set()
    entities = list(entities)
    random.shuffle(entities)
    for e in entities:
        # In there are too many roots, start getting parents to reduce the number of roots
        if len(roots) > add_parent_threshold:
            e = hier[e].parents[0]
        if e not in roots and e not in ignore and not get_all_ancestors(e,hier) & roots:
            roots.add(e)
            for r in list(roots):
                if e in get_all_ancestors(r, hier):
                    roots.remove(r)
            ignore.update(get_all_descendants(e, hier))
    return roots


def generalization_step(roots, hier, entity_level, switch_point=50):
    original_size = len(roots)
    undo = {}
    while len(roots) >= original_size:

        # Sort candidates, to find lazy strategy winner (covering least roots and deepest in the hierarchy)
        logger.debug("Computing candidates")
        if len(roots) > 250:
            candidates = []
            while not candidates:
                sample_idx = np.random.randint(0, len(roots), 100)
                roots_list = list(roots)
                candidates = list(set(
                    [(p, tuple(get_all_descendants(p, hier) & roots)) for r in [roots_list[i] for i in sample_idx] for p in hier[r].parents if
                     p not in roots]))
        else:
            candidates = list(set(
                [(p, tuple(get_all_descendants(p, hier) & roots)) for r in roots for p in hier[r].parents if
                 p not in roots]))

        logger.debug(f"Candidates size = {len(candidates)}")

        if len(roots) > switch_point:
            candidates.sort(key=lambda x: -len(x[1]) * len(entity_level) + (len(entity_level) - entity_level[x[0]] - 1))
        else:
            candidates.sort(key=lambda x: len(x[1]) * len(entity_level) + (len(entity_level) - entity_level[x[0]] - 1))

        selected, covered_roots = candidates[0]
        roots = (roots | set([selected])) - set(covered_roots)

        logger.debug(f"Number of nodes covered by selected gebneralization candidate {selected} = {len(covered_roots)}")

        #  In case the selected node covers only 1 root node, we need to be able to undo it change if necessary
        if len(covered_roots) == 1:
            if covered_roots[0] in undo:
                undo[selected] = undo[covered_roots[0]]
            else:
                undo[selected] = covered_roots[0]

        # Undoes unnecessary changes which have not been used by the last step that reduces the number of roots
        elif len(covered_roots) > 1:
            for c in covered_roots:
                if c in undo:
                    del undo[c]
            for new, old in undo.items():
                if new in roots:
                    roots.remove(new)
                    roots.add(old)
    return roots

def simplify_roots(roots, entities, hier):
    """
    Simplifies the set of roots to eliminate redundant root nodes that do not affect the coverage
    :param roots:
    :param entities:
    :param hier:
    :return:
    """
    roots_hier_coverage = {r: get_all_descendants(r, hier) for r in roots}
    roots_entity_coverage = {r: ((descs | set([r])) & entities) for r, descs in roots_hier_coverage.items()}
    original_coverage = len(set.union(*[cov for _, cov in roots_entity_coverage.items()]))
    keep_simplifying = True
    while len(roots) > 1 and keep_simplifying:
        keep_simplifying = False
        for r in list(roots):
            current_coverage = len(set.union(*[cov for rr, cov in roots_entity_coverage.items() if rr != r ]))
            if current_coverage == original_coverage:
                roots.remove(r)
                keep_simplifying = True
                break
    return roots



def compute_constraint(r_entities, r_id, labels, ied, ird, ent_hier, prop_hier, nodes_ts, entity_level,
                       descendants_per_node, count_types, constraint, min_conf=0.95):
    all_suggestions = {}
    if r_id in r_entities:
        logger.info(f"Computing constraint for {labels[ird[r_id]]}")
        iri = ird[r_id]
        entities = set(r_entities[r_id]) | get_subsumed_entities(r_id, r_entities, prop_hier)
        logger.debug(f"Number of entities = {len(entities)}")

        if entities:

            roots = get_roots_efficiently(entities, ent_hier)
            logger.debug(f"Number of roots: {len(roots)}")
            roots = set([r for r in roots if not get_all_ancestors(r, ent_hier) & roots])
            logger.debug(f"Number of cleaned roots: {len(roots)}")

            original_roots = deepcopy(roots)
            original_entity_coverage = len(entities)
            original_hier_coverage = len(set.union(*[get_all_descendants(r,ent_hier) | set([r]) for r in roots]))

            max_nodes = nodes_ts

            while len(roots) >= 1 and max_nodes >= 1:
                conf = 1.0
                while len(roots) > max_nodes:
                    roots = generalization_step(roots, ent_hier, entity_level)
                    roots = simplify_roots(roots, entities, ent_hier)

                relaxed_roots = deepcopy(roots)
                roots_hier_coverage = {r: get_all_descendants(r, ent_hier) for r in relaxed_roots}
                roots_entity_coverage = {r: ((descs | set([r])) & entities) for r, descs in roots_hier_coverage.items()}
                current_hier_coverage = len(
                    set.union(*[descs for r, descs in roots_hier_coverage.items() if r in relaxed_roots]))
                step_suggestions = []
                while relaxed_roots and conf >= min_conf:
                    specificity = 1 / (1 + math.log10((1 + len(current_hier_coverage)) / (1 + len(original_hier_coverage))))

                    # mm_roots = mm(roots, root_coverage, conf=1.0, specificity, lift, entities,
                    #                             ent_hier, entity_level)
                    mm_roots = []
                    step_suggestions.append((tuple(set([(r,len(cov)) for r, cov in roots_entity_coverage.items() if r in relaxed_roots])), (conf,mm_roots)))

                    ranked_roots = list([(r,cov) for r,cov in roots_entity_coverage.items() if r in relaxed_roots])
                    ranked_roots.sort(key=lambda x: len(x[1]))
                    dropped_root, dropped_coverage = ranked_roots[0]
                    relaxed_roots.remove(dropped_root)

                    if relaxed_roots:
                        current_hier_coverage = len(
                            set.union(*[descs for r, descs in roots_hier_coverage.items() if r in relaxed_roots]))

                        prev_conf = conf
                        conf = float(len(current_hier_coverage))/original_entity_coverage

                        # If confidence remains 1.0, ignore previous longer suggestion
                        if conf == 1.0 or conf==prev_conf:
                            logger.debug(f"Remove suggestion {step_suggestions[-1]}")
                            step_suggestions = step_suggestions[:-1]

                for s, measures in step_suggestions:
                    all_suggestions[s] = measures

                max_nodes = min(max_nodes, len(roots) - 1)

            print(f"\n\n## [{labels[iri]}]({iri})\n### {constraint}\n")
            all_suggestions = list(all_suggestions.items())
            all_suggestions.sort(key=lambda x:(x[1][0], -len(s)), reverse=True)
            for s, measures in all_suggestions:
                conf, mm_roots = measures
                mmstr = [f"{m:.3}" for m in mm_roots]
                print(f" - [conf={conf:.2f}] {mmstr} [" + ", ".join(
                        [f"[{labels[ied[r]]}]({ied[r]}): {cov}" for r, cov in s]) + "]")

        else:
            logger.warn(f"Relation {r_id} {labels[ird[r_id]]} has no assertions!")
    else:
        logger.warn(f"Relation {r_id} does not exist")



if __name__ == '__main__':

    parser = ArgumentParser("IRCoCH: Induction of Relation Constraints on Complex Hierarchies")
    parser.add_argument("input", type=str, default=None, help="Path to KB .nt dump or .npz information")
    parser.add_argument("-n", "--nodes", type=int, default=10,
                        help="Maximum number of nodes to be considered in suggestions")
    parser.add_argument("-np", "--n-processes", type=int, default=1, help="Number of process to be run parallely")

    args = parser.parse_args()

    rdflib.logger.setLevel(logging.ERROR)

    logger.info(f"Loading data from {args.input}")
    if args.input.endswith(".npz"):
        d = np.load(args.input, allow_pickle=True)
        ent_hier = d["ent_hier"].item()
        prop_hier = d["prop_hier"].item()
        rd = d["rd"].item()
        r_subjects = d["r_s"].item()
        r_objects = d["r_o"].item()
        ed = d["ed"].item()
        labels = d["labels"].item()
        domains = d["domains"].item()
        ranges = d["ranges"].item()
    elif args.input.endswith(".nt"):
        ent_hier, prop_hier, rd, r_subjects, r_objects, ed, labels, domains, ranges = load_data(args.input)
        # if not os.path.isfile(args.input.replace(".nt", ".npz")):
        np.savez(args.input.replace(".nt", ".npz"), ent_hier=ent_hier, prop_hier=prop_hier, rd=rd, r_s=r_subjects,
                 r_o=r_objects, ed=ed, labels=labels, domains=domains, ranges=ranges)

    logger.info(
        f"Data loaded: \n\t"
        f"{len(ed)} entities \n\t"
        f"{len(rd)} relations \n\t"
        f"{sum([len(s) for s in r_subjects.values()])} relation assertions \n\t"
        f"{sum([len(e.parents) for e in ent_hier.values()])} subClassOf axioms \n\t"
        f"{mean([len(e.parents) for e in ent_hier.values()])} average fan-in \n\t")

    relations = set(rd.keys())
    ied = {k: v for v, k in ed.items()}
    ird = {k: v for v, k in rd.items()}

    for e in ed.keys() | rd.keys():
        if e not in labels:
            labels[e] = shorten(e)

    roots = get_roots(ent_hier)
    levels = level_hierarchy(ent_hier, roots)

    logger.info(f"Hierarchy leveled: \n\t"
                f"{len(levels)} maximum depth \n\t"
                f"{sum([i*len(l) for i,l in enumerate(levels)])/len(ed)} average depth")

    descendants_per_node = get_number_of_descendants(levels, ent_hier)

    entity_level = {e: i for i, ents in enumerate(levels) for e in ents}

    count_subject_types = {e: 0 for e in ed.keys()}
    count_object_types = {e: 0 for e in ed.keys()}
    for r, ents in list(r_subjects.items()):
        for e in ents:
            for t in get_all_ancestors(e, ent_hier):
                count_subject_types[ied[t]] += 1
    for r, ents in list(r_objects.items()):
        for e in ents:
            for t in get_all_ancestors(e, ent_hier):
                count_object_types[ied[t]] += 1

    procs = []
    manager = Manager()
    return_dict = manager.dict()
    print("# Domain and Range Suggestions")

    pool_args = []
    for iri, r_id in rd.items():
        print(f"## [{labels[iri]}]({iri})")
        for constraint, r_entities, count_types in [("Domains", r_subjects, count_subject_types),
                                                    ("Ranges", r_objects, count_object_types)]:
            if args.n_processes <= 1:
                compute_constraint(r_entities, r_id, labels, ied, ird, ent_hier, prop_hier, args.nodes, entity_level,
                                   descendants_per_node, count_types, constraint)
            else:
                proc = Process(target=compute_constraint,
                               args=(r_entities, r_id, labels, ied, ird, ent_hier, prop_hier, args.nodes, entity_level,
                                     descendants_per_node, count_types, constraint))
                procs.append(proc)

    if args.n_processes > 1:
        remaining_procs = list(range(len(procs)))
        ran_procs = set([])
        while not all([p.exitcode == 1 for p in procs]):
            while len(remaining_procs) and sum([p.is_alive() for p in procs]) < args.n_processes:
                id = remaining_procs[random.randint(0, len(remaining_procs) - 1)]
                startproc = procs[id]
                startproc.start()
                remaining_procs.remove(id)

            time.sleep(1)
