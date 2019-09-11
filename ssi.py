#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from argparse import ArgumentParser
from multiprocessing import Process, Manager

import numpy as np
import rdflib
import time
import random

from ircoch import shorten, get_all_ancestors, level_hierarchy, load_data, get_roots, DAGNode

from scipy.sparse import coo_matrix

parents_cache = {}

def get_all_ancestors(n, hier):
    if n in parents_cache:
        return parents_cache[n]
    parents = set()
    if n in hier:
        q = [hier[n]]
        while len(q):
            nd, q = q[0], q[1:]
            for p in nd.parents:
                if p not in parents:
                    parents.add(p)
                    if p in parents_cache:
                        parents.update(parents_cache[p])
                    else:
                        q.append(hier[p])
    parents_cache[n] = parents
    return parents

def compute_constraint(r_entities, r_id, count_types, labels, ied, ird, hier, min_conf=0.95):
    if r_id in r_entities and len(r_entities[r_id]):
        iri = ird[r_id]
        output = f"\n\n## [{labels[iri]}]({iri})\n### {constraint}\n"

        total_rel_ass = sum([len(v) for k,v in r_entities.items()])
        n_relass = len(r_entities[r_id])
        rows, cols = [], []

        rd = {k: v for v, k in ird.items()}
        ed = {k: v for v, k in ied.items()}

        for i, e in enumerate(r_entities[r_id]):
            types = get_all_ancestors(e, hier)
            # type_ids = [ied[t] for t in types]

            rows += [i] * len(types)
            cols += types

        m = coo_matrix(([1]*len(rows), (rows,cols)), shape=(n_relass, len(ed)))
        m = m.tocsc()
        counts = m.sum(axis=0).tolist()[0]

        for i,c in enumerate(counts):
            conf = c/n_relass
            if conf >= min_conf:
                lift = c/total_rel_ass / (n_relass/total_rel_ass * count_types[ied[i]]/total_rel_ass)
                output += f" - [{conf}, {lift}] [{labels[ied[i]]}]({ied[i]})\n"
        print(output)







if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("input", type=str, default=None, help="Path to KB .nt dump or .npz information")
    parser.add_argument("-n", "--nodes", type=int, default=10,
                        help="Maximum number of nodes to be considered in suggestions")
    parser.add_argument("-np", "--n-processes", type=int, default=1, help="Number of process to be run parallely")

    args = parser.parse_args()

    rdflib.logger.setLevel(logging.ERROR)

    print(f"Loading data from {args.input}")
    if args.input.endswith(".npz"):
        d = np.load(args.input)
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

    print(f"Data loaded {len(ed)} entities, {len(rd)} relations and {len(ent_hier)} links in the entities hierarchy")

    relations = set(rd.keys())
    ied = {k: v for v, k in ed.items()}
    ird = {k: v for v, k in rd.items()}

    for e in ed.keys() | rd.keys():
        if e not in labels:
            labels[e] = shorten(e)

    roots = get_roots(ent_hier)
    levels = level_hierarchy(ent_hier, roots)

    entity_level = {e: i for i, ents in enumerate(levels) for e in ents}

    procs = {"Domains": {}, "Ranges": {}}
    manager = Manager()
    return_dict = manager.dict()
    print("# Domain and Range Suggestions")

    count_types = {e:0 for e in ed.keys()}
    for r, ents in (list(r_subjects.items()) + list(r_objects.items())):
        for e in ents:
            for t in get_all_ancestors(e, ent_hier):
                count_types[ied[t]] += 1



    pool_args = []
    for iri, r_id in rd.items():

        print(f"## [{labels[iri]}]({iri})")
        for constraint, r_entities in [("Domains", r_subjects), ("Ranges", r_objects)]:
            if args.n_processes <= 1:
                compute_constraint(r_entities, r_id, count_types, labels, ied, ird, ent_hier)
            else:
                proc = Process(target=compute_constraint,
                               args=(r_entities, r_id, count_types, labels, ied, ird, ent_hier))
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
