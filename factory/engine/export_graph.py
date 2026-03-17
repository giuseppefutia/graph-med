"""
Export the ontology graph from Neo4j to a pickle file (.pkl).

Extracts IcdDisease, IcdChapter, IcdGroup, HpoPhenotype, and HpoDisease nodes
with their stored embedding_label vectors and ONLY ontology hierarchy edges:
  - HAS_CHILD              (IcdDisease -> IcdDisease)
  - CHAPTER_HAS_DISEASE    (IcdChapter -> IcdDisease)
  - GROUP_HAS_DISEASE      (IcdGroup -> IcdDisease)
  - GROUP_IN_CHAPTER       (IcdGroup -> IcdChapter)
  - subClassOf             (HpoPhenotype -> HpoPhenotype, from rdflib-neo4j import)
  - HAS_PHENOTYPIC_FEATURE (HpoDisease -> HpoPhenotype)

UMLS bridge edges and mapper edges (ICD_MAPS_TO_HPO_BY_EMBEDDING,
ICD_MAPS_TO_HPO_BY_GRAPH) are deliberately excluded so the exported graph
is unbiased for GNN training.

No torch or torch_geometric required locally — output is plain numpy + pickle.
The Colab notebook reconstructs the PyG Data object from the numpy arrays.

Usage:
    python -m factory.engine.export_graph --output data/ontology_graph.pkl

Output format (single .pkl file, loaded with pickle.load):
    {
        "x":         np.ndarray [N, embedding_dim] float32,
        "edge_src":  np.ndarray [E] int64,
        "edge_tgt":  np.ndarray [E] int64,
        "node_info": [{"code": str, "name": str, "labels": list}, ...],
        "edge_types": [str, ...],   # parallel to edge_src/edge_tgt
        "stats":     {"n_icd": int, "n_hpo": int, "n_edges": int, ...}
    }
"""
import argparse
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List

from database.neo4j_db import Neo4jGraphDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Only these relationship types are included in the exported graph.
# Changing this list changes what the GNN learns about; keep it bias-free.
# Note: subClassOf is camelCase (created by rdflib-neo4j from rdfs:subClassOf).
HIERARCHY_REL_TYPES = [
    "HAS_CHILD",
    "CHAPTER_HAS_DISEASE",
    "GROUP_HAS_DISEASE",
    "GROUP_IN_CHAPTER",
    "subClassOf",
    "HAS_PHENOTYPIC_FEATURE",
]

CYPHER_NODES = """
MATCH (n)
WHERE n.embedding_label IS NOT NULL
  AND (n:IcdDisease OR n:IcdChapter OR n:IcdGroup OR n:HpoPhenotype OR n:HpoDisease)
RETURN
    elementId(n) AS node_id,
    n.id          AS code,
    n.label       AS name,
    labels(n)     AS node_labels,
    n.embedding_label AS embedding
"""

CYPHER_EDGES = """
MATCH (a)-[r:HAS_CHILD|CHAPTER_HAS_DISEASE|GROUP_HAS_DISEASE|GROUP_IN_CHAPTER|subClassOf|HAS_PHENOTYPIC_FEATURE]->(b)
WHERE elementId(a) IN $node_ids AND elementId(b) IN $node_ids
RETURN elementId(a) AS src, elementId(b) AS tgt, type(r) AS rel_type
"""


class GraphExporter(Neo4jGraphDB):
    """Exports the ontology graph to a PyG Data object."""

    def export(self, embedding_dim: int = 3584) -> Dict:
        """
        Pull nodes and hierarchy edges from Neo4j.

        Returns a dict ready for pickle.dump() — no torch dependency.
        """
        logging.info("Fetching nodes (IcdDisease, IcdGroup, HpoPhenotype, HpoDisease)...")
        all_nodes = self._fetch_nodes(CYPHER_NODES)
        if not all_nodes:
            raise RuntimeError("No nodes found — are embeddings stored in Neo4j?")

        # Count per label
        label_counts = {}
        for n in all_nodes:
            for lbl in n.get("node_labels", []):
                if lbl in ("IcdDisease", "IcdChapter", "IcdGroup", "HpoPhenotype", "HpoDisease"):
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
        for lbl, cnt in sorted(label_counts.items()):
            logging.info("  %s: %d", lbl, cnt)

        # Build index: elementId -> position in all_nodes list
        element_id_to_idx: Dict[str, int] = {
            n["node_id"]: i for i, n in enumerate(all_nodes)
        }
        node_ids = list(element_id_to_idx.keys())

        logging.info("Fetching hierarchy edges (bias-free)...")
        edges = self._fetch_edges(node_ids)
        logging.info("  Hierarchy edges: %d", len(edges))

        # Build node feature tensor
        logging.info("Building node feature tensor...")
        x_list = []
        node_info = []
        for node in all_nodes:
            emb = node.get("embedding")
            if emb:
                x_list.append(np.array(emb, dtype=np.float32))
            else:
                x_list.append(np.zeros(embedding_dim, dtype=np.float32))
            node_info.append({
                "code":   node.get("code", ""),
                "name":   node.get("name", ""),
                "labels": node.get("node_labels", []),
            })

        x = np.stack(x_list).astype(np.float32)  # [N, embedding_dim]

        # Build edge arrays (numpy, no torch required)
        srcs, tgts, rel_types = [], [], []
        for edge in edges:
            src_idx = element_id_to_idx.get(edge["src"])
            tgt_idx = element_id_to_idx.get(edge["tgt"])
            if src_idx is not None and tgt_idx is not None:
                srcs.append(src_idx)
                tgts.append(tgt_idx)
                rel_types.append(edge["rel_type"])

        edge_src = np.array(srcs, dtype=np.int64)
        edge_tgt = np.array(tgts, dtype=np.int64)

        stats = {
            "n_icd":         label_counts.get("IcdDisease", 0),
            "n_icd_chapter": label_counts.get("IcdChapter", 0),
            "n_icd_group":   label_counts.get("IcdGroup", 0),
            "n_hpo":         label_counts.get("HpoPhenotype", 0),
            "n_hpo_disease": label_counts.get("HpoDisease", 0),
            "n_edges":       len(srcs),
            "embedding_dim": embedding_dim,
            "hierarchy_rel_types": HIERARCHY_REL_TYPES,
        }

        logging.info(
            "Graph summary: %d nodes, %d hierarchy edges",
            len(all_nodes), len(srcs),
        )

        return {
            "x":          x,          # np.ndarray [N, embedding_dim] float32
            "edge_src":   edge_src,   # np.ndarray [E] int64
            "edge_tgt":   edge_tgt,   # np.ndarray [E] int64
            "node_info":  node_info,
            "edge_types": rel_types,
            "stats":      stats,
        }

    def _fetch_nodes(self, query: str) -> List[Dict]:
        with self._driver.session(database=self._database) as session:
            return [dict(r) for r in session.run(query)]

    def _fetch_edges(self, node_ids: List[str]) -> List[Dict]:
        with self._driver.session(database=self._database) as session:
            return [
                dict(r)
                for r in session.run(CYPHER_EDGES, node_ids=node_ids)
            ]


def main():
    parser = argparse.ArgumentParser(
        description="Export Neo4j ontology graph to a pickle file for GNN training in Colab."
    )
    parser.add_argument(
        "--output",
        default="data/ontology_graph.pkl",
        help="Output path (default: data/ontology_graph.pkl)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=3584,
        help="Embedding dimension for zero-filled fallback nodes (default: 3584)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exporter = GraphExporter()
    try:
        result = exporter.export(embedding_dim=args.embedding_dim)
    finally:
        exporter.close()

    with open(output_path, "wb") as f:
        pickle.dump(result, f)
    logging.info("Saved graph to %s", output_path)

    stats = result["stats"]
    n_total = stats['n_icd'] + stats['n_icd_chapter'] + stats['n_icd_group'] + stats['n_hpo'] + stats['n_hpo_disease']
    print(f"\nExport complete:")
    print(f"  Nodes : {n_total}")
    print(f"    IcdDisease    : {stats['n_icd']}")
    print(f"    IcdChapter    : {stats['n_icd_chapter']}")
    print(f"    IcdGroup      : {stats['n_icd_group']}")
    print(f"    HpoPhenotype  : {stats['n_hpo']}")
    print(f"    HpoDisease    : {stats['n_hpo_disease']}")
    print(f"  Edges : {stats['n_edges']} (hierarchy-only)")
    print(f"  Dim   : {stats['embedding_dim']}")
    print(f"  File  : {output_path}")
    print(f"\nUpload {output_path.name} to Colab and open notebook/run/03_gnn_train.ipynb")


if __name__ == "__main__":
    main()
