
import stanza
from nltk.tree import Tree
from typing import List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class Span(object):
    left: int
    right: int


@dataclass
class SubNP(object):
    text: str
    span: Span


@dataclass
class AllNPs(object):
    nps: List[str]
    spans: List[Span]
    lowest_nps: List[SubNP]

def preprocess_prompt(prompt: str) -> str:
    return prompt.lower().strip().strip(".").strip()


def get_sub_nps(tree: Tree, left: int, right: int): #-> List[SubNP]:
    if isinstance(tree, str) or len(tree.leaves()) == 1:
        return []

    sub_nps = []

    n_leaves = len(tree.leaves())
    n_subtree_leaves = [len(t.leaves()) for t in tree]
    offset = np.cumsum([0] + n_subtree_leaves)[: len(n_subtree_leaves)]
    assert right - left == n_leaves

    if tree.label() == "NP" and n_leaves > 1:
        sub_np = SubNP(
            text=" ".join(tree.leaves()),
            span=Span(left=int(left), right=int(right)),
        )
        sub_nps.append(sub_np)

    for i, subtree in enumerate(tree):
        sub_nps += get_sub_nps(
            subtree,
            left=left + offset[i],
            right=left + offset[i] + n_subtree_leaves[i],
        )
    return sub_nps


def get_all_nps(tree: Tree, full_sent: Optional[str] = None) -> AllNPs:
    start = 0
    end = len(tree.leaves())

    all_sub_nps = get_sub_nps(tree, left=start, right=end)

    lowest_nps = []
    for i in range(len(all_sub_nps)):
        span = all_sub_nps[i].span
        lowest = True
        for j in range(len(all_sub_nps)):
            span2 = all_sub_nps[j].span
            if span2.left >= span.left and span2.right <= span.right:
                lowest = False
                break
        if lowest:
            lowest_nps.append(all_sub_nps[i])

    all_nps = [all_sub_np.text for all_sub_np in all_sub_nps]
    spans = [all_sub_np.span for all_sub_np in all_sub_nps]

    if full_sent and full_sent not in all_nps:
        all_nps = [full_sent] + all_nps
        spans = [Span(left=start, right=end)] + spans

    return AllNPs(nps=all_nps, spans=spans, lowest_nps=lowest_nps)

def get(prompt, nlp):

    out = preprocess_prompt(prompt)
    doc = nlp(out)
    tree = Tree.fromstring(str(doc.sentences[0].constituency))
    all_nps = get_all_nps(tree=tree, full_sent=out)

    return all_nps