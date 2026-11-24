"""Train and run the matched first-hop scorers on a fixed lexical shortlist."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn
from torch.nn import functional as F


DIRECTIONAL_METHODS = ("r2-direction-residual", "r2-role-joint", "r2-forward-contrast")
METHODS = ("pooled", *DIRECTIONAL_METHODS)
LEGACY_POOLED_TRAIN_SHA256 = "86354d0498df69b9d250aac76ca594a37ac1eb152ac201ced3ca7b36e3473416"
BASE_SOURCE_SHA256 = "5014b3fab1a44ac451abec6d24bcc00be9242ddd64a906abb46a1cd8552aa025"
LEGACY_RERANKER_SOURCE_SHA256 = "e8368b52bea3af646a7499446ac942199cb862a8dfd298c5fc0d5df3b2750a84"
LEGACY_PAIRED_SOURCE_SHA256 = "6a721217e196c723f3a86f769b89a3cccadcae87a684f5e1b7e219c8df8f9f51"
LEGACY_CLAUSE_SOURCE_SHA256 = "cc0e1daf42a992a86088f47907708a24f4b21416b894cda4d5703bdd7a99f28f"
FROZEN_LAYER1_SOURCE_SHA256 = "017f250d47f27064c072be7dbc785e1ebb884ba482fc477b41fb6a937cf3ff21"
BASE_METHOD = "r2-direction-residual"
RERANKERS = ("factorized", "mention-joint", "soft-joint", "centered-joint")
TRAIN_RERANKER = "factorized"
PAIR_TERMS = ("smooth_l1_gap", "absolute_ce_gap", "error_weighted_gap")
STACK_MODES = ("single-continue", "parallel", "serial")
DOUBLE_LAYERS = STACK_MODES[1:]
PAIR_LOSS = "worst-view"
PAIR_LOSS_WEIGHTS = (1.0, 0.5, 0.0)
FROZEN_LAYER1 = {
    1701: (2, 1124, "2c2626c0be5ed13ab4b973de2473374750d3ec25350a2db0eea3e16a74100413"),
    2701: (6, 1093, "66d6235822ce496ad058aa90658e6a6a44d39c5251b5863fe0673c01f54c90ee"),
    3701: (7, 1122, "7a544c3c18406cef636a338aa34b006d027d0c893fe539e036ae3a3a47108c6f"),
}
INVARIANCE_BETA = 1.0
TRAIN_DOCUMENTS = 12_000
GATE_DOCUMENTS = 1_840
REVERSIBLE_DOCUMENTS = 1_208
PAIRS_PER_EPOCH = 920
SWAPPED_ROWS_PER_EPOCH = 604
UPDATES_PER_EPOCH = 188
SEEDS = (1701, 2701, 3701)
SCHEMA_VERSION = 1
NODE_COUNT = 12
CANDIDATE_COUNT = 2
TRIPLE_LENGTH = 3
PAD_ID = 0
UNKNOWN_ID = 1
MAX_SEQUENCE_LENGTH = 128
MODEL_DIM = 64
ATTENTION_HEADS = 4
FEEDFORWARD_DIM = 128
COMPONENT_COUNT = 4
LOGIT_TEMPERATURE = 0.1
DIRECTION_WEIGHT = 0.5
ADAPTER_WIDTH = 32
ALIGNMENT_TEMPERATURE = 0.25
PREDICATE_MATCH_BIAS = 2.0
EPOCHS = 8
BATCH_SIZE = 64
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.01
GRADIENT_NORM = 1.0


@dataclass(frozen=True)
class Example:
    doc_id: str
    stratum: str
    question: tuple[str, ...]
    nodes: tuple[tuple[str, ...], ...]
    candidates: tuple[int, int]
    gold: int | None


@dataclass(frozen=True)
class MentionAlignment:
    head_distances: torch.Tensor
    predicate_mentions: torch.Tensor


@dataclass(frozen=True)
class EncodedExample:
    question: torch.Tensor
    nodes: tuple[torch.Tensor, torch.Tensor]
    target: int
    alignment: MentionAlignment | None = None


@dataclass(frozen=True)
class PairIdentity:
    doc_id: str
    stratum: str
    meaning: str
    style: int
    gold: int


@dataclass(frozen=True)
class QueryBlock:
    first: EncodedExample
    second: EncodedExample
    pair: PairIdentity | None = None


def tokens(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_SEQUENCE_LENGTH:
        raise ValueError(f"expected between 1 and {MAX_SEQUENCE_LENGTH} text tokens")
    if any(not isinstance(token, str) or not token for token in value):
        raise ValueError("tokens must be nonempty strings")
    return tuple(value)


def lexical_candidates(
    question: tuple[str, ...], nodes: tuple[tuple[str, ...], ...],
) -> tuple[int, int]:
    question_tokens = set(question)
    matches = [len(question_tokens.intersection(node)) for node in nodes]
    first, second = sorted(range(NODE_COUNT), key=lambda index: (-matches[index], index))[:2]
    return first, second


def load_examples(path: Path, require_gold: bool) -> list[Example]:
    examples = []
    seen = set()
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("expected a JSON object")
                doc_id = raw["doc_id"]
                stratum = raw["stratum"]
                if not isinstance(doc_id, str) or not doc_id or doc_id in seen:
                    raise ValueError("doc_id must be nonempty and unique")
                if not isinstance(stratum, str) or not stratum:
                    raise ValueError("stratum must be a nonempty string")
                question = tokens(raw["question"])
                if not isinstance(raw["nodes"], list) or len(raw["nodes"]) != NODE_COUNT:
                    raise ValueError(f"expected {NODE_COUNT} nodes")
                nodes = tuple(tokens(node) for node in raw["nodes"])
                gold = raw["gold"] if require_gold else None
                if require_gold and (type(gold) is not int or not 0 <= gold < NODE_COUNT):
                    raise ValueError("gold must be a valid integer node index")
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

            examples.append(Example(doc_id, stratum, question, nodes, lexical_candidates(question, nodes), gold))
            seen.add(doc_id)
    if not examples:
        raise ValueError(f"no examples in {path}")
    return examples


def build_vocabulary(examples: list[Example]) -> dict[str, int]:
    vocabulary = set()
    for example in examples:
        vocabulary.update(example.question)
        for node in example.nodes:
            vocabulary.update(node)
    return {token: index for index, token in enumerate(sorted(vocabulary), start=2)}


def mention_alignment(example: Example) -> MentionAlignment | None:
    first, second = (example.nodes[index] for index in example.candidates)
    head, relation, _ = first
    other_head, other_relation, _ = second
    if head != other_head or relation == other_relation:
        return None
    question = example.question
    question_tokens = set(question)
    if not {head, relation, other_relation} <= question_tokens:
        return None

    head_positions = [index for index, token in enumerate(question) if token == head]
    distances = [min(abs(index - head_index) for head_index in head_positions) for index in range(len(question))]
    references = [(token == relation, token == other_relation) for token in question]
    return MentionAlignment(
        torch.tensor(distances, dtype=torch.float32),
        torch.tensor(references, dtype=torch.bool),
    )


def encode_examples(
    examples: list[Example], vocabulary: dict[str, int], method: str, *, with_alignment: bool = False
) -> list[EncodedExample]:
    def encode(sequence: tuple[str, ...]) -> torch.Tensor:
        return torch.tensor([vocabulary.get(token, UNKNOWN_ID) for token in sequence], dtype=torch.long)

    encoded = []
    for example in examples:
        candidates = example.candidates
        if method in DIRECTIONAL_METHODS and any(len(node) != TRIPLE_LENGTH for node in example.nodes):
            raise ValueError(f"{example.doc_id}: directional scoring requires [head, relation, tail] nodes")
        target = candidates.index(example.gold) if example.gold in candidates else -1
        alignment = mention_alignment(example) if with_alignment else None
        encoded.append(EncodedExample(
            encode(example.question), tuple(encode(example.nodes[i]) for i in candidates), target, alignment,
        ))
    return encoded


def make_clause_schedule(
    path: Path, examples: list[Example], encoded: list[EncodedExample], vocabulary: dict[str, int],
) -> tuple[list[list[QueryBlock]], list[dict]]:
    from training_pairs import epoch_blocks, load_views

    views = load_views(path, examples)
    original_ids = [item.doc_id for item in examples]
    gate_ids = [item.doc_id for item, row in zip(examples, encoded, strict=True) if row.alignment is not None]
    reversible_ids = {doc_id for doc_id, (_, swapped) in views.items() if swapped is not None}
    if len(examples) != TRAIN_DOCUMENTS or len(gate_ids) != GATE_DOCUMENTS or set(gate_ids) != views.keys():
        raise ValueError("clause views must cover exactly the original gate-positive training IDs")
    if len(reversible_ids) != REVERSIBLE_DOCUMENTS:
        raise ValueError("clause views must include the exact reversible training cohort")
    outside_slots = [(index, None, 0, 0) for index, doc_id in enumerate(original_ids) if doc_id not in views]
    outside_blocks = list(zip(outside_slots[::2], outside_slots[1::2], strict=True))
    encoded_views = {}
    pair_occurrences = Counter()
    schedules = []
    receipts = []
    for epoch in range(1, EPOCHS + 1):
        raw_blocks = epoch_blocks(views, original_ids, epoch)
        raw_pairs = raw_blocks[:PAIRS_PER_EPOCH]
        if raw_blocks[PAIRS_PER_EPOCH:] != outside_blocks:
            raise ValueError(f"epoch {epoch}: outside-gate singleton blocks differ from archive order")

        blocks = []
        active_ids = []
        cells = Counter()
        for forward, reverse in raw_pairs:
            index, meaning, style, wording = forward
            if wording != 0 or meaning not in (0, 1) or style not in (0, 1) or reverse != (index, meaning, style, 1):
                raise ValueError(f"epoch {epoch}: expected ordered forward and reverse wording of one meaning and style")
            original = examples[index]
            doc_id = original.doc_id
            gold = original.candidates[1 - original.candidates.index(original.gold)] if meaning else original.gold
            group = views[doc_id][meaning]
            if group is None:
                raise ValueError(f"{doc_id}: schedule requested a missing swapped meaning")
            if forward not in encoded_views:
                siblings = []
                for selection in (forward, reverse):
                    view = group[style][selection[3]]
                    question = tokens(view["question"])
                    if view["gold"] != gold or any(token not in vocabulary for token in question):
                        raise ValueError(f"{doc_id}: clause view label or frozen vocabulary differs")
                    if lexical_candidates(question, original.nodes) != original.candidates:
                        raise ValueError(f"{doc_id}: clause view changed ordered Top2")
                    siblings.append(Example(doc_id, original.stratum, question, original.nodes, original.candidates, gold))
                for slot, item in zip((forward, reverse), encode_examples(
                    siblings, vocabulary, BASE_METHOD, with_alignment=True,
                ), strict=True):
                    if item.target < 0 or item.alignment is None:
                        raise ValueError(f"{doc_id}: clause view lost gate eligibility or coverage")
                    encoded_views[slot] = item

            identity = PairIdentity(doc_id, original.stratum, "swapped" if meaning else "original", style, gold)
            blocks.append(QueryBlock(encoded_views[forward], encoded_views[reverse], identity))
            active_ids.append(doc_id)
            cells[(doc_id in reversible_ids, meaning, style)] += 1
            pair_occurrences[(doc_id, meaning, style)] += 1

        expected_cells = {(False, 0, style): 158 for style in (0, 1)}
        expected_cells.update({(True, meaning, style): 151 for meaning in (0, 1) for style in (0, 1)})
        if len(set(active_ids)) != PAIRS_PER_EPOCH or cells != expected_cells:
            raise ValueError(f"epoch {epoch}: active IDs, semantic balance or style balance differ")
        blocks.extend(QueryBlock(encoded[first[0]], encoded[second[0]]) for first, second in outside_blocks)
        schedules.append(blocks)
        ordered_pairs = [
            [block.pair.doc_id, block.pair.meaning, block.pair.style]
            for block in blocks[:PAIRS_PER_EPOCH]
        ]
        receipts.append({
            "epoch": epoch,
            "pairs": len(raw_pairs),
            "original_pairs": PAIRS_PER_EPOCH - SWAPPED_ROWS_PER_EPOCH // 2,
            "swapped_pairs": SWAPPED_ROWS_PER_EPOCH // 2,
            "pairs_per_style": [PAIRS_PER_EPOCH // 2, PAIRS_PER_EPOCH // 2],
            "untouched_singletons": len(outside_slots),
            "ordered_pair_identities_sha256": hashlib.sha256(json.dumps(
                ordered_pairs, ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")).hexdigest(),
        })

    expected_occurrences = {
        (doc_id, meaning, style): 1 if doc_id in reversible_ids else 2
        for doc_id in views
        for meaning in ((0, 1) if doc_id in reversible_ids else (0,))
        for style in (0, 1)
    }
    if pair_occurrences != expected_occurrences:
        raise ValueError("clause schedule did not balance all graphs, meanings and styles across epochs")
    return schedules, receipts


def mean_peak(scores: torch.Tensor) -> torch.Tensor:
    return (scores.mean(dim=-1) + scores.amax(dim=-1)) / 2


class Scorer(nn.Module):
    def __init__(self, vocabulary_size: int, method: str):
        super().__init__()
        self.method = method
        self.embedding = nn.Embedding(vocabulary_size, MODEL_DIM, padding_idx=PAD_ID)
        self.positions = nn.Embedding(MAX_SEQUENCE_LENGTH, MODEL_DIM)
        self.context = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=ATTENTION_HEADS,
            dim_feedforward=FEEDFORWARD_DIM,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.component_queries = nn.Parameter(torch.empty(COMPONENT_COUNT, MODEL_DIM))
        nn.init.normal_(self.component_queries, std=MODEL_DIM**-0.5)

    def encode(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = ids.ne(PAD_ID)
        positions = torch.arange(ids.shape[1], device=ids.device)
        embedded = self.embedding(ids) + self.positions(positions)
        return self.context(embedded, src_key_padding_mask=~valid), valid

    def rank_directionally(self, components: torch.Tensor, node_hidden: torch.Tensor) -> torch.Tensor:
        triples = node_hidden.reshape(-1, CANDIDATE_COUNT, TRIPLE_LENGTH, MODEL_DIM)
        head, relation, tail = triples.unbind(dim=2)
        normalized_components = F.normalize(components, dim=-1).transpose(1, 2)

        def similarity(node_vectors: torch.Tensor) -> torch.Tensor:
            return F.normalize(node_vectors, dim=-1) @ normalized_components

        if self.method == "r2-direction-residual":
            directed = triples.mean(dim=2) + DIRECTION_WEIGHT * (head - tail)
            return mean_peak(similarity(directed))

        forward = mean_peak(similarity(head + relation))
        if self.method == "r2-role-joint":
            head_scores = similarity(head)
            relation_scores = similarity(relation)
            pair_scores = (head_scores.unsqueeze(-1) + relation_scores.unsqueeze(-2)) / 2
            same_component = torch.eye(COMPONENT_COUNT, device=pair_scores.device, dtype=torch.bool)
            best_pair = pair_scores.masked_fill(same_component, float("-inf")).amax(dim=(-2, -1))
            return (best_pair + forward) / 2

        reverse = mean_peak(similarity(tail + relation))
        return forward + DIRECTION_WEIGHT * (forward - reverse)

    def forward(
        self, question_ids: torch.Tensor, node_ids: torch.Tensor, *, return_context: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        question_hidden, question_mask = self.encode(question_ids)
        attention = question_hidden @ self.component_queries.T / math.sqrt(MODEL_DIM)
        attention = attention.masked_fill(~question_mask.unsqueeze(-1), float("-inf")).softmax(dim=1)
        components = attention.transpose(1, 2) @ question_hidden

        node_hidden, node_mask = self.encode(node_ids)
        if self.method == "pooled":
            nodes = (node_hidden * node_mask.unsqueeze(-1)).sum(dim=1) / node_mask.sum(dim=1, keepdim=True)
            nodes = F.normalize(nodes.reshape(-1, CANDIDATE_COUNT, MODEL_DIM), dim=-1)
            query = F.normalize(components.mean(dim=1), dim=-1)
            scores = (nodes * query.unsqueeze(1)).sum(dim=-1)
        else:
            scores = self.rank_directionally(components, node_hidden)
        logits = scores / LOGIT_TEMPERATURE
        return (logits, question_hidden, node_hidden) if return_context else logits


class CandidateAlignment(nn.Module):
    def __init__(self, reranker: str):
        super().__init__()
        self.reranker_index = RERANKERS.index(reranker)
        self.question_projection = nn.Linear(MODEL_DIM, ADAPTER_WIDTH, bias=False)
        self.head_projection = nn.Linear(MODEL_DIM, ADAPTER_WIDTH, bias=False)
        self.relation_projection = nn.Linear(MODEL_DIM, ADAPTER_WIDTH, bias=False)
        self.anchor_projection = nn.Linear(ADAPTER_WIDTH, ADAPTER_WIDTH, bias=False)
        for projection in (self.question_projection, self.head_projection, self.relation_projection, self.anchor_projection):
            nn.init.xavier_uniform_(projection.weight)
        self.salience = nn.Parameter(torch.empty(ADAPTER_WIDTH))
        nn.init.normal_(self.salience, std=ADAPTER_WIDTH**-0.5)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))

    @staticmethod
    def logmeanexp(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~mask, float("-inf"))
        return masked.logsumexp(dim=-1) - mask.sum(dim=-1).to(values.dtype).log()

    def alignment_scores(
        self, question: torch.Tensor, triples: torch.Tensor, valid: torch.Tensor,
        head_distances: torch.Tensor, predicate_mentions: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.question_projection(question)
        heads = self.head_projection(triples[:, :, 0])
        relations = self.relation_projection(triples[:, :, 1])
        normalized_tokens = F.normalize(tokens, dim=-1).transpose(1, 2)
        valid_tokens = valid.unsqueeze(1)
        head_similarity = F.normalize(heads, dim=-1) @ normalized_tokens / ALIGNMENT_TEMPERATURE
        anchor = head_similarity.masked_fill(~valid_tokens, float("-inf")).softmax(dim=-1) @ tokens

        salience = (tokens.unsqueeze(1) + self.anchor_projection(anchor).unsqueeze(2)).tanh() @ self.salience
        salience = salience - self.gamma * head_distances.log1p().unsqueeze(1)
        predicate_support = predicate_mentions.transpose(1, 2) & valid_tokens
        relation_match = F.normalize(relations, dim=-1) @ normalized_tokens / ALIGNMENT_TEMPERATURE
        relation_match = relation_match + PREDICATE_MATCH_BIAS * predicate_support
        mention_support = torch.where(predicate_support.any(dim=-1, keepdim=True), predicate_support, valid_tokens)

        factorized = self.logmeanexp(salience, valid_tokens) + self.logmeanexp(relation_match, valid_tokens)
        joint = self.logmeanexp(salience + relation_match, valid_tokens)
        mention_joint = self.logmeanexp(salience + relation_match, mention_support)
        return torch.stack((factorized, mention_joint, joint, joint - factorized), dim=-1)

@dataclass(frozen=True)
class LayerScores:
    layer0_absolute_gap: torch.Tensor
    layer1_active_logits: torch.Tensor
    conditioning_confidence: torch.Tensor | None
    layer2_active_residual: torch.Tensor | None


class GatedReranker(CandidateAlignment):
    def __init__(self, base: Scorer, reranker: str):
        super().__init__(reranker)
        self.base = base.requires_grad_(False)

    def train(self, mode: bool = True) -> GatedReranker:
        super().train(mode)
        self.base.eval()
        return self

    def layer_outputs(
        self, question_ids: torch.Tensor, node_ids: torch.Tensor, indices: torch.Tensor,
        head_distances: torch.Tensor, predicate_mentions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        with torch.no_grad():
            baseline, question, facts = self.base(question_ids, node_ids, return_context=True)
        triples = facts.reshape(-1, CANDIDATE_COUNT, TRIPLE_LENGTH, MODEL_DIM)
        active_inputs = (
            question.index_select(0, indices),
            triples.index_select(0, indices),
            question_ids.index_select(0, indices).ne(PAD_ID),
            head_distances,
            predicate_mentions,
        )
        scores = self.alignment_scores(*active_inputs)
        active_baseline = baseline.index_select(0, indices)
        confidence = (active_baseline[:, 0] - active_baseline[:, 1]).abs()
        scale = (self.alpha + self.beta * confidence).sigmoid()
        active_logits = active_baseline + scale.unsqueeze(1) * scores[:, :, self.reranker_index]
        return baseline, baseline.index_copy(0, indices, active_logits), active_inputs

    def forward(
        self, question_ids: torch.Tensor, node_ids: torch.Tensor, indices: torch.Tensor,
        head_distances: torch.Tensor, predicate_mentions: torch.Tensor, *, return_layer_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LayerScores]:
        layer0, layer1, _ = self.layer_outputs(question_ids, node_ids, indices, head_distances, predicate_mentions)
        if not return_layer_scores:
            return layer1
        active0 = layer0.index_select(0, indices)
        return layer1, LayerScores(
            (active0[:, 0] - active0[:, 1]).abs(), layer1.index_select(0, indices), None, None,
        )


class StackedReranker(nn.Module):
    def __init__(self, layer1: GatedReranker, mode: str):
        super().__init__()
        if mode not in DOUBLE_LAYERS:
            raise ValueError(f"unsupported stacked scorer: {mode}")
        self.mode = mode
        self.layer1 = layer1.requires_grad_(False)
        self.layer1.eval()
        self.layer2 = CandidateAlignment(TRAIN_RERANKER)

    def train(self, mode: bool = True) -> StackedReranker:
        super().train(mode)
        self.layer1.eval()
        return self

    def second_residual(
        self, z0: torch.Tensor, z1: torch.Tensor, raw_r2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conditioning_logits = z0 if self.mode == "parallel" else z1
        confidence = (conditioning_logits[:, 0] - conditioning_logits[:, 1]).abs()
        scale = (self.layer2.alpha + self.layer2.beta * confidence).tanh()
        return scale.unsqueeze(1) * raw_r2, confidence

    def forward(
        self, question_ids: torch.Tensor, node_ids: torch.Tensor, indices: torch.Tensor,
        head_distances: torch.Tensor, predicate_mentions: torch.Tensor, *, return_layer_scores: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, LayerScores]:
        with torch.no_grad():
            layer0, layer1, active_inputs = self.layer1.layer_outputs(
                question_ids, node_ids, indices, head_distances, predicate_mentions,
            )
            active0 = layer0.index_select(0, indices)
            active1 = layer1.index_select(0, indices)
        raw_r2 = self.layer2.alignment_scores(*active_inputs)[:, :, self.layer2.reranker_index]
        residual, confidence = self.second_residual(active0, active1, raw_r2)
        scores = layer1.index_copy(0, indices, active1 + residual)
        if not return_layer_scores:
            return scores
        return scores, LayerScores((active0[:, 0] - active0[:, 1]).abs(), active1, confidence, residual)


def collate(batch: list[EncodedExample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    questions = nn.utils.rnn.pad_sequence([example.question for example in batch], batch_first=True)
    nodes = nn.utils.rnn.pad_sequence([node for example in batch for node in example.nodes], batch_first=True)
    return questions.to(device), nodes.to(device)


def collate_alignment(
    batch: list[EncodedExample], sequence_length: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = [(index, item.alignment) for index, item in enumerate(batch) if item.alignment is not None]
    distances = torch.zeros(len(selected), sequence_length)
    mentions = torch.zeros(len(selected), sequence_length, CANDIDATE_COUNT, dtype=torch.bool)
    for row, (_, alignment) in enumerate(selected):
        length = alignment.head_distances.numel()
        distances[row, :length] = alignment.head_distances
        mentions[row, :length] = alignment.predicate_mentions
    indices = torch.tensor([index for index, _ in selected], dtype=torch.long)
    return indices.to(device), distances.to(device), mentions.to(device)


@torch.inference_mode()
def predict_indices(
    model: Scorer | GatedReranker | StackedReranker, encoded: list[EncodedExample], device: torch.device
) -> list[int]:
    model.eval()
    indices = []
    for offset in range(0, len(encoded), BATCH_SIZE):
        batch = encoded[offset : offset + BATCH_SIZE]
        questions, nodes = collate(batch, device)
        if isinstance(model, (GatedReranker, StackedReranker)):
            alignments = collate_alignment(batch, questions.shape[1], device)
            logits = model(questions, nodes, *alignments)
        else:
            logits = model(questions, nodes)
        indices.extend(logits.argmax(dim=-1).tolist())
    return indices


def prediction_rows(examples: list[Example], indices: list[int]) -> list[dict]:
    return [
        {
            "doc_id": example.doc_id,
            "stratum": example.stratum,
            "prediction": example.candidates[index],
            "candidates": list(example.candidates),
        }
        for example, index in zip(examples, indices, strict=True)
    ]


def evaluate_dev(examples: list[Example], rows: list[dict], encoded: list[EncodedExample]) -> dict:
    strata: dict[str, list[int]] = {}
    by_gate = {"inside": [0, 0, 0], "outside": [0, 0, 0]}
    count = covered = correct = 0
    for example, row, item in zip(examples, rows, encoded, strict=True):
        is_covered = example.gold in example.candidates
        is_correct = row["prediction"] == example.gold
        bucket = strata.setdefault(example.stratum, [0, 0, 0])
        bucket[0] += 1
        bucket[1] += is_covered
        bucket[2] += is_correct
        gate_bucket = by_gate["inside" if item.alignment is not None else "outside"]
        gate_bucket[0] += 1
        gate_bucket[1] += is_covered
        gate_bucket[2] += is_correct
        count += 1
        covered += is_covered
        correct += is_correct

    def summarize(total: int, in_shortlist: int, hits: int) -> dict:
        return {
            "examples": total,
            "covered": in_shortlist,
            "correct": hits,
            "recall_at_2": in_shortlist / total if total else None,
            "accuracy": hits / total if total else None,
            "accuracy_given_coverage": hits / in_shortlist if in_shortlist else None,
        }

    return {
        **summarize(count, covered, correct),
        "by_stratum": {name: summarize(*values) for name, values in sorted(strata.items())},
        "by_gate": {name: summarize(*values) for name, values in by_gate.items()},
    }


def clause_invariance_loss(
    scores: torch.Tensor, target: torch.Tensor, forward_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_indices = torch.stack((forward_indices, forward_indices + 1), dim=1)
    pair_scores = scores[pair_indices]
    pair_targets = target[pair_indices].unsqueeze(-1)
    gold_logits = pair_scores.gather(dim=-1, index=pair_targets).squeeze(-1)
    other_logits = pair_scores.gather(dim=-1, index=1 - pair_targets).squeeze(-1)
    margins = gold_logits - other_logits
    gap = margins[:, 0] - margins[:, 1]
    terms = F.smooth_l1_loss(gap, torch.zeros_like(gap), beta=INVARIANCE_BETA, reduction="sum")
    return terms / max(1, forward_indices.numel()), margins


def robust_pair_terms(
    scores: torch.Tensor, target: torch.Tensor, forward_indices: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    smooth_l1_mean, margins = clause_invariance_loss(scores, target, forward_indices)
    m_forward, m_reverse = margins.unbind(dim=1)
    gap = m_forward - m_reverse
    smooth_l1 = F.smooth_l1_loss(gap, torch.zeros_like(gap), beta=INVARIANCE_BETA, reduction="none")
    absolute_ce_gap = (F.softplus(-m_forward) - F.softplus(-m_reverse)).abs()
    error_gate = torch.sigmoid(-torch.minimum(m_forward, m_reverse)).detach()
    weighted_gap = error_gate * smooth_l1
    count = max(1, forward_indices.numel())
    terms = (smooth_l1_mean, absolute_ce_gap.sum() / count, weighted_gap.sum() / count)

    with torch.no_grad():
        pair_indices = torch.stack((forward_indices, forward_indices + 1), dim=1).flatten()
        per_row_ce = F.cross_entropy(
            scores.index_select(0, pair_indices),
            target.index_select(0, pair_indices),
            reduction="none",
        ).reshape(-1, 2)
        diagnostics = torch.cat((
            margins, per_row_ce, torch.stack((smooth_l1, absolute_ce_gap, error_gate, weighted_gap), dim=1),
        ), dim=1)
    return terms, diagnostics


def layer_scalar_values(model: GatedReranker | StackedReranker) -> dict:
    def values(layer: CandidateAlignment) -> dict:
        return {name: float(getattr(layer, name).detach().item()) for name in ("alpha", "beta", "gamma")}

    if isinstance(model, StackedReranker):
        return {"layer1": values(model.layer1), "layer2": values(model.layer2)}
    return {"layer1": values(model), "layer2": None}


def pair_layer_trace(layers: LayerScores, active_targets: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        target = active_targets.unsqueeze(1)
        layer1 = layers.layer1_active_logits
        margin1 = (layer1.gather(1, target) - layer1.gather(1, 1 - target)).flatten()
        base_gap = layers.layer0_absolute_gap
        condition = layers.conditioning_confidence
        residual = layers.layer2_active_residual
        stage_gap = (
            residual[:, 0] - residual[:, 1] if residual is not None else torch.zeros_like(base_gap)
        )
        return torch.cat(tuple(tensor.reshape(-1, 2) for tensor in (
            base_gap, margin1, condition if condition is not None else torch.zeros_like(base_gap), stage_gap,
        )), dim=1)


def train_epoch(
    model: GatedReranker | StackedReranker,
    optimizer: torch.optim.Optimizer,
    blocks: list[QueryBlock],
    pair_weights: tuple[float, ...],
    device: torch.device,
    epoch: int,
) -> tuple[dict, list[dict]]:
    model.train()
    initial_scalars = layer_scalar_values(model)
    has_layer2 = isinstance(model, StackedReranker)
    order = torch.randperm(len(blocks)).tolist()
    loss_sum = ce_sum = pair_contribution_sum = 0.0
    term_sums = dict.fromkeys(PAIR_TERMS, 0.0)
    update_losses = []
    covered = updates = 0
    gated = gated_covered = gated_updates = 0
    pair_batches = both_correct = both_strictly_correct = ties = 0
    swapped_pairs = 0
    pair_records = []
    blocks_per_batch = BATCH_SIZE // 2
    for offset in range(0, len(order), blocks_per_batch):
        batch_blocks = [blocks[index] for index in order[offset : offset + blocks_per_batch]]
        batch = [slot for block in batch_blocks for slot in (block.first, block.second)]
        pairs = [(index, block) for index, block in enumerate(batch_blocks) if block.pair is not None]
        indices_forward = torch.tensor([2 * index for index, _ in pairs], device=device, dtype=torch.long)
        pair_count = len(pairs)
        eligible = sum(example.target >= 0 and example.alignment is not None for example in batch)
        if eligible != 2 * pair_count:
            raise ValueError("B8 training schedule must put all covered G questions in complete F/R pairs")
        target = torch.tensor([example.target for example in batch], device=device)
        questions, nodes = collate(batch, device)
        alignments = collate_alignment(batch, questions.shape[1], device)
        indices = alignments[0]
        optimizer.zero_grad(set_to_none=True)
        scores, layers = model(questions, nodes, *alignments, return_layer_scores=True)
        ce = F.cross_entropy(
            scores.index_select(0, indices), target.index_select(0, indices), ignore_index=-1, reduction="sum",
        ) / max(1, eligible)
        pair_terms, diagnostics = robust_pair_terms(scores, target, indices_forward)
        diagnostics = torch.cat((diagnostics, pair_layer_trace(layers, target.index_select(0, indices))), dim=1)
        weighted_pair = sum(weight * term for weight, term in zip(pair_weights, pair_terms, strict=True))
        loss = ce + weighted_pair
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM, error_if_nonfinite=True)
        optimizer.step()
        updates += 1

        batch_loss, batch_ce, batch_weighted_pair, *batch_terms = torch.stack(
            (loss, ce, weighted_pair, *pair_terms)
        ).detach().tolist()
        loss_sum += batch_loss * eligible
        ce_sum += batch_ce * eligible
        pair_contribution_sum += batch_weighted_pair * eligible
        term_values = dict(zip(PAIR_TERMS, batch_terms, strict=True))
        for name, value in term_values.items():
            term_sums[name] += value * pair_count
        update_losses.append({
            "epoch_update": updates,
            "optimizer_update": (epoch - 1) * UPDATES_PER_EPOCH + updates,
            "pairs": pair_count,
            "gated_covered": eligible,
            "ce": batch_ce,
            "terms": term_values,
            "weighted_pair_contribution": batch_weighted_pair,
            "loss": batch_loss,
        })
        pair_batches += pair_count > 0
        covered += sum(example.target >= 0 for example in batch)
        gated += indices.numel()
        gated_covered += eligible
        gated_updates += eligible > 0
        for (block_index, block), row in zip(pairs, diagnostics.tolist(), strict=True):
            (
                m_forward, m_reverse, ce_forward, ce_reverse, l1, absolute_gap, gate, weighted_gap,
                conf0_forward, conf0_reverse, margin1_forward, margin1_reverse,
                condition_forward, condition_reverse, delta_forward, delta_reverse,
            ) = row
            identity = block.pair
            strict = min(m_forward, m_reverse) > 0
            correct_forward = m_forward > 0 or (m_forward == 0 and block.first.target == 0)
            correct_reverse = m_reverse > 0 or (m_reverse == 0 and block.second.target == 0)
            tied = m_forward == 0 or m_reverse == 0
            both_correct += correct_forward and correct_reverse
            both_strictly_correct += strict
            ties += tied
            swapped_pairs += identity.meaning == "swapped"
            pair_records.append({
                "doc_id": identity.doc_id,
                "stratum": identity.stratum,
                "meaning": identity.meaning,
                "style": identity.style,
                "gold": identity.gold,
                "target_slot": block.first.target,
                "epoch": epoch,
                "epoch_update": updates,
                "optimizer_update": (epoch - 1) * UPDATES_PER_EPOCH + updates,
                "forward_batch_slot": 2 * block_index,
                "reverse_batch_slot": 2 * block_index + 1,
                "margin_forward": m_forward,
                "margin_reverse": m_reverse,
                "layer0_absolute_gap_forward": conf0_forward,
                "layer0_absolute_gap_reverse": conf0_reverse,
                "layer1_correct_margin_forward": margin1_forward,
                "layer1_correct_margin_reverse": margin1_reverse,
                "conditioning_confidence_forward": condition_forward if has_layer2 else None,
                "conditioning_confidence_reverse": condition_reverse if has_layer2 else None,
                "second_logit_gap_forward": delta_forward if has_layer2 else None,
                "second_logit_gap_reverse": delta_reverse if has_layer2 else None,
                "ce_forward": ce_forward,
                "ce_reverse": ce_reverse,
                "smooth_l1_gap": l1,
                "absolute_ce_gap": absolute_gap,
                "detached_error_gate": gate,
                "error_weighted_gap": weighted_gap,
                "correct_forward": correct_forward,
                "correct_reverse": correct_reverse,
                "both_correct": correct_forward and correct_reverse,
                "both_strictly_correct": strict,
                "either_tied": tied,
            })

    examples = 2 * len(blocks)
    pair_count = len(pair_records)
    swapped_rows = 2 * swapped_pairs
    if (examples, updates, gated, gated_covered, pair_count, swapped_rows) != (
        TRAIN_DOCUMENTS, UPDATES_PER_EPOCH, GATE_DOCUMENTS, GATE_DOCUMENTS, PAIRS_PER_EPOCH, SWAPPED_ROWS_PER_EPOCH,
    ):
        raise ValueError(f"epoch {epoch}: realized exposure, pair or optimizer counts differ")
    if len({record["doc_id"] for record in pair_records}) != pair_count:
        raise ValueError(f"epoch {epoch}: training pair diagnostics contain duplicate IDs")
    return {
        "examples": examples,
        "covered": covered,
        "loss_given_gated_coverage": loss_sum / gated_covered,
        "ce_loss_given_gated_coverage": ce_sum / gated_covered,
        "pair_terms_given_pairs": {name: value / pair_count for name, value in term_sums.items()},
        "pair_contribution_given_gated_coverage": pair_contribution_sum / gated_covered,
        "pair_weights": dict(zip(PAIR_TERMS, pair_weights, strict=True)),
        "optimizer_update_losses": update_losses,
        "layer_scalars": {"before": initial_scalars, "after": layer_scalar_values(model)},
        "optimizer_updates": updates,
        "inside_gate": gated,
        "outside_gate": examples - gated,
        "inside_gate_covered": gated_covered,
        "outside_gate_covered": covered - gated_covered,
        "optimizer_batches_with_gated_coverage": gated_updates,
        "optimizer_batches_without_gated_coverage": updates - gated_updates,
        "untouched_original_singletons": examples - gated,
        "original_meaning_questions": examples - swapped_rows,
        "swapped_meaning_questions": swapped_rows,
        "forward_wording_questions": pair_count,
        "reverse_wording_questions": pair_count,
        "invariance_pairs": pair_count,
        "optimizer_batches_with_invariance_pairs": pair_batches,
        "optimizer_batches_without_invariance_pairs": updates - pair_batches,
        "invariance_pairs_both_correct": both_correct,
        "invariance_pairs_both_strictly_correct": both_strictly_correct,
        "invariance_pairs_with_tie": ties,
    }, pair_records


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_write(path: Path, write, *, new_only: bool = False) -> None:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if new_only:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, lambda stream: stream.write(payload.encode("utf-8")))


def jsonl_payload(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")


def write_jsonl(path: Path, rows: list[dict], *, new_only: bool = False) -> None:
    payload = jsonl_payload(rows)
    atomic_write(path, lambda stream: stream.write(payload), new_only=new_only)


def write_pair_records(path: Path, rows: list[dict]) -> None:
    def write(stream) -> None:
        with gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as payload:
            for row in rows:
                payload.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))

    atomic_write(path, write)


def training_config(
    args: argparse.Namespace, device: torch.device, train: list[Example], dev: list[Example],
    model: GatedReranker | StackedReranker, frozen_checkpoint: dict, schedule_receipts: list[dict], initialization: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": sha256(Path(__file__)),
        "method": args.method,
        "reranker": args.reranker,
        "pair_loss": PAIR_LOSS,
        "stack_mode": args.stack_mode,
        "seed": args.seed,
        "base_checkpoint": {
            "sha256": sha256(args.base_checkpoint),
            "source_sha256": frozen_checkpoint["config"]["source_sha256"],
            "method": frozen_checkpoint["config"]["method"],
            "seed": frozen_checkpoint["config"]["seed"],
            "selected_epoch": frozen_checkpoint["epoch"],
        },
        "layer1_initialization": initialization,
        "train": {"sha256": sha256(args.train), "examples": len(train)},
        "train_clause_views": {
            "sha256": sha256(args.train_clause_views),
            "scheduler_source_sha256": sha256(Path(__file__).with_name("training_pairs.py")),
            "eligible_documents": GATE_DOCUMENTS,
            "reversible_documents": REVERSIBLE_DOCUMENTS,
            "pairs_per_epoch": PAIRS_PER_EPOCH,
            "outside_singletons_per_epoch": TRAIN_DOCUMENTS - GATE_DOCUMENTS,
            "query_slots_per_block": 2,
            "blocks_per_batch": BATCH_SIZE // 2,
            "schedule": "deterministic per-epoch meaning and style, adjacent F/R slots, unmodified archive-order outside-G singleton blocks, one seeded torch.randperm of blocks per epoch",
            "epoch_receipts": schedule_receipts,
        },
        "dev": {"sha256": sha256(args.dev), "examples": len(dev)},
        "device": str(device),
        "torch_version": str(torch.__version__),
        "architecture": {
            "model_dim": MODEL_DIM,
            "attention_heads": ATTENTION_HEADS,
            "feedforward_dim": FEEDFORWARD_DIM,
            "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
            "question_components": COMPONENT_COUNT,
            "logit_temperature": LOGIT_TEMPERATURE,
            "direction_weight": DIRECTION_WEIGHT,
            "directional_input": ["head", "relation", "tail"],
            "parameters": sum(param.numel() for param in model.parameters()),
            "frozen_parameters": sum(param.numel() for param in model.parameters() if not param.requires_grad),
            "trainable_parameters": sum(param.numel() for param in model.parameters() if param.requires_grad),
            "adapter_width": ADAPTER_WIDTH,
            "alignment_temperature": ALIGNMENT_TEMPERATURE,
            "predicate_match_bias": PREDICATE_MATCH_BIAS,
            "layer1_initialization": "full frozen B10 worst-view model state",
            "trainable_layer": "layer1" if args.stack_mode == "single-continue" else "layer2",
            "layer2_initialization": None if args.stack_mode == "single-continue" else "same seeded factorized adapter with zero alpha and beta",
            "layer2_conditioning": {"single-continue": None, "parallel": "abs(layer0_candidate_logit_difference)", "serial": "abs(layer1_candidate_logit_difference)"}[args.stack_mode],
            "layer2_residual": None if args.stack_mode == "single-continue" else "tanh(alpha2 + beta2 * confidence) * raw_layer2_candidate_score",
        },
        "optimization": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "global_gradient_norm_clip": GRADIENT_NORM,
            "loss": "mean_cross_entropy_over_gate_positive_rows_with_gold_in_fixed_shortlist",
            "pair_terms_over_complete_same_answer_pairs": {
                "smooth_l1_gap": "smooth_l1(m_forward-m_reverse,0,beta=1)",
                "absolute_ce_gap": "abs(softplus(-m_forward)-softplus(-m_reverse))",
                "error_weighted_gap": "detach(sigmoid(-min(m_forward,m_reverse)))*smooth_l1_gap",
            },
            "invariance_smooth_l1_beta": INVARIANCE_BETA,
            "pair_weights": dict(zip(PAIR_TERMS, PAIR_LOSS_WEIGHTS, strict=True)),
            "combination": "mean_gated_covered_ce_plus_weighted_means_of_three_pair_terms",
            "empty_pair_batch": "differentiable_zero_pair_loss_and_ordinary_optimizer_update",
            "metric_denominators": {
                "loss_ce_pair_contribution": "batch_values_weighted_by_gated_covered_queries",
                "unweighted_pair_terms": "batch_values_weighted_by_complete_same_answer_pairs",
            },
            "online_diagnostics": "pre_update_logits_at_training_step; strict_correct_both_margins_positive; resolved_correct_uses_original_candidate_index_tie",
            "empty_gate_batch": "differentiable_zero_and_ordinary_optimizer_update",
        },
        "retrieval": {
            "candidates": CANDIDATE_COUNT,
            "lexical_score": "number of shared distinct text tokens, exact case-sensitive match",
            "lexical_tie": "smaller node index",
            "learned_tie": "earlier lexical shortlist position",
            "gate": "raw_tokens_same_head_distinct_relations_and_all_three_tokens_appear_in_question",
            "outside_gate": "frozen_baseline_logits_without_adapter_adjustment",
        },
        "selection": "maximize all-example dev accuracy over newly trained epochs 1–8, preferring the earlier epoch for ties. Initial replay has no selection weight.",
    }


def checkpoint_path(directory: Path, epoch: int) -> Path:
    return directory / "checkpoints" / f"epoch-{epoch:02d}.pt"


def read_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["config"]["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"incompatible checkpoint schema in {path}")
    return checkpoint


def publish_epoch_outputs(directory: Path, checkpoint: dict) -> None:
    history = checkpoint["history"]
    epoch = checkpoint["epoch"]
    best_epoch = max(history, key=lambda item: item["dev"]["correct"])["epoch"]
    best_path = checkpoint_path(directory, best_epoch)
    best = checkpoint if best_epoch == epoch else read_checkpoint(best_path)
    best_sha = sha256(best_path)
    selected_path = directory / "best.pt"
    if not selected_path.is_file() or sha256(selected_path) != best_sha:
        def copy_best(stream) -> None:
            with best_path.open("rb") as original:
                shutil.copyfileobj(original, stream)

        atomic_write(selected_path, copy_best)
    write_json(directory / "metrics.json", history)
    write_jsonl(directory / "dev_predictions.jsonl", best["dev_predictions"])
    write_json(
        directory / "best.json",
        {
            "checkpoint": "best.pt",
            "source_checkpoint": best_path.relative_to(directory).as_posix(),
            "sha256": best_sha,
            "epoch": best_epoch,
            "method": checkpoint["config"]["method"],
            "seed": checkpoint["config"]["seed"],
            "dev": history[best_epoch - 1]["dev"],
        },
    )
    write_json(
        directory / "status.json",
        {"status": "complete" if epoch == EPOCHS else "running", "epochs_completed": epoch},
    )


def state_tensors_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode("ascii")
        digest.update(len(header).to_bytes(4, "little"))
        digest.update(header)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(bytes(raw.tolist()))
    return digest.hexdigest()


def load_layer1_initialization(
    args: argparse.Namespace, model: GatedReranker, vocabulary: dict[str, int], schedule_receipts: list[dict],
) -> dict:
    epoch, dev_hits, expected_sha = FROZEN_LAYER1[args.seed]
    actual_sha = sha256(args.layer1_checkpoint)
    if actual_sha != expected_sha:
        raise ValueError("layer 1 checkpoint bytes do not match the frozen B10 best for this seed")
    with args.layer1_receipt.open(encoding="utf-8") as stream:
        receipt = json.load(stream)
    if (
        receipt["checkpoint"] != "best.pt" or receipt["seed"] != args.seed
        or receipt["epoch"] != epoch or receipt["sha256"] != actual_sha
        or receipt["method"] != BASE_METHOD or receipt["dev"]["correct"] != dev_hits
    ):
        raise ValueError("layer 1 checkpoint and frozen B10 best receipt disagree")
    checkpoint = read_checkpoint(args.layer1_checkpoint)
    config = checkpoint["config"]
    if (
        config["source_sha256"] != FROZEN_LAYER1_SOURCE_SHA256 or config["pair_loss"] != PAIR_LOSS
        or config["method"] != BASE_METHOD or config["reranker"] != args.reranker
        or config["seed"] != args.seed or checkpoint["epoch"] != epoch or checkpoint["vocabulary"] != vocabulary
        or checkpoint["history"][epoch - 1]["dev"] != receipt["dev"]
        or config["base_checkpoint"]["sha256"] != sha256(args.base_checkpoint)
        or config["train"]["sha256"] != sha256(args.train) or config["dev"]["sha256"] != sha256(args.dev)
        or config["train_clause_views"]["sha256"] != sha256(args.train_clause_views)
        or config["train_clause_views"]["scheduler_source_sha256"] != sha256(Path(__file__).with_name("training_pairs.py"))
        or config["train_clause_views"]["epoch_receipts"] != schedule_receipts
    ):
        raise ValueError("layer 1 checkpoint must be the B10 worst-view best with the same seed, data and block schedule")
    for name, tensor in model.base.state_dict().items():
        layer1_tensor = checkpoint["model"][f"base.{name}"]
        if tensor.dtype != layer1_tensor.dtype or not torch.equal(tensor, layer1_tensor):
            raise ValueError(f"B10 checkpoint changed frozen B2 backbone tensor: {name}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model_sha = state_tensors_sha256(model.state_dict())
    if model_sha != state_tensors_sha256(checkpoint["model"]):
        raise ValueError("loaded layer 1 bytes differ from the frozen B10 model state")
    return {
        "loaded_layer1_model_tensor_sha256": model_sha,
        "checkpoint_sha256": actual_sha,
        "best_receipt_sha256": sha256(args.layer1_receipt),
        "source_sha256": FROZEN_LAYER1_SOURCE_SHA256,
        "source_arm": PAIR_LOSS,
        "seed": args.seed,
        "source_epoch": epoch,
        "source_dev_correct": dev_hits,
        "optimizer": "fresh AdamW",
        "training_rng": "replay seeded B8 block permutations from the beginning",
        "initial_replay_eligible_for_selection": False,
    }


def train(args: argparse.Namespace, device: torch.device) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_examples = load_examples(args.train, require_gold=True)
    dev_examples = load_examples(args.dev, require_gold=True)
    vocabulary = build_vocabulary(train_examples)
    train_encoded = encode_examples(train_examples, vocabulary, args.method, with_alignment=True)
    dev_encoded = encode_examples(dev_examples, vocabulary, args.method, with_alignment=True)
    schedules, schedule_receipts = make_clause_schedule(args.train_clause_views, train_examples, train_encoded, vocabulary)

    frozen = read_checkpoint(args.base_checkpoint)
    frozen_config = frozen["config"]
    if (
        frozen_config["source_sha256"] != BASE_SOURCE_SHA256
        or frozen_config["method"] != BASE_METHOD
        or frozen_config["seed"] != args.seed
        or frozen["vocabulary"] != vocabulary
    ):
        raise ValueError("base checkpoint is not the expected seed and vocabulary of the frozen B2 scorer")
    base = Scorer(len(vocabulary) + 2, BASE_METHOD)
    base.load_state_dict(frozen["model"])
    # Adapter initialization must not change the fixed seed's training-document permutations.
    with torch.random.fork_rng(devices=[]):
        layer1 = GatedReranker(base, args.reranker)
        model = StackedReranker(layer1, args.stack_mode) if args.stack_mode in DOUBLE_LAYERS else layer1
    initialization = load_layer1_initialization(args, layer1, vocabulary, schedule_receipts)
    initialization["initial_layer2_model_tensor_sha256"] = (
        state_tensors_sha256(model.layer2.state_dict()) if isinstance(model, StackedReranker) else None
    )
    model.to(device)
    initial_predictions = jsonl_payload(prediction_rows(dev_examples, predict_indices(model, dev_encoded, device)))
    if initial_predictions != args.layer1_dev_predictions.read_bytes():
        raise ValueError("initial model predictions must replay frozen B10 original DEV byte for byte")
    initialization["initial_dev_replay_sha256"] = hashlib.sha256(initial_predictions).hexdigest()

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, foreach=False, fused=False,
    )
    initialization["fresh_optimizer_initial_state_entries"] = len(optimizer.state)
    config = training_config(args, device, train_examples, dev_examples, model, frozen, schedule_receipts, initialization)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists():
        if not args.resume:
            raise FileExistsError(f"{config_path} already exists; use --resume")
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume configuration differs from the saved run")
    elif any(not (args.resume and item.name.startswith(".config.json.")) for item in args.output.iterdir()):
        raise ValueError(f"output directory is not a recognized training run: {args.output}")
    else:
        write_json(config_path, config)

    initial_path = args.output / "initial-dev-replay.jsonl"
    if initial_path.exists():
        if initial_path.read_bytes() != initial_predictions:
            raise ValueError("saved initial prediction replay differs")
    else:
        atomic_write(initial_path, lambda stream: stream.write(initial_predictions), new_only=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    diagnostics = args.output / "pair-diagnostics"
    diagnostics.mkdir(exist_ok=True)
    checkpoints = sorted((args.output / "checkpoints").glob("epoch-*.pt"))
    history = []
    if checkpoints:
        last = read_checkpoint(checkpoints[-1])
        if last["config"] != config or last["vocabulary"] != vocabulary:
            raise ValueError("checkpoint configuration or vocabulary differs")
        history = last["history"]
        expected = [checkpoint_path(args.output, epoch) for epoch in range(1, last["epoch"] + 1)]
        if checkpoints != expected or len(history) != last["epoch"]:
            raise ValueError("checkpoint sequence is incomplete")
        for item in history:
            pair_receipt = item["train"]["online_pair_diagnostics"]
            path = diagnostics / f"epoch-{item['epoch']:02d}.jsonl.gz"
            if pair_receipt != {
                "path": path.relative_to(args.output).as_posix(), "sha256": sha256(path), "rows": PAIRS_PER_EPOCH,
            }:
                raise ValueError(f"online training pair diagnostics differ from their checkpoint: {path}")
        model.load_state_dict(last["model"])
        optimizer.load_state_dict(last["optimizer"])
        random.setstate(last["rng"]["python"])
        torch.set_rng_state(last["rng"]["torch"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(last["rng"]["cuda"], device)
        publish_epoch_outputs(args.output, last)
        print(f"resumed {args.method} seed={args.seed} after epoch {last['epoch']}", flush=True)
    else:
        write_json(args.output / "status.json", {"status": "running", "epochs_completed": 0})

    for epoch in range(len(history) + 1, EPOCHS + 1):
        training_metrics, pair_records = train_epoch(
            model, optimizer, schedules[epoch - 1], PAIR_LOSS_WEIGHTS, device, epoch,
        )
        if isinstance(model, StackedReranker):
            layer1_sha = state_tensors_sha256(model.layer1.state_dict())
            if layer1_sha != initialization["loaded_layer1_model_tensor_sha256"]:
                raise ValueError("frozen layer 1 tensors changed during stacked training")
            training_metrics["frozen_layer1_model_tensor_sha256"] = layer1_sha
        pair_path = diagnostics / f"epoch-{epoch:02d}.jsonl.gz"
        write_pair_records(pair_path, pair_records)
        training_metrics["online_pair_diagnostics"] = {
            "path": pair_path.relative_to(args.output).as_posix(),
            "sha256": sha256(pair_path),
            "rows": len(pair_records),
        }
        rows = prediction_rows(dev_examples, predict_indices(model, dev_encoded, device))
        metrics = {"epoch": epoch, "train": training_metrics, "dev": evaluate_dev(dev_examples, rows, dev_encoded)}
        history.append(metrics)
        checkpoint = {
            "epoch": epoch,
            "config": config,
            "vocabulary": vocabulary,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
            "history": history,
            "dev_predictions": rows,
        }
        atomic_write(checkpoint_path(args.output, epoch), lambda stream: torch.save(checkpoint, stream))
        publish_epoch_outputs(args.output, checkpoint)
        print(json.dumps(metrics, ensure_ascii=True, allow_nan=False), flush=True)


def predict(args: argparse.Namespace, device: torch.device) -> None:
    if args.output.exists():
        raise FileExistsError(f"prediction output already exists: {args.output}")
    examples = load_examples(args.input, require_gold=False)
    if args.lexical:
        indices = [0] * len(examples)
    else:
        checkpoint = read_checkpoint(args.checkpoint)
        method = checkpoint["config"]["method"]
        if method not in METHODS:
            raise ValueError(f"unsupported checkpoint scorer: {method}")
        training_source = checkpoint["config"]["source_sha256"]
        legacy_pooled = method == "pooled" and training_source == LEGACY_POOLED_TRAIN_SHA256
        legacy_residual = method == BASE_METHOD and training_source == BASE_SOURCE_SHA256
        legacy_reranker = method == BASE_METHOD and training_source in {
            LEGACY_RERANKER_SOURCE_SHA256, LEGACY_PAIRED_SOURCE_SHA256, LEGACY_CLAUSE_SOURCE_SHA256,
            FROZEN_LAYER1_SOURCE_SHA256,
        }
        if not (legacy_pooled or legacy_residual or legacy_reranker) and training_source != sha256(Path(__file__)):
            raise ValueError("prediction code differs from the checkpoint training code")
        vocabulary = checkpoint["vocabulary"]
        model = Scorer(len(vocabulary) + 2, method)
        reranker = checkpoint["config"].get("reranker")
        if reranker is not None:
            if method != BASE_METHOD:
                raise ValueError("unsupported reranker checkpoint backbone")
            model = GatedReranker(model, reranker)
            stack_mode = checkpoint["config"].get("stack_mode")
            if stack_mode in DOUBLE_LAYERS:
                model = StackedReranker(model, stack_mode)
        model.to(device)
        model.load_state_dict(checkpoint["model"])
        encoded = encode_examples(examples, vocabulary, method, with_alignment=reranker is not None)
        indices = predict_indices(model, encoded, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, prediction_rows(examples, indices), new_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--method", required=True, choices=(BASE_METHOD,))
    train_parser.add_argument("--reranker", required=True, choices=(TRAIN_RERANKER,))
    train_parser.add_argument("--stack-mode", required=True, choices=STACK_MODES)
    train_parser.add_argument("--base-checkpoint", required=True, type=Path)
    train_parser.add_argument("--layer1-checkpoint", required=True, type=Path)
    train_parser.add_argument("--layer1-receipt", required=True, type=Path)
    train_parser.add_argument("--layer1-dev-predictions", required=True, type=Path)
    train_parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    train_parser.add_argument("--train", required=True, type=Path)
    train_parser.add_argument("--train-clause-views", required=True, type=Path)
    train_parser.add_argument("--dev", required=True, type=Path)
    train_parser.add_argument("--output", required=True, type=Path)
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--device", default="auto")
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--input", required=True, type=Path)
    predict_parser.add_argument("--output", required=True, type=Path)
    predict_parser.add_argument("--device", default="auto")
    prediction_mode = predict_parser.add_mutually_exclusive_group(required=True)
    prediction_mode.add_argument("--checkpoint", type=Path)
    prediction_mode.add_argument("--lexical", action="store_true")
    args = parser.parse_args()

    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.command == "train":
        train(args, device)
    else:
        predict(args, device)


if __name__ == "__main__":
    main()
