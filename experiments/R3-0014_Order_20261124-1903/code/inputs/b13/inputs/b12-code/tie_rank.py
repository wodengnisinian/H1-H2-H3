"""Question-aware tie breaking for the SpiderWeb ordered lexical Top-2."""

MODES = ("original", "direction", "first_relation", "combined")

# Fixed from the public two-step task and training-language conventions.
# "最后一人" names a destination, so only "最后一段" is a second-hop marker.
FIRST_MARKERS = {"先", "首先", "第一步", "开头", "此前"}
SECOND_MARKERS = {"再", "随后", "之后"}
SECOND_PHRASES = {("第二", "段"), ("最后", "一段"), ("结束", "于")}
BOUNDARIES = {"。", "？", "！", "：", "；", ".", "?", "!", ":", ";"}


def parse_question(question, nodes):
    """Return (anchor, first_relation). Either component may be unknown."""
    words = set(question)
    entities = {entity for head, _, tail in nodes for entity in (head, tail)}
    anchors = entities & words
    anchor = next(iter(anchors)) if len(anchors) == 1 else None

    relations = {relation for _, relation, _ in nodes}
    mentions = [(i, word) for i, word in enumerate(question) if word in relations]
    if len(mentions) != 2 or mentions[0][1] == mentions[1][1]:
        return anchor, None

    # Each cue belongs to the following relation in its sentence. A comma can
    # separate an anchor from its relation without ending the instruction.
    # An unmarked step can be inferred only if the other step is explicit.
    first_candidates = set()
    lo = 0
    for pos, (end, relation) in enumerate(mentions):
        cues = set()
        for i in range(lo, end):
            token = question[i]
            if token in BOUNDARIES:
                cues.clear()
            elif token in FIRST_MARKERS:
                cues.add(1)
            elif token in SECOND_MARKERS or (token, question[i + 1]) in SECOND_PHRASES:
                cues.add(2)
        if len(cues) > 1:
            return anchor, None
        if cues == {1}:
            first_candidates.add(relation)
        elif cues == {2}:
            first_candidates.add(mentions[1 - pos][1])
        lo = end + 1
    first = next(iter(first_candidates)) if len(first_candidates) == 1 else None
    return anchor, first


def rank_candidates(question, nodes, mode):
    """Select two zero-based indices. Semantic features break lexical ties only."""
    if mode not in MODES:
        raise ValueError(f"Unsupported candidate ranking mode: {mode}")
    if len(nodes) < 2:
        raise ValueError("Two distinct shortlist candidates are required.")
    words = set(question)
    anchor, first = (None, None) if mode == "original" else parse_question(question, nodes)
    use_anchor = mode in ("direction", "combined") and anchor is not None
    use_relation = mode in ("first_relation", "combined") and first is not None

    def sort_key(index):
        head, relation, _ = nodes[index]
        return (-len(words.intersection(nodes[index])),
                -int(use_anchor and head == anchor),
                -int(use_relation and relation == first),
                index)

    return sorted(range(len(nodes)), key=sort_key)[:2]
