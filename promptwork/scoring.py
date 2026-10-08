"""Shared-state dialogue scoring with explicit experimental user weights."""
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence
import numpy as np
from .core import GaussianPosterior
from .embedders import Embedder
from .segment import split_evidence


@dataclass(frozen=True)
class IGConfig:
    sigma0_sq: float = 1.0                                                              #start uncertainty
    sigma_sq: float = 0.25                                                              #noise of observation
    beta: float = 1.0                                                                   #relevance weight
    eta: float = 0.05                                                                   #relevance cut-off
    min_chunk_chars: int = 12                                                           #min length for chunck
    keep_short_turns: bool = True                                                       
    state_roles: Literal['both', 'assistant_only', 'user_only'] = 'both'                #whose lines update uncertainty
    user_relevance: Literal['none', 'first_user', 'previous_assistant'] = 'none'        #comparing answers to which line (for weights)
                                                                                        #none - weights=1
                                                                                        #first user - all weights calculated against initial question
                                                                                        #prev assistant - against assistant's reply
    join_wrapped_lines: bool = True

    def __post_init__(self):                                                            #validation
        for name in ('sigma0_sq', 'sigma_sq', 'beta'):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not np.isfinite(self.eta) or not 0 <= self.eta <= 1:
            raise ValueError('eta must be between 0 and 1')
        if not isinstance(self.min_chunk_chars, int) or self.min_chunk_chars < 0:
            raise ValueError('min_chunk_chars must be a nonnegative integer')
        if self.state_roles not in ('both', 'assistant_only', 'user_only'):
            raise ValueError('state_roles must be both, assistant_only, or user_only')
        if self.user_relevance not in ('none', 'first_user', 'previous_assistant'):
            raise ValueError('Unknown user_relevance')


@dataclass
class TurnScore:
    index: int              #position in output
    role: str               #role (user, assistant)
    n_chunks: int           #chunks left
    n_active_chunks: int    #chunks processed w weight!=0
    weights: list[float]    #final chunk weights
    ig: float               #ig of turn
    cumulative_ig: float    #cumulative ig


@dataclass
class DialogueScore:
    turns: list[TurnScore] = field(default_factory=list)
    total: float = 0.0

    def by_role(self) -> dict[str, float]:      #sum of ig for user and assistant separately
        return {role: sum(t.ig for t in self.turns if t.role == role)
                for role in ('user', 'assistant')}

    @property
    def user_share(self) -> float:              #user ig/total ig
        return self.by_role()['user'] / self.total if self.total else 0.0


def _weights(z, anchor, config):                                    #anchor - against what we are comparing
    if anchor is None:                                              #for no comparison weight=1
        return np.ones(len(z), dtype=np.float64)
    norms = np.linalg.norm(z, axis=1) * np.linalg.norm(anchor)      #norms multiplication for cosine similarity (below)
    cosine = np.divide(z @ anchor, norms, out=np.zeros(len(z)), where=norms > 0)    #angle between embeddings
    weights = np.clip(cosine, 0, 1) ** config.beta                  #max(0, cos) - from paper
    weights[weights < config.eta] = 0                               #discarding those below the threshold
    return weights


def score_dialogue(turns: Sequence[Mapping[str, str]], embedder: Embedder,
                   config: IGConfig = IGConfig()) -> DialogueScore:
    """Score supported turns in original order; all gains are natural-log nats.

    Ignored roles produce no result row. Row indices refer to the input sequence.
    Disabled turns report weights of 1, zero gain, and zero active chunks.
    """
    prepared = []                                                       #part 1 - prep
    texts = dict.fromkeys([])                                           #empty dict (ordered set of rows for embeddings)
    first_user = previous_user = previous_assistant = None              
    for index, turn in enumerate(turns):                                #iterating turns
        role = turn.get('role')                                         #getting role
        if role not in ('user', 'assistant'):                           #filtering out unneeded roles
            continue
        content = turn['content']                                       #getting the sentence itself
        if not isinstance(content, str):
            raise TypeError('Turn content must be a string')            #sanity check
        text = content.strip()                                          #stripping off white space
        chunks = split_evidence(content, config.min_chunk_chars, config.keep_short_turns,
                                config.join_wrapped_lines)              #getting chunks
        anchor = None                                                   #enabled - choosing whether the turn updates state
        enabled = ((role == 'user' and config.state_roles in ('both', 'user_only')) or
                   (role == 'assistant' and config.state_roles in ('both', 'assistant_only')))
        if enabled and role == 'assistant':                             #for assistant weights are calculated against
            anchor = previous_user                                      #last user's reply (as in paper)
        elif enabled:
            if config.user_relevance == 'first_user':                   #for user two options, initial q and prev reply
                anchor = first_user
            elif config.user_relevance == 'previous_assistant':
                anchor = previous_assistant
        prepared.append((index, role, chunks, anchor, enabled))         #saving tuple of corresp. roles, chunks, anchor etc.
        if enabled and chunks:                                          #for enabled (w influence) and non-empty chunks:
            for chunk in chunks:                                        
                texts.setdefault(chunk, None)                           #adding unique chunks to 'texts' dictionary
            if anchor is not None:
                texts.setdefault(anchor, None)                          #adding unique anchors to 'texts'
        if text:                                                        
            if role == 'user':                                          #reassigning anchors (first/prev...)
                if first_user is None:
                    first_user = text
                previous_user = text
            else:
                previous_assistant = text
    embeddings = {}                                                         #part 2 - embeddings
    posterior = None
    if texts:                                                               #if texts are non empty
        batch = list(texts)                                                 #creating batch
        vectors = np.asarray(embedder.embed(batch), dtype=np.float64)       #creating embeddings
        if vectors.ndim != 2 or vectors.shape[0] != len(batch) or vectors.shape[1] == 0:
            raise ValueError('Embedder must return shape (n, d) with d > 0') #checking shape
        if not np.all(np.isfinite(vectors)):
            raise ValueError('Embedder returned nonfinite values')          #checking finiteness of values
        embeddings = dict(zip(batch, vectors))                              #a dict line->vector
        posterior = GaussianPosterior(vectors.shape[1], config.sigma0_sq)   #creating uncertanty value
    result = DialogueScore()                                                #part 3 - calculating
    for index, role, chunks, anchor, enabled in prepared:                   #iterating over those tuples of roles, chunks, anchor etc.
        weights = np.ones(len(chunks), dtype=np.float64)                    #initiating weights
        ig = 0.0                                                            #initiating ig
        if enabled and chunks:                                              #for enabled and non-empty chunks
            z = np.stack([embeddings[chunk] for chunk in chunks])           #z - matrix of stacked raw embeddings
            weights = _weights(z, embeddings[anchor] if anchor is not None else None, config) #calc weights
            ig = posterior.update(z, weights / config.sigma_sq)             #updating uncertainty and calc ig
        result.total += ig                                                  #appending ig
        result.turns.append(TurnScore(index, role, len(chunks),             
                           int(np.count_nonzero(weights)) if enabled else 0,
                           weights.tolist(), ig, result.total))
    return result
