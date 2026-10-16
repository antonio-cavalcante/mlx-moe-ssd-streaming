"""A multi-turn conversation that keeps its KV cache between turns.

Every turn feeds the model only the NEW tokens: the previous turns (including the reasoning the model
produced) are already in the cache. The new tokens are whatever the chat template adds when the next user
message is appended to the history. The sliding-window layers cannot be rewound once they wrap, so the
cache is append-only; if its state is ever in doubt (e.g. Ctrl-C in the middle of a step) the session
rebuilds it from the stored messages on the next turn.
"""
import time
from dataclasses import dataclass

import mlx.core as mx
from mlx_lm.generate import stream_generate

from turns import TurnParser


class ContextFull(Exception):
    pass


@dataclass
class Turn:
    answer: str
    reasoning: str
    prompt_tokens: int       # tokens fed this turn (new tokens only)
    generation_tokens: int
    prompt_s: float          # time to the first generated token
    generation_tps: float
    finish: str              # "stop", "length" or "interrupted" (Ctrl-C or cancelled)
    context: int             # tokens in the cache afterwards
    rebuilt: bool            # the cache had to be rebuilt from the messages first
    reused: int = 0          # tokens that were already in the cache and not fed again


class ChatSession:
    def __init__(self, model, tok, reasoning_effort="medium", sampler=None, max_context=8192,
                 small_step=8, small_below=48, big_step=2048):
        self.model, self.tok, self.sampler = model, tok, sampler
        self.effort, self.max_context = reasoning_effort, max_context
        # A big prefill chunk reads nearly every expert from the SSD (~20 s whatever its size), while up to
        # `small_step` tokens at a time use the cached decode path. Short turns go the small way.
        self.small_step, self.small_below, self.big_step = small_step, small_below, big_step
        self.eot = tok.convert_tokens_to_ids("<|eot|>")
        self.eos = set(tok.eos_token_ids) | {self.eot}
        self.reset()

    def reset(self, reasoning_effort=None):
        if reasoning_effort:
            self.effort = reasoning_effort
        self.messages = []                    # [{"role", "content"}], the answers only
        self.cache = self.model.make_cache()
        self.history = []                     # token ids whose keys/values are in the cache
        self._dirty = False
        self._head = None                     # the system prompt the cache was built under

    def restore(self, messages, reasoning_effort=None):
        """Adopt a conversation sent by a stateless client; the cache is rebuilt on the next send()."""
        self.reset(reasoning_effort)
        self.messages = [dict(m) for m in messages]
        self._dirty = bool(self.messages)

    @property
    def context(self):
        return len(self.history)

    def _render(self, messages, generation):
        return self.tok.apply_chat_template(messages, add_generation_prompt=generation, tokenize=False,
                                            reasoning_effort=self.effort)

    def _system_prompt(self, messages):
        """The rendered system block (it holds the effort and the date, so it can change under the cache)."""
        lead = messages[:1] if messages and messages[0]["role"] in ("system", "developer") else []
        return self._render(lead + [{"role": "user", "content": ""}], False)   # (a template needs a message)

    def send(self, text, on_chunks=None, max_tokens=2048, cancel=None):
        """Run one turn. `on_chunks([(channel, text), ...])` gets the reasoning/answer text as it streams.
        `cancel()` is polled after every token; when it returns True the turn ends early."""
        rebuilt = self._dirty
        if self._dirty:
            self.cache, self.history, self._dirty = self.model.make_cache(), [], False
        user = {"role": "user", "content": text}
        prev = self._render(self.messages, False) if self.history else ""
        full = self._render(self.messages + [user], True)
        head = self._system_prompt(self.messages)
        if self.history and (head != self._head or not full.startswith(prev)):
            # the system prompt changed under the cache (reasoning effort, or the date): start over
            self.cache, self.history, rebuilt, prev = self.model.make_cache(), [], True, ""
        if not self.history:
            self._head = head
        reused = len(self.history)
        ids = self.tok.encode(full[len(prev):], add_special_tokens=False)
        if self.history and self.history[-1] not in self.eos:
            ids = [self.eot] + ids            # the previous turn was cut short: close it first
        room = self.max_context - len(self.history) - len(ids)
        if room < 16:
            raise ContextFull(f"{len(self.history)} tokens in context + {len(ids)} new "
                              f"> limit {self.max_context}")
        max_tokens = min(max_tokens, room)
        step = self.small_step if len(ids) <= self.small_below else self.big_step

        parser, parts, gen, last, finish = TurnParser(in_header=True), {"answer": [], "reasoning": []}, [], None, "stop"

        def emit(chunks):
            for channel, t in chunks:
                parts[channel].append(t)
            if on_chunks and chunks:
                on_chunks(chunks)

        t0, t_first = time.perf_counter(), None
        try:
            for resp in stream_generate(self.model, self.tok, mx.array(ids), prompt_cache=self.cache,
                                        max_tokens=max_tokens, sampler=self.sampler, prefill_step_size=step):
                if t_first is None:
                    t_first = time.perf_counter()
                gen.append(resp.token)
                last = resp
                emit(parser.feed(resp.text))
                if cancel and cancel():
                    finish = "interrupted"
                    break
            emit(parser.flush())
            if finish != "interrupted":
                finish = last.finish_reason if last and last.finish_reason else "stop"
        except KeyboardInterrupt:
            finish = "interrupted"
        answer = "".join(parts["answer"])

        expected = self.history + ids + gen   # every yielded token has been fed to the model
        if {c.offset for c in self.cache} == {len(expected)}:
            self.history = expected
        else:
            self._dirty = True                # cache state unknown: rebuild on the next turn
        if finish == "interrupted" and not answer:
            self._dirty = True                # nothing worth keeping of this turn
        else:
            self.messages += [user, {"role": "assistant", "content": answer}]
        t_end = time.perf_counter()
        gps = last.generation_tps if last else 0.0
        return Turn(answer, "".join(parts["reasoning"]), len(ids), len(gen),
                    (t_first or t_end) - t0, gps, finish, self.context, rebuilt, reused)
