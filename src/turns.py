"""Split the model's streamed text into reasoning and answer.

An assistant turn looks like
    <|start|>assistant to=self<|message|>...reasoning...<|eom|>
    <|start|>assistant to=user<|message|>...answer...<|eot|>
The chat template ends the prompt with `<|start|>assistant`, so generation begins inside the header.
Messages addressed to "user" are the answer; anything else (self, commentary, tools) is "reasoning".
"""

START, MESSAGE = "<|start|>", "<|message|>"
ENDS = ("<|eom|>", "<|eot|>", "<|end_of_text|>")
MARKERS = (START, MESSAGE) + ENDS


class TurnParser:
    def __init__(self, in_header=True):
        self.in_header = in_header  # True after a chat-template prompt; False for raw text
        self.header = ""
        self.channel = "answer"
        self.buf = ""

    def _emit(self, text, out):
        if not text:
            return
        if self.in_header:
            self.header += text
        else:
            out.append((self.channel, text))

    def feed(self, text):
        """Returns [(channel, text), ...] for the text that is complete enough to show."""
        self.buf += text
        out = []
        while True:
            hits = [(self.buf.find(m), m) for m in MARKERS if m in self.buf]
            if not hits:
                break
            i, m = min(hits)
            self._emit(self.buf[:i], out)
            self.buf = self.buf[i + len(m):]
            if m == START:
                self.in_header, self.header = True, ""
            elif m == MESSAGE:
                to = self.header.partition("to=")[2].strip()
                self.channel = "answer" if to in ("user", "") else "reasoning"
                self.in_header = False
            # an end marker just closes the message; the next <|start|> opens the next one
        keep = 0  # hold back a possible half-received marker at the end
        for m in MARKERS:
            for n in range(min(len(m) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(m[:n]):
                    keep = max(keep, n)
                    break
        if keep:
            self._emit(self.buf[:-keep], out)
            self.buf = self.buf[-keep:]
        else:
            self._emit(self.buf, out)
            self.buf = ""
        return out

    def flush(self):
        out = []
        self._emit(self.buf, out)
        self.buf = ""
        return out
