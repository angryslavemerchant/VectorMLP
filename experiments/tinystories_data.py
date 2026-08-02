"""TinyStories corpus prep: download, train a small BPE, encode to uint16 bins.

Everything is cached, so the first run pays for it and later runs start
instantly. Run it directly to prepare the data before training:

    python experiments/tinystories_data.py
    python experiments/tinystories_data.py --vocab 8192

Why a custom tokenizer rather than the GPT-Neo one TinyStories ships with:
that vocabulary is 50257, which at d_model=384 is a 19.3M embedding matrix
against a 10.6M transformer body. The embedding would be two thirds of the
model and the FFN a fifth, which is exactly the dilution that made the CIFAR
head comparisons uninformative. At vocab 4096 the embedding is 1.6M and the
FFN stays the dominant block, which is the point of the experiment.

TinyStories was written with a deliberately restricted vocabulary, so a 4k BPE
loses very little, and the merge rules converge on a sample — the tokenizer is
trained on the first --tokenizer-mb megabytes rather than the whole corpus.
"""

import argparse
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'tinystories'
EOT = '<|endoftext|>'


def raw_text(split):
    """Return an iterable of story strings for 'train' or 'validation'."""
    from datasets import load_dataset
    ds = load_dataset('roneneldan/TinyStories', split=split)
    return ds


def train_tokenizer(vocab_size, tokenizer_mb, path):
    """Byte-level BPE. Trained on a prefix of the corpus, not all of it."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    budget = tokenizer_mb * 1024 * 1024

    def sample():
        used = 0
        for row in raw_text('train'):
            text = row['text']
            used += len(text)
            if used > budget:
                return
            yield text

    tok.train_from_iterator(sample(), trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    print(f'tokenizer: {tok.get_vocab_size()} tokens -> {path}', flush=True)
    return tok


def load_tokenizer(vocab_size=4096, tokenizer_mb=100):
    from tokenizers import Tokenizer
    path = DATA / f'bpe{vocab_size}.json'
    if path.exists():
        return Tokenizer.from_file(str(path))
    return train_tokenizer(vocab_size, tokenizer_mb, path)


def encode_split(tok, split, out_path, eot_id):
    """Encode a split into a flat uint16 .bin, one EOT between stories."""
    ids = []
    total = 0
    with open(out_path, 'wb') as f:
        for i, row in enumerate(raw_text(split)):
            ids.extend(tok.encode(row['text']).ids)
            ids.append(eot_id)
            if len(ids) >= 1_000_000:
                np.asarray(ids, dtype=np.uint16).tofile(f)
                total += len(ids)
                ids = []
                if i % 200_000 == 0:
                    print(f'  {split}: {i:,} stories, {total:,} tokens',
                          flush=True)
        if ids:
            np.asarray(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
    print(f'{split}: {total:,} tokens -> {out_path}', flush=True)
    return total


def prepare(vocab_size=4096, tokenizer_mb=100):
    """Ensure tokenizer + train.bin/val.bin exist. Returns (tok, paths)."""
    DATA.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(vocab_size, tokenizer_mb)
    eot_id = tok.token_to_id(EOT)
    assert eot_id is not None, 'tokenizer is missing the EOT token'

    paths = {}
    for split, name in (('train', 'train'), ('validation', 'val')):
        p = DATA / f'{name}{vocab_size}.bin'
        if not p.exists() or p.stat().st_size == 0:
            encode_split(tok, split, p, eot_id)
        paths[name] = p
    return tok, paths


def memmap(path):
    return np.memmap(path, dtype=np.uint16, mode='r')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab', type=int, default=4096)
    ap.add_argument('--tokenizer-mb', type=int, default=100,
                    help='corpus prefix used to TRAIN the tokenizer (MB)')
    args = ap.parse_args()

    tok, paths = prepare(args.vocab, args.tokenizer_mb)
    for name, p in paths.items():
        n = os.path.getsize(p) // 2
        print(f'{name}: {n:,} tokens ({os.path.getsize(p)/1e9:.2f} GB)')

    sample = 'Once upon a time, a little girl named Lily found a shiny red ball.'
    ids = tok.encode(sample).ids
    print(f'\nsample: {len(sample)} chars -> {len(ids)} tokens '
          f'({len(sample)/len(ids):.2f} chars/token)')
    print(f'roundtrip: {tok.decode(ids)!r}')
