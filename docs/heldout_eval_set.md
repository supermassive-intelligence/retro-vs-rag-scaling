# Held-out evaluation set: `wiki_postdump`

1,000 English Wikipedia articles created after the datastore's snapshot, so
none of them can appear in any index. This is the text Protocol B is scored on
(see [`eval_protocol.md`](eval_protocol.md)).

- **Location:** `data/eval/wiki_postdump/` on `blackwell-maxq-0`
  (`articles.jsonl` + `manifest.json`)
- **Built by:** `scripts/build_heldout_eval.py`
- **Command:** `python -m scripts.build_heldout_eval --n-articles 1000 --out data/eval/wiki_postdump`
- **Code:** `f6c1653`, clean tree.
- **`articles.jsonl` SHA-256:** `2f321f78313b50831522316eec18ba17415565c1a547e28bd09e6a4104792626`
- **Deterministic:** an earlier build from `6fead9e` with uncommitted changes
  produced a byte-identical file, with the same rejections.

**Current runs use the first 500 articles** (`eval_lm --max-articles 500`).
The builder keeps articles in shuffle order and stops at `--n-articles`, so
the first 500 lines are exactly what `--n-articles 500` would build.

## Why post-dump articles

The indexes come from Wikipedia as of **2023-11-01**. If we evaluated on
articles that are also in the index, retrieval would just find the exact text
the model is asked to predict. The score would then measure "is this article
in the index?", not whether retrieval helps.

## How the articles were chosen

1. **Where the datastore's snapshot ends.** `wikimedia/wikipedia` `20231101.en`
   (revision `b04c8d1c`) has 6,407,814 articles, and its largest page ID is
   **75,200,227**. Wikipedia assigns page IDs at creation, and they only
   increase.
2. **Newer articles.** In `omarkamali/wikipedia-monthly` `20260301.en`
   (revision `9cd30b1f`, 7,155,624 articles), **473,926** articles have a page
   ID above the cutoff, so they were created after the snapshot. Of those,
   97,773 have at least 4,000 characters.
3. **Sample.** Shuffle with seed 0, then keep articles with at least 1,024
   Qwen2.5 tokens, one full evaluation window.
4. **Check creation dates.** For each article, the Wikipedia API returns the
   timestamp of its first revision. Articles created before 2023-11-01, or
   since deleted, are dropped.

The script stopped once 1,000 articles passed. Along the way it rejected:

| Reason | Count |
|---|---|
| Shorter than 1,024 tokens | 99 |
| Created before the cutoff | 9 |
| Page since deleted | 12 |

## What's in it

- **Articles:** 1,000
- **Tokens:** 3,041,085 (Qwen2.5 tokenizer)
- **UTF-8 bytes:** 11,050,732
- **Tokens per article:** median 1,814, minimum 1,024, maximum 63,997
- **Evaluation windows** (1,024 tokens, 512 stride): 4,478
- **Scored tokens under Protocol B:** 2,292,736
- **Creation dates:** 2023-11-01 to 2026-03-03, median 2024-12-06

| Created in | Articles |
|---|---|
| 2023 (Nov–Dec) | 95 |
| 2024 | 435 |
| 2025 | 411 |
| 2026 (Jan–Mar) | 59 |

Example titles: *Celmisia gracilenta*, *List of Kaeloo episodes*, *Music of
Clair Obscur: Expedition 33*, *HK (comic book)*, *Opposition to devolution in
the United Kingdom*.

## Checks

- All 1,000 have a page ID above 75,200,227.
- All 1,000 were created on or after 2023-11-01.
- There are no duplicate articles.
- None appear in `wikipedia_1k` (1,000 docs) or `wikipedia_100k` (100,000 docs).

## Caveats

- **Base-model exposure.** Qwen2.5 was released in September 2024, so the 2023
  articles and some of the 2024 ones may be in its training data. That would
  make the no-retrieval baseline look better than it should and understate
  retrieval gains. `--created-after 2024-10-01` keeps only articles newer than
  both the snapshot and the model.
- **Slightly different text cleaning.** The eval snapshot comes from a
  different pipeline than the datastore, so small formatting differences exist
  (for example, `( ; )` where the datastore has `(; )`). These shift absolute
  bpb slightly, but identically for every arm and index size.
- **Longer articles only.** Requiring 1,024 tokens excludes short articles, so
  the set leans toward longer, more developed ones.

## Build notes

- The dataset files are downloaded whole and read locally, cached in
  `/home/normal/sai/hf-cache` (`HF_HOME`). Reading a single column over HTTP
  measured about 5× slower (40 s against 8.7 s for one 421 MB file).
- Wikipedia API calls are made 0.5 s apart. On HTTP 429 or 503 the script
  waits for the server's `Retry-After`, with up to 8 attempts. The first run
  sent about 5 requests per second and was rate-limited after about 200
  articles.
