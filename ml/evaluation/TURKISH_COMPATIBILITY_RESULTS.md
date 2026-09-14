# Turkish compatibility results

Date: 2026-09-15

Base commit: 0de00368470c84e66a55a49ee8d9e9bdec0e857e

Decision: **E — Turkish compatibility still fails acceptance**

## Protocol

The development corpus has six source-disjoint groups and seven videos: three calibration groups and three held-out groups. Its 72 natural Turkish queries comprise 42 calibration and 30 held-out queries across Visual, Speech and Hybrid intent. Queries include Turkish characters, suffixes, apostrophes, colloquial omissions and mixed technical terms. All source metadata, checksums and ASR observations are stored in the manifest.

The production baseline was frozen before candidate selection. Router and retrieval choices used calibration data only. The personal acceptance data was neither inspected for individual rule creation nor used for selection. Because the development held-out gates failed, the personal acceptance suite was not rerun.

## Stage diagnosis on held-out sources

| Stage | Native Turkish baseline R@5 | English equivalent R@5 | Selected cheap candidate R@5 | Semantic diagnostic R@5 |
| --- | ---: | ---: | ---: | ---: |
| Visual, forced | 93.33% | 100.00% | 86.67% | — |
| Speech, forced | 75.00% | 12.50% | 87.50% | 100.00% |
| Hybrid, forced | 71.43% | 57.14% | 71.43% | 85.71% |
| AUTO | — | — | 80.00% | 86.67% |

Production Turkish routing was 50.00% on calibration and 53.33% on held-out data; it selected no Speech route. The selected cheap 14-feature linear router reached 88.10% in source-leave-one-out calibration and 76.67% on held-out sources. This is the largest remaining compatibility loss.

Direct Turkish CLIP retrieval was already strong: its held-out forced-Visual R@5 was 93.33%, 6.67 points behind paired English queries. The calibration-selected lexical visual adaptation reduced held-out R@5 to 86.67%, so query translation or expansion should not be applied indiscriminately to the visual path.

BM25 behaved differently. Cheap suffix and lexical handling improved Turkish Speech R@5 from 75.00% to 87.50%. Translating the paired query to English while searching Turkish ASR text reduced R@5 to 12.50%, confirming that blanket Turkish-to-English translation is incompatible with Turkish transcripts. Hybrid retrieval remained at 71.43% with the cheap candidate.

## Selected cheap candidate

| Metric | R@1 | R@3 | R@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Visual | 53.33% | 86.67% | 86.67% | 67.78% |
| Speech | 87.50% | 87.50% | 87.50% | 87.50% |
| Hybrid | 57.14% | 71.43% | 71.43% | 61.90% |
| AUTO | 56.67% | 80.00% | 80.00% | 66.67% |

The cheap candidate added 0.124 ms median and 0.159 ms p95 latency, no measurable RSS increase, a 1,771-byte router and a 1,758-byte adapter artifact. English queries remain on the unchanged production path, yielding a measured structural regression of 0 percentage points.

## Multilingual semantic diagnostic

After the cheap candidate missed its gates, a non-promotable diagnostic used pinned sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 embeddings. Held-out Speech R@5 reached 100%, Hybrid R@5 reached 85.71%, and AUTO R@5 reached 86.67%. The unchanged 76.67% router still missed its gate.

The semantic component required 9.995 ms median and 14.378 ms p95 warm query latency, 266,276,864 bytes of added RSS, and 479,776,499 bytes of cached files. First cold load took 46.06 seconds and a cached repeat load took 1.57 seconds. These costs and the unresolved router prevent promotion from this diagnostic.

## Frozen promotion decision

| Gate | Requirement | Result | Pass |
| --- | ---: | ---: | --- |
| Turkish routing | at least 90% | 76.67% | No |
| Turkish AUTO R@5 | at least 85% | 80.00% | No |
| English R@5 regression | at most 2 pp | 0 pp | Yes |
| Cheap routing/adaptation latency | under 5 ms preferred | 0.124/0.159 ms median/p95 | Yes |

The candidate failed the two quality gates. Production code and behavior remain unchanged, no second frozen run was warranted, and the frozen personal acceptance run was not repeated.

Personal acceptance therefore remains: Turkish routing 55.56%, Turkish useful Top-5 77.78%, overall AUTO routing 77.78%, overall useful Top-5 83.33%, and English useful Top-5 88.89%. There are no after values because running held-out acceptance after a failed development gate would violate the protocol.

The exact recommendation is to avoid claiming reliable Turkish support in SceneMind v1.0. Add independent Turkish source groups and freeze a new router-validation split before reconsidering a compact semantic Speech branch. Do not tune this held-out set, do not add blanket translation, and do not modify the already-usable direct Turkish visual path.

## Requested final report

| # | Item | Result |
| ---: | --- | --- |
| 1 | Turkish development source count | 6 source groups / 7 videos |
| 2 | Turkish development query count | 72: 42 calibration / 30 held-out |
| 3 | Baseline Turkish routing accuracy | 53.33% held-out |
| 4 | Final Turkish routing accuracy | 76.67% held-out |
| 5 | Baseline Turkish Visual R@5 | 93.33% |
| 6 | Final Turkish Visual R@5 | 86.67% cheap candidate |
| 7 | Baseline Turkish Speech R@5 | 75.00% |
| 8 | Final Turkish Speech R@5 | 87.50% cheap candidate |
| 9 | Baseline Turkish Hybrid R@5 | 71.43% |
| 10 | Final Turkish Hybrid R@5 | 71.43% cheap candidate |
| 11 | Final Turkish AUTO R@5 | 80.00% cheap candidate |
| 12 | Selected language strategy | Original Turkish routing plus path-specific cheap adaptation; no promotion |
| 13 | Translation used? | No; paired diagnostic rejected blanket translation |
| 14 | Multilingual embedding used? | Diagnostic only; not production |
| 15 | Added median/p95 latency | Cheap: 0.124/0.159 ms; semantic diagnostic: 9.995/14.378 ms |
| 16 | Added RAM | Cheap: 0 measured; semantic diagnostic: 266,276,864 bytes |
| 17 | English regression | 0 percentage points; English path unchanged |
| 18 | Promotion gate passed? | No |
| 19 | Second/frozen acceptance rerun? | No |
| 20 | Held-out Turkish routing before/after | 55.56% / no after-run |
| 21 | Held-out Turkish Top-5 before/after | 77.78% / no after-run |
| 22 | Overall useful Top-5 before/after | 83.33% / no after-run |
| 23 | Production modified? | No |
| 24 | Exact remaining blockers | Router source generalization; harmful blanket visual adaptation; semantic Speech resource cost with unresolved routing |
| 25 | Another ML milestone? | Not immediately; expand independent sources and freeze a new router-validation split first |
