# Q&A Structured-Question Evidence V1 Failures

## Summary

The frozen run contains ten incorrect outcomes: three list/count failures, three temporal failures, one ordinary failure and three false answers on hard negatives. One failure is a false abstention. The remaining nine produce an answer that is incomplete, targets the wrong fact or accepts a false premise.

## List/count

| ID | Failure | Diagnosis |
| --- | --- | --- |
| `val-pasta-06` | The kneading sequence becomes “down and up, twist down, twist over.” | The generator splits overlapping wording into three syntactic claims instead of three distinct annotated actions. Count form passes while semantic distinctness fails. |
| `val-pasta-07` | Abstains on three sauce-finishing additions. | Relevant material is spread through nearby chunks, but the generator focuses on the later cream/Parmesan finish and declares the third item absent. |
| `val-lang-08` | Gives “accident” and reliance on one data system as two reasons. | The second phrase explains the first reason; the answer omits the separate statement that international English is probably the best worldwide choice. |

## Temporal

| ID | Failure | Diagnosis |
| --- | --- | --- |
| `val-pasta-10` | Answers a cutting step after flouring the board instead of adding water after making a well. | Normalized lexical coverage localizes the generic word *flour* at 196 s rather than the semantic anchor at 68–82 s. |
| `val-pasta-12` | Describes starting the sauce instead of the ingredient preparation before it. | The base BM25 chunk crosses the anchor boundary and overrules the direction-specific target evidence. |
| `val-lang-10` | Returns the naming joke immediately after “German German,” not the annotated next topic of other German variants. | The bounded window preserves order but does not identify which post-anchor statement the question intends. |

`val-lang-12` answers “European English” correctly, but its dedicated temporal roles are wrong: the anchor is part of the target statement and its target roles point to later date-format details. It counts as answer-correct but fails temporal citation correctness.

## Ordinary

| ID | Failure | Diagnosis |
| --- | --- | --- |
| `val-lang-05` | Recommends European English rather than international English (`EN-001`). | BM25 favors the earlier European-English passage; the answer is locally grounded but targets the wrong entity. |

## Hard-negative false answers

| ID | False answer | Safety failure |
| --- | --- | --- |
| `val-pasta-15` | Says flour is added after a pasta maker rolls sheets. | The video explicitly shapes pasta by hand and says no pasta maker is needed. Expansion maps a nonexistent premise onto a nearby real event. |
| `val-pasta-17` | Calls chicken stock and frozen peas “two meats.” | Evidence contains the nouns but not the requested category relation. |
| `val-lang-17` | Calls European English and international English “benchmark results.” | Evidence contains both locale names but no benchmark or comprehension result. |

These three failures show the dominant safety defect: lexical evidence can satisfy requested count and local relevance while failing the question's category or premise. Role labels and timestamp direction cannot repair that semantic mismatch.

## Historical replay

The post-decision replay recovered `val-ui-03` (Tauri frontend/backend synchronization) and `val-ui-04` (Dioxus). It still failed `val-rewire-03` by omitting the time-in-node-graph guideline and failed `val-rewire-04` by answering “oscilloscope.” Historical replay is diagnostic only and was not used to change the frozen system.

## Required next evidence

Do not patch these examples or reuse this validation set for tuning. Any future milestone needs new sources and must predeclare tests for false-premise rejection, semantic type constraints, anchor disambiguation, base-chunk boundary leakage and list-item distinctness. Ask Video remains disabled.
