# Grounded Video Q&A V1 failures

The dominant category is **MISSED_ABSTENTION**. One of three verified unanswerable questions (`val-12`, mobile data versus Wi-Fi) produced a substantive answer. Correct abstention is 66.67% and false-answer rate is 33.33%.

`val-09` is a **TEMPORAL_FRAGMENTATION / UNSUPPORTED_CLAIM / BAD_CITATION** failure. Its first sentence correctly says DNS follows DHCP and cites the expected interval. Its second sentence claims the speaker next returns to entering an address in a browser, but cites a passage near the start of the talk. That earlier passage supports the browser topic but cannot support the temporal ordering claim.

Failure counts:

- Retrieval misses at Top-5: 0/9 answerable questions.
- Provider generation errors: 0/12 calls.
- Answers containing unsupported material claims: 2/10 substantive answers.
- Citation-completeness failure answers: 2/10 substantive answers.
- Abstention failures: 1/3 unanswerable questions.

The next experiment must use new development sources to test an explicit answerability gate or evidence-sufficiency policy. It must not tune against these validation outputs or rerun this set as an untouched holdout.
