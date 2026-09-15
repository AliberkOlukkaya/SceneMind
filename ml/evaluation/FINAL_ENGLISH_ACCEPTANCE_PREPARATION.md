# Final English acceptance preparation result

Historical note: this document records the pre-run inventory decision. Eligible media was later supplied and the completed decision C result is in [FINAL_ENGLISH_ACCEPTANCE_RESULTS.md](FINAL_ENGLISH_ACCEPTANCE_RESULTS.md).

Decision: **D — acceptance not run**.

Conservative search UX is implemented without changing retrieval. AUTO remains default; Visual, Speech, and Hybrid overrides remain available. Results are titled “Most relevant moments,” described as possible matches ranked by relevance, and no longer display raw scores. Empty candidate lists suggest another mode or wording without claiming the requested content is absent.

The final quality run was not executed because no suitable media exists locally. An exact-SHA inventory of 41 unique media candidates found 33 decodable files and eight deliberately corrupt/unreadable test artifacts. One decoded file is in the required 30–60 minute range: the known 2,700-second synthetic repetition used only for infrastructure stress. The longest real content is 1,369.633 seconds, is silent, and falls below 30 minutes; two differently encoded copies exist. None may be used as final search-quality evidence.

| Item | Result |
| --- | --- |
| Conservative UX implemented | Yes |
| Production retrieval modified | No |
| AUTO default / explicit modes | Preserved / preserved |
| English-first scope documented | Yes |
| No-match limitation documented | Yes |
| Suitable real media available | No |
| Final acceptance executed | No |
| Video duration / processing / RAM | Not measured; no eligible run |
| Routing / useful Top-1/3/5 | Not measured; no eligible run |
| Negative-query UX | Candidate moments remain visible with one restrained relevance caveat; no absence claim |
| Acceptance gate | Not evaluated |
| Remaining blocker | One real continuous 30–60 minute English-speaking video, legal for local testing and unused for tuning |
| ML experimentation | Stop for v1.0 while acceptance input is missing |
| Next milestone | Obtain the eligible video, human-review and freeze 25–40 queries, then run the normal durable product pipeline once |

The manifest template, validator/summarizer, fixed categories, failure taxonomy, metrics, and execution protocol are ready. Populated private artifacts remain under ignored `data/`.
