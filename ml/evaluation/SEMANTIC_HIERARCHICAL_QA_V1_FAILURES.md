# Semantic Hierarchical Q&A V1 failures

The BM25 baseline has four false abstentions, four incomplete answers, and two wrong-evidence answers. The candidate reduces false abstentions to two, but produces six incomplete and four wrong-evidence answers. Both systems preserve 100% correct abstention, 0% false answers, 100% grounded answers, 100% citation precision, and 0% unsupported claims.

The central failure is section discrimination. On the 44-minute conference source, repeated terms such as *course*, *pilot*, *students*, *editing*, and *Wikipedia* occur across many speakers and initiatives. Context embeddings find topically similar sections, while the fixed selector sometimes chooses the wrong pilot or a nearby red-flag category. The generator then answers safely from cited evidence but does not answer the intended relation.

The second failure is material-fact coverage. The candidate often includes evidence for part of a multi-fact answer yet omits a number, consequence, navigation step, or secondary mechanism. Mean evidence completeness improves by 9.09 points, but fully complete packages remain 87.88%, below the 90% gate. Generation can also omit a required fact even when the package contains it.

The candidate's 36.36% success on the long source is catastrophic. This prevents promotion independently of aggregate metrics. No post-validation parameter, chunk, embedding, selector, prompt, or abstention change was made.
