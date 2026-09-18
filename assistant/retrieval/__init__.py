"""From a question to the passages that may answer it.

`retrieve.py` is thin by design. It expands the question with the handful of
synonyms that close the vocabulary gap and then lets the repository do the
filtering and the ranking, because that is where the audience filter belongs.
Its one hard refusal is the important line in the package: an index built with
one embedding model and queried with another does not fail, it returns
plausible and confidently wrong passages, so the mismatch is raised at load
rather than warned about.

`candidates.py` and `compatibility.py` are the eligibility half, and they exist
because retrieving a passage that mentions a product is not permission to
recommend that product. Each candidate is assessed against the corpus per
required property, and a product whose substrate suitability is not
independently established cannot be recommended.

What is deliberately absent is the data, not the seam. `compatibility.py` can
load a structured product-to-substrate matrix and gate retrieval on it, and
`tests/test_compatibility_matrix.py` exercises that path — but no such matrix
exists in the published corpus and none was invented, so none ships under
`config/`. With no matrix loaded, `candidates._documented_rule` returns `None`
for every pair, and `None` is deliberately three-valued: it reads as unknown
rather than as permission, or an absent matrix would silently approve
everything it failed to mention. Building the matrix is partnership work; the
seam is kept clean so it can arrive without reopening this package.
"""
