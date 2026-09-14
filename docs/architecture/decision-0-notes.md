# Decision 0 — live source inventory (lime-green.co.uk)

Pulled from the live site's `sitemap.xml` and spot-checked pages. Supersedes the hypothesis in the mental-model record's §8.6, where noted.

## Confirmed

- `robots.txt` allows all crawlers and points to a real `sitemap.xml` — the record assumed this could not be confirmed and told decision 0 not to rely on a sitemap; it exists and is complete. Crawl by sitemap.
- Product pages: 6 categories (Lime Mortar, Lime Plaster, Lime Render, Insulation, Primers & Adhesives, Stone Repair), ~35 individual product pages, plus `/products-by-colour` (24 colour pages).
- Technical datasheets: one PDF per product, plus up to 6 other PDF types per product (SDS, UK/EU DoP, Carbon Footprint, EPD, LRV) — confirms the record's decision to classify by link text, not filename, and to exclude everything except the TDS.
- Knowledge base: `/support/knowledgebase`, 15 articles (record assumed "about five" — correct this in §8.4/§8.7).
- FAQ: `/support/faq`, one page, **~41 questions across 6 sections** (General, Plasters, Mortars, Renders, Warmshell Systems) — record cites "17 items" throughout (§10.2, §8.7, the completeness check in §7.6); this needs correcting wherever it appears.
- Case studies: 24, matching the named examples already in the record (Tower of London, Westminster Fire Station, Globe Theatre, etc.).

## Rough tier-one corpus size, corrected

~35 product pages + ~30 TDS PDFs + 1 FAQ page (41 Q&As, chunked per question) + 15 KB articles + contact/find-a-supplier ≈ **83 addressable units**, above the record's "about forty documents" cap. The spend-order in §8.7/§10.2 (contact → all TDS → FAQ → KB articles → remaining product pages) will need to actually cut, not just theoretically allow for cutting.

## Not yet checked

- Warmshell subsection pages (4)
- `find-a-supplier` and `contact` page structure (needed for the hand-off text used by `REFUSE`/`HANDOFF` in the architecture diagram)
- Whether datasheet PDFs keep qualifiers/caveats in the same section as their figures (the record's chunking strategy depends on this — still open per §10.8)
- Actual PDF text extraction quality (garbled units, as predicted, unverified)
