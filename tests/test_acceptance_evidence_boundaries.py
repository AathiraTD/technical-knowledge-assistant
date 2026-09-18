"""A cited sentence is not the same as a supported claim, and this is where that is enforced.

Every failure in this file is a sentence that would survive naive grounding.
The passage is real, the citation points at it, the words overlap — and the
claim is still wrong, because the evidence is about a different product, the
opposite layer order, the negated form of the predicate, or a condition the
answer dropped on the way out. Word overlap with a cited passage is a weak
property, and this suite is the collection of ways it is too weak.

Four boundaries are exercised, each with hand-written passages so the case is
exactly the shape being argued about.

**Scope, before anything is considered.** `scoped_evidence` removes passages
about other products, and the interesting part is the two exceptions it keeps:
a product explicitly named in a comparison, and a passage that stands in a
published relation to the asked product — Bond priming Ultra, Forte setting
before Tradirend. Dropping those would lose real answers; keeping anything
further would let Duro's mixing figure sit in front of an Ultra question.

**Direction and polarity.** "Solo can be applied over Ultra" is not evidence
that Ultra can be applied over Solo, and the two sentences share almost every
word — so the check cannot be lexical. `test_inverse_direction_is_not_word_overlap_evidence`
states that outright, and the long parametrised tables either side of it split
the space in two: claims that **are** supported must pass unaltered, and claims
that broaden the source in any way — dropping "only internally", "after
priming", "cured", "we cannot confirm", or flipping `cannot` to `can` — must
fail. These are `check 8`, the relationship and property-predicate check, and
this is the only suite that exercises it — `tests/test_checks.py` stops at
check 7. `docs/architecture.md` now lists all eight; it named only seven until
a documentation audit reconciled it against the span the code actually opens.

**Subject.** A property must be predicated **of the product asked about**.
"Solo Primer is waterproof" does not make Solo waterproof; "a coating for Ultra
is waterproof" does not make Ultra waterproof; "unlike Ultra, Solo is
waterproof" positively says the opposite. Adjacency is the trap here, and
`test_adjacent_salts_and_draught_are_not_waterproof_evidence` shows both halves
of it — the fabricated claim fails check 8, and quoting what the sheet really
says fails check 6, because resisting salts is not an answer to "is it
waterproof?".

**Deterministic rendering.** The rest drives `AnswerEngine.factual` with no
model at all: figures are copied from the passage rather than from anything the
engine knows about the product, a multi-field question answers each field with
its own citation and says plainly which field the evidence did not establish, a
purchase question quotes the coverage figure and refuses the arithmetic, and
caveats are appended only from documents that were actually cited.

The `engine` fixture is a `SimpleNamespace` standing in for the repository —
there is no store, no retrieval, no snapshot and no Ollama in this file. It
proves what the checks reject and what the deterministic extractor prints; it
says nothing about what retrieval would have found.
"""

from types import SimpleNamespace

import pytest

from assistant.answering.answer import AnswerEngine, run_checks, scoped_evidence
from assistant.knowledge.model import Chunk, Document, Retrieved
from assistant.answering.router import Decision, Path_


NAMES = {"products": ["Ultra", "Solo", "Duro", "Bond", "Forte", "Tradirend"]}


def passage(content, product="Ultra", section="Application", kind="datasheet"):
    """One synthetic retrieved passage, with its product and section carried in metadata."""
    url = f"https://example.invalid/{product}/{section}"
    return Retrieved(
        chunk=Chunk(canonical_url=url, version=1, chunk_index=0, section=section,
                    content=content, product=product, document_type=kind, authority=1),
        document=Document(canonical_url=url, title=product, product=product,
                          document_type=kind, authority=1),
        score=0.71,
    )


def decision(hits, product="Ultra", **kwargs):
    """A router decision that has already chosen Compose, so only the evidence varies."""
    return Decision(Path_.COMPOSE, "synthetic evidence", hits=hits,
                    slots={"product": product}, **kwargs)


@pytest.fixture
def engine():
    """An `AnswerEngine` over a stub repository: known product names, no caveats, no store."""
    repo = SimpleNamespace(snapshot=lambda: SimpleNamespace(notes=NAMES),
                           caveats=lambda _url: [])
    return AnswerEngine(repo, SimpleNamespace())


def checks(text, source, question=""):
    """Run the post-generation checks over one claim and one supporting passage."""
    return run_checks(text, [passage(source)], NAMES, [], product="Ultra",
                      asked_products=("Ultra", "Solo"), question=question)


def test_scoping_drops_other_products_even_without_own_answer():
    """Evidence about another product is removed even when that leaves nothing to say.

    Answering from the wrong product's sheet is worse than answering nothing.
    """
    unrelated = passage("Mix Duro with 8 litres.", "Duro", "Mixing")
    assert scoped_evidence(decision([unrelated]), registry=NAMES["products"]) == []


def test_scoping_keeps_explicit_comparison_and_related_primer():
    """The two exceptions: a product the question names, and one in a published relation."""
    ultra = passage("Apply Ultra to a damp surface.")
    solo = passage("Apply Solo in one coat.", "Solo")
    primer = passage("Prime Ultra with Bond.", "Bond", "Preparation")
    unrelated = passage("Mix Duro with 8 litres.", "Duro", "Mixing")
    d = decision([ultra, solo, primer, unrelated])
    assert scoped_evidence(d, registry=NAMES["products"]) == [ultra, primer]
    assert scoped_evidence(d, "Compare Ultra and Solo", NAMES["products"]) == [
        ultra, solo, primer]


def test_scoping_preserves_cross_product_finishing_relation():
    """A finishing-coat sentence naming both products survives scoping to the finish product."""
    finish = passage("Leave Forte to set before applying Tradirend.", "Forte",
                     "Finishing Coats")
    assert scoped_evidence(decision([finish], "Tradirend"),
                           registry=NAMES["products"]) == [finish]


@pytest.mark.parametrize("claim", [
    "Ultra can be applied over Solo",
    "Apply Ultra on top of Solo",
    "Ultra may be used over Solo",
    "Solo can be finished with Ultra",
    "Solo may be coated using Ultra",
    "Solo can be used under Ultra",
])
def test_inverse_direction_is_not_word_overlap_evidence(claim):
    """Six reversals of one published layer order, each sharing nearly every word with it.

    This is the case that makes a lexical support check unusable: "Solo can be
    applied over Ultra" and its inverse differ by word order alone.
    """
    failures = checks(claim + " [1].", "Solo can be applied over Ultra.")
    assert any("check 8" in f for f in failures)


@pytest.mark.parametrize(("source", "claim"), [
    ("Solo can be applied over Ultra.", "Solo can be applied over Ultra"),
    ("Ultra can be finished with Solo.", "Solo can be applied over Ultra"),
    ("Ultra can be used under Solo.", "Solo can be used over Ultra"),
    ("Solo can be applied over Ultra.", "Ultra can be finished with Solo"),
    ("Ultra can be applied over Solo.", "Ultra can be applied over Solo"),
    ("Solo can be applied over cured Ultra.", "Solo can be applied over cured Ultra"),
    ("Ultra cannot be applied over Solo.", "Ultra cannot be applied over Solo"),
    ("Ultra can be applied over Solo only internally.",
     "Ultra can be applied over Solo only internally"),
])
def test_direct_supported_direction_and_polarity_pass(source, claim):
    """Claims the passage actually supports pass, including negated and conditioned ones.

    The half of the contract that stops the check from being satisfied by
    refusing everything.
    """
    assert checks(claim + " [1].", source) == []


@pytest.mark.parametrize(("source", "claim"), [
    ("Ultra cannot be applied over Solo.", "Ultra can be applied over Solo"),
    ("Ultra can be applied over Solo.", "Ultra cannot be applied over Solo"),
    ("Do not apply Ultra over Solo.", "Apply Ultra over Solo"),
    ("Ultra might be applied over Solo.", "Ultra can be applied over Solo"),
    ("Ultra can be applied over Solo only internally.", "Ultra can be applied over Solo"),
    ("Ultra can be applied over Solo after priming.", "Ultra can be applied over Solo"),
    ("Ultra can be applied over Solo if the surface is sound.",
     "Ultra can be applied over Solo if the surface is damp"),
    ("Ultra is not proven compatible with Solo.", "Ultra is compatible with Solo"),
    ("Solo can be applied over Ultra.", "Ultra is compatible with Solo"),
    ("Solo can be applied over cured Ultra.", "Solo can be applied over Ultra"),
    ("Solo can be applied over cured Ultra.", "Ultra can be applied over cured Solo"),
    ("Ultra can be applied over Solo.", "Ultra can be applied directly over Solo"),
    ("Ultra can be applied over Solo after priming.",
     "Ultra can be applied over Solo before or after priming"),
    ("Ultra cannot be applied over Solo; do not use outdoors.",
     "Ultra can be applied over Solo; do not use outdoors"),
    ("Ultra cannot be applied over Solo and do not use outdoors.",
     "Ultra can be applied over Solo and do not use outdoors"),
    ("Can Ultra be applied over Solo?", "Ultra can be applied over Solo"),
    ("We cannot confirm Ultra can be applied over Solo.",
     "Ultra cannot be applied over Solo"),
])
def test_relationship_negation_uncertainty_and_conditions_are_not_discarded(source, claim):
    """Seventeen ways to broaden a relationship, all refused.

    Flipping polarity, dropping a restriction, upgrading "might" to "can",
    adding "directly", answering a question as though it were a statement, and
    treating "we cannot confirm X" as a statement of X.
    """
    assert any("check 8" in f for f in checks(claim + " [1].", source))


def test_adjacent_salts_and_draught_are_not_waterproof_evidence():
    """Both halves of the near-miss: the invented claim fails check 8, the honest quote fails check 6."""
    source = "Ultra resists salts and excludes draughts."
    assert any("check 8" in f for f in checks("Ultra is waterproof [1].", source))
    assert any("check 6" in f for f in checks(
        "Ultra resists salts and excludes draughts [1].", source,
        question="Is Ultra waterproof?"))


def test_waterproof_negation_is_preserved():
    """"Not waterproof" supports "not waterproof" and refuses "waterproof"."""
    assert checks("Ultra is not waterproof [1].", "Ultra is not waterproof.") == []
    assert checks("Ultra is waterproof [1].", "Ultra is not waterproof.")


@pytest.mark.parametrize("source", [
    "Ultra is not waterproof. Solo is waterproof.",
    "Ultra is not waterproof; Solo is waterproof.",
    "Ultra is not waterproof but Solo is waterproof.",
    "Ultra resists salts and Solo is waterproof.",
    "Unlike Ultra, Solo is waterproof.",
    "Solo is waterproof, unlike Ultra.",
])
def test_waterproof_support_is_bound_to_product(source):
    """Six sentence shapes where the property belongs to Solo and is claimed for Ultra.

    Including the contrastive forms — "unlike Ultra, Solo is waterproof" —
    where the passage says the opposite of the claim it is cited for.
    """
    assert any("check 8" in f for f in checks("Ultra is waterproof [1].", source))


def test_implicit_waterproof_claim_cannot_borrow_another_products_property():
    """A claim whose subject is only "it" cannot inherit the nearest product's property."""
    assert any("check 8" in f for f in checks(
        "It is waterproof [1].", "Ultra resists salts and Solo is waterproof."))


def test_an_asked_property_must_be_answered_not_just_adjacent():
    """Quoting a true neighbouring sentence is not answering the property that was asked about."""
    assert any("check 6" in f for f in checks(
        "Ultra resists salts [1].", "Ultra resists salts. Ultra is not waterproof.",
        question="Is Ultra waterproof?"))


@pytest.mark.parametrize("amount", ["2.75", "8.25", "11"])
def test_water_is_extracted_from_evidence_not_product_constants(engine, amount):
    """Three different published figures are each copied from the passage, not from a constant.

    A competing 90-litre figure for another product is in the same decision and
    must not appear.
    """
    source = f"Mix Ultra with approximately {amount} litres of water per 19 kg bag."
    d = decision([passage("Mix Solo with 90 litres.", "Solo", "Mixing"),
                  passage(source, section="Mixing")])
    answer = engine.factual(d, "How much water does Ultra need?")
    assert answer and not answer.refused
    assert source.rstrip(".") in answer.text
    assert "90 litres" not in answer.text
    assert answer.sources[0]["score"] == 0.71
    assert not answer.failed_checks


def test_numeric_checks_still_reject_fabrication_product_swap_and_lost_qualifier():
    """Checks 2, 3 and 4: an invented figure, a figure moved to another product, a dropped "approximately"."""
    hits = [passage("Mix Ultra with approximately 8 litres per 19 kg bag.",
                    section="Mixing"), passage("Solo is a finish.", "Solo")]
    for text in ("Mix Ultra with 9 litres per 19 kg bag [1].",
                 "Mix Solo with approximately 8 litres per 19 kg bag [1].",
                 "Mix Ultra with 8 litres per 19 kg bag [1]."):
        assert run_checks(text, hits, NAMES, [], product="Ultra")


def test_numeric_contrast_is_not_attribution_to_every_mentioned_product():
    """A figure in a contrastive sentence belongs to one product, not to both it names."""
    assert run_checks("Ultra needs 8 litres of water [1].", [
        passage("Unlike Ultra, Solo needs 8 litres of water.")], NAMES, [], product="Ultra")


def test_short_product_name_cannot_evade_product_scope_check():
    """Check 7 matches whole product identities, so a shared first word is not a match."""
    names = {"products": ["Ultra Insulating Base Coat", "Solo Onecoat Plaster"]}
    hits = [passage("Solo needs 8 litres of water.", "Solo Onecoat Plaster")]
    assert any("check 7" in f for f in run_checks(
        "Solo needs 8 litres of water [1].", hits, names, [], product="Ultra"))


def test_thickness_is_not_the_finishing_layer_on_same_datasheet(engine):
    """Two thicknesses in one passage: the base coat's is answered, the finish coat's is not."""
    source = ("Ultra should be applied at a uniform thickness of 12 to 28mm. "
              "Apply Solo as a finish coat at 3 to 6mm.")
    answer = engine.factual(decision([passage(source)]),
                            "What thickness should I apply Ultra at?")
    assert answer and "12 to 28mm" in answer.text
    assert "3 to 6mm" not in answer.text


def test_unnamed_finish_coat_is_not_a_basecoat_thickness(engine):
    """A "Finishing Coats" figure with no product named is not the base coat's thickness."""
    hit = passage("The finish coat should be 3 to 6mm thick.",
                  "Ultra Insulating Base Coat", "Finishing Coats")
    answer = engine.factual(decision([hit]), "How thick should Ultra be?")
    assert answer and answer.refused
    assert "3 to 6mm" not in answer.text
    assert run_checks("Ultra should be 3 to 6mm thick [1].", [hit], NAMES, [],
                      product="Ultra")


def test_finish_product_keeps_its_own_thickness(engine):
    """The same passage does answer the question when the finish product is the one asked about."""
    hit = passage("Apply Solo as a finish coat at 3 to 6mm.", "Solo", "Finishing Coats")
    answer = engine.factual(decision([hit], "Solo"), "What thickness should Solo be?")
    assert answer and not answer.refused
    assert "Apply Solo as a finish coat at 3 to 6mm" in answer.text


def test_multiask_covers_each_field_with_own_citation(engine):
    """Four fields, four citations, and the fifth reported as not established rather than omitted.

    Silence about a field the question asked for reads as though it had no
    answer; naming it as unsupported is the honest form.
    """
    hits = [
        passage("Prepare Ultra backgrounds by removing dust.", section="Preparation"),
        passage("Apply Ultra at a thickness of 12 to 28mm.", section="Application"),
        passage("Mix Ultra with approximately 8 litres of water per bag.", section="Mixing"),
        passage("Only use Ultra above 7 degrees and below 29 degrees.",
                section="Conditions"),
    ]
    answer = engine.factual(decision(hits),
        "For Ultra tell me preparation, thickness, mixing water and curing or conditions.")
    assert answer and not answer.refused
    assert all(f"[{i}]" in answer.text for i in range(1, 5))
    assert "curing: the retrieved evidence does not establish" in answer.text
    assert answer.diagnostics["unsupported_fields"] == ["ultra: curing"]
    assert "Only use Ultra above 7 degrees and below 29 degrees" in answer.text


def test_comparison_keeps_fields_and_unknowns_separate(engine):
    """Each product gets its own block, and Ultra's thickness does not fill Solo's gap."""
    hits = [
        passage("Ultra is suitable for most masonry backgrounds.", section="Description"),
        passage("Apply Ultra at a thickness of 12 to 28mm."),
        passage("Solo is suitable for lath backgrounds.", "Solo", "Description"),
    ]
    answer = engine.factual(decision(hits),
                            "Compare Ultra and Solo for suitable backgrounds and thickness.")
    assert answer and not answer.refused
    assert "solo:" in answer.text and "ultra:" in answer.text
    assert "thickness: the retrieved evidence does not establish this for solo" in answer.text
    solo_block = answer.text.split("solo:", 1)[1].split("ultra:", 1)[0]
    assert "12 to 28mm" not in solo_block
    assert "Solo is suitable for lath backgrounds.” [3]" in solo_block


def test_a_product_named_only_to_exclude_it_is_not_reported_on(engine):
    """"Don't give me figures for Solo or Duro" is not a request to cover them.

    Reading the raw sentence turned the exclusion into a three-product
    breakdown, two of them printed only to say the evidence established nothing
    for them -- an odd way to honour a request not to mention them.
    """
    hits = [
        passage("Ultra should be applied in a uniform thickness of between 10 "
                "and 30mm.", section="How to Apply"),
        passage("Mix with approximately 4 to 4.5 litres of clean water per bag.",
                section="How to Mix"),
    ]
    answer = engine.factual(
        decision(hits),
        "I'm using Lime Green Ultra. What thickness and mixing water should I "
        "use? Please don't give me figures for Solo or Duro.")

    assert answer and not answer.refused
    # The Ultra figures the datasheet publishes.
    assert "between 10 and 30mm" in answer.text
    assert "4 to 4.5 litres of clean water per bag" in answer.text
    # And no per-product block for a product named only to rule it out.
    assert "ultra:" in answer.text
    assert "solo:" not in answer.text and "duro:" not in answer.text
    assert "does not establish this for solo" not in answer.text
    assert "does not establish this for duro" not in answer.text


def test_an_exclusion_clause_does_not_disable_a_genuine_comparison(engine):
    """The narrowing must not cost the comparison it sits next to."""
    hits = [
        passage("Apply Ultra at a thickness of 12 to 28mm."),
        passage("Solo is suitable for lath backgrounds.", "Solo", "Description"),
    ]
    answer = engine.factual(decision(hits),
                            "Compare Ultra and Solo for suitable backgrounds and thickness.")
    assert answer and "solo:" in answer.text and "ultra:" in answer.text


@pytest.mark.parametrize("question", [
    "I have 42 square metres at 20mm. Exactly how many bags of Ultra do I need?",
    "I have 15 square metres at 10mm. How many bags should I buy?",
])
def test_purchase_queries_quote_coverage_without_calculation(engine, question):
    """Coverage is quoted verbatim and the bag count declined; arithmetic is not the model's job."""
    hit = passage("Ultra covers approximately 1.7 m² per bag at 12mm.",
                  section="Coverage")
    answer = engine.factual(decision([hit]), question)
    assert answer and "approximately 1.7 m² per bag at 12mm" in answer.text
    assert "cannot give an exact purchase quantity or bag count" in answer.text
    assert "thickness:" not in answer.text


def test_purchase_does_not_discard_independent_water_question(engine):
    """Refusing the sum must not swallow the answerable half of the same message."""
    hits = [passage("Ultra covers approximately 1.7 m² at 12mm.", section="Coverage"),
            passage("Mix Ultra with 8 litres of water.", section="Mixing")]
    answer = engine.factual(decision(hits),
                            "How much water does Ultra need and how many bags should I buy?")
    assert answer and "8 litres of water" in answer.text
    assert "approximately 1.7 m²" in answer.text
    assert "cannot give an exact purchase quantity" in answer.text


def test_unused_solo_source_does_not_fail_correct_ultra_claims():
    """An irrelevant passage in the decision does not turn a correct claim into a failure."""
    hits = [passage("Apply Ultra at a thickness of 12 to 28mm."),
            passage("Apply Solo at 3mm.", "Solo")]
    assert run_checks("Apply Ultra at a thickness of 12 to 28mm [1].", hits,
                      NAMES, ["thickness"], product="Ultra") == []


def test_unpublished_waterproof_lookup_is_explicitly_limited(engine):
    """An unpublished property refuses and does not print the adjacent sentence as consolation."""
    answer = engine.factual(decision([
        passage("Ultra resists salts and excludes draughts.")]), "Is Ultra waterproof?")
    assert answer and answer.refused
    assert "does not establish" in answer.text
    assert "resists salts" not in answer.text


def test_factual_lookup_does_not_override_policy_route(engine):
    """A decision already routed to Refuse is not rescued by the extractor finding a match."""
    d = decision([passage("Ultra is waterproof.")])
    d.path = Path_.REFUSE
    assert engine.factual(d, "Is Ultra waterproof?") is None


def test_requested_direction_gets_limit_instead_of_reversed_quote(engine):
    """The reversed sentence is not printed at all — quoting it would read as a yes."""
    hit = passage("Solo can be applied over Ultra.")
    answer = engine.factual(decision([hit]), "Can I use Ultra over Solo?")
    assert answer and answer.refused
    assert "does not establish the requested layer order" in answer.text
    assert "Solo can be applied over Ultra" not in answer.text


def test_supported_requested_direction_is_not_blanket_refused(engine):
    """When the order really is published, it is quoted with its condition attached."""
    hit = passage("Ultra can be applied over Solo after priming.")
    answer = engine.factual(decision([hit]), "Can I use Ultra over Solo?")
    assert answer and not answer.refused
    assert "Ultra can be applied over Solo after priming.” [1]" in answer.text


def test_generated_reverse_answer_does_not_answer_requested_order():
    """A truthful sentence in the wrong direction still fails, because it answers a different question."""
    assert any("check 8" in f for f in checks(
        "Solo can be applied over Ultra [1].", "Solo can be applied over Ultra.",
        question="Can I use Ultra over Solo?"))


def test_relationship_does_not_discard_additional_water_question(engine):
    """A compatibility question bundled with a factual one keeps both answers."""
    hits = [passage("Ultra can be applied over Solo."),
            passage("Mix Ultra with 8 litres of water.", section="Mixing")]
    answer = engine.factual(decision(hits),
                            "Can I use Ultra over Solo, and how much water does Ultra need?")
    assert answer and "Ultra can be applied over Solo." in answer.text
    assert "8 litres of water" in answer.text


def test_unknown_comparison_point_is_not_silently_dropped(engine):
    """A requested field with no evidence is named as unestablished and recorded in diagnostics."""
    hit = passage("Apply Ultra at a thickness of 12 to 28mm.")
    answer = engine.factual(decision([hit]), "Tell me Ultra thickness and certification.")
    assert answer and "12 to 28mm" in answer.text
    assert "certification: the retrieved evidence does not establish" in answer.text
    assert "ultra: certification" in answer.diagnostics["unsupported_fields"]


def test_router_identified_additional_field_gets_explicit_limit(engine):
    """Fields the router identified, not only those the extractor recognised, get the same treatment."""
    d = decision([passage("Apply Ultra at a thickness of 12 to 28mm.")],
                 evidence_terms={"thickness": ["thickness"], "strength": ["strength"]})
    answer = engine.factual(d, "Tell me Ultra thickness and strength.")
    assert answer and "strength: the retrieved evidence does not establish" in answer.text


def test_deterministic_extraction_still_runs_checks(engine, monkeypatch):
    """The checks apply to code-quoted answers too, not only to generated prose.

    `run_checks` is forced to fail, and the extracted figure must not print —
    the extract path is not exempt from the safety gate.
    """
    import assistant.answering.answer as answering

    monkeypatch.setattr(answering, "run_checks", lambda *a, **kw: ["check 2: blocked"])
    hit = passage("Mix Ultra with 8 litres of water.", section="Mixing")
    answer = engine.factual(decision([hit]), "How much water does Ultra need?")
    assert answer and answer.refused
    assert "8 litres" not in answer.text


def test_caveats_only_come_from_cited_documents(engine):
    """A caveat travels with the document that was cited, never with one merely retrieved."""
    hit = passage("Mix Ultra with 8 litres of water.", section="Mixing")
    solo = passage("Solo is a finish.", "Solo")
    engine.repo.caveats = lambda url: [
        SimpleNamespace(sentence="Never mix with dirty water." if "Ultra" in url
                        else "An unrelated Solo caveat.")]
    answer = engine.factual(decision([solo, hit]), "How much water does Ultra need?")
    assert answer and answer.caveats == ["Never mix with dirty water."]


def test_third_cited_document_keeps_its_caveat(engine):
    """The caveat of the third cited document is not lost to the first two."""
    hits = [passage("Mix Ultra with 8 litres of water.", section="Mixing"),
            passage("Apply Ultra at a thickness of 12 to 28mm."),
            passage("Only use Ultra above 7 degrees.", section="Conditions")]
    engine.repo.caveats = lambda url: (
        [SimpleNamespace(sentence="Protect from frost.")] if "Conditions" in url else [])
    answer = engine.factual(decision(hits), "Ultra water, thickness and conditions?")
    assert answer and answer.caveats == ["Protect from frost."]


def test_thickness_preserves_substrate_condition_and_warning(engine):
    """The substrate condition on a figure and the warning beside it are both printed."""
    hit = passage("On lath backgrounds, apply Ultra at a minimum thickness of 14mm. "
                  "Do not apply to frozen surfaces.")
    answer = engine.factual(decision([hit]), "How thick should Ultra be?")
    assert answer and not answer.refused
    assert "On lath backgrounds, apply Ultra at a minimum thickness of 14mm" in answer.text
    assert "Do not apply to frozen surfaces" in answer.text


@pytest.mark.parametrize("source", [
    "Ultra requires a waterproof coating.",
    "A waterproof coating can be used with Ultra.",
    "For Ultra, the waterproof coating is Solo.",
    "Ultra is applied to a waterproof substrate.",
    "Ultra mentions waterproof performance.",
    "The substrate is waterproof.",
    "Solo is a coating. It is waterproof.",
])
def test_property_predicate_not_mentions_or_datasheet_metadata(source, engine):
    """Seven sentences that mention the property near the product without predicating it of it.

    Requiring it, being applied to it, being made of it, or merely "mentioning
    waterproof performance" are all not the claim.
    """
    assert any("check 8" in f for f in checks("Ultra is waterproof [1].", source))
    answer = engine.factual(decision([passage(source)]), "Is Ultra waterproof?")
    assert answer and answer.refused


@pytest.mark.parametrize("prop", ["waterproof", "watertight", "structural", "certified"])
def test_property_subject_cannot_be_an_accessory(prop, engine):
    """A property of "Solo Primer" is not a property of Solo, for four different properties."""
    source = f"Solo Primer is {prop}."
    hit = passage(source, "Solo")
    assert run_checks(f"Solo is {prop} [1].", [hit], NAMES, [], product="Solo")
    answer = engine.factual(decision([hit], "Solo"), f"Is Solo {prop}?")
    assert answer and answer.refused


@pytest.mark.parametrize("registered", [False, True])
def test_primer_identity_not_collapsed_into_coating(registered, engine):
    """A primer is a different product whether or not it appears in the published name list.

    Asserted at all three gates — scoping, extraction and the checks — because
    the registry changing must not change the identity rule.
    """
    if registered:
        engine.names["products"] = [*NAMES["products"], "Solo Primer"]
    hit = passage("Mix Solo Primer with 8 litres of water.", "Solo Primer", "Mixing")
    answer = engine.factual(decision([hit], "Solo"), "How much water does Solo need?")
    assert answer and answer.refused
    assert scoped_evidence(decision([hit], "Solo"),
                           registry=engine.names["products"]) == []
    assert run_checks("Mix Solo with 8 litres of water [1].", [hit],
                      engine.names, [], product="Solo")


@pytest.mark.parametrize(("source", "claim"), [
    ("Ultra can be applied over Solo Primer.", "Ultra can be applied over Solo"),
    ("Ultra Primer can be applied over Solo.", "Ultra can be applied over Solo"),
    ("Ultra is mentioned in guidance for applying Bond over Solo.",
     "Ultra can be applied over Solo"),
    ("Ultra is discussed and Bond can be applied over Solo.",
     "Ultra can be applied over Solo"),
    ("Cured Ultra can be applied over Solo.", "Ultra can be applied over cured Solo"),
    ("Ultra can be applied over Solo after curing.", "Ultra can be applied over Solo"),
    ("Ultra can be applied over Solo unless it is damp.", "Ultra can be applied over Solo"),
    ("Ultra can be applied over Solo if it is sound.", "Ultra can be applied over Solo"),
    ("Ultra may or may not be applied over Solo.", "Ultra cannot be applied over Solo"),
    ("It is uncertain whether Ultra can be applied over Solo.",
     "Ultra can be applied over Solo"),
    ("Ultra can be applied over Solo; only after curing.",
     "Ultra can be applied over Solo"),
    ("Ultra can be applied over Solo. Only after curing.",
     "Ultra can be applied over Solo"),
])
def test_relationship_requires_subject_and_attached_restrictions(source, claim):
    """Twelve more relationship failures: wrong subject, wrong product, or a restriction dropped."""
    assert any("check 8" in f for f in checks(claim + " [1].", source))


@pytest.mark.parametrize("source", [
    "Ultra can be applied over Solo only internally.",
    "Ultra can be applied over Solo if the surface is sound.",
    "Ultra can be applied over Solo unless it is damp.",
    "Ultra can be applied over Solo after curing.",
    "Ultra cannot be applied over Solo.",
    "We cannot confirm Ultra can be applied over Solo.",
    "Ultra can be applied over Solo; only after curing.",
    "Ultra can be applied over Solo. Only after curing.",
])
def test_relationship_quotes_keep_all_restrictions(source, engine):
    """If a restricted relationship is quoted at all, it is quoted whole and never opens with "yes"."""
    answer = engine.factual(decision([passage(source)]), "Can I use Ultra over Solo?")
    assert answer and (answer.refused or source in answer.text)
    if not answer.refused:
        assert not answer.text.lower().startswith("yes")
        assert not answer.failed_checks


@pytest.mark.parametrize("source", [
    "Ultra can be applied over Solo Primer.",
    "Ultra Primer can be applied over Solo.",
    "Ultra is mentioned in guidance for applying Bond over Solo.",
])
def test_relationship_lookup_does_not_quote_wrong_subject(source, engine):
    """Where the sentence is about a primer or a third product, the lookup refuses rather than quotes."""
    answer = engine.factual(decision([passage(source)]), "Can I use Ultra over Solo?")
    assert answer and answer.refused


@pytest.mark.parametrize("source", [
    "Ultra is waterproof.",
    "Ultra is not waterproof.",
    "Ultra is waterproof only indoors.",
    "Ultra is waterproof if correctly cured.",
])
def test_direct_property_predicate_still_answers_verbatim(source, engine):
    """A direct predicate — positive, negative, or conditioned — is answered with the sentence itself."""
    answer = engine.factual(decision([passage(source)]), "Is Ultra waterproof?")
    assert answer and not answer.refused
    assert source in answer.text


@pytest.mark.parametrize(("source", "claim"), [
    ("Ultra can be applied over Solo if the surface is not damp.",
     "Ultra cannot be applied over Solo if the surface is not damp"),
    ("Ultra cannot be applied over Solo if the surface is not damp.",
     "Ultra can be applied over Solo if the surface is not damp"),
    ("Ultra is waterproof if the substrate is not wet.",
     "Ultra is not waterproof if the substrate is not wet"),
    ("Ultra is not waterproof if the substrate is not wet.",
     "Ultra is waterproof if the substrate is not wet"),
    ("Ultra is used with a coating applied over Solo.", "Ultra can be applied over Solo"),
])
def test_negation_and_relationship_subject_are_local_to_predicate(source, claim):
    """A "not" inside a trailing condition does not license flipping the main predicate."""
    assert any("check 8" in f for f in checks(claim + " [1].", source))


def test_other_products_true_property_does_not_answer_requested_product():
    """A true claim about Solo fails when Ultra is the product the question asked about."""
    assert run_checks("Solo is waterproof [1].", [
        passage("Ultra resists salts. Solo is waterproof.")], NAMES, [],
        product="Ultra", asked_products=("Ultra", "Solo"), question="Is Ultra waterproof?")


def test_explicit_primer_lookup_remains_supported(engine):
    """Asked about the primer by name, with the primer registered, the passage answers normally."""
    engine.names["products"] = [*NAMES["products"], "Solo Primer"]
    source = "Mix Solo Primer with 8 litres of water."
    hit = passage(source, "Solo Primer", "Mixing")
    answer = engine.factual(decision([hit], "Solo Primer"),
                            "How much water does Solo Primer need?")
    assert answer and not answer.refused
    assert source in answer.text


def test_cross_product_metadata_does_not_hide_explicit_property_subject(engine):
    """A sentence naming Ultra answers about Ultra even on a passage tagged to another product.

    Chunk metadata is a retrieval aid, not the authority on what a sentence is
    about.
    """
    source = "Ultra is waterproof."
    answer = engine.factual(decision([passage(source, "Solo")]), "Is Ultra waterproof?")
    assert answer and not answer.refused
    assert source in answer.text


@pytest.mark.parametrize(("source", "claim"), [
    ("Ultra is waterproof; only after curing.", "Ultra is waterproof"),
    ("Ultra is waterproof but only after curing.", "Ultra is waterproof"),
    ("Ultra cannot be applied over Solo and is unsuitable outdoors.",
     "Ultra can be applied over Solo and is unsuitable outdoors"),
    ("Ultra is not waterproof and is unsuitable outdoors.",
     "Ultra is waterproof and is unsuitable outdoors"),
    ("Ultra isn't waterproof.", "Ultra is waterproof"),
    ("Ultra isn’t waterproof.", "Ultra is waterproof"),
    ("Ultra may be waterproof.", "Ultra is waterproof"),
])
def test_reviewed_predicate_scope_regressions(source, claim):
    """Seven regressions from review: trailing conditions, contractions, and "may be"."""
    assert any("check 8" in f for f in checks(claim + " [1].", source))


@pytest.mark.parametrize("source", [
    "Ultra is waterproof; only after curing.",
    "Ultra is waterproof but only after curing.",
    "Ultra is not waterproof and is unsuitable outdoors.",
    "Ultra isn't waterproof.",
    "Ultra may be waterproof.",
    "Ultra can be applied over Solo if the surface is not damp.",
    "Ultra cannot be applied over Solo if the surface is not damp.",
    "Cured Ultra can be applied over Solo.",
    "Ultra can be applied over cured Solo.",
])
def test_restricted_evidence_can_still_be_quoted_without_broadening(source):
    """The same restricted sentences pass when the claim keeps every restriction.

    The pair to the tests above — without this one, they could be satisfied by
    refusing all conditioned evidence.
    """
    assert checks(source.rstrip(".") + " [1].", source) == []


@pytest.mark.parametrize(("source", "claim"), [
    ("A coating for Ultra is waterproof.", "Ultra is waterproof"),
    ("A coating containing Ultra is waterproof.", "Ultra is waterproof"),
    ("A coating for Ultra can be applied over Solo.", "Ultra can be applied over Solo"),
    ("A coating containing Ultra can be applied over Solo.", "Ultra can be applied over Solo"),
])
def test_product_in_subject_noun_phrase_is_not_necessarily_the_subject(source, claim):
    """"A coating for Ultra" and "a coating containing Ultra" are not Ultra."""
    assert any("check 8" in f for f in checks(claim + " [1].", source))


def test_comma_separated_properties_do_not_share_polarity():
    """Two clauses in one sentence keep their own polarity; the true reading still passes."""
    source = "Ultra is not waterproof, Solo is waterproof."
    claim = "Ultra is waterproof, Solo is not waterproof [1]."
    assert any("check 8" in f for f in checks(claim, source))
    assert checks(source.rstrip(".") + " [1].", source) == []


@pytest.mark.parametrize(("source", "claim"), [
    ("Ultra can be applied over Solo. Do not use outdoors.",
     "Ultra can be applied over Solo"),
    ("Ultra can be applied over Solo; do not use outdoors.",
     "Ultra can be applied over Solo"),
    ("Ultra is waterproof. Do not use outdoors.", "Ultra is waterproof"),
])
def test_attached_negative_warning_cannot_be_dropped(source, claim):
    """A warning in the following sentence or clause travels with the claim it qualifies."""
    assert any("check 8" in f for f in checks(claim + " [1].", source))
