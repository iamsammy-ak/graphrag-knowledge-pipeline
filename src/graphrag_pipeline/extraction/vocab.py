"""Shared vocabulary for extraction: entity typing hints and relation-type canonicalisation.

The offline extractor produces predicates such as ``headquarter_in`` or ``chief_executive_of`` from verb lemmas,
prepositions and role nouns. ``canonical_relation`` maps them onto a small, fixed vocabulary so that edges from
different phrasings ("was founded by", "started", "founder of") line up, and so that extraction can be scored
against a hand-labelled gold set. The LLM extractor is asked to use the same vocabulary.
"""

from __future__ import annotations

from ..schemas import EntityType

#: spaCy NER labels we keep, mapped onto our entity types. Dates, numbers and nationalities are dropped.
NER_LABELS: dict[str, EntityType] = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "PRODUCT": "PRODUCT",
    "WORK_OF_ART": "PRODUCT",
    "EVENT": "EVENT",
}

CANONICAL_RELATIONS: tuple[str, ...] = (
    "FOUNDED",
    "LEADS",
    "WORKS_FOR",
    "LOCATED_IN",
    "BORN_IN",
    "STUDIED_AT",
    "ACQUIRED",
    "SUBSIDIARY_OF",
    "INVESTED_IN",
    "FINANCED",
    "PARTNERED_WITH",
    "MAKES",
    "USES",
    "SUPPLIES",
)

#: Last words that mark an organisation name ("Velltrix Systems", "Harrowgate Capital").
ORG_HEAD_WORDS = frozenset(
    """inc ltd llc plc gmbh corp corporation company co group holdings partners capital bank labs lab systems
    technologies software robotics motors materials energy freight logistics ports optics foods bio pharma
    therapeutics analytics dynamics industries networks media studios ventures fund foundation institute
    university polytechnic college school council agency ministry airlines""".split()
)
#: Last words that mark a place name ("Vey Harbour", "Lantern Square").
LOCATION_HEAD_WORDS = frozenset(
    "city province county river valley harbour harbor bay square street island lake mountains district region".split()
)
#: Last words that mark a product name ("Glint SDK", "Halcyon Sensor").
PRODUCT_HEAD_WORDS = frozenset(
    "sdk sensor sensors lens truck analyser analyzer robot gel toolkit platform app api chip engine suite".split()
)

#: Head nouns of descriptions and definite descriptions ("a battery manufacturer", "the startup") and the entity
#: type they imply. Used both to type entities and to resolve "the company" back to a named entity.
GENERIC_NOUNS: dict[str, EntityType] = {
    **{
        word: "ORGANIZATION"
        for word in """company firm startup business operator bank fund manufacturer maker producer group university
        polytechnic institute school lab laboratory supplier developer division subsidiary""".split()
    },
    **{word: "LOCATION" for word in "city town province region port capital centre center country".split()},
    **{
        word: "PRODUCT"
        for word in """sensor truck analyser analyzer robot toolkit software application system cell lens gel device
        product unit electrolyte platform""".split()
    },
}

# predicate key -> (canonical type, reverse direction?)
_PREDICATES: dict[str, tuple[str, bool]] = {}


def _register(canonical: str, keys: str, reverse: bool = False) -> None:
    for key in keys.split():
        _PREDICATES[key] = (canonical, reverse)


_register("FOUNDED", "found co_found cofound start establish founder_of co_founder_of cofounder_of")
_register(
    "LEADS",
    """lead run head ceo_of chief_executive_of chief_executive_officer_of executive_of managing_partner_of
    rector_of president_of head_of director_of chairman_of""",
)
_register(
    "WORKS_FOR",
    """work_at work_for join join_from spend_at engineer_at researcher_at employee_of chief_technology_officer_of
    technology_officer_of cto_of chief_operating_officer_of operating_officer_of coo_of chief_financial_officer_of
    cfo_of officer_of member_of board_of chair chair_of partner_at professor_at teach_at advise""",
)
_register(
    "LOCATED_IN",
    """headquarter_in headquartered_in base_in based_in locate_in located_in situate_in lie_on lie_in capital_of
    city_in city_of port_in centre_of center_of""",
)
_register("BORN_IN", "bear_in born_in")
_register("STUDIED_AT", "study_at graduate_from graduate_of educate_at train_at attend alumnus_of student_at")
_register("ACQUIRED", "acquire buy purchase takeover_of")
_register("SUBSIDIARY_OF", "division_of subsidiary_of part_of unit_of")
_register("INVESTED_IN", "invest_in back fund investor_in")
_register("FINANCED", "lend_to finance loan_to lend")
_register("PARTNERED_WITH", "partner_with collaborate_with partnership_with")
_register("MAKES", "make develop build produce manufacture design create maker_of manufacturer_of developer_of")
_register(
    "USES", "use navigate_with read_with run_on rely_on integrate contain build_from operate ship_with customer_for"
)
_register("USES", "power", reverse=True)
_register("SUPPLIES", "supply provide supply_to supplier_of")
_register("SUPPLIES", "customer_of buy_from", reverse=True)


def normalise_key(predicate: str) -> str:
    return "_".join(predicate.lower().replace("-", "_").replace(" ", "_").split("_")).strip("_")


def canonical_relation(predicate: str, target_type: str | None = None) -> tuple[str, bool]:
    """Map a raw predicate onto the canonical vocabulary.

    Leading modifiers are dropped one at a time ("former_engineer_at" -> "engineer_at"). A predicate ending in
    ``_in`` whose target is a location falls back to LOCATED_IN ("company_in Dunmarrow"). Unknown predicates keep
    their own upper-cased name, so nothing is silently discarded. Returns ``(type, reverse)``.
    """
    key = normalise_key(predicate)
    parts = [p for p in key.split("_") if p]
    for i in range(len(parts)):
        candidate = "_".join(parts[i:])
        if candidate in _PREDICATES:
            return _PREDICATES[candidate]
    if target_type == "LOCATION" and parts and parts[-1] == "in":
        return "LOCATED_IN", False
    return key.upper(), False


def refine_type(name: str, current: EntityType) -> EntityType:
    """Correct an NER type with surface cues: organisation suffixes, place words, product words, model numbers."""
    words = name.lower().replace("-", " ").split()
    if not words:
        return current
    first, last = words[0], words[-1]
    if last in ORG_HEAD_WORDS or first in {"university", "bank"} or "institute" in words:
        return "ORGANIZATION"
    if last in LOCATION_HEAD_WORDS:
        return "LOCATION"
    if last in PRODUCT_HEAD_WORDS:
        return "PRODUCT"
    if any(ch.isdigit() for ch in name) and any(ch.isalpha() for ch in name):
        return "PRODUCT"
    return current
