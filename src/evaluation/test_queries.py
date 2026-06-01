"""30 gold-standard test queries for IKEA buying guide evaluation.

3 categories x 10 queries each:
- Factual (F01-F10): Precise retrieval of specific product specs/dimensions
- Comparative (C01-C10): Cross-product comparison requiring breadth
- Thematic (T01-T10): Deep synthesis across multiple buying guides

Gold standards derived from actual IKEA buying guide PDFs (D01-D20),
verified against extracted text.
"""

from dataclasses import dataclass, field


@dataclass
class TestQuery:
    """A test query with gold-standard answer and metadata."""
    query_id: str
    category: str  # "factual", "comparative", "thematic"
    query: str
    gold_answer: str
    relevant_docs: list[str]  # Document IDs that should be retrieved
    relevant_sections: list[str] = field(default_factory=list)


# --- Factual Queries (F01-F10) ---
# Test precision of retrieval: specific dimensions, specs, product details

FACTUAL_QUERIES = [
    TestQuery(
        query_id="F01",
        category="factual",
        query="What are the available frame widths for the PAX wardrobe system?",
        gold_answer=(
            "The PAX wardrobe system offers frames in three widths: "
            "19⅝\", 29½\", and 39¼\". These are available in two depths "
            "(13¾\" and 22⅞\") and two heights (79¼\" and 93⅛\"). "
            "Frame colors include white, gray-beige, and dark gray."
        ),
        relevant_docs=["D02"],
        relevant_sections=["Frames"],
    ),
    TestQuery(
        query_id="F02",
        category="factual",
        query="What is the maximum load per shelf for the BILLY bookcase?",
        gold_answer=(
            "The BILLY bookcase has a maximum load of 66 lbs (30 kg) per shelf "
            "for the wider 31½\" models, and 31 lbs (14 kg) per shelf for the "
            "narrower 15¾\" models. The bookcase is available in two heights: "
            "41¾\" and 79½\", with a depth of 11\"."
        ),
        relevant_docs=["D01"],
        relevant_sections=["Specifications"],
    ),
    TestQuery(
        query_id="F03",
        category="factual",
        query="What is the SEKTION kitchen cabinet work height and how is it calculated?",
        gold_answer=(
            "The SEKTION kitchen work height is 36\", which includes the base "
            "cabinet frame height of 30\", a 4½\" toekick/legs, and a 1½\" "
            "countertop. The system includes a free 25-year limited warranty "
            "and is backed by 50 years of IKEA kitchen experience."
        ),
        relevant_docs=["D07"],
        relevant_sections=["Good to know", "Cabinets"],
    ),
    TestQuery(
        query_id="F04",
        category="factual",
        query="What are the maximum load capacities for the BESTÅ TV units?",
        gold_answer=(
            "The BESTÅ 70⅞\" TV unit can support a maximum load of 55 lbs "
            "and requires eight legs or four legs and two supporting legs. "
            "The 47¼\" TV unit supports a maximum load of 35 lbs and requires "
            "six legs or four legs and one supporting leg. BESTÅ frames have a "
            "max load of 44 lbs per surface."
        ),
        relevant_docs=["D04"],
        relevant_sections=["BESTÅ TV Units"],
    ),
    TestQuery(
        query_id="F05",
        category="factual",
        query="What minimum ceiling height is required for the PAX wardrobe system?",
        gold_answer=(
            "The minimum ceiling heights required for the PAX wardrobe are: "
            "82½\" for the 22⅞\" deep × 79¼\" high frames, and 95¾\" for the "
            "22⅞\" deep × 93⅛\" high frames. Two people are required to "
            "assemble the PAX wardrobe system safely."
        ),
        relevant_docs=["D02"],
        relevant_sections=["Safety", "Assembly"],
    ),
    TestQuery(
        query_id="F06",
        category="factual",
        query="What cushion materials are used in the KIVIK sofa series?",
        gold_answer=(
            "The KIVIK sofa seat cushions are made of pocket springs, "
            "high-resilience foam, and polyester fibers, providing both "
            "relaxing softness and firm support. The KIVIK series includes "
            "a free 10-year limited warranty. The TRESUND fabric cover "
            "withstands 30,000 abrasion cycles, while KELINGE and TIBBLEBY "
            "fabrics withstand 50,000 cycles."
        ),
        relevant_docs=["D09"],
        relevant_sections=["Combinations", "Go to know"],
    ),
    TestQuery(
        query_id="F07",
        category="factual",
        query="What are the available sizes of SEKTION corner base cabinets?",
        gold_answer=(
            "SEKTION offers two corner base cabinet sizes: 38×38×30\" "
            "(W38×D38×H30\") and 47×26×30\" (W47×D26×H30\"). Corner wall "
            "cabinets are available in 26×26\" in heights of 30\" and 40\". "
            "The UTRUSTA corner base cabinet pull-out fitting (47\") and "
            "carousel (34½\") help maximize corner storage accessibility."
        ),
        relevant_docs=["D07"],
        relevant_sections=["Cabinets", "Interior Fittings"],
    ),
    TestQuery(
        query_id="F08",
        category="factual",
        query="What is special about the HEMNES series material and who designed it?",
        gold_answer=(
            "The HEMNES series is designed by Carina Bengs. All furniture in "
            "the series is made of solid wood, a natural material that ages "
            "beautifully and is both durable and renewable. The glass-door "
            "cabinet has a max load of 66 lbs per shelf, and the TV unit "
            "supports up to 110 lbs. The material in the product may be "
            "recyclable."
        ),
        relevant_docs=["D08"],
        relevant_sections=["Design", "All Parts and Prices"],
    ),
    TestQuery(
        query_id="F09",
        category="factual",
        query="What door styles are available for the ENHET laundry system?",
        gold_answer=(
            "The ENHET system offers four door and drawer front styles: "
            "White (clean, bright, and fresh expression), Oak effect (foil "
            "surface resisting moisture, scratches, and bumps with wood "
            "pattern), White frame (wide frame like an empty canvas), and "
            "Gray frame (warm and cozy feeling). ENHET doors and drawer "
            "fronts have a 10-year limited warranty."
        ),
        relevant_docs=["D15"],
        relevant_sections=["Doors and Drawer Fronts"],
    ),
    TestQuery(
        query_id="F10",
        category="factual",
        query="What integrated lighting options are available for the PAX wardrobe?",
        gold_answer=(
            "The ÖVERSIDAN LED wardrobe lighting strip with sensor is available "
            "in three sizes (18\", 28\", 38\") and three colors matching PAX frames "
            "(beige, white, dark gray). It switches on/off automatically when "
            "you open/close the door. Works best with solid PAX doors. Requires "
            "TRÅDFRI LED driver and ANSLUTA power cord. The KÖLVATTEN LED "
            "lighting with sensor automatically turns off after 60 seconds and "
            "is rechargeable via USB-C."
        ),
        relevant_docs=["D02"],
        relevant_sections=["Integrated Lighting"],
    ),
]

# --- Comparative Queries (C01-C10) ---
# Test breadth: require cross-document comparison

COMPARATIVE_QUERIES = [
    TestQuery(
        query_id="C01",
        category="comparative",
        query=(
            "Compare the BILLY bookcase and KALLAX shelving unit for "
            "storage solutions."
        ),
        gold_answer=(
            "BILLY [D01] is a traditional bookcase available in widths 15¾\" "
            "and 31½\", depth 11\", heights 41¾\" and 79½\", with max load "
            "31-66 lbs/shelf. It was launched in 1979 and has 6 color options "
            "with optional OXBERG doors. KALLAX [D03] is a modular shelving "
            "unit available in multiple sizes, with a depth of 15⅜\". It can "
            "be used as a room divider and has inserts (doors, drawers, "
            "bottle holders) with compatible DRÖNA boxes. BILLY suits "
            "traditional book storage while KALLAX offers more modular "
            "flexibility."
        ),
        relevant_docs=["D01", "D03"],
    ),
    TestQuery(
        query_id="C02",
        category="comparative",
        query=(
            "Compare the PAX wardrobe and BESTÅ storage system in terms "
            "of available sizes and load capacity."
        ),
        gold_answer=(
            "PAX [D02] offers frame widths of 19⅝\", 29½\", 39¼\" with "
            "depths 13¾\" and 22⅞\" and heights 79¼\" and 93⅛\". It requires "
            "wall mounting and minimum ceiling heights of 82½\"-95¾\". "
            "BESTÅ [D04] offers frame widths 23⅝\" and 47¼\" with depths "
            "7⅞\" and 15¾\" and heights 15\"-75⅝\". BESTÅ frames support "
            "44 lbs per surface; TV units support 35-55 lbs. PAX is designed "
            "for bedrooms/wardrobes while BESTÅ is designed for living room "
            "media and display storage."
        ),
        relevant_docs=["D02", "D04"],
    ),
    TestQuery(
        query_id="C03",
        category="comparative",
        query=(
            "How do BILLY, HEMNES, and BESTÅ compare for living room "
            "storage solutions?"
        ),
        gold_answer=(
            "BILLY [D01]: basic bookcase, 11\" depth, 31-66 lbs/shelf, "
            "6 colors, optional glass doors. HEMNES [D08]: solid wood, "
            "designed by Carina Bengs, 14⅝\" depth, 66 lbs/shelf for "
            "glass-door cabinet, TV unit supports 110 lbs, includes desks "
            "and sideboards. BESTÅ [D04]: modular system with frame "
            "widths 23⅝\"-47¼\", 44 lbs/surface, many door options "
            "(LAPPVIKEN, GLASSVIK, SINDVIK, etc.), can be wall-mounted "
            "with suspension rail, designed specifically for media storage. "
            "HEMNES offers warmth through solid wood; BESTÅ offers the "
            "most modularity; BILLY is the most affordable."
        ),
        relevant_docs=["D01", "D04", "D08"],
    ),
    TestQuery(
        query_id="C04",
        category="comparative",
        query="Compare the warranty periods across different IKEA product lines.",
        gold_answer=(
            "SEKTION kitchen cabinets [D07] have the longest warranty at "
            "25 years. PAX wardrobe [D02], KIVIK sofa [D09], and ENHET "
            "laundry [D15] all come with 10-year limited warranties. BESTÅ "
            "[D04] accessories also carry a 10-year warranty. These warranty "
            "periods reflect IKEA's confidence in product durability, with "
            "kitchens receiving the longest coverage due to their higher "
            "investment and daily-use demands."
        ),
        relevant_docs=["D02", "D04", "D07", "D09", "D15"],
    ),
    TestQuery(
        query_id="C05",
        category="comparative",
        query=(
            "Compare the door options available for PAX wardrobe versus "
            "BESTÅ storage."
        ),
        gold_answer=(
            "PAX [D02] offers hinged doors (BERGSBO, FORSAND, GRIMO, "
            "FARDAL, TYSSEDAL, ÅHEIM, REINSVOLL, TONSTAD, etc.) and "
            "sliding doors (HASVIK, GRIMO glass, SVARTISDAL, BJÖRNÖYA, "
            "AULI mirror, HOKKSUND, MEHAMN). Sliding doors include "
            "integrated soft open/closing. BESTÅ [D04] offers panel doors "
            "(LAPPVIKEN, HANVIKEN, SUTTERVIKEN, BJÖRKÖVIKEN, etc.), "
            "glass doors (GLASSVIK, SINDVIK, OSTVIK), high-gloss options "
            "(SELSVIKEN), and special finishes (RIKSVIKEN bronze, "
            "BERGSVIKEN marble, KALLVIKEN concrete, MÖRTVIKEN metal). "
            "PAX focuses on full-height wardrobe doors while BESTÅ "
            "emphasizes decorative front variety."
        ),
        relevant_docs=["D02", "D04"],
    ),
    TestQuery(
        query_id="C06",
        category="comparative",
        query="Compare KIVIK and other IKEA seating options for comfort features.",
        gold_answer=(
            "KIVIK [D09] uses pocket springs, high-resilience foam, and "
            "polyester fibers for maximum comfort. Covers include TRESUND "
            "(30K cycles, machine washable, cotton-polyester blend), "
            "KELINGE (50K cycles, 100% polyester corduroy), and TIBBLEBY "
            "(50K cycles, 100% polyester herringbone). KIVIK also offers "
            "GRANN/BOMSTAD leather with coated fabric combination. "
            "The wide armrests have thick padding for neck support. "
            "Models range from loveseat (74¾\") to 6-seat sectional with "
            "chaise configurations."
        ),
        relevant_docs=["D09"],
    ),
    TestQuery(
        query_id="C07",
        category="comparative",
        query=(
            "Compare the planning tools IKEA offers for kitchens versus "
            "wardrobes."
        ),
        gold_answer=(
            "For kitchens [D07]: IKEA offers a free measurement service, "
            "free kitchen planning service (online 90-minute appointment or "
            "in-store 2-hour appointment), and the IKEA Home Planner for "
            "self-service 3D planning with price calculation. For PAX "
            "wardrobes [D02]: IKEA offers the PAX/KOMPLEMENT planner "
            "available both in-store and online, which calculates prices and "
            "generates product lists. Both systems offer professional "
            "planning support and delivery/assembly services."
        ),
        relevant_docs=["D02", "D07"],
    ),
    TestQuery(
        query_id="C08",
        category="comparative",
        query=(
            "Compare how IKEA handles safety across the BILLY, BESTÅ, "
            "and PAX product lines."
        ),
        gold_answer=(
            "All three product lines share the core safety requirement: "
            "\"Secure It! Prevent tip-over injury\" — furniture must be "
            "secured to the wall with included restraints. BILLY [D01] "
            "requires a wall fastener. PAX [D02] frames must be mounted "
            "flush to the wall per assembly instructions; two people "
            "are required for assembly. BESTÅ [D04] requires wall "
            "fasteners for floor-standing units over 25¼\" high and "
            "suspension rails for wall-mounted units. HEMNES [D08] "
            "includes a detailed wall anchoring guide for different wall "
            "materials (drywall, plaster, masonry). All guides note that "
            "different wall materials require different hardware types."
        ),
        relevant_docs=["D01", "D02", "D04", "D08"],
    ),
    TestQuery(
        query_id="C09",
        category="comparative",
        query=(
            "Compare the interior organizer systems between PAX/KOMPLEMENT "
            "and SEKTION/MAXIMERA."
        ),
        gold_answer=(
            "PAX/KOMPLEMENT [D02] offers clothes rails, shelves, glass "
            "shelves, drawers, mesh baskets, pull-out trays, pants hangers, "
            "shoe shelves, valet hangers, drawer mats, and dividers — all "
            "sized to match 13¾\" or 22⅞\" frame depths. SEKTION/MAXIMERA "
            "[D07] offers low/medium/high drawers with built-in dampers "
            "for soft closing, wire baskets, pull-out interior fittings, "
            "shelves, corner carousels, corner cabinet pull-out fittings, "
            "and UPPDATERA/VARIERA organizer inserts. PAX focuses on "
            "clothing organization while SEKTION focuses on kitchen "
            "utility organization."
        ),
        relevant_docs=["D02", "D07"],
    ),
    TestQuery(
        query_id="C10",
        category="comparative",
        query="Compare the color options across BILLY, PAX, BESTÅ, and HEMNES.",
        gold_answer=(
            "BILLY [D01] offers 6 color finishes for the bookcase. "
            "PAX [D02] frames come in white, gray-beige, and dark gray "
            "(3 colors). BESTÅ [D04] frames are available in white, "
            "black-brown, white stained oak, and dark gray (4 finishes). "
            "HEMNES [D08] comes in white stain, black-brown, dark gray "
            "stain, and some pieces in white stain/light brown or "
            "black-brown/light brown two-tone options. Each series "
            "maintains consistent colors within its range for easy "
            "coordination of pieces."
        ),
        relevant_docs=["D01", "D02", "D04", "D08"],
    ),
]

# --- Thematic Queries (T01-T10) ---
# Test deep synthesis: require multi-document reasoning

THEMATIC_QUERIES = [
    TestQuery(
        query_id="T01",
        category="thematic",
        query=(
            "What safety considerations should I be aware of when buying "
            "IKEA furniture, and which products require wall anchoring?"
        ),
        gold_answer=(
            "IKEA emphasizes the \"Secure It!\" campaign across all tall or "
            "heavy furniture. Products requiring wall anchoring include: "
            "BILLY bookcases [D01], PAX wardrobes [D02], BESTÅ frames "
            "over 25¼\" [D04], HEMNES storage [D08], and KALLAX units "
            "[D03]. Wall materials (drywall, plaster, masonry) require "
            "different hardware types — HEMNES includes a detailed "
            "anchoring guide. PAX requires minimum ceiling heights "
            "(82½\"-95¾\") and two-person assembly. BESTÅ needs "
            "suspension rails for wall mounting. Children should never "
            "climb or hang on drawers, doors, or shelves."
        ),
        relevant_docs=["D01", "D02", "D03", "D04", "D08"],
    ),
    TestQuery(
        query_id="T02",
        category="thematic",
        query=(
            "How does IKEA approach modularity and customization across "
            "its product systems?"
        ),
        gold_answer=(
            "IKEA's modular philosophy appears across multiple systems: "
            "PAX/KOMPLEMENT [D02] lets you combine frames, doors, and "
            "interior organizers — widths 19⅝\"-39¼\" with corner solutions. "
            "BESTÅ [D04] offers frames, doors, drawer fronts, legs, and "
            "top panels that combine into custom media solutions. SEKTION "
            "[D07] uses interchangeable base/wall/high cabinets with "
            "MAXIMERA drawers and UTRUSTA shelves. KALLAX [D03] uses "
            "modular cube units with inserts. ENHET [D15] combines "
            "cabinets, doors, and accessories for laundry areas. Each "
            "system offers an online planner tool for customization."
        ),
        relevant_docs=["D02", "D03", "D04", "D07", "D15"],
    ),
    TestQuery(
        query_id="T03",
        category="thematic",
        query=(
            "What storage solutions does IKEA recommend for organizing "
            "different areas of the home?"
        ),
        gold_answer=(
            "IKEA offers specialized storage for each room: Living room — "
            "BESTÅ for media/display [D04], HEMNES for traditional style "
            "[D08], BILLY/KALLAX for books/display [D01, D03]. Bedroom — "
            "PAX/KOMPLEMENT for wardrobe organization with clothes rails, "
            "shoe shelves, and pull-out trays [D02]. Kitchen — SEKTION with "
            "MAXIMERA drawers, UTRUSTA organizers, and UPPDATERA inserts "
            "[D07]. Laundry — ENHET system for washing, drying, and "
            "cleaning storage [D15]. Each system emphasizes organizing "
            "items by frequency of use and accessibility."
        ),
        relevant_docs=["D01", "D02", "D03", "D04", "D07", "D08", "D15"],
    ),
    TestQuery(
        query_id="T04",
        category="thematic",
        query=(
            "How does IKEA address durability and material quality across "
            "its product lines?"
        ),
        gold_answer=(
            "IKEA addresses durability through material choices and "
            "testing: HEMNES [D08] uses solid wood that \"ages beautifully\" "
            "and is both durable and renewable. KIVIK [D09] fabrics are "
            "abrasion-tested (15K cycles = everyday suitable, 30K+ = very "
            "resistant); KELINGE and TIBBLEBY withstand 50K cycles. "
            "BESTÅ [D04] uses tempered glass (breaks into small pieces, "
            "not sharp fragments). SEKTION [D07] MAXIMERA drawers have "
            "built-in dampers for soft closing. ENHET [D15] door surfaces "
            "resist moisture, scratches, and bumps. Warranty periods "
            "range from 10 years (PAX, KIVIK, ENHET) to 25 years "
            "(SEKTION kitchens)."
        ),
        relevant_docs=["D04", "D07", "D08", "D09", "D15"],
    ),
    TestQuery(
        query_id="T05",
        category="thematic",
        query=(
            "What planning and design services does IKEA offer to help "
            "customers?"
        ),
        gold_answer=(
            "IKEA offers comprehensive planning support: Kitchen — free "
            "measurement service, free planning service (online 90-min or "
            "in-store 2-hour appointments), and IKEA Home Planner for 3D "
            "self-service [D07]. Wardrobe — PAX/KOMPLEMENT planner in-store "
            "or online for customization and price calculation [D02]. "
            "Living room — BESTÅ storage planner for custom combinations "
            "[D04]. Additional services include delivery, assembly (via "
            "TaskRabbit partnership [D09]), measurement, and financing "
            "(IKEA Projekt Credit Card [D07]). These tools calculate prices "
            "and generate product lists for easy ordering."
        ),
        relevant_docs=["D02", "D04", "D07", "D09"],
    ),
    TestQuery(
        query_id="T06",
        category="thematic",
        query=(
            "What design and style trends does IKEA recommend for 2025, "
            "and how do they relate to furniture choices?"
        ),
        gold_answer=(
            "The IKEA 2025 Style Guide [D12] presents two main design "
            "directions: \"Moody modernism\" — featuring neutrals with "
            "muted blues, greens, and pastels; medium-dark brown wood; "
            "contrasting chrome with natural fibers; and \"Sunny "
            "Scandinavian\" — featuring primary colors, bright pastels "
            "balanced with black and white; pine, plywood, and blonde "
            "wood. Key concepts include biophilic design (connecting with "
            "nature through plants, colors, and natural materials), "
            "decluttering for better routines, and six sleep essentials "
            "(comfort, temperature, storage, lighting, air quality, sound)."
        ),
        relevant_docs=["D12"],
    ),
    TestQuery(
        query_id="T07",
        category="thematic",
        query=(
            "How does IKEA optimize small spaces across its different "
            "product systems?"
        ),
        gold_answer=(
            "IKEA addresses small spaces through multiple strategies: "
            "Wall mounting — BESTÅ with suspension rails frees floor space "
            "[D04]; PAX shallow depth (13¾\") frames [D02]. Corner "
            "solutions — PAX add-on corner units [D02]; SEKTION corner "
            "base cabinets with carousels and pull-out fittings [D07]; "
            "HEMNES corner TV unit [D08]. Multi-function — KALLAX as room "
            "divider [D03]; HEMNES desk with cable management [D08]; "
            "KIVIK one-seat sleeper [D09]. Vertical storage — BILLY "
            "79½\" height [D01]; SEKTION high cabinets up to 90\" [D07]; "
            "BESTÅ 75⅝\" frames [D04]. ENHET laundry solutions are "
            "specifically designed for bathroom, closet, and basement "
            "spaces [D15]."
        ),
        relevant_docs=["D01", "D02", "D03", "D04", "D07", "D08", "D09", "D15"],
    ),
    TestQuery(
        query_id="T08",
        category="thematic",
        query=(
            "What interior organization accessories does IKEA offer across "
            "its storage systems?"
        ),
        gold_answer=(
            "IKEA offers extensive interior organization: PAX/KOMPLEMENT "
            "[D02] — clothes rails, shelves, mesh baskets, pull-out trays "
            "with shoe inserts, pants hangers, drawer mats, dividers, "
            "valet hangers. SEKTION/MAXIMERA [D07] — fully extendable "
            "drawers with dampers, UTRUSTA wire baskets and pull-out "
            "racks, UPPDATERA dividers and flatware trays, VARIERA shelf "
            "inserts, HÅLLBAR recycling bins. BESTÅ [D04] — shelves, "
            "glass shelves, drawer frames, soft-closing/push-open hinges. "
            "Cross-system accessories include SKUBB boxes, HEMMAFIXARE "
            "storage cases, DRÖNA boxes [D03], and KVARNVIK boxes [D08]."
        ),
        relevant_docs=["D02", "D03", "D04", "D07", "D08"],
    ),
    TestQuery(
        query_id="T09",
        category="thematic",
        query=(
            "How does IKEA handle the balance between aesthetics and "
            "functionality in its product design?"
        ),
        gold_answer=(
            "IKEA balances aesthetics and functionality throughout: "
            "HEMNES [D08] uses solid wood for natural beauty while "
            "integrating hidden cable management and adjustable shelves. "
            "BESTÅ [D04] offers decorative fronts (marble effect BERGSVIKEN, "
            "reeded glass FÄLLSVIK, bronze RIKSVIKEN) while hiding "
            "clutter and allowing remote control through glass doors. "
            "PAX [D02] combines stylish door options with functional "
            "KOMPLEMENT organizers and ÖVERSIDAN sensor lighting. "
            "SEKTION [D07] uses open storage (TORNVIKEN, VADHOLMA) to "
            "\"personalize your kitchen and create a nice break\" alongside "
            "closed storage. The 2025 Style Guide [D12] explicitly "
            "encourages bold design choices while maintaining functional "
            "sleep and storage essentials."
        ),
        relevant_docs=["D02", "D04", "D07", "D08", "D12"],
    ),
    TestQuery(
        query_id="T10",
        category="thematic",
        query=(
            "What are the key factors to consider when planning a complete "
            "room using IKEA furniture systems?"
        ),
        gold_answer=(
            "Key planning factors across IKEA systems: (1) Space "
            "measurement — PAX requires minimum ceiling heights [D02]; "
            "SEKTION needs accurate kitchen measurements [D07]. "
            "(2) Wall type — determines hardware for anchoring (drywall, "
            "plaster, masonry need different fasteners) [D08]. (3) Load "
            "capacity — varies by product: BESTÅ 44 lbs/surface [D04], "
            "BILLY 31-66 lbs/shelf [D01], HEMNES TV 110 lbs [D08]. "
            "(4) Style coordination — each series maintains consistent "
            "colors for matching pieces [D08, D04]. (5) Future flexibility "
            "— modular systems (BESTÅ, PAX, SEKTION) allow adding or "
            "reconfiguring pieces. (6) Use IKEA planning tools (online "
            "or in-store) to calculate prices and validate combinations "
            "[D02, D04, D07]."
        ),
        relevant_docs=["D01", "D02", "D04", "D07", "D08"],
    ),
]

# --- Combined query set ---
ALL_TEST_QUERIES = FACTUAL_QUERIES + COMPARATIVE_QUERIES + THEMATIC_QUERIES


def get_queries_by_category(category: str) -> list[TestQuery]:
    """Get test queries filtered by category."""
    return [q for q in ALL_TEST_QUERIES if q.category == category]


def get_query_by_id(query_id: str) -> TestQuery:
    """Get a specific test query by ID."""
    for q in ALL_TEST_QUERIES:
        if q.query_id == query_id:
            return q
    raise ValueError(f"Query ID '{query_id}' not found")
