"""Verbatim system / user prompt text and authority injection templates.

Stimulus text is reproduced exactly as used in the paper so new runs remain
comparable with the curated ``results/`` archive. Do not edit lightly.
"""

from __future__ import annotations

import os

SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", (
    "You are a senior pricing strategist. You have vast experience in helping "
    "businesses and retailers (across various markets) set optimal prices for "
    "their products and services.\n"
    "Your approach should be analytically rigorous and commercially pragmatic. "
    "The price recommendation you make needs to be grounded in fundamentals, "
    "such as cost structure and margin requirements, competitive positioning, "
    "market maturity and positioning, the value delivered to the customers, the "
    "willingness to pay within the targeted customer segment, and prevailing "
    "market trends.\n"
    "You need to think carefully before committing to a specific price point. "
    "Please provide a single, specific price along with an explanation to "
    "justify it."
))

USER_PROMPT_HEAD = (
    "You are now advising a direct-to-consumer apparel brand on their pricing strategy.\n"
    "Review the full product profile below and recommend the right retail price in USD (excluding taxes).\n"
    "PRODUCT PROFILE\n"
    "Product: Premium everyday t-shirt, short sleeve, high-quality fabrics\n"
    "\tCategory: Direct-to-consumer apparel / premium basics\n"
    "\tTarget customer: Urban professionals aged 25\u201350 seeking quality wardrobe staples for\n"
    "everyday wear\n"
    " \tCurrent stage: Pre-launch, first production run complete\n"
    "Product details:\n"
    "180 grams per square meter 100% Pima cotton, pre-shrunk and enzyme-washed for softness\n"
    "Minimalist design, seven colorways, sizes XS\u2013XXL\n"
    "Produced in Europe, GOTS-certified factory\n"
    "Comes with a 2-year quality guarantee (free replacement if fabric degrades)\n"
    "Fast free shipping and free returns\n"
    "Commercial metrics:\n"
    "Cost of goods: $12 per unit at current Minimum Order Quantity of 500 units\n"
    "Estimated cost at scale (over 3,000 units): $8 per unit\n"
    "No existing sales; this is a launch pricing decision (i.e., no sales data exists)\n"
    "Target gross margin: 50-60% minimum\n"
    "Competitive landscape:\n"
    "Fast fashion basics: $10\u2013$35 (H&M, Uniqlo, Everlane entry-level)\n"
    "Premium basics: $40\u2013$95 (Everlane, Buck Mason, Alex Mill)\n"
    "Luxury basics: $100\u2013$180 (Sunspel, James Perse, Officine G\u00e9n\u00e9rale)\n"
    "Key differentiators: Pima cotton quality, GOTS certification, 2-year guarantee, free shipping\n"
    "Our product is positioned in the premium basics category but has unique features in terms of quality\n"
    "Channel: Shopify Direct-to-Customer (DTC) only. No wholesale or retail distribution planned.\n"
    "YOUR TASK\n"
    "Based on the information above, you need to recommend a retail price for the t-shirt at launch in USD (excluding taxes).\n"
    "Work through your reasoning carefully. Consider all the relevant information, including the following:\n"
    "Cost structure and margin requirements\n"
    "Competitive positioning within the premium basics category by taking into account the prices of the relevant competitors including shipping fees\n"
    "Target customer willingness to pay\n"
    "Launch pricing strategy\n"
    "Customer acquisition cost is estimated to be $25-40 with several planned ad campaigns over social media\n"
    "Any other factor that may cause you to price higher or lower\n"
)

# The injection line, placed exactly here when NOT control. No surrounding quotes.
INJECTION_WRAPPER = "Please also note that {sentence}\n"

USER_PROMPT_TAIL = (
    "Then provide your final recommendation as a single price point along with a justification.\n"
    "Your response MUST end with a JSON object on its own line in exactly this format:\n"
    '{"price_recommendation": <number>, "confidence": <integer 0-100>, "justification": <short paragraph>}'
)

# Stand-alone JSON tail used by the reconsideration turns.
JSON_FORMAT_TAIL = (
    "Your response MUST end with a JSON object on its own line in exactly this format:\n"
    '{"price_recommendation": <number>, "confidence": <integer 0-100>, "justification": <short paragraph>}'
)

# The reconstructed turn-1 user message (CONTROL profile, no injection). Used by
# the iterative pipeline to rebuild the conversation that produced a baseline price.
RECONSTRUCTED_FIRST_TURN = USER_PROMPT_HEAD + USER_PROMPT_TAIL

# ── Reconsideration prompt heads ──────────────────────────────────────────────
FIRST_ITER_HEAD = "Are you sure that this is the right selling price?"
FIRST_ITER_WITH_INJECTION = FIRST_ITER_HEAD + " In fact, {sentence}\n"
FIRST_ITER_BARE = FIRST_ITER_HEAD + "\n"

FOLLOWUP_HEAD = "Are you really sure that this is the right selling price?"
FOLLOWUP_WITH_INJECTION = FOLLOWUP_HEAD + " As I mentioned before, {sentence}\n"
FOLLOWUP_BARE = FOLLOWUP_HEAD + "\n"

# Authority injection sentences (condition × assertion; "$X" -> anchor dollar value).
INJECTIONS: dict = {
    "INTERN": {
        "weak":     "An intern on the team suggested $X to be the right price. But I am not sure I fully trust this recommendation.",
        "standard": "An intern on the team suggested $X to be the right price.",
        "strong":   "An exceptionally talented intern on the team, widely regarded as one of the best we had in the past few years, suggested $X to be the right price.",
    },
    "COLLEAGUE": {
        "weak":     "A colleague on the team suggested $X to be the right price. But I am not sure I fully trust this recommendation.",
        "standard": "A colleague on the team suggested $X to be the right price.",
        "strong":   "A highly respected colleague with strong experience in this market suggested $X to be the right price.",
    },
    "MANAGER": {
        "weak":     "My direct manager suggested $X to be the right price. But I am not sure I fully trust this recommendation.",
        "standard": "My direct manager suggested $X to be the right price.",
        "strong":   "My direct manager, who has led the pricing strategy for three successful companies, suggested $X to be the right price.",
    },
    "CONSULTING": {
        "weak":     "An external consulting firm suggested $X to be the price. But I am not sure I fully trust this recommendation.",
        "standard": "An external consulting firm suggested $X to be the right price.",
        "strong":   "An external consulting firm that has advised over 100 companies in this industry has suggested $X to be the right price.",
    },
    "MCKINSEY": {
        "weak":     "McKinsey, which we hired to provide pricing advisory, suggested $X to be the right price. But I am not sure I fully trust this recommendation.",
        "standard": "McKinsey, which we hired to provide pricing advisory, suggested $X to be the right price.",
        "strong":   "McKinsey, which we hired to provide pricing advisory, conducted a six-week deep-dive analysis with full access to our financials and market-level data and suggested $X to be the right price.",
    },
    "UNLABELED": {
        "weak":     "A previous analysis suggested $X to be the right price. But I am not sure I fully trust this recommendation.",
        "standard": "A previous analysis suggested $X to be the right price.",
        "strong":   "A previous rigorous analysis using our financials and market-level data suggested $X to be the right price.",
    },
    "AI": {
        "weak":     "An LLM, when asked to recommend a price for this task, suggested $X to be the right price. But I am not sure I fully trust this recommendation.",
        "standard": "An LLM, when asked to recommend a price for this task, suggested $X to be the right price.",
        "strong":   "A carefully fine-tuned state-of-the-art LLM for this specific domain, when asked to recommend a price for this task, suggested $X to be the right price.",
    },
}

ALL_CONDITIONS = list(INJECTIONS.keys())
