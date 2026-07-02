MIN_WORD_COUNT = 100
MAX_WORD_COUNT = 150

BANNED_WORDS = [
    "guaranteed",
    "fda approved",
    "medical claim",
    "best seller",
    "authentic",
]

USE_CASE_SIGNALS = [
    "gift for",
    "perfect for",
    "idea for",
    "use in",
    "for a birthday",
    "for the holiday"
]
A2_EXTRACTION_VERSION = "a2_extra_v1"
A2_EXTRACTION_PROMT = """
    Analyze the following product information and extract structured SEO signals.
    [User's Own Product Information (High Priority - Please prioritize this data)]:
    {own_data}
    [Competitor Listing Data (For reference)]:
    {competitor_data}
    Instructions:
    1. Extract keywords and selling points based on the provided information.
    2. If a section is empty or contains no relevant data, ignore it.
    3. Maintain high precision and avoid marketing fluff.
    """

A3_DRAFT_VERSION = "a2_extra_v1"
A3_DRAFT_PROMT = """
    You are an expert Etsy listing copywriter. Your goal is to create high-converting, SEO-optimized listing copy.

    ### Task Parameters:
    - Tone: {tone}
    - Keywords to include: {keywords}
    - Unique Selling Points: {selling_point}
    - Reference Description: {description}
    - Constraints/Rubric: {rubric}

    ### Output Format:
    Please separate your output clearly using these tags:
    <title> [Write your title here] </title>
    <description> [Write your description here] </description>
    """

A4_PROMPT_VERSION =  "a4_soft_v1"
A4_PROMPT_PROMPT = """ You are an Etsy copy quality reviewer.
The draft below has ALREADY passed all hard rules (word count, keywords, banned terms, use-case). Score ONLY the subjective quality dimensions.
    Seller's intended tone: {tone}
    Title: {draft_title}
    Description: {draft_desc}

    score each dimension: 0-5:
    - 5 = excellent, fully meets the standard
    - 3 = acceptable but with 1-2 noticeable issues
    - 1 = clearly falls short
    For 'feedback_points', give specific, actionable revision notes. Any dimension scoring below 3 MUST have a feedback point explaining exactly what to fix."""


KEYWORD_COVERAGE_THRESHOLD = 0.8