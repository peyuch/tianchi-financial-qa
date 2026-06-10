"""Domain router: rule-based, zero-token question classification."""
import re
from config import DOMAIN_CONFIG

PROMPT_SELECTION_RULES = [
    (re.compile(r'同比|环比|增长|变化|较上年|较20\d{2}年|增速|降幅|连续\d+年'),
     "financial_reports", "reasoner_temporal"),
    (re.compile(r'计算|合计|占比|金额|每股|分红|派发'),
     "financial_reports", "reasoner_numerical"),
    (re.compile(r'计算|赔付|退保|金额|排序|共应'),
     "insurance", "reasoner_numerical"),
    (re.compile(r'同比|增长|变化|趋势|历年|较20\d{2}年|截至20\d{2}年'),
     "research", "reasoner_temporal"),
]

DEFAULT_PROMPT = "reasoner_default"


def route_domain(domain: str) -> dict:
    """Layer 1 routing: return domain-specific configuration."""
    if domain in DOMAIN_CONFIG:
        return dict(DOMAIN_CONFIG[domain])
    return dict(DOMAIN_CONFIG["insurance"])


def select_reasoning_prompt(domain: str, question_text: str) -> str:
    """Layer 2 routing: select reasoning prompt based on domain + question keywords."""
    for pattern, rule_domain, prompt_name in PROMPT_SELECTION_RULES:
        if rule_domain == domain and pattern.search(question_text):
            return prompt_name
    config = DOMAIN_CONFIG.get(domain)
    if config:
        return config["default_prompt"]
    return DEFAULT_PROMPT


def get_output_format(answer_format: str) -> dict:
    """Return output normalization rules for each answer format."""
    if answer_format == "mcq":
        return {"type": "single", "separator": "", "sort": False}
    elif answer_format == "multi":
        return {"type": "multi", "separator": "", "sort": True}
    elif answer_format == "tf":
        return {"type": "single", "separator": "", "sort": False}
    else:
        return {"type": "single", "separator": "", "sort": False}
