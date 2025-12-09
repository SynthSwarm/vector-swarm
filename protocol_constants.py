# protocol_constants.py
from enum import Enum


class Space(Enum):
    GLOBAL = "global"
    ALIGNMENT = "alignment"
    BODY = "body"


# You might want different capacities for different spaces
SPACE_CONFIG = {
    Space.GLOBAL: {"max_items": 10000, "dim": 768},  # High churn
    Space.ALIGNMENT: {"max_items": 1000, "dim": 768},  # Static/Low churn
    Space.BODY: {"max_items": 5000, "dim": 768},  # Medium churn
}
