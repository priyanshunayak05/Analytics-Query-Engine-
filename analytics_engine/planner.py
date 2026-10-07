# This file plans the query by using the LLM or feedback

from .catalog import SemanticCatalog
from .feedback import FeedbackStore
from .llm import LLMPlanner
from .models import QueryPlan


class HybridPlanner:
    def __init__(
        self,
        catalog: SemanticCatalog,
        feedback: FeedbackStore,
        llm: LLMPlanner,
    ):
        self.catalog = catalog
        self.feedback = feedback
        self.llm = llm

    def plan(self, query: str) -> QueryPlan:
        # First check if we have a correction in the feedback log
        correction = self.feedback.corrected_plan(query)
        if correction:
            # print("Found a correction in the log!")
            plan = QueryPlan.from_dict(correction, source="feedback")
            # we are very confident since a human corrected it
            plan.semantic_confidence = max(plan.semantic_confidence, 0.98)
            return plan

        # If not, ask the GenAI model to plan it
        payload = self.llm.plan(query, self.catalog, self.feedback.prompt_examples())
        return QueryPlan.from_dict(payload, source="llm")

