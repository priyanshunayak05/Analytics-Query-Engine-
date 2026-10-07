# Main engine file for orchestration
import json
from pathlib import Path
from typing import Any

from .catalog import SemanticCatalog
from .executor import AnalyticsExecutor
from .feedback import FeedbackStore
from .llm import LLMPlanner
from .loaders import load_csv, load_json
from .planner import HybridPlanner
from .validator import PlanValidationError, PlanValidator

class AnalyticsEngine:
    def __init__(
        self,
        sales_path,
        targets_path,
        dictionary_path,
        feedback_path=None,
        llm_url=None,
        llm_model=None,
        llm_api_key=None,
    ):
        # First load all the data
        sales = load_csv(sales_path)
        targets = load_csv(targets_path)
        dictionary = load_json(dictionary_path)
        
        if not sales:
            raise ValueError("sales dataset is empty. Check the path!")
            
        # The assignment says GenAI is required, so check if they passed it
        if not llm_url or not llm_model:
            raise ValueError("GenAI model configuration (llm_url, llm_model) is required per requirements")
            
        self.catalog = SemanticCatalog.build(dictionary, sales)
        self.feedback = FeedbackStore(feedback_path)
        
        # Initialize the LLM
        llm = LLMPlanner(llm_url, llm_model, llm_api_key)
        self.planner = HybridPlanner(self.catalog, self.feedback, llm)
        self.validator = PlanValidator(self.catalog)
        self.executor = AnalyticsExecutor(sales, targets)

    def answer(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            raise ValueError("query must not be empty")
            
        # Step 1: Parse the query to get a plan
        plan = self.planner.plan(query)
        # print("Generated plan:", plan) # debug
        
        # Step 2: Validate it so we don't run bad sql
        self.validator.validate(plan)

        # Step 3: Run the query
        compiled = self.executor.compile(plan)
        result = self.executor.execute(compiled)
        
        # Step 4: Calculate confidence score
        confidence = self._final_confidence(plan.semantic_confidence, plan.analysis_type, result)
        
        # add the parameters to the sql string so it's easy to read
        params_note = ""
        if compiled.params:
            params_note = f"\n-- parameters: {json.dumps(compiled.params, ensure_ascii=False)}"
            
        return {
            "query": query,
            "generated_logic": compiled.sql + params_note,
            "result": result,
            "confidence_score": confidence,
            "explanation": self._explanation(plan, compiled.description, result),
        }

    def _final_confidence(self, base: float, analysis_type: str, result) -> float:
        score = base + 0.04  # bump up a bit if it executed successfully
        
        if len(result) == 0: # empty results lower confidence
            score -= 0.12
            
        # check if it's YoY but we don't have enough data
        if analysis_type == "yoy":
            has_yoy = False
            for row in result:
                if row.get("yoy_growth_pct") is not None:
                    has_yoy = True
            if not has_yoy:
                score -= 0.18
                
        return round(max(0.0, min(1.0, score)), 2)

    def _explanation(self, plan: Any, execution: str, result) -> str:
        # build the explanation string piece by piece
        understood = f"Understood as {plan.analysis_type.replace('_', ' ')} analysis of {plan.metric}"
        
        if len(plan.dimensions) > 0:
            understood += f" by {', '.join(plan.dimensions)}"
            
        if len(plan.filters) > 0:
            # I used a list comprehension here to make it shorter
            rendered = ", ".join(f"{item.field} {item.operator} {item.value}" for item in plan.filters)
            understood += f", filtered to {rendered}"
            
        text = f"{understood}. Generated a validated, parameterized SQLite query to {execution.lower()}."
        
        if plan.source == "llm":
            text += " The semantic plan was produced by the configured GenAI model and passed the allowlist validator."
        elif plan.source == "feedback":
            text += " An accepted correction from the feedback log supplied the semantic plan."
            
        # check yoy missing data again for the explanation
        if plan.analysis_type == "yoy":
            has_yoy = False
            for row in result:
                if row.get("yoy_growth_pct") is not None:
                    has_yoy = True
            if not has_yoy:
                text += " A growth rate cannot be calculated because the filtered data contains no prior-year comparison."
                
        # append any assumptions made by the model
        if getattr(plan, 'assumptions', None):
            if len(plan.assumptions) > 0:
                text += " Assumptions: " + " ".join(plan.assumptions)
                
        return text

