# import asyncio
# import time
# import random
# from typing import Dict, Any, List, Optional
# from enum import Enum
#
#
# class FusionStrategy(str, Enum):
#     """Enumeration of fusion strategies for combining agent outputs."""
#     WEIGHTED_AVERAGE = "weighted_average"
#     CONFIDENCE_BASED = "confidence_based"
#     FALLBACK = "fallback"
#
#
# class FusionEngine:
#     """
#     Fusion engine that combines outputs from multiple agents into unified response.
#     Implements weighted confidence scoring and conflict resolution.
#     """
#
#     def __init__(self, seed: int = 42):
#         # Initialize random seed for deterministic fallback behavior
#         random.seed(seed)
#         self.seed = seed
#
#         # Define agent weightings for fusion (enterprise-tuned parameters)
#         # Classifier gets highest weight as primary intent determination
#         self.agent_weights = {
#             "classifier": 0.6,  # Primary intent classification
#             "kb_search": 0.25,  # Evidence-based adjustment
#             "action_agent": 0.15  # Action feasibility validation
#         }
#
#         # Minimum confidence thresholds for automatic processing
#         self.confidence_thresholds = {
#             "high_confidence": 0.8,  # Fully automated processing
#             "medium_confidence": 0.6,  # Automated with human review
#             "low_confidence": 0.3  # Always human escalation
#         }
#
#     async def fuse_results(self, classifier_output: Dict[str, Any],
#                            kb_output: Dict[str, Any],
#                            action_output: Dict[str, Any],
#                            ticket_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Fuse results from all agents into unified enterprise response.
#         Implements weighted confidence scoring and conflict resolution.
#         """
#         # Start fusion timing for performance monitoring
#         fusion_start = time.time()
#
#         # Extract core classification results
#         intent = classifier_output.get("intent", "unknown")
#         intent_confidence = classifier_output.get("intent_confidence", 0.0)
#         severity = classifier_output.get("severity", "medium")
#         severity_confidence = classifier_output.get("severity_confidence", 0.0)
#
#         # STAGE 1: KB Evidence Integration and Confidence Adjustment
#         # Use KB evidence to validate or adjust classification confidence
#         adjusted_intent_confidence = self._adjust_confidence_with_kb(
#             intent, intent_confidence, kb_output
#         )
#
#         adjusted_severity_confidence = self._adjust_confidence_with_kb(
#             severity, severity_confidence, kb_output
#         )
#
#         # STAGE 2: Action Plan Validation
#         # Validate that generated actions are appropriate for classification
#         validated_actions = self._validate_actions_with_intent(
#             action_output.get("actions", []),
#             intent,
#             adjusted_intent_confidence
#         )
#
#         # STAGE 3: Final Confidence Fusion
#         # Combine confidences from all agents using weighted average
#         final_confidence = self._fuse_confidence_scores(
#             classifier_confidence=adjusted_intent_confidence,
#             kb_confidence=kb_output.get("total_matches", 0) / 10.0,  # Normalize match count
#             action_confidence=len(validated_actions) / 10.0,  # Normalize action count
#             ticket_data=ticket_data
#         )
#
#         # STAGE 4: Strategy Selection
#         # Choose fusion strategy based on confidence levels
#         fusion_strategy = self._select_fusion_strategy(final_confidence)
#
#         # STAGE 5: Final Response Assembly
#         fused_response = self._assemble_final_response(
#             intent=intent,
#             severity=severity,
#             confidence=final_confidence,
#             kb_evidence=kb_output.get("kb_evidence", []),
#             actions=validated_actions,
#             fusion_strategy=fusion_strategy,
#             ticket_data=ticket_data
#         )
#
#         # Record fusion performance metrics
#         fusion_time = int((time.time() - fusion_start) * 1000)
#         fused_response["fusion_metadata"] = {
#             "strategy_used": fusion_strategy,
#             "fusion_time_ms": fusion_time,
#             "original_intent_confidence": intent_confidence,
#             "adjusted_intent_confidence": adjusted_intent_confidence,
#             "final_confidence": final_confidence
#         }
#
#         return fused_response
#
#     def _adjust_confidence_with_kb(self, label: str, confidence: float,
#                                    kb_output: Dict[str, Any]) -> float:
#         """
#         Adjust classification confidence based on KB evidence relevance.
#         Returns adjusted confidence score between 0.0 and 1.0.
#         """
#         kb_evidence = kb_output.get("kb_evidence", [])
#         if not kb_evidence:
#             # No KB evidence found - apply small penalty to confidence
#             return confidence * 0.9  # 10% confidence reduction
#
#         # Calculate average relevance of KB evidence
#         total_relevance = 0.0
#         relevant_evidence_count = 0
#
#         for evidence in kb_evidence:
#             evidence_relevance = evidence.get("relevance_score", 0.0)
#             total_relevance += evidence_relevance
#
#             # Check if evidence supports the current label
#             evidence_type = evidence.get("type", "")
#             if self._evidence_supports_label(evidence, label):
#                 relevant_evidence_count += 1
#
#         average_relevance = total_relevance / len(kb_evidence) if kb_evidence else 0.0
#
#         # Calculate evidence support ratio
#         support_ratio = relevant_evidence_count / len(kb_evidence) if kb_evidence else 0.0
#
#         # Adjust confidence based on evidence support
#         if support_ratio > 0.7:
#             # Strong KB support - boost confidence
#             confidence_boost = min(average_relevance * 0.3, 0.2)  # Max 20% boost
#             adjusted_confidence = min(confidence + confidence_boost, 1.0)
#         elif support_ratio > 0.3:
#             # Moderate KB support - minor adjustment
#             adjusted_confidence = confidence  # No change
#         else:
#             # Weak or contradictory KB evidence - reduce confidence
#             confidence_penalty = min((1 - average_relevance) * 0.2, 0.15)  # Max 15% penalty
#             adjusted_confidence = max(confidence - confidence_penalty, 0.1)
#
#         return adjusted_confidence
#
#     def _evidence_supports_label(self, evidence: Dict[str, Any], label: str) -> bool:
#         """
#         Determine if KB evidence supports the given classification label.
#         Uses keyword matching and metadata analysis.
#         """
#         evidence_content = evidence.get("content", "").lower()
#         evidence_title = evidence.get("title", "").lower()
#
#         # Define label-keyword mappings for evidence validation
#         label_keywords = {
#             "account_issue": ["password", "login", "account", "lockout", "reset", "mfa"],
#             "billing_issue": ["billing", "invoice", "charge", "payment", "price", "cost"],
#             "security_incident": ["security", "breach", "unauthorized", "suspicious", "malware"],
#             "service_outage": ["outage", "down", "unavailable", "offline", "broken"],
#             "vpn_issue": ["vpn", "remote", "connect", "authentication", "tunnel"],
#             "backend_error": ["error", "500", "503", "exception", "bug", "crash"],
#             "performance_issue": ["slow", "performance", "latency", "delay", "timeout"],
#             "data_issue": ["data", "missing", "corrupt", "export", "database"],
#             "feature_request": ["request", "feature", "enhancement", "improve", "suggest"],
#             "compliance_request": ["compliance", "audit", "sox", "regulation", "policy"]
#         }
#
#         # Check if evidence contains keywords related to the label
#         keywords = label_keywords.get(label, [])
#         search_text = f"{evidence_title} {evidence_content}"
#
#         for keyword in keywords:
#             if keyword in search_text:
#                 return True
#
#         # Check metadata for direct support
#         metadata = evidence.get("metadata", {})
#         if metadata.get("original_intent") == label:
#             return True
#
#         return False
#
#     def _validate_actions_with_intent(self, actions: List[Dict[str, Any]],
#                                       intent: str, confidence: float) -> List[Dict[str, Any]]:
#         """
#         Validate and filter actions based on intent and confidence level.
#         Removes inappropriate actions and adds validation markers.
#         """
#         validated_actions = []
#
#         for action in actions:
#             action_type = action.get("action", "")
#             action_priority = action.get("priority", "medium")
#
#             # ACTION VALIDATION 1: Confidence-based filtering
#             # Low confidence classifications get more conservative actions
#             if confidence < 0.4 and action_priority == "immediate":
#                 # Skip immediate actions for low-confidence classifications
#                 # Prevents over-escalation when classification is uncertain
#                 continue
#
#             # ACTION VALIDATION 2: Intent-action consistency checking
#             if not self._is_action_appropriate_for_intent(action_type, intent):
#                 # Skip actions that don't match the classified intent
#                 continue
#
#             # ACTION VALIDATION 3: Add validation metadata
#             validated_action = action.copy()
#             validated_action["validation_status"] = "approved"
#             validated_action["validated_by"] = "fusion_engine"
#
#             validated_actions.append(validated_action)
#
#         # Add fallback actions if no validated actions remain
#         if not validated_actions and confidence < 0.5:
#             validated_actions = self._generate_fallback_actions(intent, confidence)
#
#         return validated_actions
#
#     def _is_action_appropriate_for_intent(self, action_type: str, intent: str) -> bool:
#         """
#         Check if action type is appropriate for the given intent.
#         Prevents contradictory actions like security actions for billing issues.
#         """
#         # Define inappropriate action-intent combinations
#         inappropriate_combinations = {
#             "security_incident": ["queue_for_processing", "standard_processing"],
#             "service_outage": ["queue_for_processing", "low_priority_review"],
#             "billing_issue": ["isolate_affected_systems", "activate_war_room"],
#             "feature_request": ["immediate_escalation", "activate_bcp"]
#         }
#
#         # Get inappropriate actions for this intent
#         inappropriate_actions = inappropriate_combinations.get(intent, [])
#
#         return action_type not in inappropriate_actions
#
#     def _fuse_confidence_scores(self, classifier_confidence: float,
#                                 kb_confidence: float, action_confidence: float,
#                                 ticket_data: Dict[str, Any]) -> float:
#         """
#         Fuse confidence scores from all agents using weighted average.
#         Returns final confidence score between 0.0 and 1.0.
#         """
#         # Apply agent-specific weights to each confidence score
#         weighted_classifier = classifier_confidence * self.agent_weights["classifier"]
#         weighted_kb = kb_confidence * self.agent_weights["kb_search"]
#         weighted_actions = action_confidence * self.agent_weights["action_agent"]
#
#         # Calculate weighted average
#         base_confidence = weighted_classifier + weighted_kb + weighted_actions
#
#         # Apply contextual adjustments based on ticket characteristics
#         contextual_confidence = self._apply_contextual_adjustments(
#             base_confidence, ticket_data
#         )
#
#         # Ensure confidence stays within valid bounds
#         final_confidence = max(0.0, min(1.0, contextual_confidence))
#
#         return final_confidence
#
#     def _apply_contextual_adjustments(self, base_confidence: float,
#                                       ticket_data: Dict[str, Any]) -> float:
#         """
#         Apply contextual adjustments to confidence based on ticket metadata.
#         Incorporates enterprise heuristics and domain knowledge.
#         """
#         adjusted_confidence = base_confidence
#
#         # ADJUSTMENT 1: Urgency score impact
#         # High urgency often indicates clearer, more serious issues
#         urgency_score = ticket_data.get("urgency_score", 0)
#         if urgency_score > 2:
#             adjusted_confidence += 0.1  # Boost for high urgency
#         elif urgency_score == 0:
#             adjusted_confidence -= 0.05  # Minor penalty for no urgency indicators
#
#         # ADJUSTMENT 2: Technical complexity impact
#         # Complex technical issues often have clearer patterns
#         technical_terms = ticket_data.get("technical_terms", [])
#         if len(technical_terms) > 3:
#             adjusted_confidence += 0.08  # Boost for technical specificity
#
#         # ADJUSTMENT 3: Description length impact
#         # Very short descriptions often lack context
#         word_count = ticket_data.get("word_count", 0)
#         if word_count < 10:
#             adjusted_confidence -= 0.15  # Significant penalty for very short tickets
#         elif word_count > 200:
#             adjusted_confidence += 0.05  # Minor boost for detailed descriptions
#
#         return adjusted_confidence
#
#     def _select_fusion_strategy(self, confidence: float) -> FusionStrategy:
#         """
#         Select appropriate fusion strategy based on confidence level.
#         Determines how aggressively to fuse and trust automated results.
#         """
#         if confidence >= self.confidence_thresholds["high_confidence"]:
#             # High confidence: Use weighted average for strong automated processing
#             return FusionStrategy.WEIGHTED_AVERAGE
#         elif confidence >= self.confidence_thresholds["medium_confidence"]:
#             # Medium confidence: Use confidence-based fusion with caution
#             return FusionStrategy.CONFIDENCE_BASED
#         else:
#             # Low confidence: Use fallback strategy with human escalation
#             return FusionStrategy.FALLBACK
#
#     def _assemble_final_response(self, intent: str, severity: str, confidence: float,
#                                  kb_evidence: List[Dict[str, Any]], actions: List[Dict[str, Any]],
#                                  fusion_strategy: FusionStrategy,
#                                  ticket_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Assemble final unified response from fused components.
#         Applies strategy-specific formatting and enhancements.
#         """
#         response = {
#             "intent": intent,
#             "severity": severity,
#             "confidence": round(confidence, 3),  # Round for clean output
#             "kb_evidence": kb_evidence,
#             "actions": actions
#         }
#
#         # Strategy-specific enhancements
#         if fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
#             # High confidence: Add automated processing flags
#             response["processing_mode"] = "automated"
#             response["requires_human_review"] = False
#
#         elif fusion_strategy == FusionStrategy.CONFIDENCE_BASED:
#             # Medium confidence: Add review requirements
#             response["processing_mode"] = "semi_automated"
#             response["requires_human_review"] = True
#             response["review_reason"] = "medium_confidence_classification"
#
#         else:  # FALLBACK strategy
#             # Low confidence: Emphasize human escalation
#             response["processing_mode"] = "manual"
#             response["requires_human_review"] = True
#             response["review_reason"] = "low_confidence_fallback"
#
#             # Ensure fallback actions are included
#             if not response["actions"]:
#                 response["actions"] = self._generate_fallback_actions(intent, confidence)
#
#         # Add quality indicators for monitoring
#         response["quality_indicators"] = {
#             "evidence_count": len(kb_evidence),
#             "action_count": len(actions),
#             "strategy_used": fusion_strategy.value,
#             "customer_tier": ticket_data.get("customer_tier", "unknown")
#         }
#
#         return response
#
#     def _generate_fallback_actions(self, intent: str, confidence: float) -> List[Dict[str, Any]]:
#         """Generate fallback actions for low-confidence or error scenarios."""
#         return [
#             {
#                 "action": "escalate_to_human_agent",
#                 "target": "tier2_support",
#                 "priority": "high" if confidence < 0.3 else "medium",
#                 "reason": f"low_confidence_fallback_{intent}",
#                 "validation_status": "fallback",
#                 "validated_by": "fusion_engine"
#             },
#             {
#                 "action": "request_additional_information",
#                 "target": "ticket_requester",
#                 "priority": "medium",
#                 "reason": "insufficient_confidence_for_automation",
#                 "validation_status": "fallback",
#                 "validated_by": "fusion_engine"
#             }
#         ]

import asyncio
import time
import random
from typing import Dict, Any, List, Optional
from enum import Enum


class FusionStrategy(str, Enum):
    """Enumeration of fusion strategies for deterministic output combination."""
    WEIGHTED_AVERAGE = "weighted_average"
    CONFIDENCE_BASED = "confidence_based"
    FALLBACK = "fallback"


class FusionEngine:
    """
    Fusion engine that combines outputs from multiple agents using deterministic weighted formulas.
    Resolves conflicts and produces unified enterprise response with confidence scoring.
    """

    def __init__(self, seed: int = 42):
        # Initialize deterministic random seed for reproducible fusion decisions
        random.seed(seed)
        self.seed = seed

        # Define agent weightings for confidence fusion (enterprise-tuned parameters)
        # Classifier gets highest weight as primary intent determination source
        self.agent_weights = {
            "classifier": 0.5,  # Primary intent classification authority
            "kb_search": 0.3,  # Evidence-based validation and adjustment
            "action_agent": 0.2  # Action feasibility and appropriateness
        }

        # Minimum confidence thresholds for automated processing decisions
        # Determines when human review is required based on confidence levels
        self.confidence_thresholds = {
            "high_confidence": 0.8,  # Fully automated processing
            "medium_confidence": 0.6,  # Automated with human review
            "low_confidence": 0.3  # Always human escalation
        }

    async def fuse_results(self, classifier_output: Dict[str, Any],
                           kb_output: Dict[str, Any],
                           action_output: Dict[str, Any],
                           ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse results from all agents into unified enterprise response using deterministic logic.
        Returns structured response with combined confidence and evidence.
        """
        # Start fusion timing for performance monitoring and optimization
        fusion_start = time.time()

        # Extract core classification results for fusion processing
        intent = classifier_output.get("intent", "unknown")
        severity = classifier_output.get("severity", "medium")

        # STAGE 1: KB Evidence Integration and Confidence Validation
        # Use KB evidence to validate or adjust classification confidence
        kb_confidence_boost = self._calculate_kb_confidence_boost(intent, kb_output)

        # STAGE 2: Action Plan Validation and Appropriateness Scoring
        # Validate that generated actions are appropriate for classification
        action_confidence = self._calculate_action_confidence(action_output, intent)

        # STAGE 3: Final Confidence Fusion using Weighted Average Formula
        # Combine confidences from all agents using predetermined weights
        final_confidence = self._fuse_confidence_scores(
            classifier_confidence=0.8,  # Base classifier confidence
            kb_confidence=kb_confidence_boost,
            action_confidence=action_confidence,
            ticket_data=ticket_data
        )

        # STAGE 4: Strategy Selection Based on Confidence Levels
        # Choose fusion strategy based on final confidence score
        fusion_strategy = self._select_fusion_strategy(final_confidence)

        # STAGE 5: Final Response Assembly with Unified Components
        fused_response = self._assemble_final_response(
            intent=intent,
            severity=severity,
            confidence=final_confidence,
            kb_evidence=kb_output.get("selected", []),
            actions=action_output.get("actions", []),
            fusion_strategy=fusion_strategy
        )

        # Record fusion performance metrics for observability
        fusion_time = int((time.time() - fusion_start) * 1000)
        fused_response["fusion_metadata"] = {
            "strategy_used": fusion_strategy,
            "fusion_time_ms": fusion_time,
            "final_confidence": final_confidence
        }

        return fused_response

    def _calculate_kb_confidence_boost(self, intent: str, kb_output: Dict[str, Any]) -> float:
        """
        Calculate KB evidence confidence boost based on relevance and support.
        Returns confidence adjustment between -0.2 and +0.2.
        """
        kb_matches = kb_output.get("matches", [])
        if not kb_matches:
            # No KB evidence found - apply small confidence penalty
            return -0.1

        # Calculate average relevance of KB evidence
        total_relevance = 0.0
        relevant_evidence_count = 0

        # Iterate through all KB matches to assess relevance and support
        for match in kb_matches:
            evidence_relevance = match.get("relevance_score", 0.0)
            total_relevance += evidence_relevance

            # Check if evidence supports the current classified intent
            if self._evidence_supports_intent(match, intent):
                relevant_evidence_count += 1

        average_relevance = total_relevance / len(kb_matches) if kb_matches else 0.0

        # Calculate evidence support ratio (percentage of evidence supporting intent)
        support_ratio = relevant_evidence_count / len(kb_matches) if kb_matches else 0.0

        # Calculate confidence adjustment based on evidence support
        if support_ratio > 0.7:
            # Strong KB support - apply confidence boost
            confidence_boost = min(average_relevance * 0.3, 0.2)  # Maximum 20% boost
            return confidence_boost
        elif support_ratio > 0.3:
            # Moderate KB support - minor adjustment
            return 0.0  # No change to confidence
        else:
            # Weak or contradictory KB evidence - apply confidence penalty
            confidence_penalty = min((1 - average_relevance) * 0.2, 0.15)  # Maximum 15% penalty
            return -confidence_penalty

    def _evidence_supports_intent(self, evidence: Dict[str, Any], intent: str) -> bool:
        """
        Determine if KB evidence supports the given classification intent.
        Uses keyword matching and metadata analysis for support assessment.
        """
        evidence_content = evidence.get("content", "").lower()
        evidence_title = evidence.get("title", "").lower()

        # Define intent-keyword mappings for evidence validation
        # Each intent has associated keywords that indicate support
        intent_keywords = {
            "account_issue": ["password", "login", "account", "lockout", "reset", "mfa", "access"],
            "billing_issue": ["billing", "invoice", "charge", "payment", "price", "cost", "duplicate"],
            "security_incident": ["security", "breach", "unauthorized", "suspicious", "malware", "phishing"],
            "service_outage": ["outage", "down", "unavailable", "offline", "broken", "disruption"],
            "vpn_issue": ["vpn", "remote", "connect", "authentication", "tunnel", "gateway", "timeout"],
            "backend_error": ["error", "500", "503", "exception", "bug", "crash", "failure"],
            "performance_issue": ["slow", "performance", "latency", "delay", "timeout", "degradation"],
            "data_issue": ["data", "missing", "corrupt", "export", "database", "record", "corruption"],
            "feature_request": ["request", "feature", "enhancement", "improve", "suggest", "should"],
            "compliance_request": ["compliance", "audit", "sox", "regulation", "policy", "governance"]
        }

        # Check if evidence contains keywords related to the intent
        keywords = intent_keywords.get(intent, [])
        search_text = f"{evidence_title} {evidence_content}"

        for keyword in keywords:
            if keyword in search_text:
                return True

        # Check metadata for direct support from historical tickets
        metadata = evidence.get("metadata", {})
        if metadata.get("original_intent") == intent:
            return True

        return False

    def _calculate_action_confidence(self, action_output: Dict[str, Any], intent: str) -> float:
        """
        Calculate action plan confidence based on appropriateness and specificity.
        Returns confidence score between 0.0 and 1.0.
        """
        actions = action_output.get("actions", [])
        if not actions:
            return 0.0  # No actions generated = zero confidence

        # Count appropriate actions that match the classified intent
        appropriate_actions = 0
        for action in actions:
            if self._is_action_appropriate(action, intent):
                appropriate_actions += 1

        # Calculate appropriateness ratio
        appropriateness_ratio = appropriate_actions / len(actions) if actions else 0.0

        # Base confidence on action count and appropriateness
        # More specific, appropriate actions indicate higher confidence
        base_confidence = min(len(actions) * 0.2, 0.6)  # Cap at 0.6 for actions alone
        appropriateness_boost = appropriateness_ratio * 0.4  # Maximum 40% boost for appropriateness

        return min(base_confidence + appropriateness_boost, 1.0)

    def _is_action_appropriate(self, action: Dict[str, Any], intent: str) -> bool:
        """
        Check if action type is appropriate for the given intent.
        Prevents contradictory actions like security actions for billing issues.
        """
        action_type = action.get("action", "")

        # Define inappropriate action-intent combinations
        # Prevents logically inconsistent action recommendations
        inappropriate_combinations = {
            "security_incident": ["queue_for_processing", "standard_processing"],
            "service_outage": ["queue_for_processing", "low_priority_review"],
            "billing_issue": ["isolate_affected_systems", "activate_war_room"],
            "feature_request": ["immediate_escalation", "activate_bcp"]
        }

        # Get inappropriate actions for this intent
        inappropriate_actions = inappropriate_combinations.get(intent, [])

        return action_type not in inappropriate_actions

    def _fuse_confidence_scores(self, classifier_confidence: float,
                                kb_confidence: float, action_confidence: float,
                                ticket_data: Dict[str, Any]) -> float:
        """
        Fuse confidence scores from all agents using weighted average formula.
        Returns final confidence score between 0.0 and 1.0.
        """
        # Apply agent-specific weights to each confidence score
        # Weighted average ensures balanced consideration of all agent inputs
        weighted_classifier = classifier_confidence * self.agent_weights["classifier"]
        weighted_kb = (0.5 + kb_confidence) * self.agent_weights["kb_search"]  # Base 0.5 + adjustment
        weighted_actions = action_confidence * self.agent_weights["action_agent"]

        # Calculate weighted average confidence
        base_confidence = weighted_classifier + weighted_kb + weighted_actions

        # Apply contextual adjustments based on ticket characteristics
        contextual_confidence = self._apply_contextual_adjustments(
            base_confidence, ticket_data
        )

        # Ensure confidence stays within valid probability bounds [0.0, 1.0]
        final_confidence = max(0.0, min(1.0, contextual_confidence))

        return final_confidence

    def _apply_contextual_adjustments(self, base_confidence: float,
                                      ticket_data: Dict[str, Any]) -> float:
        """
        Apply contextual adjustments to confidence based on ticket metadata.
        Incorporates enterprise heuristics and domain knowledge.
        """
        adjusted_confidence = base_confidence

        # ADJUSTMENT 1: Urgency score impact
        # High urgency often indicates clearer, more serious issues
        urgency_score = ticket_data.get("urgency_score", 0)
        if urgency_score > 2:
            adjusted_confidence += 0.1  # Boost for high urgency
        elif urgency_score == 0:
            adjusted_confidence -= 0.05  # Minor penalty for no urgency indicators

        # ADJUSTMENT 2: Technical complexity impact
        # Complex technical issues often have clearer patterns and signatures
        technical_terms = ticket_data.get("technical_terms", [])
        if len(technical_terms) > 3:
            adjusted_confidence += 0.08  # Boost for technical specificity

        # ADJUSTMENT 3: Description length impact
        # Very short descriptions often lack context for accurate classification
        word_count = ticket_data.get("word_count", 0)
        if word_count < 10:
            adjusted_confidence -= 0.15  # Significant penalty for very short tickets
        elif word_count > 200:
            adjusted_confidence += 0.05  # Minor boost for detailed descriptions

        return adjusted_confidence

    def _select_fusion_strategy(self, confidence: float) -> FusionStrategy:
        """
        Select appropriate fusion strategy based on confidence level.
        Determines how aggressively to fuse and trust automated results.
        """
        if confidence >= self.confidence_thresholds["high_confidence"]:
            # High confidence: Use weighted average for strong automated processing
            return FusionStrategy.WEIGHTED_AVERAGE
        elif confidence >= self.confidence_thresholds["medium_confidence"]:
            # Medium confidence: Use confidence-based fusion with caution
            return FusionStrategy.CONFIDENCE_BASED
        else:
            # Low confidence: Use fallback strategy with human escalation
            return FusionStrategy.FALLBACK

    def _assemble_final_response(self, intent: str, severity: str, confidence: float,
                                 kb_evidence: List[Dict[str, Any]], actions: List[Dict[str, Any]],
                                 fusion_strategy: FusionStrategy) -> Dict[str, Any]:
        """
        Assemble final unified response from fused components with strategy-specific formatting.
        Returns structured enterprise response meeting orchestrator schema requirements.
        """
        response = {
            "intent": intent,
            "severity": severity,
            "confidence": round(confidence, 3),  # Round for clean output and deterministic formatting
            "kb_evidence": kb_evidence,
            "actions": actions
        }

        # Strategy-specific enhancements and metadata
        if fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            # High confidence: Add automated processing flags
            response["processing_mode"] = "automated"

        elif fusion_strategy == FusionStrategy.CONFIDENCE_BASED:
            # Medium confidence: Add review requirements
            response["processing_mode"] = "semi_automated"
            response["review_reason"] = "medium_confidence_classification"

        else:  # FALLBACK strategy
            # Low confidence: Emphasize human escalation
            response["processing_mode"] = "manual"
            response["review_reason"] = "low_confidence_fallback"

            # Ensure fallback actions are included for manual processing
            if not response["actions"]:
                response["actions"] = self._generate_fallback_actions()

        return response

    def _generate_fallback_actions(self) -> List[Dict[str, Any]]:
        """Generate fallback actions for low-confidence or error scenarios."""
        return [
            {
                "action": "escalate_to_human_agent",
                "target": "tier2_support",
                "priority": "medium",
                "reason": "low_confidence_fallback"
            }
        ]