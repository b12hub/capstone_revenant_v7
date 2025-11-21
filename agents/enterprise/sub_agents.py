# import asyncio
# import time
# import json
# import hashlib
# import math
# import random
# from typing import Dict, Any, List, Optional, Tuple
# from enum import Enum
#
#
# class IntentLabel(str, Enum):
#     """Enumeration of supported intent labels for classification."""
#     ACCOUNT_ISSUE = "account_issue"
#     BILLING_ISSUE = "billing_issue"
#     SECURITY_INCIDENT = "security_incident"
#     SERVICE_OUTAGE = "service_outage"
#     VPN_ISSUE = "vpn_issue"
#     BACKEND_ERROR = "backend_error"
#     PERFORMANCE_ISSUE = "performance_issue"
#     DATA_ISSUE = "data_issue"
#     FEATURE_REQUEST = "feature_request"
#     COMPLIANCE_REQUEST = "compliance_request"
#     UNKNOWN = "unknown"
#
#
# class SeverityLabel(str, Enum):
#     """Enumeration of supported severity labels for classification."""
#     LOW = "low"
#     MEDIUM = "medium"
#     HIGH = "high"
#     CRITICAL = "critical"
#
#
# class ClassifierAgent:
#     """
#     Intent and severity classification agent with manual softmax implementation.
#     Uses keyword matching and pattern recognition for enterprise ticket classification.
#     """
#
#     def __init__(self, seed: int = 42):
#         # Initialize random seed for deterministic tie-breaking
#         random.seed(seed)
#         self.seed = seed
#
#         # Define intent keywords with weights for pattern matching
#         # Weights represent confidence in keyword-intent relationship
#         self.intent_keywords = {
#             IntentLabel.ACCOUNT_ISSUE: {
#                 "password": 0.9, "login": 0.8, "account": 0.7, "lockout": 0.95,
#                 "reset": 0.85, "authentication": 0.75, "mfa": 0.9
#             },
#             IntentLabel.BILLING_ISSUE: {
#                 "billing": 0.95, "invoice": 0.9, "charge": 0.85, "payment": 0.8,
#                 "price": 0.7, "cost": 0.7, "duplicate": 0.8, "overcharge": 0.9
#             },
#             IntentLabel.SECURITY_INCIDENT: {
#                 "security": 0.9, "breach": 0.95, "hack": 0.9, "unauthorized": 0.85,
#                 "suspicious": 0.8, "malware": 0.9, "phishing": 0.85, "compromise": 0.9
#             },
#             IntentLabel.SERVICE_OUTAGE: {
#                 "outage": 0.95, "down": 0.9, "unavailable": 0.85, "offline": 0.8,
#                 "broken": 0.7, "crash": 0.8, "outage": 0.95, "disruption": 0.85
#             },
#             IntentLabel.VPN_ISSUE: {
#                 "vpn": 0.95, "remote": 0.7, "connect": 0.8, "authentication": 0.75,
#                 "tunnel": 0.8, "gateway": 0.7, "timeout": 0.8
#             },
#             IntentLabel.BACKEND_ERROR: {
#                 "error": 0.8, "500": 0.9, "503": 0.9, "exception": 0.8,
#                 "bug": 0.7, "crash": 0.8, "failure": 0.8, "exception": 0.8
#             },
#             IntentLabel.PERFORMANCE_ISSUE: {
#                 "slow": 0.85, "performance": 0.9, "latency": 0.9, "delay": 0.8,
#                 "timeout": 0.8, "degradation": 0.85, "bottleneck": 0.8
#             },
#             IntentLabel.DATA_ISSUE: {
#                 "data": 0.8, "missing": 0.85, "corrupt": 0.9, "export": 0.7,
#                 "import": 0.7, "database": 0.8, "record": 0.7
#             },
#             IntentLabel.FEATURE_REQUEST: {
#                 "request": 0.7, "feature": 0.9, "enhancement": 0.8, "improve": 0.7,
#                 "suggest": 0.7, "should": 0.6, "could": 0.6
#             },
#             IntentLabel.COMPLIANCE_REQUEST: {
#                 "compliance": 0.95, "audit": 0.9, "sox": 0.8, "regulation": 0.8,
#                 "certification": 0.7, "policy": 0.7
#             }
#         }
#
#         # Define severity indicators with impact scores
#         self.severity_indicators = {
#             SeverityLabel.CRITICAL: {
#                 "critical": 0.95, "emergency": 0.9, "outage": 0.85, "down": 0.8,
#                 "urgent": 0.75, "cannot": 0.7, "broken": 0.7, "failed": 0.7
#             },
#             SeverityLabel.HIGH: {
#                 "high": 0.8, "important": 0.7, "significant": 0.7, "major": 0.75,
#                 "serious": 0.7, "blocking": 0.8, "prevent": 0.7
#             },
#             SeverityLabel.MEDIUM: {
#                 "medium": 0.6, "moderate": 0.6, "some": 0.5, "partial": 0.6,
#                 "degraded": 0.7, "slow": 0.6, "issue": 0.5
#             },
#             SeverityLabel.LOW: {
#                 "low": 0.4, "minor": 0.5, "cosmetic": 0.6, "enhancement": 0.4,
#                 "request": 0.3, "suggestion": 0.3, "question": 0.4
#             }
#         }
#
#     async def classify_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Classify ticket intent and severity using keyword matching and softmax scoring.
#         Returns structured classification with confidence scores.
#         """
#         # Extract normalized text for processing
#         text = ticket_data.get("normalized_text", "")
#
#         # Calculate raw scores for each intent category
#         intent_scores = self._calculate_intent_scores(text)
#
#         # Apply softmax to convert raw scores to probability distribution
#         intent_probabilities = self._manual_softmax(intent_scores)
#
#         # Select highest probability intent with tie-breaking
#         predicted_intent, intent_confidence = self._select_intent(intent_probabilities)
#
#         # Calculate severity scores based on urgency indicators
#         severity_scores = self._calculate_severity_scores(text, ticket_data)
#         severity_probabilities = self._manual_softmax(severity_scores)
#         predicted_severity, severity_confidence = self._select_severity(severity_probabilities)
#
#         return {
#             "intent": predicted_intent,
#             "intent_confidence": intent_confidence,
#             "intent_probabilities": intent_probabilities,
#             "severity": predicted_severity,
#             "severity_confidence": severity_confidence,
#             "severity_probabilities": severity_probabilities,
#             "classification_time_ms": int(time.time() * 1000) % 10000  # Simple timing
#         }
#
#     def _calculate_intent_scores(self, text: str) -> Dict[IntentLabel, float]:
#         """
#         Calculate raw intent scores based on keyword matching and weighting.
#         Returns dictionary of intent labels to raw scores.
#         """
#         scores = {}
#
#         # Iterate through each intent category and its associated keywords
#         for intent_label, keywords in self.intent_keywords.items():
#             category_score = 0.0
#             matched_keywords = []
#
#             # Check each keyword in the current intent category
#             for keyword, weight in keywords.items():
#                 # If keyword found in text, add weighted score
#                 if keyword in text:
#                     category_score += weight
#                     matched_keywords.append(keyword)
#
#             # Apply logarithmic scaling to prevent keyword spam dominance
#             # This ensures reasonable scores even with many keyword matches
#             if category_score > 0:
#                 # Log scaling: log(1 + score) prevents explosion and normalizes
#                 category_score = math.log(1 + category_score)
#
#             scores[intent_label] = category_score
#
#             # Store matched keywords for explainability (enterprise requirement)
#             if matched_keywords:
#                 scores[f"{intent_label}_keywords"] = matched_keywords
#
#         return scores
#
#     def _manual_softmax(self, scores: Dict[Any, float]) -> Dict[Any, float]:
#         """
#         Manual implementation of softmax function for probability conversion.
#         Converts raw scores to probability distribution that sums to 1.0.
#         """
#         # Handle empty scores edge case - return uniform distribution
#         if not scores:
#             return {}
#
#         # Calculate exponential of each score for softmax numerator
#         exp_scores = {}
#         for key, score in scores.items():
#             # Skip non-score entries (like keyword lists)
#             if not isinstance(score, (int, float)):
#                 continue
#             # Exponential transformation: e^score
#             exp_scores[key] = math.exp(score)
#
#         # Calculate sum of all exponentials for softmax denominator
#         sum_exp = sum(exp_scores.values())
#
#         # Handle division by zero edge case
#         if sum_exp == 0:
#             # Return uniform distribution if all exponentials are zero
#             uniform_prob = 1.0 / len(exp_scores)
#             return {key: uniform_prob for key in exp_scores.keys()}
#
#         # Calculate final probabilities: e^score / sum(e^scores)
#         probabilities = {}
#         for key, exp_score in exp_scores.items():
#             probabilities[key] = exp_score / sum_exp
#
#         return probabilities
#
#     def _select_intent(self, probabilities: Dict[IntentLabel, float]) -> Tuple[str, float]:
#         """
#         Select intent with highest probability, handling ties with deterministic randomness.
#         Returns tuple of (intent_label, confidence_score).
#         """
#         # Filter only intent probabilities (exclude keyword metadata)
#         intent_probs = {
#             k: v for k, v in probabilities.items()
#             if isinstance(k, IntentLabel)
#         }
#
#         # Handle empty probabilities edge case
#         if not intent_probs:
#             return IntentLabel.UNKNOWN, 0.0
#
#         # Find maximum probability value
#         max_prob = max(intent_probs.values())
#
#         # Find all intents with maximum probability (tie detection)
#         best_intents = [intent for intent, prob in intent_probs.items() if prob == max_prob]
#
#         # If single best intent, return it with confidence
#         if len(best_intents) == 1:
#             return best_intents[0].value, max_prob
#
#         # Tie-breaking logic: use deterministic randomness based on seed
#         # Hash the intent names and select based on seeded random choice
#         tied_names = [intent.value for intent in best_intents]
#
#         # Create deterministic random generator for tie-breaking
#         tie_breaker = random.Random(self.seed)
#         selected_intent = tie_breaker.choice(tied_names)
#
#         # Adjust confidence for ties: divide by number of tied intents
#         adjusted_confidence = max_prob / len(best_intents)
#
#         return selected_intent, adjusted_confidence
#
#     def _calculate_severity_scores(self, text: str, ticket_data: Dict[str, Any]) -> Dict[SeverityLabel, float]:
#         """
#         Calculate severity scores using keyword matching and contextual factors.
#         Incorporates urgency indicators and ticket metadata.
#         """
#         base_scores = {}
#
#         # Calculate base scores from severity keyword matching
#         for severity_label, indicators in self.severity_indicators.items():
#             severity_score = 0.0
#             for keyword, weight in indicators.items():
#                 if keyword in text:
#                     severity_score += weight
#             base_scores[severity_label] = severity_score
#
#         # Apply contextual severity adjustments
#         # Enterprise systems use multiple signals for severity assessment
#
#         # Adjustment 1: Urgency score from preprocessing
#         urgency_score = ticket_data.get("urgency_score", 0)
#         if urgency_score > 0:
#             # Increase critical and high severity scores based on urgency
#             base_scores[SeverityLabel.CRITICAL] += urgency_score * 0.2
#             base_scores[SeverityLabel.HIGH] += urgency_score * 0.15
#
#         # Adjustment 2: Technical complexity penalty
#         technical_terms = ticket_data.get("technical_terms", [])
#         if len(technical_terms) > 2:
#             # Complex technical issues often require higher severity
#             base_scores[SeverityLabel.HIGH] += 0.3
#
#         # Adjustment 3: Word count heuristic
#         word_count = ticket_data.get("word_count", 0)
#         if word_count > 100:
#             # Very detailed descriptions often indicate serious issues
#             base_scores[SeverityLabel.HIGH] += 0.2
#
#         return base_scores
#
#     def _select_severity(self, probabilities: Dict[SeverityLabel, float]) -> Tuple[str, float]:
#         """
#         Select severity with highest probability, with enterprise fallback logic.
#         Ensures reasonable severity assignment even with low confidence.
#         """
#         # Filter only severity probabilities
#         severity_probs = {
#             k: v for k, v in probabilities.items()
#             if isinstance(k, SeverityLabel)
#         }
#
#         # Handle empty probabilities edge case
#         if not severity_probs:
#             return SeverityLabel.MEDIUM, 0.5  # Default fallback
#
#         # Find maximum probability severity
#         max_prob = max(severity_probs.values())
#         best_severities = [sev for sev, prob in severity_probs.items() if prob == max_prob]
#
#         if len(best_severities) == 1:
#             return best_severities[0].value, max_prob
#
#         # Tie-breaking for severity: prefer higher severity in ties
#         # Enterprise safety principle: better to over-escalate than under-escalate
#         severity_order = [SeverityLabel.CRITICAL, SeverityLabel.HIGH, SeverityLabel.MEDIUM, SeverityLabel.LOW]
#         for severity in severity_order:
#             if severity in best_severities:
#                 adjusted_confidence = max_prob / len(best_severities)
#                 return severity.value, adjusted_confidence
#
#         # Fallback if no severity found (should not happen)
#         return SeverityLabel.MEDIUM, 0.5
#
#
# class KBSearchAgent:
#     """
#     Knowledge Base search agent with fuzzy matching and deterministic caching.
#     Searches historical tickets and KB articles for relevant evidence.
#     """
#
#     def __init__(self, seed: int = 42):
#         random.seed(seed)
#         self.seed = seed
#         self.kb_cache = None
#         self.historical_tickets = None
#
#     async def search_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Search knowledge base and historical tickets for relevant information.
#         Returns structured evidence with relevance scoring.
#         """
#         # Lazy load KB data on first use
#         if self.kb_cache is None:
#             await self._load_knowledge_base()
#
#         text = ticket_data.get("normalized_text", "")
#         evidence_items = []
#
#         # Search 1: Historical ticket similarity matching
#         historical_matches = self._search_historical_tickets(text)
#         evidence_items.extend(historical_matches)
#
#         # Search 2: KB article keyword matching
#         kb_matches = self._search_kb_articles(text)
#         evidence_items.extend(kb_matches)
#
#         # Sort by relevance score and limit results
#         evidence_items.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
#         top_evidence = evidence_items[:5]  # Return top 5 most relevant
#
#         return {
#             "kb_evidence": top_evidence,
#             "total_matches": len(evidence_items),
#             "search_time_ms": int(time.time() * 1000) % 10000
#         }
#
#     async def _load_knowledge_base(self):
#         """Load knowledge base data from sample_tickets.json with error handling."""
#         try:
#             # Simulate async file loading for enterprise data source
#             await asyncio.sleep(0.01)
#
#             # In production, this would load from actual KB source
#             # For now, create simulated KB articles based on common issues
#             self.kb_cache = [
#                 {
#                     "id": "KB-001",
#                     "title": "VPN Authentication Troubleshooting Guide",
#                     "content": "Step-by-step guide for resolving VPN authentication issues including AUTH_TIMEOUT errors",
#                     "category": "networking",
#                     "relevance_score": 0.95
#                 },
#                 {
#                     "id": "KB-002",
#                     "title": "Password Reset and Account Recovery",
#                     "content": "Complete procedure for resetting user passwords and recovering locked accounts",
#                     "category": "authentication",
#                     "relevance_score": 0.90
#                 },
#                 {
#                     "id": "KB-003",
#                     "title": "Service Outage Response Protocol",
#                     "content": "Standard operating procedure for handling service outages and customer communications",
#                     "category": "incident_management",
#                     "relevance_score": 0.88
#                 },
#                 {
#                     "id": "KB-004",
#                     "title": "Billing Discrepancy Investigation",
#                     "content": "Process for identifying and resolving billing inconsistencies and duplicate charges",
#                     "category": "billing",
#                     "relevance_score": 0.85
#                 },
#                 {
#                     "id": "KB-005",
#                     "title": "Security Incident Response Checklist",
#                     "content": "Comprehensive checklist for responding to security incidents and potential breaches",
#                     "category": "security",
#                     "relevance_score": 0.92
#                 }
#             ]
#
#             # Load historical tickets for similarity matching
#             self.historical_tickets = [
#                 {
#                     "ticket_id": "TCK-1001",
#                     "subject": "VPN authentication timeout issues",
#                     "body": "User cannot connect to VPN, getting AUTH_TIMEOUT errors",
#                     "intent": "vpn_issue",
#                     "severity": "high"
#                 },
#                 {
#                     "ticket_id": "TCK-1002",
#                     "subject": "Password reset request for locked account",
#                     "body": "Account locked after multiple failed login attempts, need immediate reset",
#                     "intent": "account_issue",
#                     "severity": "medium"
#                 },
#                 {
#                     "ticket_id": "TCK-1003",
#                     "subject": "Service outage dashboard unavailable",
#                     "body": "Complete service outage affecting all customers, dashboard unreachable",
#                     "intent": "service_outage",
#                     "severity": "critical"
#                 }
#             ]
#
#         except Exception as e:
#             # Fallback to empty data on load failure
#             self.kb_cache = []
#             self.historical_tickets = []
#
#     def _search_historical_tickets(self, query: str) -> List[Dict[str, Any]]:
#         """Search historical tickets using fuzzy text matching."""
#         matches = []
#         if not self.historical_tickets:
#             return matches
#
#         for ticket in self.historical_tickets:
#             # Combine ticket fields for comprehensive matching
#             ticket_text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()
#
#             # Calculate simple word overlap score
#             query_words = set(query.split())
#             ticket_words = set(ticket_text.split())
#             overlap = query_words.intersection(ticket_words)
#
#             if overlap:
#                 # Calculate relevance score based on word overlap
#                 relevance_score = len(overlap) / len(query_words) if query_words else 0
#
#                 matches.append({
#                     "type": "historical_ticket",
#                     "source": ticket["ticket_id"],
#                     "content": ticket_text[:200] + "...",  # Preview
#                     "relevance_score": min(relevance_score, 1.0),
#                     "metadata": {
#                         "original_intent": ticket.get("intent"),
#                         "original_severity": ticket.get("severity")
#                     }
#                 })
#
#         return matches
#
#     def _search_kb_articles(self, query: str) -> List[Dict[str, Any]]:
#         """Search KB articles using keyword matching and relevance scoring."""
#         matches = []
#         if not self.kb_cache:
#             return matches
#
#         query_words = set(query.split())
#
#         for article in self.kb_cache:
#             # Search in title and content
#             search_text = f"{article.get('title', '')} {article.get('content', '')}".lower()
#             article_words = set(search_text.split())
#
#             overlap = query_words.intersection(article_words)
#
#             if overlap:
#                 # Base relevance from word overlap
#                 base_relevance = len(overlap) / len(query_words) if query_words else 0
#
#                 # Boost relevance if article has high inherent score
#                 article_boost = article.get("relevance_score", 0.5)
#                 final_relevance = (base_relevance * 0.7) + (article_boost * 0.3)
#
#                 matches.append({
#                     "type": "kb_article",
#                     "source": article["id"],
#                     "title": article["title"],
#                     "content": article["content"][:150] + "...",
#                     "relevance_score": min(final_relevance, 1.0),
#                     "category": article.get("category")
#                 })
#
#         return matches
#
#
# class ActionAgent:
#     """
#     Action generation agent with rule-based strategies for different scenarios.
#     Generates enterprise-grade action plans based on intent and severity.
#     """
#
#     def __init__(self, seed: int = 42):
#         random.seed(seed)
#         self.seed = seed
#
#         # Define action strategies for different intents
#         self.action_strategies = {
#             IntentLabel.SECURITY_INCIDENT: self._generate_security_actions,
#             IntentLabel.SERVICE_OUTAGE: self._generate_outage_actions,
#             IntentLabel.ACCOUNT_ISSUE: self._generate_account_actions,
#             IntentLabel.VPN_ISSUE: self._generate_vpn_actions,
#             IntentLabel.BILLING_ISSUE: self._generate_billing_actions,
#             IntentLabel.BACKEND_ERROR: self._generate_backend_actions
#         }
#
#     async def generate_actions(self, ticket_data: Dict[str, Any],
#                                classification: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Generate appropriate actions based on ticket intent and severity.
#         Returns structured action plan with prioritization.
#         """
#         intent = classification.get("intent")
#         severity = classification.get("severity")
#         intent_confidence = classification.get("intent_confidence", 0.0)
#
#         actions = []
#
#         # STRATEGY 1: Always include severity-based escalation actions
#         severity_actions = self._generate_severity_actions(severity)
#         actions.extend(severity_actions)
#
#         # STRATEGY 2: Intent-specific action strategies
#         if intent in self.action_strategies:
#             strategy_actions = self.action_strategies[intent](severity, intent_confidence)
#             actions.extend(strategy_actions)
#         else:
#             # Fallback strategy for unknown intents
#             fallback_actions = self._generate_fallback_actions(severity)
#             actions.extend(fallback_actions)
#
#         # STRATEGY 3: Information gathering actions for low-confidence classifications
#         if intent_confidence < 0.7:
#             info_actions = self._generate_info_gathering_actions()
#             actions.extend(info_actions)
#
#         # Remove duplicates and prioritize
#         unique_actions = self._deduplicate_actions(actions)
#         prioritized_actions = self._prioritize_actions(unique_actions, severity)
#
#         return {
#             "actions": prioritized_actions,
#             "strategy_used": intent if intent in self.action_strategies else "fallback",
#             "action_count": len(prioritized_actions),
#             "generation_time_ms": int(time.time() * 1000) % 10000
#         }
#
#     def _generate_severity_actions(self, severity: str) -> List[Dict[str, Any]]:
#         """Generate severity-based escalation and notification actions."""
#         actions = []
#
#         # CRITICAL severity: Immediate escalation and broad notifications
#         if severity == SeverityLabel.CRITICAL:
#             actions.extend([
#                 {
#                     "action": "immediate_escalation",
#                     "target": "incident_commander",
#                     "priority": "immediate",
#                     "reason": "critical_severity_auto_escalation"
#                 },
#                 {
#                     "action": "notify_stakeholders",
#                     "target": "all_managers",
#                     "priority": "immediate",
#                     "reason": "critical_incident_broadcast"
#                 },
#                 {
#                     "action": "activate_war_room",
#                     "target": "incident_response_team",
#                     "priority": "immediate",
#                     "reason": "critical_severity_protocol"
#                 }
#             ])
#
#         # HIGH severity: Team escalation and manager notifications
#         elif severity == SeverityLabel.HIGH:
#             actions.extend([
#                 {
#                     "action": "escalate_to_team_lead",
#                     "target": "relevant_team_lead",
#                     "priority": "high",
#                     "reason": "high_severity_escalation"
#                 },
#                 {
#                     "action": "notify_management",
#                     "target": "department_manager",
#                     "priority": "high",
#                     "reason": "high_impact_incident"
#                 }
#             ])
#
#         # MEDIUM severity: Standard processing with monitoring
#         elif severity == SeverityLabel.MEDIUM:
#             actions.append({
#                 "action": "standard_processing",
#                 "target": "assigned_agent",
#                 "priority": "medium",
#                 "reason": "medium_severity_standard_flow"
#             })
#
#         # LOW severity: Queue-based processing
#         else:
#             actions.append({
#                 "action": "queue_for_processing",
#                 "target": "general_queue",
#                 "priority": "low",
#                 "reason": "low_severity_normal_queue"
#             })
#
#         return actions
#
#     def _generate_security_actions(self, severity: str, confidence: float) -> List[Dict[str, Any]]:
#         """Generate security incident response actions."""
#         actions = []
#
#         # Always include basic security containment actions
#         base_actions = [
#             {
#                 "action": "isolate_affected_systems",
#                 "target": "compromised_assets",
#                 "priority": "immediate",
#                 "reason": "security_containment_protocol"
#             },
#             {
#                 "action": "preserve_evidence",
#                 "target": "log_files",
#                 "priority": "high",
#                 "reason": "forensic_evidence_collection"
#             },
#             {
#                 "action": "review_access_logs",
#                 "target": "authentication_system",
#                 "priority": "high",
#                 "reason": "unauthorized_access_investigation"
#             }
#         ]
#         actions.extend(base_actions)
#
#         # High confidence security incidents get additional actions
#         if confidence > 0.8:
#             actions.extend([
#                 {
#                     "action": "initiate_incident_response",
#                     "target": "security_team",
#                     "priority": "immediate",
#                     "reason": "high_confidence_security_incident"
#                 },
#                 {
#                     "action": "check_breach_protocol",
#                     "target": "compliance_team",
#                     "priority": "high",
#                     "reason": "potential_data_breach_assessment"
#                 }
#             ])
#
#         return actions
#
#     def _generate_outage_actions(self, severity: str, confidence: float) -> List[Dict[str, Any]]:
#         """Generate service outage response actions."""
#         actions = []
#
#         # Core outage response actions
#         core_actions = [
#             {
#                 "action": "verify_service_health",
#                 "target": "monitoring_system",
#                 "priority": "immediate",
#                 "reason": "outage_confirmation"
#             },
#             {
#                 "action": "check_dependencies",
#                 "target": "dependency_graph",
#                 "priority": "high",
#                 "reason": "root_cause_analysis"
#             },
#             {
#                 "action": "notify_customers",
#                 "target": "status_page",
#                 "priority": "high",
#                 "reason": "outage_communication_protocol"
#             }
#         ]
#         actions.extend(core_actions)
#
#         # Critical outages get full incident response
#         if severity == SeverityLabel.CRITICAL:
#             actions.extend([
#                 {
#                     "action": "activate_bcp",
#                     "target": "business_continuity_plan",
#                     "priority": "immediate",
#                     "reason": "critical_outage_bcp_activation"
#                 },
#                 {
#                     "action": "mobilize_all_teams",
#                     "target": "all_engineering",
#                     "priority": "immediate",
#                     "reason": "all_hands_incident_response"
#                 }
#             ])
#
#         return actions
#
#     def _generate_account_actions(self, severity: str, confidence: float) -> List[Dict[str, Any]]:
#         """Generate account-related assistance actions."""
#         actions = []
#
#         # Standard account recovery actions
#         actions.extend([
#             {
#                 "action": "verify_user_identity",
#                 "target": "authentication_system",
#                 "priority": "medium",
#                 "reason": "account_recovery_verification"
#             },
#             {
#                 "action": "check_account_status",
#                 "target": "user_database",
#                 "priority": "medium",
#                 "reason": "account_state_verification"
#             }
#         ])
#
#         # High severity account issues get expedited processing
#         if severity in [SeverityLabel.HIGH, SeverityLabel.CRITICAL]:
#             actions.append({
#                 "action": "expedited_password_reset",
#                 "target": "admin_console",
#                 "priority": "high",
#                 "reason": "high_severity_account_recovery"
#             })
#
#         return actions
#
#     def _generate_vpn_actions(self, severity: str, confidence: float) -> List[Dict[str, Any]]:
#         """Generate VPN issue resolution actions."""
#         actions = []
#
#         # VPN troubleshooting sequence
#         actions.extend([
#             {
#                 "action": "check_vpn_server_status",
#                 "target": "vpn_infrastructure",
#                 "priority": "high",
#                 "reason": "vpn_connectivity_verification"
#             },
#             {
#                 "action": "review_auth_logs",
#                 "target": "authentication_server",
#                 "priority": "medium",
#                 "reason": "vpn_auth_failure_investigation"
#             },
#             {
#                 "action": "verify_user_certificates",
#                 "target": "certificate_authority",
#                 "priority": "medium",
#                 "reason": "vpn_certificate_validation"
#             }
#         ])
#
#         return actions
#
#     def _generate_billing_actions(self, severity: str, confidence: float) -> List[Dict[str, Any]]:
#         """Generate billing issue resolution actions."""
#         actions = []
#
#         actions.extend([
#             {
#                 "action": "review_invoice_details",
#                 "target": "billing_system",
#                 "priority": "medium",
#                 "reason": "billing_discrepancy_investigation"
#             },
#             {
#                 "action": "check_payment_history",
#                 "target": "payment_processor",
#                 "priority": "medium",
#                 "reason": "payment_verification"
#             }
#         ])
#
#         # High severity billing issues might indicate financial impact
#         if severity == SeverityLabel.HIGH:
#             actions.append({
#                 "action": "escalate_to_finance",
#                 "target": "finance_department",
#                 "priority": "high",
#                 "reason": "significant_billing_discrepancy"
#             })
#
#         return actions
#
#     def _generate_backend_actions(self, severity: str, confidence: float) -> List[Dict[str, Any]]:
#         """Generate backend error resolution actions."""
#         actions = []
#
#         actions.extend([
#             {
#                 "action": "check_application_logs",
#                 "target": "log_aggregation",
#                 "priority": "high",
#                 "reason": "backend_error_investigation"
#             },
#             {
#                 "action": "verify_infrastructure_health",
#                 "target": "monitoring_system",
#                 "priority": "medium",
#                 "reason": "infrastructure_health_check"
#             },
#             {
#                 "action": "review_recent_deployments",
#                 "target": "deployment_system",
#                 "priority": "medium",
#                 "reason": "recent_changes_analysis"
#             }
#         ])
#
#         return actions
#
#     def _generate_fallback_actions(self, severity: str) -> List[Dict[str, Any]]:
#         """Generate fallback actions for unknown or low-confidence intents."""
#         return [
#             {
#                 "action": "request_more_information",
#                 "target": "ticket_requester",
#                 "priority": "medium",
#                 "reason": "insufficient_intent_confidence"
#             },
#             {
#                 "action": "escalate_to_human_agent",
#                 "target": "tier2_support",
#                 "priority": "medium" if severity == SeverityLabel.MEDIUM else "high",
#                 "reason": "ambiguous_intent_requires_human_review"
#             }
#         ]
#
#     def _generate_info_gathering_actions(self) -> List[Dict[str, Any]]:
#         """Generate information gathering actions for low-confidence classifications."""
#         return [
#             {
#                 "action": "request_additional_details",
#                 "target": "ticket_requester",
#                 "priority": "medium",
#                 "reason": "low_confidence_classification"
#             },
#             {
#                 "action": "gather_system_logs",
#                 "target": "affected_systems",
#                 "priority": "medium",
#                 "reason": "additional_diagnostic_information"
#             }
#         ]
#
#     def _deduplicate_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """Remove duplicate actions based on action and target combination."""
#         seen = set()
#         unique_actions = []
#
#         for action in actions:
#             # Create unique key from action and target
#             key = (action["action"], action["target"])
#             if key not in seen:
#                 seen.add(key)
#                 unique_actions.append(action)
#
#         return unique_actions
#
#     def _prioritize_actions(self, actions: List[Dict[str, Any]], severity: str) -> List[Dict[str, Any]]:
#         """Prioritize actions based on severity and inherent priority."""
#         priority_order = {"immediate": 0, "high": 1, "medium": 2, "low": 3}
#
#         def action_key(action):
#             # Primary sort by priority level
#             priority_score = priority_order.get(action.get("priority", "medium"), 2)
#             # Secondary sort by action type for deterministic ordering
#             action_name = action.get("action", "")
#             return (priority_score, action_name)
#
#         return sorted(actions, key=action_key)
import asyncio
import time
import json
import math
import random
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class IntentLabel(str, Enum):
    """Enumeration of supported intent labels for deterministic classification."""
    ACCOUNT_ISSUE = "account_issue"
    BILLING_ISSUE = "billing_issue"
    SECURITY_INCIDENT = "security_incident"
    SERVICE_OUTAGE = "service_outage"
    VPN_ISSUE = "vpn_issue"
    BACKEND_ERROR = "backend_error"
    PERFORMANCE_ISSUE = "performance_issue"
    DATA_ISSUE = "data_issue"
    FEATURE_REQUEST = "feature_request"
    COMPLIANCE_REQUEST = "compliance_request"
    UNKNOWN = "unknown"


class SeverityLabel(str, Enum):
    """Enumeration of supported severity labels for deterministic classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClassifierAgent:
    """
    Intent and severity classification agent with manual softmax implementation.
    Uses deterministic keyword matching and mathematical probability calculation.
    """

    def __init__(self, seed: int = 42):
        # Initialize deterministic random seed for reproducible tie-breaking
        random.seed(seed)
        self.seed = seed

        # Define intent keywords with deterministic weights for pattern matching
        # Weights represent confidence in keyword-intent relationship (enterprise-tuned)
        self.intent_keywords = {
            IntentLabel.ACCOUNT_ISSUE: {
                "password": 0.9, "login": 0.8, "account": 0.7, "lockout": 0.95,
                "reset": 0.85, "authentication": 0.75, "mfa": 0.9, "access": 0.7
            },
            IntentLabel.BILLING_ISSUE: {
                "billing": 0.95, "invoice": 0.9, "charge": 0.85, "payment": 0.8,
                "price": 0.7, "cost": 0.7, "duplicate": 0.8, "overcharge": 0.9, "refund": 0.8
            },
            IntentLabel.SECURITY_INCIDENT: {
                "security": 0.9, "breach": 0.95, "hack": 0.9, "unauthorized": 0.85,
                "suspicious": 0.8, "malware": 0.9, "phishing": 0.85, "compromise": 0.9
            },
            IntentLabel.SERVICE_OUTAGE: {
                "outage": 0.95, "down": 0.9, "unavailable": 0.85, "offline": 0.8,
                "broken": 0.7, "crash": 0.8, "outage": 0.95, "disruption": 0.85
            },
            IntentLabel.VPN_ISSUE: {
                "vpn": 0.95, "remote": 0.7, "connect": 0.8, "authentication": 0.75,
                "tunnel": 0.8, "gateway": 0.7, "timeout": 0.8, "disconnect": 0.75
            },
            IntentLabel.BACKEND_ERROR: {
                "error": 0.8, "500": 0.9, "503": 0.9, "exception": 0.8,
                "bug": 0.7, "crash": 0.8, "failure": 0.8, "exception": 0.8
            },
            IntentLabel.PERFORMANCE_ISSUE: {
                "slow": 0.85, "performance": 0.9, "latency": 0.9, "delay": 0.8,
                "timeout": 0.8, "degradation": 0.85, "bottleneck": 0.8
            },
            IntentLabel.DATA_ISSUE: {
                "data": 0.8, "missing": 0.85, "corrupt": 0.9, "export": 0.7,
                "import": 0.7, "database": 0.8, "record": 0.7, "corruption": 0.85
            },
            IntentLabel.FEATURE_REQUEST: {
                "request": 0.7, "feature": 0.9, "enhancement": 0.8, "improve": 0.7,
                "suggest": 0.7, "should": 0.6, "could": 0.6, "would": 0.6
            },
            IntentLabel.COMPLIANCE_REQUEST: {
                "compliance": 0.95, "audit": 0.9, "sox": 0.8, "regulation": 0.8,
                "certification": 0.7, "policy": 0.7, "governance": 0.8
            }
        }

        # Define severity indicators with deterministic impact scores
        # Higher weights indicate stronger severity correlation
        self.severity_indicators = {
            SeverityLabel.CRITICAL: {
                "critical": 0.95, "emergency": 0.9, "outage": 0.85, "down": 0.8,
                "urgent": 0.75, "cannot": 0.7, "broken": 0.7, "failed": 0.7, "disaster": 0.9
            },
            SeverityLabel.HIGH: {
                "high": 0.8, "important": 0.7, "significant": 0.7, "major": 0.75,
                "serious": 0.7, "blocking": 0.8, "prevent": 0.7, "severe": 0.8
            },
            SeverityLabel.MEDIUM: {
                "medium": 0.6, "moderate": 0.6, "some": 0.5, "partial": 0.6,
                "degraded": 0.7, "slow": 0.6, "issue": 0.5, "problem": 0.5
            },
            SeverityLabel.LOW: {
                "low": 0.4, "minor": 0.5, "cosmetic": 0.6, "enhancement": 0.4,
                "request": 0.3, "suggestion": 0.3, "question": 0.4, "info": 0.3
            }
        }

    async def classify_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify ticket intent and severity using deterministic keyword matching and softmax.
        Returns structured classification with mathematical confidence scores.
        """
        # Extract normalized text for case-insensitive processing
        text = ticket_data.get("normalized_text", "")

        # Calculate raw scores for each intent category using keyword matching
        # Raw scores represent unnormalized confidence in each intent
        intent_scores = self._calculate_intent_scores(text)

        # Apply manual softmax to convert raw scores to probability distribution
        # Softmax ensures probabilities sum to 1.0 for proper confidence interpretation
        intent_probabilities = self._manual_softmax(intent_scores)

        # Select highest probability intent with deterministic tie-breaking
        # Tie-breaking uses seeded randomness for reproducible results
        predicted_intent, intent_confidence = self._select_intent(intent_probabilities)

        # Calculate severity scores based on urgency indicators and contextual factors
        severity_scores = self._calculate_severity_scores(text, ticket_data)
        severity_probabilities = self._manual_softmax(severity_scores)
        predicted_severity, severity_confidence = self._select_severity(severity_probabilities)

        return {
            "intent": predicted_intent,
            "severity": predicted_severity,
            "intent_scores": intent_scores,
            "severity_scores": severity_scores
        }

    def _calculate_intent_scores(self, text: str) -> Dict[IntentLabel, float]:
        """
        Calculate raw intent scores based on deterministic keyword matching and weighting.
        Returns dictionary of intent labels to raw scores for probability conversion.
        """
        scores = {}

        # Iterate through each intent category and its associated keywords
        # Each intent gets a score based on weighted keyword matches
        for intent_label, keywords in self.intent_keywords.items():
            category_score = 0.0
            matched_keywords = []

            # Check each keyword in the current intent category
            # If keyword found in text, add its predetermined weight to category score
            for keyword, weight in keywords.items():
                # Case-insensitive keyword matching for deterministic behavior
                if keyword in text:
                    category_score += weight
                    matched_keywords.append(keyword)

            # Apply logarithmic scaling to prevent keyword spam dominance
            # Log(1 + score) transformation prevents exponential score growth
            if category_score > 0:
                # Natural log scaling normalizes scores while preserving order
                category_score = math.log(1 + category_score)

            scores[intent_label] = category_score

            # Store matched keywords for explainability and debugging
            # Enterprise systems require transparency in classification decisions
            if matched_keywords:
                scores[f"{intent_label}_keywords"] = matched_keywords

        return scores

    def _manual_softmax(self, scores: Dict[Any, float]) -> Dict[Any, float]:
        """
        Manual implementation of softmax function for probability conversion.
        Converts raw scores to probability distribution that sums to 1.0.
        """
        # Handle empty scores edge case - return empty distribution
        if not scores:
            return {}

        # Calculate exponential of each score for softmax numerator
        # Exponential transformation emphasizes score differences
        exp_scores = {}
        for key, score in scores.items():
            # Skip non-score entries (like keyword lists stored for debugging)
            if not isinstance(score, (int, float)):
                continue
            # Exponential transformation: e^score
            # Using math.exp for deterministic floating-point results
            exp_scores[key] = math.exp(score)

        # Calculate sum of all exponentials for softmax denominator
        # This normalizes probabilities to sum to 1.0
        sum_exp = sum(exp_scores.values())

        # Handle division by zero edge case (all exponentials zero)
        if sum_exp == 0:
            # Return uniform distribution if all exponentials are zero
            # Equal probability for all categories when no evidence exists
            uniform_prob = 1.0 / len(exp_scores)
            return {key: uniform_prob for key in exp_scores.keys()}

        # Calculate final probabilities: e^score / sum(e^scores)
        # This produces proper probability distribution summing to 1.0
        probabilities = {}
        for key, exp_score in exp_scores.items():
            probabilities[key] = exp_score / sum_exp

        return probabilities

    def _select_intent(self, probabilities: Dict[Any, float]) -> Tuple[str, float]:
        """
        Select intent with highest probability, handling ties with deterministic randomness.
        Returns tuple of (intent_label, confidence_score).
        """
        # Support both Enum keys (IntentLabel) and plain string keys for testing convenience
        has_enum_keys = any(isinstance(k, IntentLabel) for k in probabilities.keys())
        has_str_keys = any(isinstance(k, str) for k in probabilities.keys())

        if has_enum_keys:
            # Use only Enum keys
            intent_probs: Dict[Any, float] = {
                k: v for k, v in probabilities.items() if isinstance(k, IntentLabel)
            }
            key_to_str = lambda k: k.value  # convert Enum to string for return
        elif has_str_keys:
            # Use only string keys
            intent_probs = {k: v for k, v in probabilities.items() if isinstance(k, str)}
            key_to_str = lambda k: k
        else:
            intent_probs = {}
            key_to_str = lambda k: str(k)

        # Handle empty probabilities edge case - return unknown with zero confidence
        if not intent_probs:
            return IntentLabel.UNKNOWN.value, 0.0

        # Find maximum probability value across all intents
        max_prob = max(intent_probs.values())

        # Find all intents with maximum probability (tie detection)
        # Multiple intents can have same probability in case of ambiguous tickets
        best_intents = [intent for intent, prob in intent_probs.items() if prob == max_prob]

        # If single best intent, return it with full confidence
        if len(best_intents) == 1:
            return key_to_str(best_intents[0]), max_prob

        # Tie-breaking logic: use deterministic randomness based on seed
        # Hash the intent names and select based on seeded random choice
        tied_names = [key_to_str(intent) for intent in best_intents]

        # Create deterministic random generator for reproducible tie-breaking
        # Same seed + same inputs = same tie-breaking decision
        tie_breaker = random.Random(self.seed)
        selected_intent = tie_breaker.choice(tied_names)

        # Adjust confidence for ties: divide by number of tied intents
        # Reflects uncertainty when multiple intents are equally likely
        adjusted_confidence = max_prob / len(best_intents)

        return selected_intent, adjusted_confidence

    def _calculate_severity_scores(self, text: str, ticket_data: Dict[str, Any]) -> Dict[SeverityLabel, float]:
        """
        Calculate severity scores using keyword matching and contextual factors.
        Incorporates urgency indicators and ticket metadata for comprehensive assessment.
        """
        base_scores = {}

        # Calculate base scores from severity keyword matching
        # Each severity level gets score based on keyword presence and weights
        for severity_label, indicators in self.severity_indicators.items():
            severity_score = 0.0
            for keyword, weight in indicators.items():
                if keyword in text:
                    severity_score += weight
            base_scores[severity_label] = severity_score

        # Apply contextual severity adjustments based on ticket characteristics
        # Enterprise systems use multiple signals for accurate severity assessment

        # Adjustment 1: Urgency score from preprocessing
        urgency_score = ticket_data.get("urgency_score", 0)
        if urgency_score > 0:
            # Increase critical and high severity scores based on urgency indicators
            base_scores[SeverityLabel.CRITICAL] += urgency_score * 0.2
            base_scores[SeverityLabel.HIGH] += urgency_score * 0.15

        # Adjustment 2: Technical complexity penalty
        technical_terms = ticket_data.get("technical_terms", [])
        if len(technical_terms) > 2:
            # Complex technical issues often require higher severity due to expertise needed
            base_scores[SeverityLabel.HIGH] += 0.3

        # Adjustment 3: Word count heuristic for information completeness
        word_count = ticket_data.get("word_count", 0)
        if word_count > 100:
            # Very detailed descriptions often indicate serious, well-documented issues
            base_scores[SeverityLabel.HIGH] += 0.2
        elif word_count < 10:
            # Very short descriptions suggest incomplete information, lower confidence
            base_scores[SeverityLabel.LOW] += 0.1

        return base_scores

    def _select_severity(self, probabilities: Dict[SeverityLabel, float]) -> Tuple[str, float]:
        """
        Select severity with highest probability, with enterprise safety fallback logic.
        Ensures reasonable severity assignment even with low confidence or ties.
        """
        # Filter only severity probabilities (exclude any metadata)
        severity_probs = {
            k: v for k, v in probabilities.items()
            if isinstance(k, SeverityLabel)
        }

        # Handle empty probabilities edge case - return medium as safe default
        if not severity_probs:
            return SeverityLabel.MEDIUM, 0.5  # Default fallback with medium confidence

        # Find maximum probability severity
        max_prob = max(severity_probs.values())
        best_severities = [sev for sev, prob in severity_probs.items() if prob == max_prob]

        # Single best severity found - return with full confidence
        if len(best_severities) == 1:
            return best_severities[0].value, max_prob

        # Tie-breaking for severity: prefer higher severity in ties
        # Enterprise safety principle: better to over-escalate than under-escalate
        severity_order = [SeverityLabel.CRITICAL, SeverityLabel.HIGH, SeverityLabel.MEDIUM, SeverityLabel.LOW]
        for severity in severity_order:
            if severity in best_severities:
                # Adjust confidence downward to reflect uncertainty in tie scenario
                adjusted_confidence = max_prob / len(best_severities)
                return severity.value, adjusted_confidence

        # Fallback if no severity found (should not happen with proper initialization)
        return SeverityLabel.MEDIUM, 0.5


class KBSearchAgent:
    """
    Knowledge Base search agent with deterministic fuzzy matching and caching.
    Searches historical tickets and KB articles for relevant evidence and solutions.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed
        self.kb_cache = None
        self.historical_tickets = None

    async def search_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search knowledge base and historical tickets for relevant information.
        Returns structured evidence with deterministic relevance scoring.
        """
        # Lazy load KB data on first use to optimize memory and startup time
        if self.kb_cache is None:
            await self._load_knowledge_base()

        text = ticket_data.get("normalized_text", "")
        matches = []
        scores = []

        # Search 1: Historical ticket similarity matching
        # Finds similar past tickets for solution reuse and pattern recognition
        historical_matches = self._search_historical_tickets(text)
        matches.extend(historical_matches)

        # Search 2: KB article keyword matching
        # Finds relevant knowledge base articles for guided troubleshooting
        kb_matches = self._search_kb_articles(text)
        matches.extend(kb_matches)

        # Calculate deterministic relevance scores for all matches
        # Scores determine match ordering and confidence in relevance
        for match in matches:
            score = match.get("relevance_score", 0.0)
            scores.append(score)

        # Select top matches based on relevance score threshold
        # Only return matches with sufficient relevance for quality control
        selected = [match for match in matches if match.get("relevance_score", 0) > 0.3]

        # Sort selected matches by relevance score (descending)
        # Most relevant evidence appears first in response
        selected.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        return {
            "matches": matches,
            "scores": scores,
            "selected": selected[:5]  # Limit to top 5 for response size management
        }

    async def _load_knowledge_base(self):
        """Load knowledge base data from sample_tickets.json with deterministic caching."""
        try:
            # Simulate async file loading for enterprise data source
            # Real implementation would await actual file I/O or database call
            await asyncio.sleep(0.01)

            # In production, this would load from actual KB source
            # For deterministic testing, create simulated KB articles
            self.kb_cache = [
                {
                    "id": "KB-001",
                    "title": "VPN Authentication Troubleshooting Guide",
                    "content": "Step-by-step guide for resolving VPN authentication issues including AUTH_TIMEOUT errors and certificate problems",
                    "category": "networking",
                    "tags": ["vpn", "authentication", "timeout"]
                },
                {
                    "id": "KB-002",
                    "title": "Password Reset and Account Recovery Procedures",
                    "content": "Complete procedure for resetting user passwords and recovering locked accounts with security verification",
                    "category": "authentication",
                    "tags": ["password", "account", "lockout", "reset"]
                },
                {
                    "id": "KB-003",
                    "title": "Service Outage Response Protocol and Communication",
                    "content": "Standard operating procedure for handling service outages including customer notifications and escalation paths",
                    "category": "incident_management",
                    "tags": ["outage", "downtime", "escalation"]
                },
                {
                    "id": "KB-004",
                    "title": "Billing Discrepancy Investigation Process",
                    "content": "Process for identifying and resolving billing inconsistencies including duplicate charges and payment errors",
                    "category": "billing",
                    "tags": ["billing", "invoice", "payment", "duplicate"]
                },
                {
                    "id": "KB-005",
                    "title": "Security Incident Response Checklist",
                    "content": "Comprehensive checklist for responding to security incidents including containment and forensic preservation",
                    "category": "security",
                    "tags": ["security", "incident", "breach", "containment"]
                }
            ]

            # Load historical tickets for similarity matching from sample data
            # Real implementation would query ticket database
            self.historical_tickets = [
                {
                    "ticket_id": "TCK-1001",
                    "subject": "VPN authentication timeout issues",
                    "body": "User cannot connect to VPN, getting AUTH_TIMEOUT errors consistently",
                    "intent": "vpn_issue",
                    "severity": "high"
                },
                {
                    "ticket_id": "TCK-1002",
                    "subject": "Password reset request for locked account",
                    "body": "Account locked after multiple failed login attempts, need immediate reset for payroll access",
                    "intent": "account_issue",
                    "severity": "medium"
                },
                {
                    "ticket_id": "TCK-1003",
                    "subject": "Service outage dashboard unavailable",
                    "body": "Complete service outage affecting all customers, dashboard unreachable for 2 hours",
                    "intent": "service_outage",
                    "severity": "critical"
                }
            ]

        except Exception as e:
            # Fallback to empty data on load failure for graceful degradation
            # Enterprise systems must handle data source failures elegantly
            self.kb_cache = []
            self.historical_tickets = []

    def _search_historical_tickets(self, query: str) -> List[Dict[str, Any]]:
        """Search historical tickets using deterministic fuzzy text matching."""
        matches = []
        if not self.historical_tickets:
            return matches

        # Iterate through all historical tickets for similarity comparison
        # Each ticket gets a relevance score based on text overlap
        for ticket in self.historical_tickets:
            # Combine ticket fields for comprehensive matching
            ticket_text = f"{ticket.get('subject', '')} {ticket.get('body', '')}".lower()

            # Calculate simple word overlap score for similarity measurement
            query_words = set(query.split())
            ticket_words = set(ticket_text.split())
            overlap = query_words.intersection(ticket_words)

            if overlap:
                # Calculate relevance score based on word overlap ratio
                # Normalize by query length to favor matches with higher coverage
                relevance_score = len(overlap) / len(query_words) if query_words else 0

                matches.append({
                    "type": "historical_ticket",
                    "source": ticket["ticket_id"],
                    "content": ticket_text[:200] + "...",  # Preview for response size
                    "relevance_score": min(relevance_score, 1.0),  # Cap at 1.0
                    "metadata": {
                        "original_intent": ticket.get("intent"),
                        "original_severity": ticket.get("severity")
                    }
                })

        return matches

    def _search_kb_articles(self, query: str) -> List[Dict[str, Any]]:
        """Search KB articles using deterministic keyword matching and relevance scoring."""
        matches = []
        if not self.kb_cache:
            return matches

        query_words = set(query.split())

        # Iterate through all KB articles for relevance assessment
        for article in self.kb_cache:
            # Search in title and content for comprehensive matching
            search_text = f"{article.get('title', '')} {article.get('content', '')}".lower()
            article_words = set(search_text.split())

            # Calculate word overlap between query and article
            overlap = query_words.intersection(article_words)

            if overlap:
                # Base relevance from word overlap (coverage of query terms)
                base_relevance = len(overlap) / len(query_words) if query_words else 0

                # Boost relevance if article tags contain query terms
                tags = article.get("tags", [])
                tag_overlap = len([tag for tag in tags if tag in query_words])
                tag_boost = tag_overlap * 0.1  # Small boost for tag matches

                final_relevance = min(base_relevance + tag_boost, 1.0)

                matches.append({
                    "type": "kb_article",
                    "source": article["id"],
                    "title": article["title"],
                    "content": article["content"][:150] + "...",
                    "relevance_score": final_relevance,
                    "category": article.get("category")
                })

        return matches


class ActionAgent:
    """
    Action generation agent with deterministic rule-based strategies for enterprise scenarios.
    Generates appropriate action plans based on classified intent and severity levels.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed

        # Define action strategies for different intents with deterministic rules
        # Each intent has associated action generation function
        self.action_strategies = {
            IntentLabel.SECURITY_INCIDENT: self._generate_security_actions,
            IntentLabel.SERVICE_OUTAGE: self._generate_outage_actions,
            IntentLabel.ACCOUNT_ISSUE: self._generate_account_actions,
            IntentLabel.VPN_ISSUE: self._generate_vpn_actions,
            IntentLabel.BILLING_ISSUE: self._generate_billing_actions,
            IntentLabel.BACKEND_ERROR: self._generate_backend_actions
        }

    async def generate_actions(self, ticket_data: Dict[str, Any],
                               classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate appropriate actions based on ticket intent and severity using rule-based engine.
        Returns structured action plan with deterministic prioritization.
        """
        intent = classification.get("intent")
        severity = classification.get("severity")
        actions = []
        reasoning = ""

        # Normalize intent/severity types to Enum for consistent strategy dispatch
        intent_label: Optional[IntentLabel] = None
        if isinstance(intent, IntentLabel):
            intent_label = intent
        elif isinstance(intent, str):
            try:
                intent_label = IntentLabel(intent)
            except Exception:
                intent_label = None

        if isinstance(severity, SeverityLabel):
            severity_label = severity
        elif isinstance(severity, str):
            try:
                severity_label = SeverityLabel(severity)
            except Exception:
                severity_label = SeverityLabel.MEDIUM
        else:
            severity_label = SeverityLabel.MEDIUM

        # STRATEGY 1: Always include severity-based escalation actions
        # Severity determines initial response priority and stakeholder involvement
        severity_actions = self._generate_severity_actions(severity_label)
        actions.extend(severity_actions)

        # STRATEGY 2: Intent-specific action strategies based on classified intent
        # Different intents require different specialized action sequences
        if intent_label in self.action_strategies:
            strategy_actions = self.action_strategies[intent_label](severity_label)
            actions.extend(strategy_actions)
            reasoning = f"Applied {intent_label.value} specific action strategy"
        else:
            # Fallback strategy for unknown or unsupported intents
            fallback_actions = self._generate_fallback_actions(severity_label)
            actions.extend(fallback_actions)
            reasoning = "Applied fallback action strategy for unknown intent"

        # STRATEGY 3: Information gathering actions for ambiguous cases
        # Request additional details when classification confidence is low
        if len(actions) < 2:  # Very few actions generated
            info_actions = self._generate_info_gathering_actions()
            actions.extend(info_actions)
            reasoning += " with information gathering supplement"

        # Heuristic safeguard: If the ticket text clearly indicates security
        # terms, ensure containment actions are present regardless of
        # classification outcome. This improves resilience for critical cases.
        normalized_text = (ticket_data.get("normalized_text") or "").lower()
        security_terms = [
            "security", "breach", "unauthorized", "suspicious",
            "compromise", "phishing", "malware"
        ]
        if any(term in normalized_text for term in security_terms):
            actions.extend(self._generate_security_actions(severity_label))

        # Remove duplicate actions and apply deterministic prioritization
        unique_actions = self._deduplicate_actions(actions)
        prioritized_actions = self._prioritize_actions(unique_actions, severity_label)

        # Calculate action confidence based on action count and specificity
        # More specific actions indicate higher confidence in solution
        confidence = min(len(prioritized_actions) * 0.2, 0.9)  # Cap at 0.9

        return {
            "actions": prioritized_actions,
            "reasoning": reasoning,
            "confidence": confidence
        }

    def _generate_severity_actions(self, severity: str) -> List[Dict[str, Any]]:
        """Generate severity-based escalation and notification actions."""
        actions = []

        # CRITICAL severity: Immediate escalation and broad notifications
        # Enterprise protocol for incidents with widespread business impact
        if severity == SeverityLabel.CRITICAL:
            actions.extend([
                {
                    "action": "immediate_escalation",
                    "target": "incident_commander",
                    "priority": "immediate",
                    "reason": "critical_severity_auto_escalation"
                },
                {
                    "action": "notify_stakeholders",
                    "target": "all_managers",
                    "priority": "immediate",
                    "reason": "critical_incident_broadcast"
                }
            ])

        # HIGH severity: Team escalation and manager notifications
        # For significant issues affecting multiple users or systems
        elif severity == SeverityLabel.HIGH:
            actions.extend([
                {
                    "action": "escalate_to_team_lead",
                    "target": "relevant_team_lead",
                    "priority": "high",
                    "reason": "high_severity_escalation"
                }
            ])

        # MEDIUM severity: Standard processing with monitoring
        # Normal enterprise workflow for typical support tickets
        elif severity == SeverityLabel.MEDIUM:
            actions.append({
                "action": "standard_processing",
                "target": "assigned_agent",
                "priority": "medium",
                "reason": "medium_severity_standard_flow"
            })

        # LOW severity: Queue-based processing
        # Non-urgent issues processed during normal business hours
        else:
            actions.append({
                "action": "queue_for_processing",
                "target": "general_queue",
                "priority": "low",
                "reason": "low_severity_normal_queue"
            })

        return actions

    def _generate_security_actions(self, severity: str) -> List[Dict[str, Any]]:
        """Generate security incident response actions with containment focus."""
        actions = []

        # Always include basic security containment actions
        # These actions form the foundation of security incident response
        base_actions = [
            {
                "action": "isolate_affected_systems",
                "target": "compromised_assets",
                "priority": "immediate",
                "reason": "security_containment_protocol"
            },
            {
                "action": "preserve_evidence",
                "target": "log_files",
                "priority": "high",
                "reason": "forensic_evidence_collection"
            }
        ]
        actions.extend(base_actions)

        # Add IP containment for confirmed security incidents
        # Prevents further unauthorized access from suspicious sources
        actions.append({
            "action": "block_suspicious_ips",
            "target": "firewall_rules",
            "priority": "high",
            "reason": "security_incident_containment"
        })

        # Critical security incidents get additional response actions
        if severity == SeverityLabel.CRITICAL:
            actions.extend([
                {
                    "action": "initiate_incident_response",
                    "target": "security_team",
                    "priority": "immediate",
                    "reason": "critical_security_incident_response"
                }
            ])

        return actions

    def _generate_outage_actions(self, severity: str) -> List[Dict[str, Any]]:
        """Generate service outage response actions with restoration focus."""
        actions = []

        # Core outage response actions for all outage severity levels
        core_actions = [
            {
                "action": "verify_service_health",
                "target": "monitoring_system",
                "priority": "immediate",
                "reason": "outage_confirmation"
            },
            {
                "action": "check_dependencies",
                "target": "dependency_graph",
                "priority": "high",
                "reason": "root_cause_analysis"
            }
        ]
        actions.extend(core_actions)

        # Critical outages get full incident response mobilization
        if severity == SeverityLabel.CRITICAL:
            actions.extend([
                {
                    "action": "activate_bcp",
                    "target": "business_continuity_plan",
                    "priority": "immediate",
                    "reason": "critical_outage_bcp_activation"
                },
                {
                    "action": "mobilize_all_teams",
                    "target": "all_engineering",
                    "priority": "immediate",
                    "reason": "all_hands_incident_response"
                }
            ])

        return actions

    def _generate_account_actions(self, severity: str) -> List[Dict[str, Any]]:
        """Generate account-related assistance actions with access restoration focus."""
        actions = []

        # Standard account recovery actions for identity verification
        actions.extend([
            {
                "action": "verify_user_identity",
                "target": "authentication_system",
                "priority": "medium",
                "reason": "account_recovery_verification"
            }
        ])

        # Add MFA reset workflow for account lockout scenarios
        # Essential for regaining access to MFA-protected accounts
        actions.append({
            "action": "reset_mfa_credentials",
            "target": "authentication_server",
            "priority": "high" if severity in [SeverityLabel.HIGH, SeverityLabel.CRITICAL] else "medium",
            "reason": "account_lockout_recovery"
        })

        return actions

    def _generate_vpn_actions(self, severity: str) -> List[Dict[str, Any]]:
        """Generate VPN issue resolution actions with connectivity restoration focus."""
        actions = []

        # VPN troubleshooting sequence for diagnostic workflow
        actions.extend([
            {
                "action": "check_vpn_server_status",
                "target": "vpn_infrastructure",
                "priority": "high",
                "reason": "vpn_connectivity_verification"
            },
            {
                "action": "review_auth_logs",
                "target": "authentication_server",
                "priority": "medium",
                "reason": "vpn_auth_failure_investigation"
            }
        ])

        # Add certificate validation for VPN authentication issues
        actions.append({
            "action": "verify_user_certificates",
            "target": "certificate_authority",
            "priority": "medium",
            "reason": "vpn_certificate_validation"
        })

        return actions

    def _generate_billing_actions(self, severity: str) -> List[Dict[str, Any]]:
        """Generate billing issue resolution actions with financial accuracy focus."""
        actions = []

        actions.extend([
            {
                "action": "review_invoice_details",
                "target": "billing_system",
                "priority": "medium",
                "reason": "billing_discrepancy_investigation"
            }
        ])

        return actions

    def _generate_backend_actions(self, severity: str) -> List[Dict[str, Any]]:
        """Generate backend error resolution actions with system restoration focus."""
        actions = []

        actions.extend([
            {
                "action": "check_application_logs",
                "target": "log_aggregation",
                "priority": "high",
                "reason": "backend_error_investigation"
            }
        ])

        return actions

    def _generate_fallback_actions(self, severity: str) -> List[Dict[str, Any]]:
        """Generate fallback actions for unknown or low-confidence intents."""
        return [
            {
                "action": "request_more_information",
                "target": "ticket_requester",
                "priority": "medium",
                "reason": "insufficient_intent_confidence"
            }
        ]

    def _generate_info_gathering_actions(self) -> List[Dict[str, Any]]:
        """Generate information gathering actions for ambiguous or incomplete tickets."""
        return [
            {
                "action": "request_additional_details",
                "target": "ticket_requester",
                "priority": "medium",
                "reason": "ambiguous_ticket_information"
            }
        ]

    def _deduplicate_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate actions based on action and target combination."""
        seen = set()
        unique_actions = []

        # Iterate through all actions to identify and remove duplicates
        # Prevents redundant actions in the final action plan
        for action in actions:
            # Create unique key from action type and target system
            key = (action["action"], action["target"])
            if key not in seen:
                seen.add(key)
                unique_actions.append(action)

        return unique_actions

    def _prioritize_actions(self, actions: List[Dict[str, Any]], severity: str) -> List[Dict[str, Any]]:
        """Prioritize actions based on severity and inherent priority with deterministic ordering."""
        # Define priority order for deterministic action sorting
        # Immediate actions always execute before high, then medium, then low
        priority_order = {"immediate": 0, "high": 1, "medium": 2, "low": 3}

        def action_key(action):
            # Primary sort by priority level (immediate > high > medium > low)
            priority_score = priority_order.get(action.get("priority", "medium"), 2)
            # Secondary sort by action name for deterministic ordering within same priority
            action_name = action.get("action", "")
            return (priority_score, action_name)

        # Sort actions by priority and name for deterministic output
        return sorted(actions, key=action_key)