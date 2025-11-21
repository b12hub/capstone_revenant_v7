#!/usr/bin/env python3
"""
Comprehensive test suite for Enterprise AI Agent Pipeline
Tests all components: orchestrator, sub-agents, fusion engine, and full integration
"""

import asyncio
import pytest
import json
import time
import tempfile
import os
import sys
from typing import Dict, Any, List
import random

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.enterprise.enterprise_agent import EnterpriseOrchestratorAgent, PipelineStage
from agents.enterprise.sub_agents import ClassifierAgent, KBSearchAgent, ActionAgent, IntentLabel, SeverityLabel
from agents.enterprise.fusion import FusionEngine


class TestFullEnterprisePipeline:
    """Comprehensive test suite for the entire enterprise pipeline ecosystem."""

    @pytest.fixture
    def sample_vpn_ticket(self) -> Dict[str, Any]:
        """Provide realistic VPN issue ticket for testing."""
        return {
            "ticket_id": "TCK-2024-VPN-001",
            "subject": "Critical: VPN Authentication Failure - AUTH_TIMEOUT preventing remote access",
            "body": "User cannot establish VPN connection from remote location since 08:30 AM EST. Consistent AUTH_TIMEOUT errors during authentication. User attempted: restarting VPN client (Cisco AnyConnect 4.10), rebooting workstation, switching networks. Last successful connection was yesterday at 17:45. This is blocking access to critical financial systems for quarterly reporting.",
            "customer_id": "CUST-104",
            "customer_tier": "enterprise",
            "priority": "high",
            "sla_level": "platinum"
        }

    @pytest.fixture
    def sample_security_ticket(self) -> Dict[str, Any]:
        """Provide realistic security incident ticket for testing."""
        return {
            "ticket_id": "TCK-2024-SEC-001",
            "subject": "SECURITY: Suspicious admin account activity from international IPs",
            "body": "Security monitoring flagged unusual login patterns for service admin account. Multiple authentication attempts from Germany and Singapore IP addresses. Account typically only accessed from corporate network. Potential credential compromise detected.",
            "customer_id": "CUST-078",
            "customer_tier": "premium",
            "priority": "critical",
            "sla_level": "gold"
        }

    @pytest.fixture
    def sample_billing_ticket(self) -> Dict[str, Any]:
        """Provide realistic billing issue ticket for testing."""
        return {
            "ticket_id": "TCK-2024-BILL-001",
            "subject": "Billing discrepancy - duplicate charges on enterprise plan",
            "body": "Monthly invoice shows duplicate line items for premium support. Charged $24,000 instead of $12,000. Accounting department requires resolution before month-end closing tomorrow.",
            "customer_id": "CUST-022",
            "customer_tier": "premium",
            "priority": "medium",
            "sla_level": "silver"
        }

    @pytest.fixture
    def minimal_valid_ticket(self) -> Dict[str, Any]:
        """Provide minimal valid ticket for edge case testing."""
        return {
            "ticket_id": "MIN-001",
            "subject": "help",
            "body": "problem",
            "customer_id": "CUST-999"
        }

    @pytest.fixture
    def invalid_ticket(self) -> Dict[str, Any]:
        """Provide invalid ticket for error handling testing."""
        return {
            "ticket_id": "",
            "subject": "",
            "body": "",
            "customer_id": ""
        }

    # =========================================================================
    # ENTERPRISE ORCHESTRATOR AGENT TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_orchestrator_full_pipeline_vpn_issue(self, sample_vpn_ticket):
        """Test complete pipeline execution with VPN issue ticket."""
        # Initialize orchestrator with deterministic seed
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Execute full pipeline
        start_time = time.time()
        result = await orchestrator.execute_pipeline(sample_vpn_ticket)
        end_time = time.time()

        # Validate response structure
        assert result is not None, "Pipeline should return result object"

        # Test required fields exist
        required_fields = ["ticket_id", "intent", "severity", "confidence", "actions", "kb_evidence", "timing_ms",
                           "trace_id"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Test field types and constraints
        assert result["ticket_id"] == "TCK-2024-VPN-001", "Ticket ID should be preserved"
        assert isinstance(result["intent"], str), "Intent should be string"
        assert result["intent"] in ["vpn_issue", "account_issue", "security_incident", "service_outage",
                                    "billing_issue", "unknown"], "Valid intent"
        assert result["severity"] in ["low", "medium", "high", "critical"], "Valid severity"
        assert isinstance(result["confidence"], float), "Confidence should be float"
        assert 0.0 <= result["confidence"] <= 1.0, "Confidence should be normalized"
        assert isinstance(result["actions"], list), "Actions should be list"
        assert len(result["actions"]) > 0, "Should generate at least one action"
        assert isinstance(result["kb_evidence"], list), "KB evidence should be list"
        assert isinstance(result["timing_ms"], int), "Timing should be integer"
        assert result["timing_ms"] > 0, "Should have positive processing time"
        assert result["trace_id"].startswith("trace_"), "Valid trace ID format"

        # Test performance characteristics
        actual_duration_ms = (end_time - start_time) * 1000
        assert abs(actual_duration_ms - result["timing_ms"]) < 100, "Timing should be accurate"
        assert result["timing_ms"] < 10000, "Pipeline should complete within 10 seconds"

    @pytest.mark.asyncio
    async def test_orchestrator_security_incident_handling(self, sample_security_ticket):
        """Test pipeline handling of security incidents with appropriate actions."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
        result = await orchestrator.execute_pipeline(sample_security_ticket)

        # Security incidents should generate containment actions
        security_actions = [action for action in result["actions"]
                            if action["action"] in ["isolate_affected_systems", "block_suspicious_ips",
                                                    "preserve_evidence"]]
        assert len(security_actions) > 0, "Security incidents should generate containment actions"

        # Should have reasonable confidence for security classification
        assert result["confidence"] > 0.3, "Security incidents should have reasonable confidence"

    @pytest.mark.asyncio
    async def test_orchestrator_billing_issue_handling(self, sample_billing_ticket):
        """Test pipeline handling of billing issues with financial actions."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
        result = await orchestrator.execute_pipeline(sample_billing_ticket)

        # Billing issues should generate financial investigation actions
        billing_actions = [action for action in result["actions"]
                           if "billing" in action["action"] or "invoice" in action["action"] or action[
                               "target"] == "billing_system"]
        assert len(billing_actions) > 0, "Billing issues should generate financial actions"

    @pytest.mark.asyncio
    async def test_orchestrator_minimal_ticket_handling(self, minimal_valid_ticket):
        """Test pipeline handling of minimal valid tickets with fallback behavior."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
        result = await orchestrator.execute_pipeline(minimal_valid_ticket)

        # Should process successfully with fallback values
        assert result["ticket_id"] == "MIN-001"
        assert result["intent"] != "unknown", "Should classify even minimal tickets"
        assert len(result["actions"]) > 0, "Should generate fallback actions"
        assert result["confidence"] > 0.1, "Should have minimal confidence"

    @pytest.mark.asyncio
    async def test_orchestrator_invalid_ticket_handling(self, invalid_ticket):
        """Test pipeline error handling with invalid ticket data."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
        result = await orchestrator.execute_pipeline(invalid_ticket)

        # Should handle gracefully with low confidence
        assert result["confidence"] < 0.5, "Invalid tickets should have low confidence"
        assert "ticket_id" in result, "Should maintain response structure even on error"

    @pytest.mark.asyncio
    async def test_orchestrator_deterministic_behavior(self, sample_vpn_ticket):
        """Test that pipeline produces identical results with same seed."""
        # First execution
        orchestrator1 = EnterpriseOrchestratorAgent(seed=12345, simulate_failure=False)
        result1 = await orchestrator1.execute_pipeline(sample_vpn_ticket)

        # Second execution with same seed
        orchestrator2 = EnterpriseOrchestratorAgent(seed=12345, simulate_failure=False)
        result2 = await orchestrator2.execute_pipeline(sample_vpn_ticket)

        # Should produce identical results
        assert result1["intent"] == result2["intent"], "Intent classification should be deterministic"
        assert result1["severity"] == result2["severity"], "Severity assessment should be deterministic"
        assert result1["confidence"] == result2["confidence"], "Confidence scoring should be deterministic"
        assert len(result1["actions"]) == len(result2["actions"]), "Action count should be deterministic"

        # Actions should be identical in type and order
        for i, (act1, act2) in enumerate(zip(result1["actions"], result2["actions"])):
            assert act1["action"] == act2["action"], f"Action {i} should be deterministic"
            assert act1["target"] == act2["target"], f"Action target {i} should be deterministic"

    @pytest.mark.asyncio
    async def test_orchestrator_failure_simulation(self, sample_vpn_ticket):
        """Test pipeline behavior with failure simulation enabled."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=True)
        result = await orchestrator.execute_pipeline(sample_vpn_ticket)

        # Should complete processing even with failure simulation
        assert "ticket_id" in result, "Should maintain response structure"
        assert "intent" in result, "Should still classify intent"
        assert "actions" in result, "Should still generate actions"

    @pytest.mark.asyncio
    async def test_orchestrator_batch_processing(self, sample_vpn_ticket, sample_security_ticket):
        """Test batch processing of multiple tickets."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Create batch of tickets
        batch_tickets = [
            sample_vpn_ticket,
            sample_security_ticket,
            {**sample_vpn_ticket, "ticket_id": "TCK-2024-VPN-002"},
            {**sample_security_ticket, "ticket_id": "TCK-2024-SEC-002"}
        ]

        results = await orchestrator.process_batch(batch_tickets)

        # Should return one result per ticket
        assert len(results) == len(batch_tickets)

        # All results should have valid structure
        for result in results:
            assert "ticket_id" in result
            assert "intent" in result
            assert "actions" in result
            assert "confidence" in result

        # Verify all tickets were processed
        processed_ids = {r["ticket_id"] for r in results}
        expected_ids = {t["ticket_id"] for t in batch_tickets}
        assert processed_ids == expected_ids, "All tickets should be processed"

    @pytest.mark.asyncio
    async def test_orchestrator_memory_persistence(self, sample_vpn_ticket):
        """Test conversation memory persists across multiple tickets."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Process first ticket
        result1 = await orchestrator.execute_pipeline(sample_vpn_ticket)

        # Process second ticket from same customer
        ticket2 = sample_vpn_ticket.copy()
        ticket2["ticket_id"] = "TCK-2024-VPN-MEM-001"
        ticket2["subject"] = "Follow-up VPN connectivity issue"

        result2 = await orchestrator.execute_pipeline(ticket2)

        # Memory should be maintained between executions
        customer_id = sample_vpn_ticket["customer_id"]
        assert customer_id in orchestrator.conversation_memory, "Customer memory should be created"

        memory = orchestrator.conversation_memory[customer_id]
        assert memory["ticket_count"] == 2, "Should track multiple tickets per customer"
        assert len(memory["recent_intents"]) == 2, "Should track recent intents"
        assert memory["last_interaction"] is not None, "Should track last interaction time"

    # =========================================================================
    # CLASSIFIER AGENT TESTS
    # =========================================================================

    def test_classifier_agent_initialization(self):
        """Test classifier agent initialization and configuration."""
        classifier = ClassifierAgent(seed=42)

        # Should have intent keywords configured
        assert hasattr(classifier, 'intent_keywords')
        assert len(classifier.intent_keywords) > 0

        # Should have severity indicators configured
        assert hasattr(classifier, 'severity_indicators')
        assert len(classifier.severity_indicators) > 0

        # Should use deterministic seed
        assert classifier.seed == 42

    @pytest.mark.asyncio
    async def test_classifier_agent_vpn_classification(self):
        """Test classifier agent with VPN-related text."""
        classifier = ClassifierAgent(seed=42)

        test_data = {
            "normalized_text": "vpn authentication timeout error cannot connect remote access",
            "urgency_score": 2,
            "technical_terms": ["vpn", "authentication", "timeout"],
            "word_count": 8
        }

        result = await classifier.classify_ticket(test_data)

        # Should classify as VPN issue
        assert result["intent"] == "vpn_issue"
        assert result["severity"] in ["medium", "high", "critical"]
        assert "intent_scores" in result
        assert "severity_scores" in result

        # VPN intent should have high score
        vpn_score = result["intent_scores"].get(IntentLabel.VPN_ISSUE, 0)
        assert vpn_score > 0, "VPN issue should have positive score"

    @pytest.mark.asyncio
    async def test_classifier_agent_security_classification(self):
        """Test classifier agent with security-related text."""
        classifier = ClassifierAgent(seed=42)

        test_data = {
            "normalized_text": "security breach unauthorized access suspicious login attempts germany singapore",
            "urgency_score": 3,
            "technical_terms": ["security", "breach", "unauthorized"],
            "word_count": 12
        }

        result = await classifier.classify_ticket(test_data)

        # Should classify as security incident
        assert result["intent"] == "security_incident"
        assert result["severity"] in ["high", "critical"]  # Security issues are high severity

        # Security intent should have high score
        security_score = result["intent_scores"].get(IntentLabel.SECURITY_INCIDENT, 0)
        assert security_score > 0, "Security incident should have positive score"

    def test_classifier_manual_softmax(self):
        """Test manual softmax implementation produces valid probability distribution."""
        classifier = ClassifierAgent(seed=42)

        # Test with simple scores
        test_scores = {"A": 1.0, "B": 2.0, "C": 3.0}
        probabilities = classifier._manual_softmax(test_scores)

        # Should return valid probability distribution
        assert len(probabilities) == 3
        assert all(0.0 <= p <= 1.0 for p in probabilities.values())

        # Probabilities should sum to 1.0
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 0.0001, f"Probabilities should sum to 1.0, got {total_prob}"

        # Higher scores should have higher probabilities
        assert probabilities["C"] > probabilities["B"] > probabilities["A"]

    def test_classifier_tie_breaking(self):
        """Test deterministic tie-breaking in classifier."""
        classifier = ClassifierAgent(seed=42)

        # Create tie scenario
        tied_probabilities = {
            IntentLabel.VPN_ISSUE: 0.5,
            IntentLabel.ACCOUNT_ISSUE: 0.5,
            IntentLabel.SECURITY_INCIDENT: 0.5
        }

        selected_intent, confidence = classifier._select_intent(tied_probabilities)

        # Should select one intent and adjust confidence for tie
        assert selected_intent in ["vpn_issue", "account_issue", "security_incident"]
        assert confidence == 0.5 / 3  # Confidence divided by number of tied intents

    # =========================================================================
    # KB SEARCH AGENT TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_kb_search_agent_initialization(self):
        """Test KB search agent initialization and data loading."""
        kb_agent = KBSearchAgent(seed=42)

        # Should initialize with None cache
        assert kb_agent.kb_cache is None
        assert kb_agent.historical_tickets is None

        # Should load data on first search
        test_data = {"normalized_text": "vpn timeout", "ticket_id": "TEST-001"}
        result = await kb_agent.search_ticket(test_data)

        # Should now have loaded data
        assert kb_agent.kb_cache is not None
        assert kb_agent.historical_tickets is not None

        # Should return structured results
        assert "matches" in result
        assert "scores" in result
        assert "selected" in result
        assert isinstance(result["matches"], list)
        assert isinstance(result["scores"], list)
        assert isinstance(result["selected"], list)

    @pytest.mark.asyncio
    async def test_kb_search_agent_vpn_search(self):
        """Test KB search agent with VPN-related queries."""
        kb_agent = KBSearchAgent(seed=42)

        test_data = {
            "normalized_text": "vpn authentication timeout cannot connect",
            "ticket_id": "TEST-VPN-001"
        }

        result = await kb_agent.search_ticket(test_data)

        # Should find VPN-related matches
        vpn_matches = [match for match in result["selected"]
                       if "vpn" in match.get("content", "").lower()
                       or "vpn" in match.get("title", "").lower()]

        assert len(vpn_matches) > 0, "Should find VPN-related KB articles"

        # Should have relevance scores
        for match in result["selected"]:
            assert "relevance_score" in match
            assert 0.0 <= match["relevance_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_kb_search_agent_empty_query(self):
        """Test KB search agent with empty or nonsensical query."""
        kb_agent = KBSearchAgent(seed=42)

        test_data = {
            "normalized_text": "asdf qwer zxcv",  # Nonsensical text
            "ticket_id": "TEST-EMPTY-001"
        }

        result = await kb_agent.search_ticket(test_data)

        # Should return empty or low-relevance results
        assert "matches" in result
        assert "selected" in result
        # May have some matches due to partial word matches, but relevance should be low

    # =========================================================================
    # ACTION AGENT TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_action_agent_security_actions(self):
        """Test action agent generates appropriate security actions."""
        action_agent = ActionAgent(seed=42)

        classification = {
            "intent": "security_incident",
            "severity": "critical"
        }

        ticket_data = {
            "normalized_text": "security breach unauthorized access"
        }

        result = await action_agent.generate_actions(ticket_data, classification)

        # Should generate security-specific actions
        assert "actions" in result
        assert "reasoning" in result
        assert "confidence" in result

        security_actions = [action for action in result["actions"]
                            if action["action"] in ["isolate_affected_systems", "block_suspicious_ips",
                                                    "preserve_evidence"]]
        assert len(security_actions) > 0, "Should generate security containment actions"

        # Should have reasonable confidence
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_action_agent_vpn_actions(self):
        """Test action agent generates appropriate VPN actions."""
        action_agent = ActionAgent(seed=42)

        classification = {
            "intent": "vpn_issue",
            "severity": "high"
        }

        ticket_data = {
            "normalized_text": "vpn authentication timeout"
        }

        result = await action_agent.generate_actions(ticket_data, classification)

        # Should generate VPN-specific actions
        vpn_actions = [action for action in result["actions"]
                       if
                       action["action"] in ["check_vpn_server_status", "review_auth_logs", "verify_user_certificates"]]
        assert len(vpn_actions) > 0, "Should generate VPN troubleshooting actions"

    @pytest.mark.asyncio
    async def test_action_agent_unknown_intent(self):
        """Test action agent fallback behavior for unknown intents."""
        action_agent = ActionAgent(seed=42)

        classification = {
            "intent": "unknown_intent",
            "severity": "medium"
        }

        ticket_data = {
            "normalized_text": "some random issue"
        }

        result = await action_agent.generate_actions(ticket_data, classification)

        # Should generate fallback actions
        assert len(result["actions"]) > 0, "Should generate fallback actions for unknown intents"
        fallback_actions = [action for action in result["actions"]
                            if action["action"] in ["request_more_information", "escalate_to_human_agent"]]
        assert len(fallback_actions) > 0, "Should include information gathering actions"

    def test_action_agent_deduplication(self):
        """Test action deduplication logic."""
        action_agent = ActionAgent(seed=42)

        # Create duplicate actions
        duplicate_actions = [
            {"action": "check_status", "target": "system_a", "priority": "high", "reason": "test"},
            {"action": "check_status", "target": "system_a", "priority": "high", "reason": "test"},  # Duplicate
            {"action": "restart_service", "target": "system_b", "priority": "medium", "reason": "test"}
        ]

        unique_actions = action_agent._deduplicate_actions(duplicate_actions)

        # Should remove duplicates
        assert len(unique_actions) == 2, "Should remove duplicate actions"

        # Should preserve unique actions
        action_types = {action["action"] for action in unique_actions}
        assert "check_status" in action_types
        assert "restart_service" in action_types

    def test_action_agent_prioritization(self):
        """Test action prioritization logic."""
        action_agent = ActionAgent(seed=42)

        # Create mixed priority actions
        mixed_actions = [
            {"action": "low_priority", "target": "system", "priority": "low", "reason": "test"},
            {"action": "high_priority", "target": "system", "priority": "high", "reason": "test"},
            {"action": "immediate_priority", "target": "system", "priority": "immediate", "reason": "test"},
            {"action": "medium_priority", "target": "system", "priority": "medium", "reason": "test"}
        ]

        prioritized_actions = action_agent._prioritize_actions(mixed_actions, "high")

        # Should sort by priority (immediate > high > medium > low)
        assert prioritized_actions[0]["priority"] == "immediate"
        assert prioritized_actions[1]["priority"] == "high"
        assert prioritized_actions[2]["priority"] == "medium"
        assert prioritized_actions[3]["priority"] == "low"

    # =========================================================================
    # FUSION ENGINE TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_fusion_engine_confidence_calculation(self):
        """Test fusion engine confidence calculation with weighted formula."""
        fusion_engine = FusionEngine(seed=42)

        # Mock agent outputs
        classifier_output = {
            "intent": "vpn_issue",
            "severity": "high"
        }

        kb_output = {
            "matches": [
                {"relevance_score": 0.9, "type": "kb_article", "content": "vpn troubleshooting guide"}
            ],
            "selected": [
                {"relevance_score": 0.9, "type": "kb_article", "content": "vpn troubleshooting guide"}
            ]
        }

        action_output = {
            "actions": [
                {"action": "check_vpn_status", "target": "vpn_infrastructure", "priority": "high"}
            ],
            "confidence": 0.8
        }

        ticket_data = {
            "normalized_text": "vpn timeout error",
            "urgency_score": 2,
            "technical_terms": ["vpn", "authentication"],
            "word_count": 25
        }

        result = await fusion_engine.fuse_results(
            classifier_output, kb_output, action_output, ticket_data
        )

        # Should produce fused confidence score
        assert "confidence" in result
        confidence = result["confidence"]
        assert 0.0 <= confidence <= 1.0, "Confidence should be normalized"

        # Should include all required response fields
        assert result["intent"] == "vpn_issue"
        assert result["severity"] == "high"
        assert "actions" in result
        assert "kb_evidence" in result

        # Should include fusion metadata
        assert "fusion_metadata" in result
        assert "strategy_used" in result["fusion_metadata"]

    def test_fusion_evidence_support_validation(self):
        """Test KB evidence support validation in fusion engine."""
        fusion_engine = FusionEngine(seed=42)

        # Test evidence that supports VPN intent
        vpn_evidence = {
            "content": "vpn authentication timeout troubleshooting guide",
            "title": "VPN Issues Resolution",
            "type": "kb_article"
        }

        supports_vpn = fusion_engine._evidence_supports_intent(vpn_evidence, "vpn_issue")
        assert supports_vpn, "Should recognize VPN-related evidence"

        # Test evidence that doesn't support billing intent
        supports_billing = fusion_engine._evidence_supports_intent(vpn_evidence, "billing_issue")
        assert not supports_billing, "Should not misclassify VPN evidence as billing-related"

    def test_fusion_action_appropriateness(self):
        """Test action appropriateness validation in fusion engine."""
        fusion_engine = FusionEngine(seed=42)

        # Test appropriate action for security incident
        security_action = {"action": "isolate_affected_systems", "target": "compromised_assets",
                           "priority": "immediate"}
        is_appropriate = fusion_engine._is_action_appropriate(security_action, "security_incident")
        assert is_appropriate, "Security actions should be appropriate for security incidents"

        # Test inappropriate action for billing issue
        inappropriate_action = {"action": "isolate_affected_systems", "target": "compromised_assets",
                                "priority": "immediate"}
        is_appropriate = fusion_engine._is_action_appropriate(inappropriate_action, "billing_issue")
        assert not is_appropriate, "Security actions should not be appropriate for billing issues"

    # =========================================================================
    # INTEGRATION AND PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_full_integration_performance(self, sample_vpn_ticket):
        """Test full pipeline integration and performance characteristics."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Measure performance across multiple runs
        execution_times = []
        confidences = []

        for i in range(3):  # Multiple runs for consistency
            ticket = sample_vpn_ticket.copy()
            ticket["ticket_id"] = f"PERF-TEST-{i}"

            start_time = time.time()
            result = await orchestrator.execute_pipeline(ticket)
            end_time = time.time()

            execution_times.append((end_time - start_time) * 1000)  # Convert to ms
            confidences.append(result["confidence"])

            # Validate each result
            assert result["ticket_id"] == f"PERF-TEST-{i}"
            assert len(result["actions"]) > 0
            assert 0.0 <= result["confidence"] <= 1.0

        # Performance should be consistent
        avg_execution_time = sum(execution_times) / len(execution_times)
        assert avg_execution_time < 5000, "Average execution should be under 5 seconds"

        # Confidence should be reasonably consistent
        confidence_variance = max(confidences) - min(confidences)
        assert confidence_variance < 0.3, "Confidence should be reasonably consistent across runs"

    @pytest.mark.asyncio
    async def test_error_resilience(self):
        """Test pipeline resilience to various error conditions."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Test with malformed data
        malformed_tickets = [
            {},  # Empty ticket
            {"ticket_id": "TEST"},  # Missing required fields
            {"subject": "test", "body": "test"},  # Missing ticket_id and customer_id
            None  # None input
        ]

        for ticket in malformed_tickets:
            try:
                result = await orchestrator.execute_pipeline(ticket)
                # Should handle gracefully and return structured response
                assert "ticket_id" in result
                assert "confidence" in result
                assert result["confidence"] < 0.5  # Low confidence for malformed data
            except Exception as e:
                pytest.fail(f"Pipeline should handle malformed ticket without crashing: {e}")

    @pytest.mark.asyncio
    async def test_deterministic_reproducibility(self, sample_vpn_ticket):
        """Test that same inputs with same seed produce identical outputs."""
        results = []

        # Run pipeline multiple times with same seed and inputs
        for i in range(3):
            orchestrator = EnterpriseOrchestratorAgent(seed=12345, simulate_failure=False)
            result = await orchestrator.execute_pipeline(sample_vpn_ticket)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i]["intent"] == results[0]["intent"], f"Run {i} intent differs"
            assert results[i]["severity"] == results[0]["severity"], f"Run {i} severity differs"
            assert results[i]["confidence"] == results[0]["confidence"], f"Run {i} confidence differs"
            assert len(results[i]["actions"]) == len(results[0]["actions"]), f"Run {i} action count differs"

            # Check action consistency
            for j, (action1, action2) in enumerate(zip(results[i]["actions"], results[0]["actions"])):
                assert action1["action"] == action2["action"], f"Run {i} action {j} differs"

    # =========================================================================
    # ENTERPRISE FEATURE TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_enterprise_memory_isolation(self, sample_vpn_ticket, sample_security_ticket):
        """Test that customer memory is properly isolated between different customers."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Process ticket for customer A
        result_a = await orchestrator.execute_pipeline(sample_vpn_ticket)

        # Process ticket for customer B
        result_b = await orchestrator.execute_pipeline(sample_security_ticket)

        # Memory should be isolated between customers
        customer_a_id = sample_vpn_ticket["customer_id"]
        customer_b_id = sample_security_ticket["customer_id"]

        assert customer_a_id in orchestrator.conversation_memory
        assert customer_b_id in orchestrator.conversation_memory

        memory_a = orchestrator.conversation_memory[customer_a_id]
        memory_b = orchestrator.conversation_memory[customer_b_id]

        assert memory_a["ticket_count"] == 1
        assert memory_b["ticket_count"] == 1
        assert memory_a["recent_intents"] != memory_b["recent_intents"]

    @pytest.mark.asyncio
    async def test_enterprise_traceability(self, sample_vpn_ticket):
        """Test that pipeline provides proper traceability for enterprise auditing."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
        result = await orchestrator.execute_pipeline(sample_vpn_ticket)

        # Should provide complete traceability information
        assert "trace_id" in result
        assert result["trace_id"].startswith("trace_")
        assert len(result["trace_id"]) > 10  # Reasonable length

        # Should provide timing information for performance monitoring
        assert "timing_ms" in result
        assert isinstance(result["timing_ms"], int)
        assert result["timing_ms"] > 0

        # Should provide structured actions with reasons
        for action in result["actions"]:
            assert "action" in action
            assert "target" in action
            assert "priority" in action
            assert "reason" in action

    @pytest.mark.asyncio
    async def test_enterprise_scalability(self):
        """Test that pipeline can handle multiple concurrent requests."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Create multiple test tickets
        test_tickets = []
        for i in range(5):
            ticket = {
                "ticket_id": f"SCALE-TEST-{i}",
                "subject": f"Test ticket {i}",
                "body": f"This is test ticket number {i} with some content",
                "customer_id": f"CUST-{i % 3}"  # 3 different customers
            }
            test_tickets.append(ticket)

        # Process batch concurrently
        results = await orchestrator.process_batch(test_tickets)

        # Should process all tickets
        assert len(results) == len(test_tickets)

        # All results should be valid
        for result in results:
            assert "ticket_id" in result
            assert "intent" in result
            assert "actions" in result
            assert "confidence" in result
            assert 0.0 <= result["confidence"] <= 1.0


def run_comprehensive_test_suite():
    """Run the comprehensive test suite and report results."""
    import subprocess
    import sys

    print("🚀 RUNNING COMPREHENSIVE ENTERPRISE PIPELINE TEST SUITE")
    print("=" * 60)

    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, "-m", "pytest",
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ], capture_output=True, text=True)

    # Print results
    print("STDOUT:")
    print(result.stdout)

    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print("=" * 60)
    print(f"EXIT CODE: {result.returncode}")

    if result.returncode == 0:
        print("🎉 ALL TESTS PASSED! Pipeline is production-ready.")
    else:
        print("❌ SOME TESTS FAILED. Check output above for details.")

    return result.returncode


if __name__ == "__main__":
    # Run the test suite when executed directly
    exit_code = run_comprehensive_test_suite()
    exit(exit_code)


#
# 1. Quick Test Run:
# bash
# python test_full_enterprise_pipeline.py
# 2. Detailed Test Run:
# bash
# python -m pytest test_full_enterprise_pipeline.py -v --tb=long
# 3. Run Specific Test Categories:
# bash
# # Test only orchestrator
# python -m pytest test_full_enterprise_pipeline.py::TestFullEnterprisePipeline::test_orchestrator_full_pipeline_vpn_issue -v
#
# # Test only classifier
# python -m pytest test_full_enterprise_pipeline.py::TestFullEnterprisePipeline::test_classifier_agent_vpn_classification -v
#
# # Test performance
# python -m pytest test_full_enterprise_pipeline.py::TestFullEnterprisePipeline::test_full_integration_performance -v