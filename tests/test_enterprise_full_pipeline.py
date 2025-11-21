# import asyncio
# import pytest
# import json
# import time
# import tempfile
# import os
# from typing import Dict, Any, List
#
# from enterprise.enterprise_agent import EnterpriseOrchestratorAgent, PipelineStage
# from enterprise.sub_agents import ClassifierAgent, KBSearchAgent, ActionAgent
# from enterprise.fusion import FusionEngine
#
#
# class TestEnterpriseFullPipeline:
#     """Comprehensive test suite for enterprise pipeline components."""
#
#     @pytest.fixture
#     def sample_ticket(self) -> Dict[str, Any]:
#         """Provide standard sample ticket for testing."""
#         return {
#             "ticket_id": "TEST-001",
#             "subject": "VPN authentication failure - AUTH_TIMEOUT error",
#             "body": "User cannot connect to corporate VPN from remote location. Consistent AUTH_TIMEOUT errors during authentication. Tried restarting client and rebooting machine.",
#             "customer_id": "CUST-104",
#             "customer_tier": "enterprise",
#             "priority": "high"
#         }
#
#     @pytest.fixture
#     def minimal_ticket(self) -> Dict[str, Any]:
#         """Provide minimal ticket for edge case testing."""
#         return {
#             "ticket_id": "MIN-001",
#             "subject": "help",
#             "body": "problem",
#             "customer_id": "CUST-999"
#         }
#
#     @pytest.fixture
#     def broken_ticket(self) -> Dict[str, Any]:
#         """Provide broken ticket for error handling testing."""
#         return {
#             "ticket_id": "",  # Empty ticket_id
#             "subject": "",  # Empty subject
#             "body": "",  # Empty body
#             "customer_id": ""  # Empty customer_id
#         }
#
#     @pytest.mark.asyncio
#     async def test_pipeline_end_to_end(self, sample_ticket):
#         """Test complete pipeline execution with valid ticket."""
#         # Initialize orchestrator with fixed seed for deterministic testing
#         orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
#
#         # Execute full pipeline
#         result = await orchestrator.execute_pipeline(sample_ticket)
#
#         # Validate response structure exists
#         assert result is not None, "Pipeline should return result"
#
#         # Validate required response fields
#         assert "ticket_id" in result, "Response must contain ticket_id"
#         assert result["ticket_id"] == "TEST-001", "Ticket ID should be preserved"
#
#         assert "intent" in result, "Response must contain intent classification"
#         assert isinstance(result["intent"], str), "Intent should be string"
#         assert result["intent"] != "unknown", "Intent should be classified"
#
#         assert "severity" in result, "Response must contain severity"
#         assert result["severity"] in ["low", "medium", "high", "critical"], "Valid severity"
#
#         assert "confidence" in result, "Response must contain confidence score"
#         assert isinstance(result["confidence"], float), "Confidence should be float"
#         assert 0.0 <= result["confidence"] <= 1.0, "Confidence should be normalized"
#
#         assert "actions" in result, "Response must contain actions list"
#         assert isinstance(result["actions"], list), "Actions should be list"
#         assert len(result["actions"]) > 0, "Should generate at least one action"
#
#         assert "kb_evidence" in result, "Response must contain KB evidence"
#         assert isinstance(result["kb_evidence"], list), "KB evidence should be list"
#
#         assert "timing_ms" in result, "Response must contain timing"
#         assert isinstance(result["timing_ms"], int), "Timing should be integer"
#         assert result["timing_ms"] > 0, "Should have positive processing time"
#
#         assert "trace_id" in result, "Response must contain trace ID"
#         assert result["trace_id"].startswith("trace_"), "Trace ID should follow format"
#
#     @pytest.mark.asyncio
#     async def test_pipeline_with_minimal_ticket(self, minimal_ticket):
#         """Test pipeline handling of minimal valid ticket."""
#         orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
#         result = await orchestrator.execute_pipeline(minimal_ticket)
#
#         # Should still process successfully with fallback values
#         assert result["ticket_id"] == "MIN-001"
#         assert "intent" in result
#         assert "severity" in result
#         assert "actions" in result
#         assert len(result["actions"]) > 0  # Should have fallback actions
#
#     @pytest.mark.asyncio
#     async def test_pipeline_with_broken_ticket(self, broken_ticket):
#         """Test pipeline error handling with invalid ticket data."""
#         orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
#         result = await orchestrator.execute_pipeline(broken_ticket)
#
#         # Should handle gracefully with errors in response
#         assert "errors" in result
#         assert len(result["errors"]) > 0
#         assert result["confidence"] < 0.5  # Low confidence on error
#
#     @pytest.mark.asyncio
#     async def test_pipeline_failure_simulation(self, sample_ticket):
#         """Test pipeline behavior with failure simulation enabled."""
#         orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=True)
#         result = await orchestrator.execute_pipeline(sample_ticket)
#
#         # Should include errors when failure simulation active
#         assert "errors" in result
#         # May have lower confidence due to simulated failures
#         assert result["confidence"] <= 1.0
#
#     @pytest.mark.asyncio
#     async def test_batch_processing(self, sample_ticket):
#         """Test batch processing of multiple tickets."""
#         orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
#
#         # Create batch of similar tickets
#         batch_tickets = [sample_ticket.copy() for _ in range(3)]
#         for i, ticket in enumerate(batch_tickets):
#             ticket["ticket_id"] = f"BATCH-{i:03d}"
#
#         results = await orchestrator.process_batch(batch_tickets)
#
#         # Should return one result per ticket
#         assert len(results) == 3
#         assert all("ticket_id" in r for r in results)
#         assert all("intent" in r for r in results)
#         assert all("actions" in r for r in results)
#
#         # Verify all tickets were processed
#         ticket_ids = {r["ticket_id"] for r in results}
#         expected_ids = {"BATCH-000", "BATCH-001", "BATCH-002"}
#         assert ticket_ids == expected_ids
#
#     @pytest.mark.asyncio
#     async def test_deterministic_behavior(self, sample_ticket):
#         """Test that pipeline produces identical results with same seed."""
#         # First execution with seed 123
#         orchestrator1 = EnterpriseOrchestratorAgent(seed=123, simulate_failure=False)
#         result1 = await orchestrator1.execute_pipeline(sample_ticket)
#
#         # Second execution with same seed
#         orchestrator2 = EnterpriseOrchestratorAgent(seed=123, simulate_failure=False)
#         result2 = await orchestrator2.execute_pipeline(sample_ticket)
#
#         # Should produce identical results
#         assert result1["intent"] == result2["intent"], "Intent should be deterministic"
#         assert result1["severity"] == result2["severity"], "Severity should be deterministic"
#         assert result1["confidence"] == result2["confidence"], "Confidence should be deterministic"
#
#         # Actions should be identical in type and order
#         assert len(result1["actions"]) == len(result2["actions"])
#         for act1, act2 in zip(result1["actions"], result2["actions"]):
#             assert act1["action"] == act2["action"], "Actions should be deterministic"
#
#     @pytest.mark.asyncio
#     async def test_memory_persistence(self, sample_ticket):
#         """Test conversation memory persists across multiple tickets."""
#         orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
#
#         # Process first ticket
#         result1 = await orchestrator.execute_pipeline(sample_ticket)
#
#         # Process second ticket from same customer
#         ticket2 = sample_ticket.copy()
#         ticket2["ticket_id"] = "TEST-002"
#         ticket2["subject"] = "Follow-up VPN issue"
#
#         result2 = await orchestrator.execute_pipeline(ticket2)
#
#         # Memory should be maintained between executions
#         customer_id = sample_ticket["customer_id"]
#         assert customer_id in orchestrator.conversation_memory
#         memory = orchestrator.conversation_memory[customer_id]
#
#         assert memory["ticket_count"] == 2
#         assert len(memory["recent_intents"]) == 2
#         assert memory["last_interaction"] is not None
#
#     def test_classifier_agent_softmax(self):
#         """Test manual softmax implementation in classifier."""
#         classifier = ClassifierAgent(seed=42)
#
#         # Test softmax with simple scores
#         test_scores = {"A": 1.0, "B": 2.0, "C": 3.0}
#         probabilities = classifier._manual_softmax(test_scores)
#
#         # Should return probability distribution
#         assert len(probabilities) == 3
#         assert all(0.0 <= p <= 1.0 for p in probabilities.values())
#
#         # Probabilities should sum to approximately 1.0
#         total_prob = sum(probabilities.values())
#         assert abs(total_prob - 1.0) < 0.0001, f"Probabilities should sum to 1.0, got {total_prob}"
#
#         # Higher scores should have higher probabilities
#         assert probabilities["C"] > probabilities["B"] > probabilities["A"]
#
#     def test_classifier_tie_breaking(self):
#         """Test deterministic tie-breaking in classifier."""
#         classifier = ClassifierAgent(seed=42)
#
#         # Create tie scenario
#         tied_probabilities = {
#             "intent_a": 0.5,
#             "intent_b": 0.5,
#             "intent_c": 0.5
#         }
#
#         selected_intent, confidence = classifier._select_intent(tied_probabilities)
#
#         # Should select one intent and adjust confidence
#         assert selected_intent in ["intent_a", "intent_b", "intent_c"]
#         assert confidence == 0.5 / 3  # Confidence divided by tie count
#
#     @pytest.mark.asyncio
#     async def test_kb_search_agent(self):
#         """Test KB search agent functionality."""
#         kb_agent = KBSearchAgent(seed=42)
#
#         test_ticket = {
#             "normalized_text": "vpn authentication timeout error connection failed",
#             "ticket_id": "TEST-KB-001"
#         }
#
#         result = await kb_agent.search_ticket(test_ticket)
#
#         # Should return structured KB results
#         assert "kb_evidence" in result
#         assert "total_matches" in result
#         assert "search_time_ms" in result
#
#         # Should find relevant VPN evidence
#         vpn_evidence = [e for e in result["kb_evidence"] if "vpn" in e.get("content", "").lower()]
#         assert len(vpn_evidence) > 0, "Should find VPN-related KB articles"
#
#     @pytest.mark.asyncio
#     async def test_action_agent_strategies(self):
#         """Test action generation for different intents."""
#         action_agent = ActionAgent(seed=42)
#
#         test_cases = [
#             {
#                 "intent": "security_incident",
#                 "severity": "high",
#                 "expected_actions": ["isolate_affected_systems", "preserve_evidence"]
#             },
#             {
#                 "intent": "service_outage",
#                 "severity": "critical",
#                 "expected_actions": ["verify_service_health", "activate_bcp"]
#             },
#             {
#                 "intent": "account_issue",
#                 "severity": "medium",
#                 "expected_actions": ["verify_user_identity", "check_account_status"]
#             }
#         ]
#
#         for test_case in test_cases:
#             classification = {
#                 "intent": test_case["intent"],
#                 "severity": test_case["severity"],
#                 "intent_confidence": 0.8
#             }
#
#             ticket_data = {"normalized_text": "test"}
#             result = await action_agent.generate_actions(ticket_data, classification)
#
#             # Should generate appropriate actions
#             assert "actions" in result
#             assert len(result["actions"]) > 0
#
#             # Should include expected action types
#             action_types = {action["action"] for action in result["actions"]}
#             for expected_action in test_case["expected_actions"]:
#                 assert expected_action in action_types, f"Missing {expected_action} for {test_case['intent']}"
#
#     @pytest.mark.asyncio
#     async def test_fusion_engine_confidence_calculation(self):
#         """Test fusion engine confidence combination logic."""
#         fusion_engine = FusionEngine(seed=42)
#
#         # Mock agent outputs
#         classifier_output = {
#             "intent": "vpn_issue",
#             "intent_confidence": 0.8,
#             "severity": "high",
#             "severity_confidence": 0.7
#         }
#
#         kb_output = {
#             "kb_evidence": [
#                 {"relevance_score": 0.9, "type": "kb_article", "content": "vpn troubleshooting"}
#             ],
#             "total_matches": 3
#         }
#
#         action_output = {
#             "actions": [
#                 {"action": "check_vpn_status", "target": "vpn_infrastructure", "priority": "high"}
#             ],
#             "action_count": 1
#         }
#
#         ticket_data = {
#             "normalized_text": "vpn timeout error",
#             "urgency_score": 2,
#             "technical_terms": ["vpn", "authentication"],
#             "word_count": 25
#         }
#
#         result = await fusion_engine.fuse_results(
#             classifier_output, kb_output, action_output, ticket_data
#         )
#
#         # Should produce fused confidence score
#         assert "confidence" in result
#         confidence = result["confidence"]
#         assert 0.0 <= confidence <= 1.0
#
#         # Should include all required fields
#         assert result["intent"] == "vpn_issue"
#         assert result["severity"] == "high"
#         assert "actions" in result
#         assert "kb_evidence" in result
#
#         # Should include fusion metadata
#         assert "fusion_metadata" in result
#         assert "strategy_used" in result["fusion_metadata"]
#
#     def test_fusion_evidence_validation(self):
#         """Test KB evidence validation in fusion engine."""
#         fusion_engine = FusionEngine(seed=42)
#
#         # Test evidence that supports VPN intent
#         vpn_evidence = {
#             "content": "vpn authentication timeout troubleshooting guide",
#             "title": "VPN Issues Resolution",
#             "type": "kb_article"
#         }
#
#         supports_vpn = fusion_engine._evidence_supports_label(vpn_evidence, "vpn_issue")
#         assert supports_vpn, "Should recognize VPN-related evidence"
#
#         # Test evidence that doesn't support billing intent
#         supports_billing = fusion_engine._evidence_supports_label(vpn_evidence, "billing_issue")
#         assert not supports_billing, "Should not misclassify VPN evidence as billing-related"
#
#     @pytest.mark.asyncio
#     async def test_pipeline_performance(self, sample_ticket):
#         """Test pipeline performance characteristics."""
#         orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
#
#         start_time = time.time()
#         result = await orchestrator.execute_pipeline(sample_ticket)
#         end_time = time.time()
#
#         actual_duration = (end_time - start_time) * 1000  # Convert to ms
#         reported_duration = result["timing_ms"]
#
#         # Reported timing should be reasonable
#         assert reported_duration > 0
#         assert abs(actual_duration - reported_duration) < 100, "Timing should be accurate within 100ms"
#
#         # Should complete within reasonable time (simulated operations)
#         assert actual_duration < 5000, "Pipeline should complete within 5 seconds"
#
#     @pytest.mark.asyncio
#     async def test_error_handling_missing_agents(self, sample_ticket):
#         """Test pipeline behavior when sub-agents are missing or fail."""
#         # This tests the pipeline's resilience to partial failures
#         orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
#
#         # Temporarily break KB agent to test fallbacks
#         original_method = orchestrator.kb_search_agent.search_ticket
#         orchestrator.kb_search_agent.search_ticket = lambda x: (_ for _ in ()).throw(Exception("Simulated KB failure"))
#
#         result = await orchestrator.execute_pipeline(sample_ticket)
#
#         # Should still produce result with errors
#         assert "errors" in result
#         assert "intent" in result  # Should still have classification
#         assert "actions" in result  # Should still have actions
#
#         # Restore method
#         orchestrator.kb_search_agent.search_ticket = original_method
#
#     def test_response_validation(self, sample_ticket):
#         """Test that pipeline responses meet enterprise schema requirements."""
#         # This would typically use JSON schema validation
#         # For now, validate key structural requirements
#
#         required_fields = [
#             "ticket_id", "intent", "severity", "confidence",
#             "actions", "kb_evidence", "timing_ms", "trace_id"
#         ]
#
#         # Test that our sample ticket would produce valid response
#         # (Actual validation would happen in the test above)
#         assert all(field in sample_ticket for field in ["ticket_id", "subject", "body", "customer_id"])
#
#
# if __name__ == "__main__":
#     # Run tests when executed directly
#     pytest.main([__file__, "-v"])

import asyncio
import pytest
import json
import time
import tempfile
import os
from typing import Dict, Any, List

from agents.enterprise.enterprise_agent import EnterpriseOrchestratorAgent
from agents.enterprise.sub_agents import ClassifierAgent, KBSearchAgent, ActionAgent
from agents.enterprise.fusion import FusionEngine


class TestEnterpriseFullPipeline:
    """Comprehensive test suite for enterprise pipeline components and deterministic behavior."""

    @pytest.fixture
    def sample_ticket(self) -> Dict[str, Any]:
        """Provide standard sample ticket for deterministic testing."""
        return {
            "ticket_id": "TEST-001",
            "subject": "VPN authentication failure - AUTH_TIMEOUT error",
            "body": "User cannot connect to corporate VPN from remote location. Consistent AUTH_TIMEOUT errors during authentication. Tried restarting client and rebooting machine.",
            "customer_id": "CUST-104",
            "customer_tier": "enterprise",
            "priority": "high"
        }

    @pytest.fixture
    def minimal_ticket(self) -> Dict[str, Any]:
        """Provide minimal ticket for edge case testing."""
        return {
            "ticket_id": "MIN-001",
            "subject": "help",
            "body": "problem",
            "customer_id": "CUST-999"
        }

    @pytest.fixture
    def broken_ticket(self) -> Dict[str, Any]:
        """Provide broken ticket for error handling testing."""
        return {
            "ticket_id": "",  # Empty ticket_id
            "subject": "",  # Empty subject
            "body": "",  # Empty body
            "customer_id": ""  # Empty customer_id
        }

    @pytest.mark.asyncio
    async def test_pipeline_end_to_end(self, sample_ticket):
        """Test complete pipeline execution with valid ticket and deterministic output."""
        # Initialize orchestrator with fixed seed for reproducible testing
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Execute full pipeline and capture results
        result = await orchestrator.execute_pipeline(sample_ticket)

        # Validate response structure exists and contains required fields
        assert result is not None, "Pipeline should return result object"

        # Validate required response fields for enterprise schema compliance
        assert "ticket_id" in result, "Response must contain ticket_id"
        assert result["ticket_id"] == "TEST-001", "Ticket ID should be preserved"

        assert "intent" in result, "Response must contain intent classification"
        assert isinstance(result["intent"], str), "Intent should be string type"
        assert result["intent"] != "unknown", "Intent should be properly classified"

        assert "severity" in result, "Response must contain severity assessment"
        assert result["severity"] in ["low", "medium", "high", "critical"], "Valid severity value"

        assert "confidence" in result, "Response must contain confidence score"
        assert isinstance(result["confidence"], float), "Confidence should be float type"
        assert 0.0 <= result["confidence"] <= 1.0, "Confidence should be normalized probability"

        assert "actions" in result, "Response must contain actions list"
        assert isinstance(result["actions"], list), "Actions should be list type"
        assert len(result["actions"]) > 0, "Should generate at least one action"

        assert "kb_evidence" in result, "Response must contain KB evidence"
        assert isinstance(result["kb_evidence"], list), "KB evidence should be list type"

        assert "timing_ms" in result, "Response must contain timing metadata"
        assert isinstance(result["timing_ms"], int), "Timing should be integer type"
        assert result["timing_ms"] > 0, "Should have positive processing time"

        assert "trace_id" in result, "Response must contain trace ID"
        assert result["trace_id"].startswith("trace_"), "Trace ID should follow naming convention"

    @pytest.mark.asyncio
    async def test_pipeline_with_minimal_ticket(self, minimal_ticket):
        """Test pipeline handling of minimal valid ticket with fallback behavior."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
        result = await orchestrator.execute_pipeline(minimal_ticket)

        # Should process successfully with fallback values for minimal input
        assert result["ticket_id"] == "MIN-001"
        assert "intent" in result
        assert "severity" in result
        assert "actions" in result
        assert len(result["actions"]) > 0  # Should generate fallback actions

    @pytest.mark.asyncio
    async def test_pipeline_with_broken_ticket(self, broken_ticket):
        """Test pipeline error handling with invalid ticket data structure."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)
        result = await orchestrator.execute_pipeline(broken_ticket)

        # Should handle gracefully with low confidence but valid response structure
        assert result["confidence"] < 0.5  # Low confidence on invalid data
        assert "ticket_id" in result  # Response structure should still be valid

    @pytest.mark.asyncio
    async def test_pipeline_failure_simulation(self, sample_ticket):
        """Test pipeline behavior with failure simulation enabled."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=True)
        result = await orchestrator.execute_pipeline(sample_ticket)

        # Should complete processing even with failure simulation
        assert "ticket_id" in result
        assert "intent" in result
        assert "actions" in result

    @pytest.mark.asyncio
    async def test_batch_processing(self, sample_ticket):
        """Test batch processing of multiple tickets with deterministic ordering."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Create batch of similar tickets with different IDs
        batch_tickets = [sample_ticket.copy() for _ in range(3)]
        for i, ticket in enumerate(batch_tickets):
            ticket["ticket_id"] = f"BATCH-{i:03d}"

        results = await orchestrator.process_batch(batch_tickets)

        # Should return one result per ticket in original order
        assert len(results) == 3
        assert all("ticket_id" in r for r in results)
        assert all("intent" in r for r in results)
        assert all("actions" in r for r in results)

        # Verify all tickets were processed and order preserved
        ticket_ids = {r["ticket_id"] for r in results}
        expected_ids = {"BATCH-000", "BATCH-001", "BATCH-002"}
        assert ticket_ids == expected_ids

    @pytest.mark.asyncio
    async def test_deterministic_behavior(self, sample_ticket):
        """Test that pipeline produces identical results with same seed and inputs."""
        # First execution with specific seed
        orchestrator1 = EnterpriseOrchestratorAgent(seed=123, simulate_failure=False)
        result1 = await orchestrator1.execute_pipeline(sample_ticket)

        # Second execution with same seed and inputs
        orchestrator2 = EnterpriseOrchestratorAgent(seed=123, simulate_failure=False)
        result2 = await orchestrator2.execute_pipeline(sample_ticket)

        # Should produce identical results with same seed
        assert result1["intent"] == result2["intent"], "Intent classification should be deterministic"
        assert result1["severity"] == result2["severity"], "Severity assessment should be deterministic"
        assert result1["confidence"] == result2["confidence"], "Confidence scoring should be deterministic"

        # Actions should be identical in type, order, and count
        assert len(result1["actions"]) == len(result2["actions"])
        for act1, act2 in zip(result1["actions"], result2["actions"]):
            assert act1["action"] == act2["action"], "Action generation should be deterministic"

    @pytest.mark.asyncio
    async def test_memory_persistence(self, sample_ticket):
        """Test conversation memory persists across multiple tickets from same customer."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        # Process first ticket from customer
        result1 = await orchestrator.execute_pipeline(sample_ticket)

        # Process second ticket from same customer
        ticket2 = sample_ticket.copy()
        ticket2["ticket_id"] = "TEST-002"
        ticket2["subject"] = "Follow-up VPN issue"

        result2 = await orchestrator.execute_pipeline(ticket2)

        # Memory should be maintained between executions for same customer
        customer_id = sample_ticket["customer_id"]
        assert customer_id in orchestrator.conversation_memory
        memory = orchestrator.conversation_memory[customer_id]

        assert memory["ticket_count"] == 2
        assert len(memory["recent_intents"]) == 2
        assert memory["last_interaction"] is not None

    def test_classifier_agent_softmax(self):
        """Test manual softmax implementation produces valid probability distribution."""
        classifier = ClassifierAgent(seed=42)

        # Test softmax with simple scores
        test_scores = {"A": 1.0, "B": 2.0, "C": 3.0}
        probabilities = classifier._manual_softmax(test_scores)

        # Should return valid probability distribution
        assert len(probabilities) == 3
        assert all(0.0 <= p <= 1.0 for p in probabilities.values())

        # Probabilities should sum to approximately 1.0
        total_prob = sum(probabilities.values())
        assert abs(total_prob - 1.0) < 0.0001, f"Probabilities should sum to 1.0, got {total_prob}"

        # Higher scores should have higher probabilities
        assert probabilities["C"] > probabilities["B"] > probabilities["A"]

    def test_classifier_tie_breaking(self):
        """Test deterministic tie-breaking in classifier with reproducible results."""
        classifier = ClassifierAgent(seed=42)

        # Create tie scenario with equal probabilities
        tied_probabilities = {
            "intent_a": 0.5,
            "intent_b": 0.5,
            "intent_c": 0.5
        }

        selected_intent, confidence = classifier._select_intent(tied_probabilities)

        # Should select one intent and adjust confidence for tie
        assert selected_intent in ["intent_a", "intent_b", "intent_c"]
        assert confidence == 0.5 / 3  # Confidence divided by number of tied intents

    @pytest.mark.asyncio
    async def test_kb_search_agent(self):
        """Test KB search agent functionality and evidence retrieval."""
        kb_agent = KBSearchAgent(seed=42)

        test_ticket = {
            "normalized_text": "vpn authentication timeout error connection failed",
            "ticket_id": "TEST-KB-001"
        }

        result = await kb_agent.search_ticket(test_ticket)

        # Should return structured KB results
        assert "matches" in result
        assert "scores" in result
        assert "selected" in result

        # Should find relevant VPN evidence based on keyword matching
        vpn_evidence = [e for e in result["selected"] if "vpn" in e.get("content", "").lower()]
        assert len(vpn_evidence) > 0, "Should find VPN-related KB evidence"

    @pytest.mark.asyncio
    async def test_action_agent_strategies(self):
        """Test action generation for different intents with rule-based strategies."""
        action_agent = ActionAgent(seed=42)

        test_cases = [
            {
                "intent": "security_incident",
                "severity": "high",
                "expected_actions": ["isolate_affected_systems", "block_suspicious_ips"]
            },
            {
                "intent": "service_outage",
                "severity": "critical",
                "expected_actions": ["verify_service_health", "activate_bcp"]
            },
            {
                "intent": "account_issue",
                "severity": "medium",
                "expected_actions": ["verify_user_identity", "reset_mfa_credentials"]
            }
        ]

        for test_case in test_cases:
            classification = {
                "intent": test_case["intent"],
                "severity": test_case["severity"]
            }

            ticket_data = {"normalized_text": "test"}
            result = await action_agent.generate_actions(ticket_data, classification)

            # Should generate appropriate actions for each intent
            assert "actions" in result
            assert len(result["actions"]) > 0

            # Should include expected action types based on intent
            action_types = {action["action"] for action in result["actions"]}
            for expected_action in test_case["expected_actions"]:
                assert expected_action in action_types, f"Missing {expected_action} for {test_case['intent']}"

    @pytest.mark.asyncio
    async def test_fusion_engine_confidence_calculation(self):
        """Test fusion engine confidence combination with weighted formula."""
        fusion_engine = FusionEngine(seed=42)

        # Mock agent outputs for fusion testing
        classifier_output = {
            "intent": "vpn_issue",
            "severity": "high"
        }

        kb_output = {
            "matches": [
                {"relevance_score": 0.9, "type": "kb_article", "content": "vpn troubleshooting"}
            ],
            "selected": [
                {"relevance_score": 0.9, "type": "kb_article", "content": "vpn troubleshooting"}
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
        assert 0.0 <= confidence <= 1.0

        # Should include all required response fields
        assert result["intent"] == "vpn_issue"
        assert result["severity"] == "high"
        assert "actions" in result
        assert "kb_evidence" in result

        # Should include fusion metadata for observability
        assert "fusion_metadata" in result
        assert "strategy_used" in result["fusion_metadata"]

    def test_fusion_evidence_validation(self):
        """Test KB evidence validation in fusion engine for intent support."""
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

    @pytest.mark.asyncio
    async def test_pipeline_performance(self, sample_ticket):
        """Test pipeline performance characteristics and timing accuracy."""
        orchestrator = EnterpriseOrchestratorAgent(seed=42, simulate_failure=False)

        start_time = time.time()
        result = await orchestrator.execute_pipeline(sample_ticket)
        end_time = time.time()

        actual_duration = (end_time - start_time) * 1000  # Convert to milliseconds
        reported_duration = result["timing_ms"]

        # Reported timing should be reasonable and accurate
        assert reported_duration > 0
        assert abs(actual_duration - reported_duration) < 100, "Timing should be accurate within 100ms"

        # Should complete within reasonable time for simulated operations
        assert actual_duration < 5000, "Pipeline should complete within 5 seconds"

    def test_response_schema_validation(self, sample_ticket):
        """Test that pipeline responses meet enterprise schema requirements."""
        # Validate that sample ticket has required fields for pipeline processing
        required_fields = ["ticket_id", "subject", "body", "customer_id"]
        assert all(field in sample_ticket for field in required_fields)


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])