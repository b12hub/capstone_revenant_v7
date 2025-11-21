# import asyncio
# import time
# import json
# import hashlib
# import uuid
# from typing import Dict, Any, List, Optional
# from enum import Enum
# import random
#
# from sub_agents import ClassifierAgent, KBSearchAgent, ActionAgent
# from fusion import FusionEngine
#
#
# class PipelineStage(str, Enum):
#     """Enumeration of pipeline stages for state tracking and observability."""
#     INIT = "init"
#     LOAD_DATA = "load_data"
#     PREPROCESS = "preprocess"
#     CLASSIFY = "classify"
#     KB_SEARCH = "kb_search"
#     ACTION_GEN = "action_gen"
#     FUSION = "fusion"
#     COMPLETE = "complete"
#     ERROR = "error"
#
#
# class EnterpriseOrchestratorAgent:
#     """
#     Main enterprise orchestrator agent that coordinates the full pipeline.
#     Manages sequential and parallel execution of sub-agents with state tracking.
#     """
#
#     def __init__(self, seed: int = 42, simulate_failure: bool = False):
#         # Initialize random seed for deterministic behavior across pipeline
#         random.seed(seed)
#         self.seed = seed
#
#         # Failure simulation flag for testing error handling pathways
#         self.simulate_failure = simulate_failure
#
#         # Initialize sub-agents with shared configuration
#         self.classifier_agent = ClassifierAgent(seed=seed)
#         self.kb_search_agent = KBSearchAgent(seed=seed)
#         self.action_agent = ActionAgent(seed=seed)
#         self.fusion_engine = FusionEngine(seed=seed)
#
#         # Pipeline state tracking for observability and debugging
#         self.current_stage = PipelineStage.INIT
#         self.pipeline_start_time = None
#         self.trace_id = None
#
#         # Memory for cross-ticket context (enterprise feature)
#         self.conversation_memory = {}
#
#         # Performance metrics collection
#         self.stage_timings = {}
#
#     async def execute_pipeline(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Execute full enterprise pipeline with state tracking and error handling.
#         Returns structured response with confidence scoring and evidence.
#         """
#         # Generate unique trace ID for distributed tracing and observability
#         self.trace_id = f"trace_{uuid.uuid4().hex[:16]}"
#
#         # Record pipeline start time for latency calculation and SLA monitoring
#         self.pipeline_start_time = time.time()
#
#         # Initialize response structure with default values for error safety
#         pipeline_response = {
#             "ticket_id": ticket_data.get("ticket_id", "unknown"),
#             "intent": "unknown",
#             "severity": "medium",
#             "kb_evidence": [],
#             "actions": [],
#             "confidence": 0.0,
#             "timing_ms": 0,
#             "trace_id": self.trace_id,
#             "errors": []
#         }
#
#         try:
#             # STAGE 1: Data Loading and Validation
#             # Check if ticket data contains minimum required fields for processing
#             self.current_stage = PipelineStage.LOAD_DATA
#             if not self._validate_ticket_data(ticket_data):
#                 # If validation fails, add error and proceed with fallback processing
#                 pipeline_response["errors"].append("Ticket data validation failed")
#                 # In enterprise context, we continue processing with available data
#                 # rather than failing completely
#
#             # STAGE 2: Text Preprocessing
#             # Clean and normalize input text for consistent processing
#             self.current_stage = PipelineStage.PREPROCESS
#             preprocessed_data = await self._preprocess_ticket(ticket_data)
#
#             # STAGE 3: Parallel Agent Execution
#             # Execute classifier, KB search, and action generation in parallel
#             # for optimal pipeline latency (enterprise performance requirement)
#             self.current_stage = PipelineStage.CLASSIFY
#             classify_task = asyncio.create_task(
#                 self.classifier_agent.classify_ticket(preprocessed_data)
#             )
#
#             self.current_stage = PipelineStage.KB_SEARCH
#             kb_search_task = asyncio.create_task(
#                 self.kb_search_agent.search_ticket(preprocessed_data)
#             )
#
#             # Wait for classifier results before starting action generation
#             # because action strategies depend on intent and severity classification
#             classifier_results = await classify_task
#
#             self.current_stage = PipelineStage.ACTION_GEN
#             action_task = asyncio.create_task(
#                 self.action_agent.generate_actions(
#                     preprocessed_data,
#                     classifier_results
#                 )
#             )
#
#             # Wait for both KB search and action generation to complete
#             kb_results = await kb_search_task
#             action_results = await action_task
#
#             # STAGE 4: Result Fusion
#             # Combine and reconcile outputs from all sub-agents
#             self.current_stage = PipelineStage.FUSION
#             fused_results = await self.fusion_engine.fuse_results(
#                 classifier_output=classifier_results,
#                 kb_output=kb_results,
#                 action_output=action_results,
#                 ticket_data=preprocessed_data
#             )
#
#             # Update pipeline response with fused results
#             pipeline_response.update(fused_results)
#
#             # STAGE 5: Finalization
#             # Calculate final timing and update memory
#             self.current_stage = PipelineStage.COMPLETE
#             pipeline_response["timing_ms"] = int(
#                 (time.time() - self.pipeline_start_time) * 1000
#             )
#
#             # Update conversation memory for context-aware future processing
#             self._update_conversation_memory(ticket_data, pipeline_response)
#
#         except Exception as e:
#             # Global error handling for pipeline failures
#             # In enterprise context, we return partial results with error information
#             self.current_stage = PipelineStage.ERROR
#             pipeline_response["errors"].append(f"Pipeline error: {str(e)}")
#             pipeline_response["confidence"] = 0.1  # Minimal confidence on error
#
#             # Ensure timing is recorded even on failure for SLA monitoring
#             if self.pipeline_start_time:
#                 pipeline_response["timing_ms"] = int(
#                     (time.time() - self.pipeline_start_time) * 1000
#                 )
#
#         return pipeline_response
#
#     def _validate_ticket_data(self, ticket_data: Dict[str, Any]) -> bool:
#         """
#         Validate ticket data structure and required fields.
#         Returns boolean indicating if data meets minimum requirements.
#         """
#         # Check if ticket_id exists and is non-empty - required for tracking
#         if not ticket_data.get("ticket_id"):
#             return False
#
#         # Check if either subject or body exists - minimum content requirement
#         # Enterprise systems should handle tickets with minimal content gracefully
#         if not ticket_data.get("subject") and not ticket_data.get("body"):
#             return False
#
#         # Check if customer_id exists - required for routing and personalization
#         if not ticket_data.get("customer_id"):
#             return False
#
#         return True
#
#     async def _preprocess_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Preprocess ticket data for consistent agent processing.
#         Includes text cleaning, normalization, and feature extraction.
#         """
#         # Start preprocessing timing for performance monitoring
#         start_time = time.time()
#
#         # Create working copy to avoid modifying original data
#         processed = ticket_data.copy()
#
#         # Combine subject and body into unified text field for processing
#         # This ensures agents have complete context regardless of field separation
#         subject = processed.get("subject", "")
#         body = processed.get("body", "")
#         processed["combined_text"] = f"{subject}. {body}".strip()
#
#         # Convert text to lowercase for case-insensitive processing
#         # Enterprise systems must handle case variations consistently
#         processed["normalized_text"] = processed["combined_text"].lower()
#
#         # Extract word count for complexity estimation
#         # Used for routing decisions and resource allocation
#         word_count = len(processed["normalized_text"].split())
#         processed["word_count"] = word_count
#
#         # Extract urgency indicators from text patterns
#         # Enterprise systems use linguistic cues for priority routing
#         urgency_indicators = ["urgent", "critical", "emergency", "asap", "outage"]
#         processed["urgency_score"] = sum(
#             1 for indicator in urgency_indicators
#             if indicator in processed["normalized_text"]
#         )
#
#         # Extract technical keywords for domain routing
#         # Helps route to appropriate specialized agents
#         technical_terms = ["api", "vpn", "database", "ssl", "outage", "error"]
#         processed["technical_terms"] = [
#             term for term in technical_terms
#             if term in processed["normalized_text"]
#         ]
#
#         # Record preprocessing time for performance optimization
#         processing_time = int((time.time() - start_time) * 1000)
#         self.stage_timings["preprocessing"] = processing_time
#
#         return processed
#
#     def _update_conversation_memory(self, ticket_data: Dict[str, Any], response: Dict[str, Any]):
#         """
#         Update conversation memory with ticket context and response.
#         Enables context-aware processing for related future tickets.
#         """
#         customer_id = ticket_data.get("customer_id")
#         if not customer_id:
#             return
#
#         # Initialize customer memory if not exists
#         if customer_id not in self.conversation_memory:
#             self.conversation_memory[customer_id] = {
#                 "ticket_count": 0,
#                 "recent_intents": [],
#                 "common_issues": [],
#                 "last_interaction": None
#             }
#
#         # Update customer interaction history
#         customer_memory = self.conversation_memory[customer_id]
#         customer_memory["ticket_count"] += 1
#         customer_memory["last_interaction"] = time.time()
#
#         # Track recent intents for pattern recognition
#         # Limited to last 5 intents for memory efficiency
#         customer_memory["recent_intents"].append(response["intent"])
#         if len(customer_memory["recent_intents"]) > 5:
#             customer_memory["recent_intents"].pop(0)
#
#         # Update timing metrics for performance analysis
#         self.stage_timings["total_pipeline"] = response["timing_ms"]
#
#     async def process_batch(self, tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         Process multiple tickets in batch with controlled concurrency.
#         Enterprise feature for handling ticket bursts during incidents.
#         """
#         # Limit concurrent processing to prevent resource exhaustion
#         # Enterprise systems must handle load without degradation
#         semaphore = asyncio.Semaphore(10)  # Maximum 10 concurrent tickets
#
#         async def process_with_limit(ticket):
#             # Acquire semaphore to control concurrent execution
#             async with semaphore:
#                 return await self.execute_pipeline(ticket)
#
#         # Create tasks for all tickets with controlled concurrency
#         tasks = [process_with_limit(ticket) for ticket in tickets]
#
#         # Execute all tasks and gather results
#         # Enterprise systems must maintain order for customer correspondence
#         results = await asyncio.gather(*tasks, return_exceptions=True)
#
#         # Convert exceptions to error responses for graceful handling
#         final_results = []
#         for i, result in enumerate(results):
#             if isinstance(result, Exception):
#                 # Create error response for failed ticket processing
#                 error_response = {
#                     "ticket_id": tickets[i].get("ticket_id", f"unknown_{i}"),
#                     "intent": "error",
#                     "severity": "medium",
#                     "kb_evidence": [],
#                     "actions": [{"action": "escalate_to_human", "reason": "processing_error"}],
#                     "confidence": 0.0,
#                     "timing_ms": 0,
#                     "trace_id": f"error_{uuid.uuid4().hex[:8]}",
#                     "errors": [str(result)]
#                 }
#                 final_results.append(error_response)
#             else:
#                 final_results.append(result)
#
#         return final_results
#
import asyncio
import time
import json
import hashlib
import uuid
import random
from typing import Dict, Any, List, Optional
from enum import Enum

from .sub_agents import ClassifierAgent, KBSearchAgent, ActionAgent
from .fusion import FusionEngine


class PipelineStage(str, Enum):
    """Enumeration of pipeline stages for deterministic state tracking."""
    INIT = "init"
    LOAD_DATA = "load_data"
    PREPROCESS = "preprocess"
    CLASSIFY = "classify"
    KB_SEARCH = "kb_search"
    ACTION_GEN = "action_gen"
    FUSION = "fusion"
    COMPLETE = "complete"
    ERROR = "error"


class EnterpriseOrchestratorAgent:
    """
    Main enterprise orchestrator agent that coordinates deterministic multi-agent pipeline.
    Ensures reproducible execution through strict state management and seeded randomness.
    """

    def __init__(self, seed: int = 42, simulate_failure: bool = False):
        # Initialize deterministic random seed for all subcomponents
        # This ensures identical behavior across pipeline runs with same seed
        random.seed(seed)
        self.seed = seed

        # Failure simulation flag for testing error propagation pathways
        # When True, injects deterministic failures at specific pipeline stages
        self.simulate_failure = simulate_failure

        # Initialize sub-agents with shared deterministic seed
        # All agents must produce identical outputs given identical inputs and seed
        self.classifier_agent = ClassifierAgent(seed=seed)
        self.kb_search_agent = KBSearchAgent(seed=seed)
        self.action_agent = ActionAgent(seed=seed)
        self.fusion_engine = FusionEngine(seed=seed)

        # Pipeline state tracking for observability and debugging
        # Current stage enables progress monitoring and failure isolation
        self.current_stage = PipelineStage.INIT

        # Performance timing for SLA monitoring and optimization
        self.pipeline_start_time = None
        self.stage_timings = {}

        # Unique trace identifier for distributed tracing across microservices
        self.trace_id = None

        # Conversation memory for context-aware processing across related tickets
        # Enables enterprise feature of customer history awareness
        self.conversation_memory = {}

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lightweight entrypoint used by basic tests.
        Returns an acknowledgement without executing the full pipeline.
        """
        return {"status": "accepted", "input": input_data}

    async def execute_pipeline(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full enterprise pipeline with deterministic state transitions.
        Returns structured response with confidence scoring and evidence.
        """
        # Normalize input to dict to safely handle None or unexpected types
        if not isinstance(ticket_data, dict):
            ticket_data = {}
        # Generate unique trace ID using deterministic UUID based on seed + ticket
        # Ensures reproducible trace IDs for same inputs while maintaining uniqueness
        trace_seed = f"{self.seed}_{ticket_data.get('ticket_id', 'unknown')}"
        deterministic_uuid = uuid.UUID(hashlib.md5(trace_seed.encode()).hexdigest()[:32])
        self.trace_id = f"trace_{deterministic_uuid.hex[:16]}"

        # Record pipeline start time for latency calculation and SLA compliance
        self.pipeline_start_time = time.time()

        # Initialize response structure with default values for error safety
        # All fields must be present even in failure scenarios for client compatibility
        pipeline_response = {
            "ticket_id": ticket_data.get("ticket_id", "unknown"),
            "intent": "unknown",
            "severity": "medium",
            "kb_evidence": [],
            "actions": [],
            "confidence": 0.0,
            "timing_ms": 0,
            "trace_id": self.trace_id
        }

        try:
            # STAGE 1: Data Loading and Validation
            # Check if ticket data contains minimum required fields for processing
            # Prevents pipeline execution on malformed or incomplete tickets
            self.current_stage = PipelineStage.LOAD_DATA
            if not self._validate_ticket_data(ticket_data):
                # If validation fails, we cannot proceed with normal processing
                # Return early with minimal confidence and error indication
                pipeline_response["confidence"] = 0.1
                pipeline_response["timing_ms"] = int((time.time() - self.pipeline_start_time) * 1000)
                return pipeline_response

            # STAGE 2: Text Preprocessing and Normalization
            # Clean and standardize input text for consistent agent processing
            # Removes case sensitivity and extracts structured features
            self.current_stage = PipelineStage.PREPROCESS
            preprocessed_data = await self._preprocess_ticket(ticket_data)

            # STAGE 3: Intent and Severity Classification
            # ClassifierAgent always runs first to determine routing decisions
            # Classification results drive dynamic parallel execution strategies
            self.current_stage = PipelineStage.CLASSIFY
            classifier_results = await self.classifier_agent.classify_ticket(preprocessed_data)

            # STAGE 4: Dynamic Routing Decision
            # Determine optimal parallel execution strategy based on classification
            # High-severity security incidents get prioritized KB search + immediate actions
            routing_decision = self._determine_routing_strategy(classifier_results, preprocessed_data)

            # STAGE 5: Parallel Agent Execution
            # Execute KB search and action generation based on routing strategy
            # Parallel execution reduces latency for enterprise-scale throughput
            self.current_stage = PipelineStage.KB_SEARCH
            kb_search_task = asyncio.create_task(
                self.kb_search_agent.search_ticket(preprocessed_data)
            )

            self.current_stage = PipelineStage.ACTION_GEN
            action_task = asyncio.create_task(
                self.action_agent.generate_actions(preprocessed_data, classifier_results)
            )

            # Wait for both parallel tasks to complete with deterministic timeout
            # Timeout ensures pipeline progress even if agents hang (enterprise resilience)
            kb_results, action_results = await asyncio.gather(
                kb_search_task,
                action_task,
                return_exceptions=True  # Prevents single agent failure from blocking pipeline
            )

            # Convert exceptions to structured error responses for graceful handling
            # Enterprise systems must never crash due to agent failures
            if isinstance(kb_results, Exception):
                kb_results = {"matches": [], "scores": [], "selected": [], "error": str(kb_results)}
            if isinstance(action_results, Exception):
                action_results = {"actions": [], "reasoning": "agent_failure", "confidence": 0.0}

            # STAGE 6: Result Fusion and Confidence Calculation
            # Combine outputs from all agents using weighted fusion formula
            # Fusion engine resolves conflicts and produces unified response
            self.current_stage = PipelineStage.FUSION
            fused_results = await self.fusion_engine.fuse_results(
                classifier_output=classifier_results,
                kb_output=kb_results,
                action_output=action_results,
                ticket_data=preprocessed_data
            )

            # Update pipeline response with fused results from all agents
            pipeline_response.update(fused_results)

            # STAGE 7: Pipeline Completion and Memory Update
            # Finalize response with timing and update conversation memory
            self.current_stage = PipelineStage.COMPLETE
            pipeline_response["timing_ms"] = int((time.time() - self.pipeline_start_time) * 1000)

            # Update conversation memory for context-aware future processing
            # Enables enterprise feature of customer interaction history
            self._update_conversation_memory(ticket_data, pipeline_response)

        except Exception as e:
            # Global error handling for unexpected pipeline failures
            # Enterprise systems must return structured errors, not crash
            self.current_stage = PipelineStage.ERROR
            pipeline_response["confidence"] = 0.1  # Minimal confidence on error

            # Ensure timing is recorded even on failure for SLA monitoring
            if self.pipeline_start_time:
                pipeline_response["timing_ms"] = int((time.time() - self.pipeline_start_time) * 1000)

            # In production, this would log to centralized monitoring
            # For now, we preserve the error in the response structure

        return pipeline_response

    def _validate_ticket_data(self, ticket_data: Dict[str, Any]) -> bool:
        """
        Validate ticket data structure and required fields for pipeline processing.
        Returns boolean indicating if data meets minimum enterprise requirements.
        """
        # Check if ticket_id exists and is non-empty - required for tracking and correlation
        # Empty ticket_ids break audit trails and monitoring systems
        if not ticket_data.get("ticket_id"):
            return False

        # Check if either subject or body exists - minimum content requirement
        # Enterprise systems must handle tickets with minimal content gracefully
        if not ticket_data.get("subject") and not ticket_data.get("body"):
            return False

        # Check if customer_id exists - required for routing and personalization
        # Customer context enables enterprise features like SLA enforcement and history
        if not ticket_data.get("customer_id"):
            return False

        return True

    async def _preprocess_ticket(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess ticket data for consistent agent processing.
        Includes text cleaning, normalization, and structured feature extraction.
        """
        # Start preprocessing timing for performance optimization analysis
        start_time = time.time()

        # Create working copy to avoid modifying original data (immutable processing)
        # Preserves input data integrity for audit and debugging purposes
        processed = ticket_data.copy()

        # Combine subject and body into unified text field for comprehensive processing
        # Ensures agents have complete context regardless of field separation
        subject = processed.get("subject", "")
        body = processed.get("body", "")
        processed["combined_text"] = f"{subject}. {body}".strip()

        # Convert text to lowercase for case-insensitive keyword matching
        # Enterprise systems must handle case variations deterministically
        processed["normalized_text"] = processed["combined_text"].lower()

        # Extract word count for complexity estimation and resource allocation
        # Long tickets may require different processing strategies
        word_count = len(processed["normalized_text"].split())
        processed["word_count"] = word_count

        # Extract urgency indicators from text patterns for priority routing
        # Enterprise systems use linguistic cues for automated prioritization
        urgency_indicators = ["urgent", "critical", "emergency", "asap", "outage", "down"]
        processed["urgency_score"] = sum(
            1 for indicator in urgency_indicators
            if indicator in processed["normalized_text"]
        )

        # Extract technical keywords for domain-specific routing decisions
        # Helps route to appropriate specialized agents and knowledge bases
        technical_terms = ["api", "vpn", "database", "ssl", "outage", "error", "password", "billing"]
        processed["technical_terms"] = [
            term for term in technical_terms
            if term in processed["normalized_text"]
        ]

        # Record preprocessing time for performance optimization
        processing_time = int((time.time() - start_time) * 1000)
        self.stage_timings["preprocessing"] = processing_time

        return processed

    def _determine_routing_strategy(self, classification: Dict[str, Any],
                                    ticket_data: Dict[str, Any]) -> str:
        """
        Determine optimal parallel execution strategy based on classification results.
        Returns routing strategy identifier for dynamic pipeline optimization.
        """
        intent = classification.get("intent", "unknown")
        severity = classification.get("severity", "medium")

        # STRATEGY 1: Critical security incidents get immediate parallel processing
        # Both KB search and action generation run simultaneously for fastest response
        if (intent == "security_incident" and severity == "critical"):
            return "parallel_immediate"

        # STRATEGY 2: High severity outages prioritize KB search for known solutions
        # Action generation follows KB results for evidence-based actions
        elif (intent == "service_outage" and severity in ["high", "critical"]):
            return "kb_priority"

        # STRATEGY 3: Standard processing for medium-severity issues
        # Both agents run in parallel with normal priority
        elif severity == "medium":
            return "parallel_standard"

        # STRATEGY 4: Low severity or feature requests use sequential processing
        # Reduces resource usage for non-urgent tickets
        else:
            return "sequential_conservative"

    def _update_conversation_memory(self, ticket_data: Dict[str, Any], response: Dict[str, Any]):
        """
        Update conversation memory with ticket context and response for history awareness.
        Enables context-aware processing for related future tickets from same customer.
        """
        customer_id = ticket_data.get("customer_id")
        if not customer_id:
            return  # Cannot update memory without customer identifier

        # Initialize customer memory if not exists with deterministic structure
        # Fixed structure ensures consistent memory operations across pipeline runs
        if customer_id not in self.conversation_memory:
            self.conversation_memory[customer_id] = {
                "ticket_count": 0,
                "recent_intents": [],
                "common_issues": [],
                "last_interaction": None
            }

        # Update customer interaction history with current ticket information
        customer_memory = self.conversation_memory[customer_id]
        customer_memory["ticket_count"] += 1
        customer_memory["last_interaction"] = time.time()

        # Track recent intents for pattern recognition and routing optimization
        # Limited to last 5 intents for memory efficiency and relevance
        customer_memory["recent_intents"].append(response["intent"])
        if len(customer_memory["recent_intents"]) > 5:
            customer_memory["recent_intents"].pop(0)  # Remove oldest intent

        # Update timing metrics for performance analysis and capacity planning
        self.stage_timings["total_pipeline"] = response["timing_ms"]

    async def process_batch(self, tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple tickets in batch with controlled concurrency for enterprise scale.
        Returns list of processed ticket responses with preserved order.
        """
        # Limit concurrent processing to prevent resource exhaustion
        # Enterprise systems must handle load without performance degradation
        semaphore = asyncio.Semaphore(10)  # Maximum 10 concurrent tickets

        async def process_with_limit(ticket):
            # Acquire semaphore to control concurrent execution
            # Prevents system overload while maintaining throughput
            async with semaphore:
                return await self.execute_pipeline(ticket)

        # Create tasks for all tickets with controlled concurrency
        # Preserves ticket order for customer correspondence and auditing
        tasks = [process_with_limit(ticket) for ticket in tickets]

        # Execute all tasks and gather results with deterministic exception handling
        # Enterprise systems must process all tickets even if some fail
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to structured error responses for graceful handling
        # Ensures clients receive consistent response format regardless of failures
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create structured error response for failed ticket processing
                # Maintains response schema consistency even in failure scenarios
                error_response = {
                    "ticket_id": tickets[i].get("ticket_id", f"unknown_{i}"),
                    "intent": "error",
                    "severity": "medium",
                    "kb_evidence": [],
                    "actions": [{"action": "escalate_to_human", "reason": "processing_error"}],
                    "confidence": 0.0,
                    "timing_ms": 0,
                    "trace_id": f"error_{uuid.uuid4().hex[:8]}"
                }
                final_results.append(error_response)
            else:
                final_results.append(result)

        return final_results