import asyncio
import time
from typing import Dict, Any, List, Optional
from enum import Enum


class ToolStatus(str, Enum):
    """Enumeration of possible tool execution statuses."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ToolErrorCode(str, Enum):
    """Enumeration of standardized error codes for tool failures."""
    INVALID_QUERY = "invalid_query"
    NETWORK_TIMEOUT = "network_timeout"
    UNKNOWN_ACTION = "unknown_action"
    EXECUTION_FAILED = "execution_failed"
    RATE_LIMITED = "rate_limited"
    EMPTY_RESULT = "empty_result"


class FailMode(str, Enum):
    """Enumeration of possible failure modes for testing and simulation."""
    NONE = "none"
    TIMEOUT = "timeout"
    EMPTY_RESULT = "empty_result"
    INVALID_INPUT = "invalid_input"


class ToolError(Exception):
    """Custom exception class for tool-specific errors with structured details."""

    def __init__(self, message: str, error_code: ToolErrorCode, details: Dict[str, Any] = None):
        # Initialize with human-readable message and machine-readable error code
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class BaseTool:
    """Base class for all tools with standardized response format and error handling."""

    def __init__(self, tool_name: str, timeout_seconds: float = 5.0):
        # Store tool identifier for metadata and debugging
        self.tool_name = tool_name
        # Set maximum execution time before timeout error
        self.timeout_seconds = timeout_seconds

    async def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool with standardized response format including timing and metadata."""
        # Record start time for latency calculation
        start_time = time.time()

        try:
            # Extract fail_mode from payload to determine if we should simulate failures
            # Default to NONE if not specified
            fail_mode = payload.get("fail_mode", FailMode.NONE)

            # Simulate realistic network latency variation based on tool name hash
            # This creates deterministic but varied latency for different tools
            simulated_latency = 0.1 + (hash(self.tool_name) % 10) * 0.02
            await asyncio.sleep(simulated_latency)

            # Call the tool-specific implementation with the fail_mode parameter
            result = await self._execute_impl(payload, fail_mode)

            # Calculate actual execution time in milliseconds for performance monitoring
            latency_ms = int((time.time() - start_time) * 1000)

            # Construct successful response with standardized structure
            return {
                "status": ToolStatus.SUCCESS,
                "result": result,
                "metadata": {
                    "tool_name": self.tool_name,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "fail_mode": fail_mode,
                    "error_code": None
                }
            }

        except ToolError as e:
            # Handle expected tool errors with structured error information
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "status": ToolStatus.ERROR,
                "result": None,
                "metadata": {
                    "tool_name": self.tool_name,
                    "latency_ms": latency_ms,
                    "timestamp": time.time(),
                    "fail_mode": payload.get("fail_mode", FailMode.NONE),
                    "error_code": e.error_code,
                    "error_details": e.details
                },
                "error_message": e.message
            }

    async def _execute_impl(self, payload: Dict[str, Any], fail_mode: FailMode) -> Any:
        """Tool-specific implementation to be overridden by subclasses."""
        raise NotImplementedError


class KBSearchTool(BaseTool):
    """
    Deterministic knowledge-base search tool with configurable failure modes.
    Simulates enterprise knowledge base search with realistic response patterns.
    """

    def __init__(self):
        # Initialize with tool name and reasonable timeout for search operations
        super().__init__("kb_search", timeout_seconds=3.0)
        # Pre-load knowledge base data for deterministic search results
        self._kb_data = self._initialize_knowledge_base()

    def _initialize_knowledge_base(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize deterministic KB data with structured articles and metadata."""
        # Create categorized knowledge base with realistic article structure
        # Each category contains articles with relevance scores for ranking
        return {
            "password": [
                {
                    "id": "KB-001",
                    "title": "Password Reset Procedure - Step by Step Guide",
                    "content": "Complete guide for resetting user passwords via admin portal including security verification steps",
                    "relevance_score": 0.95,
                    "category": "authentication",
                    "last_updated": "2025-10-15"
                },
                {
                    "id": "KB-045",
                    "title": "Account Lockout Troubleshooting and Resolution",
                    "content": "Comprehensive guide for resolving common account lockout scenarios and prevention strategies",
                    "relevance_score": 0.87,
                    "category": "authentication",
                    "last_updated": "2025-09-22"
                }
            ],
            "vpn": [
                {
                    "id": "KB-013",
                    "title": "VPN Connectivity Troubleshooting Guide",
                    "content": "Comprehensive guide for diagnosing and resolving VPN connection issues including AUTH_TIMEOUT errors",
                    "relevance_score": 0.92,
                    "category": "networking",
                    "last_updated": "2025-11-10"
                },
                {
                    "id": "KB-027",
                    "title": "AUTH_TIMEOUT Error Resolution Steps",
                    "content": "Specific troubleshooting steps to resolve VPN authentication timeout errors and common root causes",
                    "relevance_score": 0.88,
                    "category": "networking",
                    "last_updated": "2025-10-30"
                }
            ],
            "billing": [
                {
                    "id": "KB-102",
                    "title": "Billing Discrepancy Investigation Process",
                    "content": "Standard operating procedure for identifying and resolving billing inconsistencies and duplicate charges",
                    "relevance_score": 0.90,
                    "category": "billing",
                    "last_updated": "2025-09-05"
                }
            ],
            "api": [
                {
                    "id": "KB-404",
                    "title": "API 5xx Error Troubleshooting Checklist",
                    "content": "Detailed debugging guide for internal server errors in API endpoints including gateway timeouts",
                    "relevance_score": 0.85,
                    "category": "development",
                    "last_updated": "2025-11-18"
                }
            ],
            "outage": [
                {
                    "id": "KB-210",
                    "title": "Service Outage Response Protocol and Communication",
                    "content": "Standard operating procedure for service disruption events including escalation paths and customer communication",
                    "relevance_score": 0.96,
                    "category": "incident_management",
                    "last_updated": "2025-08-14"
                }
            ]
        }

    async def _execute_impl(self, payload: Dict[str, Any], fail_mode: FailMode) -> Dict[str, Any]:
        """Execute KB search with deterministic results and configurable failure simulation."""
        # Extract search query from payload, default to empty string if not provided
        query = payload.get("query", "").lower().strip()

        # Handle configured failure modes before normal processing
        if fail_mode == FailMode.TIMEOUT:
            # Simulate network timeout by sleeping longer than tool timeout
            await asyncio.sleep(4.0)  # 4 seconds exceeds 3-second timeout
            raise ToolError(
                "KB search timeout - service unavailable",
                ToolErrorCode.NETWORK_TIMEOUT,
                {"query": query, "timeout_seconds": self.timeout_seconds}
            )
        elif fail_mode == FailMode.EMPTY_RESULT:
            # Return empty results regardless of query to simulate no matches found
            return {
                "hits": [],
                "total_matches": 0,
                "query": query,
                "search_time_ms": 50
            }
        elif fail_mode == FailMode.INVALID_INPUT:
            # Force invalid input error by modifying query to trigger validation failure
            query = ""  # Empty query will trigger validation error

        # Input validation - check if query string is empty before processing
        # This prevents unnecessary computation and avoids matching logic on empty input
        if not query:
            raise ToolError(
                "Empty search query provided",
                ToolErrorCode.INVALID_QUERY,
                {"required_field": "query", "min_length": 1}
            )

        # Input validation - check if query meets minimum length requirement
        # Short queries often return too many irrelevant results in real search systems
        if len(query) < 2:
            raise ToolError(
                "Search query too short (minimum 2 characters)",
                ToolErrorCode.INVALID_QUERY,
                {"min_length": 2, "provided_length": len(query), "query": query}
            )

        # Simulate rate limiting for specific testing patterns
        # This helps test how the system handles rate limit responses
        if "test" in query and "spam" in query:
            raise ToolError(
                "Rate limit exceeded for this query pattern",
                ToolErrorCode.RATE_LIMITED,
                {"retry_after_seconds": 60, "query": query}
            )

        # Simulate network timeout for excessively long queries
        # Long queries might indicate malicious or poorly formed requests in production
        if len(query) > 100:
            await asyncio.sleep(4.0)  # Exceeds timeout threshold
            raise ToolError(
                "KB search timeout - query too complex",
                ToolErrorCode.NETWORK_TIMEOUT,
                {"max_query_length": 100, "actual_length": len(query)}
            )

        # Initialize empty results list to collect matching articles
        hits = []

        # Primary search logic: iterate through each category in knowledge base
        # For each category, check if category name appears in the query
        for category, articles in self._kb_data.items():
            # Check if current category is mentioned in the search query
            # This provides category-based matching as primary search mechanism
            if category in query:
                # Filter articles within matching category by relevance threshold
                # Only return articles with high enough relevance score for quality control
                relevant_articles = [
                    article for article in articles
                    if article["relevance_score"] > 0.8  # Relevance threshold filter
                ]
                # Add filtered articles to results collection
                hits.extend(relevant_articles)

        # Secondary search logic: if no category matches found, search article content
        # This provides fallback matching when category names don't match query
        if not hits:
            # Iterate through all categories and articles for content-based matching
            for category, articles in self._kb_data.items():
                # Check each article's title and content for query terms
                for article in articles:
                    # Search in both title and content fields for broader matching
                    if (query in article["content"].lower() or
                            query in article["title"].lower()):
                        # Add article to results if query found in title or content
                        hits.append(article)

        # Sort final results by relevance score in descending order
        # This ensures most relevant articles appear first in response
        hits.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Calculate deterministic search time based on query characteristics
        # Longer queries and more results simulate increased processing time
        search_time_ms = 45 + len(query) + (len(hits) * 2)

        return {
            "hits": hits[:5],  # Limit to top 5 results for performance and usability
            "total_matches": len(hits),
            "query": query,
            "search_time_ms": search_time_ms
        }


class ExecuteActionTool(BaseTool):
    """
    Safe remediation execution tool with deterministic outcomes and failure simulation.
    Simulates enterprise actions without performing real system changes.
    """

    def __init__(self):
        # Initialize with tool name and longer timeout for action execution
        super().__init__("execute_action", timeout_seconds=10.0)
        # Pre-define supported actions with metadata for deterministic behavior
        self._supported_actions = self._initialize_supported_actions()

    def _initialize_supported_actions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize deterministic action definitions with risk and approval metadata."""
        # Define all supported actions with complete metadata for consistent responses
        return {
            "reset_password": {
                "description": "Generate password reset token and send notification",
                "risk_level": "low",
                "approval_required": False,
                "simulated_result": "Password reset token generated and sent to user registered email",
                "confidence": 0.95,
                "execution_time": 0.5
            },
            "restart_service": {
                "description": "Graceful service restart with health checks",
                "risk_level": "medium",
                "approval_required": True,
                "simulated_result": "Service restart initiated - estimated downtime: 30 seconds",
                "confidence": 0.90,
                "execution_time": 1.5
            },
            "clear_cache": {
                "description": "Clear application and CDN cache layers",
                "risk_level": "low",
                "approval_required": False,
                "simulated_result": "Cache cleared successfully - 2.4GB memory freed across 3 cache layers",
                "confidence": 0.98,
                "execution_time": 0.8
            },
            "escalate_incident": {
                "description": "Escalate to incident response team with context",
                "risk_level": "high",
                "approval_required": True,
                "simulated_result": "Incident escalated to SRE team - ticket #INC-789 created with high priority",
                "confidence": 0.99,
                "execution_time": 0.3
            },
            "block_ip": {
                "description": "Block suspicious IP address in firewall",
                "risk_level": "medium",
                "approval_required": True,
                "simulated_result": "IP address 192.168.92.45 added to firewall blocklist for 24 hours",
                "confidence": 0.85,
                "execution_time": 1.2
            }
        }

    async def _execute_impl(self, payload: Dict[str, Any], fail_mode: FailMode) -> Dict[str, Any]:
        """Execute remediation action with deterministic simulation and failure modes."""
        # Extract action and target from payload with default values
        action = payload.get("action")
        target = payload.get("target", "unknown")

        # Handle configured failure modes before normal processing
        if fail_mode == FailMode.TIMEOUT:
            # Simulate action timeout by exceeding tool timeout threshold
            await asyncio.sleep(12.0)  # 12 seconds exceeds 10-second timeout
            raise ToolError(
                "Action execution timeout - service unresponsive",
                ToolErrorCode.NETWORK_TIMEOUT,
                {"action": action, "target": target}
            )
        elif fail_mode == FailMode.INVALID_INPUT:
            # Force invalid input by clearing required action parameter
            action = None

        # Input validation - check if action parameter is provided and not empty
        # Action is required parameter for all remediation executions
        if not action:
            raise ToolError(
                "No action specified in payload",
                ToolErrorCode.INVALID_QUERY,
                {"required_field": "action", "provided_payload": payload}
            )

        # Action validation - verify requested action is in supported actions list
        # Prevents execution of unknown or unsupported remediation actions
        if action not in self._supported_actions:
            raise ToolError(
                f"Unsupported action requested: {action}",
                ToolErrorCode.UNKNOWN_ACTION,
                {
                    "supported_actions": list(self._supported_actions.keys()),
                    "provided_action": action,
                    "target": target
                }
            )

        # Retrieve action configuration for the requested action
        action_config = self._supported_actions[action]

        # Simulate execution failure for specific target patterns (deterministic testing)
        # This allows testing error handling without modifying normal execution logic
        if "fail" in target.lower():
            raise ToolError(
                f"Action execution failed for target: {target}",
                ToolErrorCode.EXECUTION_FAILED,
                {
                    "action": action,
                    "target": target,
                    "failure_reason": "Simulated failure for testing error handling"
                }
            )

        # Handle empty result failure mode by returning minimal success response
        if fail_mode == FailMode.EMPTY_RESULT:
            return {
                "action": action,
                "target": target,
                "result": "Action completed (empty result simulation)",
                "confidence": 0.0,
                "risk_level": "unknown",
                "approval_required": False,
                "execution_time_seconds": 0.1,
                "action_id": "ACT-0000"
            }

        # Simulate realistic action execution time based on configured duration
        # Different actions have different execution times in real systems
        execution_time = action_config["execution_time"]
        await asyncio.sleep(execution_time)

        # Generate deterministic action ID based on action and target hash
        # Provides consistent tracking ID for the same action-target combinations
        action_id = f"ACT-{abs(hash(action + target)) % 10000:04d}"

        return {
            "action": action,
            "target": target,
            "result": action_config["simulated_result"],
            "confidence": action_config["confidence"],
            "risk_level": action_config["risk_level"],
            "approval_required": action_config["approval_required"],
            "execution_time_seconds": execution_time,
            "action_id": action_id
        }


class ToolRegistry:
    """Central registry for managing and accessing all available tools."""

    def __init__(self):
        # Initialize empty tools dictionary
        self._tools = {}
        # Pre-register default tools for immediate availability
        self._initialize_default_tools()

    def _initialize_default_tools(self):
        """Initialize tool registry with default tool instances."""
        # Register knowledge base search tool with standard configuration
        self.register("kb_search", KBSearchTool())
        # Register action execution tool with standard configuration
        self.register("execute_action", ExecuteActionTool())

    def register(self, name: str, tool: BaseTool):
        """Register a tool instance with the registry using specified name."""
        # Store tool instance in registry dictionary for name-based lookup
        self._tools[name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool instance by name from the registry."""
        # Look up tool by name, returns None if not found (safe handling)
        return self._tools.get(name)

    def list_available(self) -> List[str]:
        """List all registered tool names for discovery and validation."""
        # Return sorted list of all registered tool names for consistent ordering
        return sorted(list(self._tools.keys()))


# Global tool registry instance for application-wide access
# Provides singleton pattern for tool management across the system
tool_registry = ToolRegistry()