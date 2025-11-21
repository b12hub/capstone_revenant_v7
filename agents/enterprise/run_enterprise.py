# #!/usr/bin/env python3
# """
# Enterprise Pipeline Main Executable
# Runs full enterprise agent pipeline with CLI interface and performance monitoring.
# """
#
# import asyncio
# import json
# import time
# import argparse
# import sys
# from typing import Dict, Any, List
#
# from enterprise.enterprise_agent import EnterpriseOrchestratorAgent
#
#
# def load_ticket_data(file_path: str) -> Dict[str, Any]:
#     """
#     Load ticket data from JSON file with comprehensive error handling.
#     Returns parsed ticket data or exits with error message.
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         # Validate basic structure
#         if not isinstance(data, dict):
#             print(f"❌ ERROR: Ticket data must be a JSON object, got {type(data)}")
#             sys.exit(1)
#
#         return data
#
#     except FileNotFoundError:
#         print(f"❌ ERROR: Ticket file not found: {file_path}")
#         sys.exit(1)
#     except json.JSONDecodeError as e:
#         print(f"❌ ERROR: Invalid JSON in {file_path}: {e}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"❌ ERROR: Failed to load {file_path}: {e}")
#         sys.exit(1)
#
#
# def load_sample_dataset(file_path: str) -> List[Dict[str, Any]]:
#     """
#     Load sample ticket dataset for batch processing.
#     Returns list of ticket data or exits with error message.
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#
#         # Validate array structure
#         if not isinstance(data, list):
#             print(f"❌ ERROR: Dataset must be a JSON array, got {type(data)}")
#             sys.exit(1)
#
#         return data
#
#     except FileNotFoundError:
#         print(f"❌ ERROR: Dataset file not found: {file_path}")
#         sys.exit(1)
#     except json.JSONDecodeError as e:
#         print(f"❌ ERROR: Invalid JSON in {file_path}: {e}")
#         sys.exit(1)
#     except Exception as e:
#         print(f"❌ ERROR: Failed to load {file_path}: {e}")
#         sys.exit(1)
#
#
# def print_colored(text: str, color: str = "white") -> None:
#     """
#     Print colored text to terminal for better readability.
#     Supports basic colors for different message types.
#     """
#     colors = {
#         "red": "\033[91m",
#         "green": "\033[92m",
#         "yellow": "\033[93m",
#         "blue": "\033[94m",
#         "magenta": "\033[95m",
#         "cyan": "\033[96m",
#         "white": "\033[97m",
#         "reset": "\033[0m"
#     }
#
#     color_code = colors.get(color, colors["white"])
#     reset_code = colors["reset"]
#
#     print(f"{color_code}{text}{reset_code}")
#
#
# def format_response(response: Dict[str, Any], verbose: bool = False) -> str:
#     """
#     Format pipeline response for human-readable output.
#     Includes color coding and conditional detail display.
#     """
#     output_lines = []
#
#     # Header with ticket information
#     output_lines.append("🎫 ENTERPRISE PIPELINE RESULTS")
#     output_lines.append("=" * 50)
#
#     # Basic classification results
#     ticket_id = response.get("ticket_id", "unknown")
#     intent = response.get("intent", "unknown")
#     severity = response.get("severity", "medium")
#     confidence = response.get("confidence", 0.0)
#
#     output_lines.append(f"📋 Ticket: {ticket_id}")
#     output_lines.append(f"🎯 Intent: {intent}")
#     output_lines.append(f"🚨 Severity: {severity}")
#     output_lines.append(f"📊 Confidence: {confidence:.3f}")
#
#     # Color code confidence level
#     if confidence > 0.8:
#         conf_color = "green"
#     elif confidence > 0.6:
#         conf_color = "yellow"
#     else:
#         conf_color = "red"
#
#     output_lines.append(f"📈 Confidence Level: {conf_color}")
#
#     # Timing information
#     timing_ms = response.get("timing_ms", 0)
#     output_lines.append(f"⏱️  Processing Time: {timing_ms}ms")
#
#     # Trace ID for observability
#     trace_id = response.get("trace_id", "unknown")
#     output_lines.append(f"🔍 Trace ID: {trace_id}")
#
#     # Actions section
#     actions = response.get("actions", [])
#     output_lines.append("")
#     output_lines.append("🚀 RECOMMENDED ACTIONS")
#     output_lines.append("-" * 30)
#
#     if actions:
#         for i, action in enumerate(actions, 1):
#             action_name = action.get("action", "unknown")
#             target = action.get("target", "unknown")
#             priority = action.get("priority", "medium")
#             reason = action.get("reason", "")
#
#             output_lines.append(f"{i}. {action_name} → {target} [{priority}]")
#             if verbose and reason:
#                 output_lines.append(f"   📝 Reason: {reason}")
#     else:
#         output_lines.append("No actions generated")
#
#     # KB Evidence section (verbose mode only)
#     if verbose:
#         kb_evidence = response.get("kb_evidence", [])
#         output_lines.append("")
#         output_lines.append("📚 KNOWLEDGE BASE EVIDENCE")
#         output_lines.append("-" * 30)
#
#         if kb_evidence:
#             for i, evidence in enumerate(kb_evidence, 1):
#                 evidence_type = evidence.get("type", "unknown")
#                 source = evidence.get("source", "unknown")
#                 relevance = evidence.get("relevance_score", 0.0)
#                 content_preview = evidence.get("content", "")[:100] + "..."
#
#                 output_lines.append(f"{i}. [{evidence_type}] {source} (relevance: {relevance:.3f})")
#                 output_lines.append(f"   {content_preview}")
#         else:
#             output_lines.append("No relevant KB evidence found")
#
#     # Errors section (if any)
#     errors = response.get("errors", [])
#     if errors:
#         output_lines.append("")
#         output_lines.append("❌ ERRORS")
#         output_lines.append("-" * 30)
#         for error in errors:
#             output_lines.append(f"• {error}")
#
#     # Fusion metadata (verbose mode only)
#     if verbose:
#         fusion_meta = response.get("fusion_metadata", {})
#         if fusion_meta:
#             output_lines.append("")
#             output_lines.append("🔧 FUSION METADATA")
#             output_lines.append("-" * 30)
#             strategy = fusion_meta.get("strategy_used", "unknown")
#             fusion_time = fusion_meta.get("fusion_time_ms", 0)
#             original_conf = fusion_meta.get("original_intent_confidence", 0.0)
#             final_conf = fusion_meta.get("final_confidence", 0.0)
#
#             output_lines.append(f"Strategy: {strategy}")
#             output_lines.append(f"Fusion Time: {fusion_time}ms")
#             output_lines.append(f"Original Confidence: {original_conf:.3f}")
#             output_lines.append(f"Final Confidence: {final_conf:.3f}")
#
#     return "\n".join(output_lines)
#
#
# async def run_single_ticket(ticket_file: str, seed: int, simulate_failure: bool,
#                             verbose: bool) -> None:
#     """
#     Run pipeline for single ticket with performance monitoring and formatted output.
#     """
#     print_colored("🚀 Starting Enterprise Pipeline...", "cyan")
#     print_colored(f"📁 Loading ticket: {ticket_file}", "blue")
#
#     # Load ticket data with error handling
#     ticket_data = load_ticket_data(ticket_file)
#
#     # Initialize orchestrator with configuration
#     orchestrator = EnterpriseOrchestratorAgent(
#         seed=seed,
#         simulate_failure=simulate_failure
#     )
#
#     # Execute pipeline and measure performance
#     start_time = time.time()
#
#     try:
#         response = await orchestrator.execute_pipeline(ticket_data)
#         processing_time = time.time() - start_time
#
#         # Print formatted results
#         formatted_output = format_response(response, verbose)
#         print_colored(formatted_output, "white")
#
#         # Print performance summary
#         print_colored("\n📊 PERFORMANCE SUMMARY", "green")
#         print_colored(f"Total processing time: {processing_time:.3f}s", "green")
#         print_colored(f"Pipeline latency: {response.get('timing_ms', 0)}ms", "green")
#
#         # Print confidence assessment
#         confidence = response.get("confidence", 0.0)
#         if confidence > 0.8:
#             assessment = "High confidence - suitable for automated processing"
#             color = "green"
#         elif confidence > 0.6:
#             assessment = "Medium confidence - recommend human review"
#             color = "yellow"
#         else:
#             assessment = "Low confidence - requires human intervention"
#             color = "red"
#
#         print_colored(f"Confidence assessment: {assessment}", color)
#
#     except Exception as e:
#         print_colored(f"❌ Pipeline execution failed: {e}", "red")
#         sys.exit(1)
#
#
# async def run_batch_processing(dataset_file: str, seed: int, simulate_failure: bool,
#                                verbose: bool) -> None:
#     """
#     Run pipeline for batch of tickets with performance analytics.
#     """
#     print_colored("📊 Starting Batch Processing...", "cyan")
#     print_colored(f"📁 Loading dataset: {dataset_file}", "blue")
#
#     # Load dataset with error handling
#     dataset = load_sample_dataset(dataset_file)
#
#     # Initialize orchestrator
#     orchestrator = EnterpriseOrchestratorAgent(
#         seed=seed,
#         simulate_failure=simulate_failure
#     )
#
#     # Process batch and measure performance
#     start_time = time.time()
#
#     try:
#         results = await orchestrator.process_batch(dataset)
#         total_time = time.time() - start_time
#
#         # Calculate batch statistics
#         total_tickets = len(results)
#         successful_tickets = len([r for r in results if not r.get("errors")])
#         avg_confidence = sum(r.get("confidence", 0) for r in results) / total_tickets
#         avg_processing_time = total_time / total_tickets
#
#         # Print batch summary
#         print_colored("\n📈 BATCH PROCESSING SUMMARY", "green")
#         print_colored("=" * 40, "green")
#         print_colored(f"Total tickets processed: {total_tickets}", "white")
#         print_colored(f"Successful processing: {successful_tickets}/{total_tickets}",
#                       "green" if successful_tickets == total_tickets else "yellow")
#         print_colored(f"Average confidence: {avg_confidence:.3f}", "white")
#         print_colored(f"Average processing time: {avg_processing_time:.3f}s per ticket", "white")
#         print_colored(f"Total batch time: {total_time:.3f}s", "white")
#
#         # Print intent distribution
#         print_colored("\n🎯 INTENT DISTRIBUTION", "cyan")
#         intents = {}
#         for result in results:
#             intent = result.get("intent", "unknown")
#             intents[intent] = intents.get(intent, 0) + 1
#
#         for intent, count in sorted(intents.items()):
#             percentage = (count / total_tickets) * 100
#             print_colored(f"  {intent}: {count} tickets ({percentage:.1f}%)", "white")
#
#         # Print sample results if verbose
#         if verbose and results:
#             print_colored("\n🔍 SAMPLE RESULTS", "cyan")
#             sample_result = results[0]  # First ticket as sample
#             formatted_sample = format_response(sample_result, verbose=False)
#             print_colored(formatted_sample, "white")
#
#     except Exception as e:
#         print_colored(f"❌ Batch processing failed: {e}", "red")
#         sys.exit(1)
#
#
# def main():
#     """Main entry point with CLI argument parsing and execution coordination."""
#     parser = argparse.ArgumentParser(
#         description="Enterprise AI Agent Pipeline",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Process single ticket
#   python run_enterprise.py --ticket examples/example_task.json
#
#   # Process batch dataset
#   python run_enterprise.py --batch examples/sample_tickets.json
#
#   # Enable verbose output and failure simulation
#   python run_enterprise.py --ticket examples/example_task.json --verbose --simulate_failure
#
#   # Set custom random seed for reproducibility
#   python run_enterprise.py --ticket examples/example_task.json --seed 12345
#         """
#     )
#
#     # Input source arguments (mutually exclusive)
#     input_group = parser.add_mutually_exclusive_group(required=True)
#     input_group.add_argument(
#         "--ticket",
#         type=str,
#         help="Path to single ticket JSON file"
#     )
#     input_group.add_argument(
#         "--batch",
#         type=str,
#         help="Path to batch tickets JSON file"
#     )
#
#     # Configuration arguments
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=42,
#         help="Random seed for deterministic behavior (default: 42)"
#     )
#     parser.add_argument(
#         "--simulate_failure",
#         action="store_true",
#         help="Simulate pipeline failures for testing"
#     )
#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         help="Enable verbose output with detailed metadata"
#     )
#
#     # Parse arguments
#     args = parser.parse_args()
#
#     # Execute appropriate pipeline
#     try:
#         if args.ticket:
#             asyncio.run(run_single_ticket(
#                 args.ticket, args.seed, args.simulate_failure, args.verbose
#             ))
#         else:
#             asyncio.run(run_batch_processing(
#                 args.batch, args.seed, args.simulate_failure, args.verbose
#             ))
#
#     except KeyboardInterrupt:
#         print_colored("\n⏹️  Pipeline interrupted by user", "yellow")
#         sys.exit(1)
#     except Exception as e:
#         print_colored(f"\n❌ Unexpected error: {e}", "red")
#         sys.exit(1)
#
#
# if __name__ == "__main__":
#     main()

# !/usr/bin/env python3
"""
Enterprise Pipeline Main Executable
Runs full deterministic enterprise agent pipeline with CLI interface and performance monitoring.
"""

import asyncio
import json
import time
import argparse
import sys
from typing import Dict, Any, List

from agents.enterprise.enterprise_agent import EnterpriseOrchestratorAgent


def load_ticket_data(file_path: str) -> Dict[str, Any]:
    """
    Load ticket data from JSON file with comprehensive error handling.
    Returns parsed ticket data or exits with structured error message.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate basic structure is a JSON object (dictionary)
        # Prevents pipeline execution on malformed data structures
        if not isinstance(data, dict):
            print(f"ERROR: Ticket data must be a JSON object, got {type(data)}")
            sys.exit(1)

        return data

    except FileNotFoundError:
        print(f"ERROR: Ticket file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load {file_path}: {e}")
        sys.exit(1)


def load_sample_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load sample ticket dataset for batch processing with validation.
    Returns list of ticket data or exits with error message.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate array structure for batch processing
        # Batch processing requires list of ticket objects
        if not isinstance(data, list):
            print(f"ERROR: Dataset must be a JSON array, got {type(data)}")
            sys.exit(1)

        return data

    except FileNotFoundError:
        print(f"ERROR: Dataset file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {file_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load {file_path}: {e}")
        sys.exit(1)


async def run_single_ticket(ticket_file: str, seed: int, simulate_failure: bool,
                            verbose: bool) -> None:
    """
    Run pipeline for single ticket with performance monitoring and formatted output.
    Demonstrates end-to-end enterprise pipeline execution.
    """
    print("Starting Enterprise Pipeline...")
    print(f"Loading ticket: {ticket_file}")

    # Load ticket data with comprehensive error handling
    ticket_data = load_ticket_data(ticket_file)

    # Initialize orchestrator with deterministic configuration
    orchestrator = EnterpriseOrchestratorAgent(
        seed=seed,
        simulate_failure=simulate_failure
    )

    # Execute pipeline and measure end-to-end performance
    start_time = time.time()

    try:
        response = await orchestrator.execute_pipeline(ticket_data)
        processing_time = time.time() - start_time

        # Print structured JSON response for machine consumption
        print("\nPipeline Results:")
        print(json.dumps(response, indent=2))

        # Print human-readable performance summary
        print(f"\nPerformance Summary:")
        print(f"Total processing time: {processing_time:.3f}s")
        print(f"Pipeline latency: {response.get('timing_ms', 0)}ms")

        # Print confidence assessment for operational guidance
        confidence = response.get("confidence", 0.0)
        if confidence > 0.8:
            assessment = "High confidence - suitable for automated processing"
        elif confidence > 0.6:
            assessment = "Medium confidence - recommend human review"
        else:
            assessment = "Low confidence - requires human intervention"

        print(f"Confidence assessment: {assessment}")

    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)


async def run_batch_processing(dataset_file: str, seed: int, simulate_failure: bool,
                               verbose: bool) -> None:
    """
    Run pipeline for batch of tickets with performance analytics and aggregate statistics.
    Demonstrates enterprise-scale processing capabilities.
    """
    print("Starting Batch Processing...")
    print(f"Loading dataset: {dataset_file}")

    # Load dataset with comprehensive error handling
    dataset = load_sample_dataset(dataset_file)

    # Initialize orchestrator with deterministic configuration
    orchestrator = EnterpriseOrchestratorAgent(
        seed=seed,
        simulate_failure=simulate_failure
    )

    # Process batch and measure aggregate performance
    start_time = time.time()

    try:
        results = await orchestrator.process_batch(dataset)
        total_time = time.time() - start_time

        # Calculate batch statistics for performance analysis
        total_tickets = len(results)
        successful_tickets = len([r for r in results if r.get("confidence", 0) > 0.3])
        avg_confidence = sum(r.get("confidence", 0) for r in results) / total_tickets
        avg_processing_time = total_time / total_tickets

        # Print batch summary with key metrics
        print(f"\nBatch Processing Summary:")
        print(f"Total tickets processed: {total_tickets}")
        print(f"Successful processing: {successful_tickets}/{total_tickets}")
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average processing time: {avg_processing_time:.3f}s per ticket")
        print(f"Total batch time: {total_time:.3f}s")

        # Print intent distribution for operational insights
        print(f"\nIntent Distribution:")
        intents = {}
        for result in results:
            intent = result.get("intent", "unknown")
            intents[intent] = intents.get(intent, 0) + 1

        for intent, count in sorted(intents.items()):
            percentage = (count / total_tickets) * 100
            print(f"  {intent}: {count} tickets ({percentage:.1f}%)")

    except Exception as e:
        print(f"Batch processing failed: {e}")
        sys.exit(1)


def main():
    """Main entry point with CLI argument parsing and execution coordination."""
    parser = argparse.ArgumentParser(
        description="Enterprise AI Agent Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input source arguments (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ticket",
        type=str,
        help="Path to single ticket JSON file"
    )
    input_group.add_argument(
        "--batch",
        type=str,
        help="Path to batch tickets JSON file"
    )

    # Configuration arguments for deterministic behavior
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic behavior (default: 42)"
    )
    parser.add_argument(
        "--simulate_failure",
        action="store_true",
        help="Simulate pipeline failures for testing resilience"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed metadata"
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Execute appropriate pipeline based on input arguments
    try:
        if args.ticket:
            asyncio.run(run_single_ticket(
                args.ticket, args.seed, args.simulate_failure, args.verbose
            ))
        else:
            asyncio.run(run_batch_processing(
                args.batch, args.seed, args.simulate_failure, args.verbose
            ))

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()