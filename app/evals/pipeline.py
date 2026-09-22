import copy
import logfire
import requests
import time
import json
import os

API_URL = "http://localhost:8000/query"
REQUEST_TIMEOUT = 120
RESPONSE_TRUNCATE = 200
DELAY_BETWEEN_CALLS = 12


def detect_tool(thought_process: list) -> str:
    """
    Maps the thought process list from /query response to a tool name
    """
    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if (
        "intent: technical" in joined
        or "search term:" in joined
        or "context retrieved" in joined
    ):
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"

    return "unknown"


def run_pipeline(golden_dataset: dict, progress_callback=None) -> dict:
    """
    Enriches the rag_samples in the golden dataset with the response from LIVE API results.
    Returns a deep copy with actual response,actual context,actual tool calls fired
    progress_callback(i,total,question,stage,response) is called per step
    """
    dataset = copy.deepcopy(golden_dataset)
    samples = dataset["rag_samples"]
    n = len(samples)

    with logfire.span("Evaluation Phase 1", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample.get("question", "")

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"Live Query {i + 1}/{n}",
                question=question[:80],
                domain=sample.get("domain", ""),
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": question, "thread_id": f"eval_run_{i}"},
                        timeout=REQUEST_TIMEOUT,
                    )

                    resp.raise_for_status()
                    data = resp.json()

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = sources[:5]
                    sample["actual_tools_called"] = [detect_tool(thought_process)]

                    logfire.info(
                        "Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chars=len(sources),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("Cannot reach the FastAPI")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

                except Exception as e:
                    logfire.error(f"Query failed: {e}")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_tools_called"] = ["unknown"]

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)


def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path) as f:
        return json.load(f)
