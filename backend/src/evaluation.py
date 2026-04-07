import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


def run_evaluation(qa_chain, test_questions: list[str], ground_truths: list[str]):
    """
    Evaluates the RAG pipeline using RAGAS metrics.
    """
    results = []
    for question, truth in zip(test_questions, ground_truths):
        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "")
        contexts = [doc.page_content for doc in result.get("source_documents", [])]
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": truth,
        })

    dataset = Dataset.from_list(results)
    score = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    return score


if __name__ == "__main__":
    pass
