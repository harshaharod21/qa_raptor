from src.data_ingestion import data_ingest
from src.Raptor_index import llm, recursive_embed_cluster_summarize
from src.Question_Ans import question_answer


def main():

    data = data_ingest()

    model_llm=llm()

    results = recursive_embed_cluster_summarize(data, level=1, n_levels=3)

    qa_results=question_answer(results,model_llm)
    print(qa_results)


if __name__== "__main__":
    main()





