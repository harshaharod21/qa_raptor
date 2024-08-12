from langchain.prompts import ChatPromptTemplate
from langchain import LLMChain

from langchain_core.output_parsers import StrOutputParser


# Define the prompt template
def question_answer(results,model_llm):

    template="""
        You are an AI assistant. Given the following summary, generate a question that could be asked about it, 
        followed by a detailed answer.Don't output anything like "Here is a question and answer based on the summary",just provide the question and the answer in the given format below

        Summary: {summary}

        Question: 
        Answer:
        """
    
    

    prompt = ChatPromptTemplate.from_template(template)

    # Create the chain
    chain = prompt | model_llm | StrOutputParser()



    # Dictionary to store Q&A pairs for each level
    qa_results = {}

    # Iterate through each level in the results
    for level in results.keys():
        level_summaries = results[level][1]["summaries"]  # Access summaries at the current level
        level_qa_pairs = []  # List to store Q&A pairs for the current level

        for summary in level_summaries:
            qa_pair = chain.invoke({"summary": summary})  # Generate Q&A pair for each summary
            level_qa_pairs.append(qa_pair)

        # Store the Q&A pairs for the current level
        qa_results[level] = level_qa_pairs

    return qa_results

    # Now `qa_results` contains the Q&A pairs for each level



