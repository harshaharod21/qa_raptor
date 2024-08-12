import os
import numpy as np
from langchain_core.prompts import PromptTemplate

from getpass import getpass
from langchain_groq import ChatGroq
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import umap

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
from typing import Optional
import numpy as np
import umap
from dotenv import load_dotenv
load_dotenv()
from langchain_core.runnables import RunnablePassthrough



def llm():
    "Define LLM to be used"
    
    GROQ_API_KEY= os.environ['GROQ_API_KEY']

    model_llm= ChatGroq(groq_api_key=GROQ_API_KEY,
              model_name='llama3-70b-8192')
    return model_llm

    

RANDOM_SEED=42

def global_cluster_embeddings(
            embeddings:np.ndarray,
            dim: int,
            n_neighbors: Optional[int] = None,
            metric:str = "cosine",

    ) -> np.ndarray:
        
        """perform global dimensionality reduction on the embeddings using UMAP.
        
        parameters:
        - embedding: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - n_neighbors: Optional; the number of neighbors to consider for each point.
                    If not provided, it defaults to the square root of the number of embeddings.
        - metric: The distance metric to use for UMAP 
        
        Returns: A numpy array of the embeddings reduced to the specified dimensionality"""

        if n_neighbors is None:
            n_neighbors= int((len(embeddings)-1)** 0.5)
        return umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric 
        ).fit_transform(embeddings)
    
    #Local clustering

def local_cluster_embeddings(
            embeddings: np.ndarray, dim:int, num_neighbors:int=10, metric:str="cosine"
    )-> np.ndarray:
        """"Docs  """

        return umap.UMAP(
            n_neighbors=num_neighbors, n_components=dim,metric=metric
        ).fit_transform(embeddings)
    
    #To get optimal number of of clusters


def get_optimal_clusters(
            embeddings: np.ndarray, max_clusters: int = 50, random_state: int= RANDOM_SEED
    ) -> int:
        
        max_clusters=min(max_clusters,len(embeddings))
        n_clusters= np.arange(1, max_clusters)
        bics=[]
        
        for n in n_clusters:
            gm= GaussianMixture(n_components=n,random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]
    #to perform cluster embeddibg using a GMM

def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state:int=0):

        """DOCS"""

        n_clusters= get_optimal_clusters(embeddings)
        gm= GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob>threshold)[0] for prob in probs]
        return labels, n_clusters

def perform_clustering(embeddings:np.ndarray, dim: int, threshold: float)-> List[np.ndarray]:
        """DOCS"""

        if len(embeddings) <= dim+1:
            #avoid clustering when theres insufficient data
            return [np.array([0]) for _ in range(len(embeddings))]
        # Global dimensionality reduction

        reduced_embeddings_global= global_cluster_embeddings(embeddings, dim)

        #Global clustering

        global_clusters, n_global_clusters =GMM_cluster(
            reduced_embeddings_global,threshold
        )

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        #Iterate through each global cluster to perform local clustering

        for i in range(n_global_clusters):
            #Extract embeddings belonging to the current global cluster

            global_cluster_embeddings_= embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_)<= dim +1:
                #handle small clusters with direct assignmemt
                local_clusters= [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters=1
            else:
                #Local dimensiioanlity reduction and clustering
                reduced_embeddings_local= local_cluster_embeddings(
                    global_cluster_embeddings_,dim
                )
                local_clusters, n_local_clusters = GMM_cluster(
                    reduced_embeddings_local,threshold
                )

            # Assign local cluster IDs, adjusting for total clusters already processed

            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        return all_local_clusters
    
    #Embedding text docs

def embed_text(docs):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(docs)
        return embeddings


def  cluster_embed_text(texts):
        """DOCS"""

        text_embedding= embed_text(texts)
        cluster_labels= perform_clustering(
            text_embedding,10,0.1
        )
        df_store=pd.DataFrame() # to store the results
        df_store["text"]=texts
        
        
        df_store["embd"]= list(text_embedding)
        df_store["cluster"]= cluster_labels 
        return df_store

def fmt_txt(df: pd.DataFrame) -> str:
        """
        Formats the text documents in a DataFrame into a single string.

        Parameters:
        - df: DataFrame containing the 'text' column with text documents to format.

        Returns:
        - A single string where all text documents are joined by a specific delimiter.
        """
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)
    
def embed_cluster_summarize(model_llm,
        texts:List[str], level:int
)-> Tuple[pd.DataFrame,pd.DataFrame]:
        
        """DOCS"""

        df_clusters = cluster_embed_text(texts)
        
        expanded_list= []

        # Expand DataFrame entries to document-cluster pairings for straightforward processing
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embd": row["embd"], "cluster": cluster}
                )
        # Create a new DataFrame from the expanded list
        expanded_df = pd.DataFrame(expanded_list)

        # Retrieve unique cluster identifiers for processing
        all_clusters = expanded_df["cluster"].unique()

        print(f"--Generated {len(all_clusters)} clusters--")


            # Summarization
        template = """Here is a document.

        Give a detailed summary of the documentation provided.Strictly avoid starting with any such phrases in the output "Here is a detailed summary of the document provided:"or "The documentation appears to be a collection ".Just provide the detailed summary and no other statements.

        Documentation:
        {context}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model_llm | StrOutputParser()     #where is model and stroutputparser?
        
        # Format text within each cluster for summarization
        summaries = []
        for i in all_clusters:
            df_cluster = expanded_df[expanded_df["cluster"] == i]
            formatted_txt = fmt_txt(df_cluster)
            summaries.append(chain.invoke({"context": formatted_txt}))

    # Create a DataFrame to store summaries with their corresponding cluster and level
        df_summary = pd.DataFrame(
                {
                    "summaries": summaries,
                    "level": [level] * len(summaries),
                    "cluster": list(all_clusters),
                }
            )
        
        return df_clusters, df_summary

def recursive_embed_cluster_summarize(
        texts: List[str], level: int = 1, n_levels: int = 3 
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Recursively embeds, clusters, and summarizes texts up to a specified level or until
        the number of unique clusters becomes 1, storing the results at each level.

        Parameters:
        - texts: List[str], texts to be processed.
        - level: int, current recursion level (starts at 1).
        - n_levels: int, maximum depth of recursion.

        Returns:
        - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
        levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
        """
        results = {}  # Dictionary to store results at each level
        GROQ_API_KEY= os.environ['GROQ_API_KEY']


        model_llm= ChatGroq(groq_api_key=GROQ_API_KEY,
                model_name='llama3-70b-8192')

        # Perform embedding, clustering, and summarization for the current level
        df_clusters, df_summary = embed_cluster_summarize(model_llm,texts, level)

        # Store the results of the current level
        results[level] = (df_clusters, df_summary)

        # Determine if further recursion is possible and meaningful
        unique_clusters = df_summary["cluster"].nunique()
        
        if level < n_levels and unique_clusters > 1:
            # Use summaries as the input texts for the next level of recursion
            new_texts = df_summary["summaries"].tolist()
            next_level_results = recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )
            # Merge the results from the next level into the current results dictionary
            results.update(next_level_results)
            
        if level == n_levels or unique_clusters == 1:
            

            f_summary= df_summary["summaries"].tolist()
            f_embed=embed_text(f_summary)
            
            df_final_embeddings = pd.DataFrame(
                {
                    "text": f_summary,
                    "embd": list(f_embed),
                    "cluster": df_summary["cluster"],
                }
            )
            
            # Append the final summaries embeddings to df_clusters
            df_clusters = pd.concat([df_clusters, df_final_embeddings])
            
            
            results[level]=(df_clusters,df_summary)
            


        return results




                



