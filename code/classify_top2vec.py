import sys
import argparse

def get_topic_for_doc(readme: str, top2vec_model_path: str):
    """
    Adds a document to a top2vec model and retrieves the cosine similarity to the closest topic, as well as
    a boolean value whether it has been assigned as bioinformatics.

    Parameters
    ----------
    readme : str
        The input repository readme as a plaintext string.
    top2vec_model_path : str
        Filepath of the top2vec model.
    Returns
    -------
    cosine_similarity : float
        Cosine similarity of the document to the most similar topic.
    assigned_bioinformatics: bool
        Returns True if document has been assigned to the bioinformatics topic.
    """
    from top2vec import Top2Vec
    import json
    model = None
    try:
        model = Top2Vec.load("Data/top2vec.model")
    except:
        sys.exit(f"Could not load model at {top2vec_model_path}")
    model.add_documents([readme])
    documents, doc_scores, doc_ids = model.query_documents(query=readme, num_docs=1)

    topic_sizes, topic_nums = model.get_topic_sizes()
    topic_words, word_scores, topic_scores, topic_nums_bioinformatics = model.search_topics(keywords=["bioinformatics"], num_topics=len(topic_nums))
    bioinformatics_topic = topic_nums_bioinformatics[0]
    topic_sizes, topic_nums = model.get_topic_sizes()

    for topic_num in topic_nums:
        documents, document_scores, document_ids = model.search_documents_by_topic(topic_num, num_docs=topic_sizes[topic_num])
        if doc_ids[0] in document_ids:
            cosine_score = document_scores[list(document_ids).index(doc_ids[0])]
            bioinformatics = False
            if topic_num == bioinformatics_topic:
                bioinformatics = True
            else:
                bioinformatics = False
            return (cosine_score, bioinformatics)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--readme', type=str,
        help="Path to the json dict containing repository data related to bioinformatics.")
    parser.add_argument('-i', '--top2vec_model_path', type=str,
        help="Path to the json dict containing repository data not related to bioinformatics.")
    args = parser.parse_args()

    cosine_score, bioinformatics = get_topic_for_doc(args.readme, args.top2vec_model_path)
    print(f"Categorized as bioinformatics: {bioinformatics}, cosine similarty with most similar topic: {cosine_score}")