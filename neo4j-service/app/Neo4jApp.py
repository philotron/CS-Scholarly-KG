import os
from neo4j import GraphDatabase
from fuzzywuzzy import process

class Neo4jApp:


    def __init__(self):
        uri = os.getenv("uri", "neo4j://0.0.0.0:7687")
        user = os.getenv("user", "neo4j")
        password = os.getenv("password", "neo4j-connect")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def get_children_nodes(self, topic_name):
        with self.driver.session() as session:
            subtopics = session.execute_read(self._get_children_nodes, topic_name)
            return subtopics

    def get_parent_nodes(self, topic_name):
        with self.driver.session() as session:
            subtopics = session.execute_read(self._get_parent_nodes, topic_name)
            return subtopics
        
    def get_topic_definition(self, topic_name):
        with self.driver.session() as session:
            topic = session.execute_read(self._get_topic_definition, topic_name)
            return topic
        
   # Just for back up if we want to be more dynamic 
    def get_nlp_taxonomy(self):
        with self.driver.session() as session:
            taxonomy = session.execute_read(self._get_nlp_taxonomy)
            return taxonomy
        
    def get_paper_details(self, paper_id):
        with self.driver.session() as session:
            paper_details = session.execute_read(self._get_paper_details, paper_id=int(paper_id))
            return paper_details
        
    def get_clusters_by_topic(self, topic_name):
        with self.driver.session() as session:
            clusters = session.read_transaction(self._retrieve_clusters_of_topic, topic_name)
            return clusters
        
    def get_subclusters_by_cluster_id(self, cluster_id):
        with self.driver.session() as session:
            subclusters = session.read_transaction(self._retrieve_subclusters, cluster_id)
            return subclusters    
        
    def get_publications_by_cluster_id(self, cluster_id):
        with self.driver.session() as session:
            publications = session.read_transaction(self._retrieve_publications, cluster_id)
            return publications    
        

    def get_publications_by_cluster_and_subclusters(self, cluster_id):
        with self.driver.session() as session:
            publications = session.read_transaction(self._retrieve_publications_and_subclusters, cluster_id)
            return publications
    
    # retrieve the clusters of a subtopic
    @staticmethod
    def _retrieve_clusters_of_topic(tx, topic_name):
        query = """
        MATCH (topic:FieldOfStudy)-[:HAS_CLUSTER]->(cluster)
        WHERE toUpper(topic.label) = toUpper($topic_name)
        OPTIONAL MATCH (cluster)-[:HAS_PUBLICATION]->(pub:Publication)
        WITH cluster, COUNT(DISTINCT pub) AS publicationsCount
        RETURN cluster.id AS ClusterID, cluster.tag AS Tag, cluster.zephyr_cluster_tag AS ZephyrTag, cluster.level AS Level, publicationsCount
        ORDER BY publicationsCount DESC
        """
        result = tx.run(query, topic_name=topic_name)
        clusters = [{"ClusterID": record["ClusterID"], "Tag": record["Tag"], "ZephyrTag": record["ZephyrTag"], "Level": record["Level"]} for record in result]
        return clusters

    # retrieve the subclusters of a cluster, results include cluster IDs, cluster tags, zephyr tags and cluster level 
    @staticmethod
    def _retrieve_subclusters(tx, cluster_id):
        query = """
        MATCH (parent:Cluster {id: $cluster_id})-[:HAS_SUBCLUSTER]->(subcluster)
        OPTIONAL MATCH (subcluster)-[:HAS_PUBLICATION]->(pub:Publication)
        WITH subcluster, COUNT(DISTINCT pub) AS publicationsCount
        RETURN subcluster.id AS SubclusterID, subcluster.tag AS Tag, subcluster.zephyr_cluster_tag AS ZephyrTag, subcluster.level AS Level, publicationsCount
        ORDER BY publicationsCount DESC
        """
        result = tx.run(query, cluster_id=cluster_id)
        subclusters = [{"ClusterID": record["SubclusterID"], "Tag": record["Tag"], "ZephyrTag": record["ZephyrTag"], "Level": record["Level"]} for record in result]
        return subclusters
    
    # retrieve the publications under a specific cluster, results include publications' IDs and the titles
    @staticmethod
    def _retrieve_publications(tx, cluster_id):
        query = """
        MATCH (cluster:Cluster {id: $cluster_id})-[:HAS_PUBLICATION]->(pub:Publication)
        RETURN id(pub) AS PublicationID
        """
        result = tx.run(query, cluster_id=cluster_id)
        publications = [{"Publication ID": record["PublicationID"]} for record in result]
        return publications    
    
    # retrieve the detail information of publications under a specific clusters
    @staticmethod
    def _retrieve_publications_and_subclusters(tx, cluster_id):
        query = """
        MATCH (cluster:Cluster {id: $cluster_id})
        CALL {
            WITH cluster
            MATCH (cluster)-[:HAS_SUBCLUSTER*0..]->(subcluster)-[:HAS_PUBLICATION]->(pub:Publication)
            RETURN pub
        }
        RETURN id(pub) AS PublicationID, pub.publicationTitle AS Title, pub.publicationAbstract AS Abstract
        """
        result = tx.run(query, cluster_id=cluster_id)
        publications = [{"PublicationID": record["PublicationID"], "Title": record["Title"], "Abstract": record["Abstract"]} for record in result]
        return publications

    @staticmethod
    def _get_fuzzy_topic(topic_name):
        topics = ['Speech & Audio in NLP', 'Multimodality', 'Visual Data in NLP',
                  'Structured Data in NLP', 'Programming Languages in NLP',
                  'Natural Language Interfaces', 'Question Answering',
                  'Dialogue Systems & Conversational Agents', 'Semantic Text Processing',
                  'Discourse & Pragmatics', 'Representation Learning', 'Knowledge Representation',
                  'Text Complexity', 'Semantic Search', 'Word Sense Disambiguation',
                  'Semantic Parsing', 'Language Models', 'Semantic Similarity',
                  'Sentiment Analysis', 'Opinion Mining', 'Stylistic Analysis',
                  'Intent Recognition', 'Emotion Analysis', 'Aspect-based Sentiment Analysis',
                  'Polarity Analysis', 'Syntactic Text Processing', 'Tagging', 'Morphology',
                  'Chunking', 'Phonology', 'Text Error Correction', 'Text Segmentation',
                  'Typology', 'Syntactic Parsing', 'Phonetics', 'Text Normalization',
                  'Linguistics & Cognitive NLP', 'Linguistic Theories', 'Cognitive Modeling',
                  'Psycholinguistics', 'Responsible & Trustworthy NLP', 'Responsible NLP',
                  'Ethical NLP', 'Low-Resource NLP', 'Robustness in NLP', 'Green & Sustainable NLP',
                  'Explainability & Interpretability in NLP', 'Reasoning', 'Textual Inference',
                  'Commonsense Reasoning', 'Numerical Reasoning', 'Knowledge Graph Reasoning',
                  'Machine Reading Comprehension', 'Fact & Claim Verification', 'Argument Mining',
                  'Multilinguality', 'Cross-Lingual Transfer', 'Machine Translation',
                  'Code-Switching', 'Information Retrieval', 'Indexing', 'Document Retrieval',
                  'Text Classification', 'Passage Retrieval', 'Information Extraction & Text Mining',
                  'Coreference Resolution', 'Text Clustering', 'Named Entity Recognition',
                  'Event Extraction', 'Open Information Extraction', 'Term Extraction',
                  'Topic Modeling', 'Relation Extraction', 'Text Generation',
                  'Data-to-Text Generation', 'Question Generation', 'Dialogue Response Generation',
                  'Captioning', 'Paraphrassing', 'Paraphrasing', 'Text Style Transfer',
                  'Code Generation', 'Summarization', 'Speech Recognition']
        threshold = 80
        best_match = process.extractOne(topic_name, topics)
        if best_match[1] < threshold:
            return ""
        else:
            return best_match[0]

    # query the definitions of a topic from the database
    @staticmethod
    def _get_topic_definition(tx, topic_name: str):
        topic_name = Neo4jApp._get_fuzzy_topic(topic_name)
        if topic_name == "":
            return topic_name
        # print(topic_name)
        query = (
            "MATCH (n:FieldOfStudy {label: $topic_name }) "
            "RETURN n.description as description "
            "LIMIT 100 "
        )
        topic_definition = tx.run(query, topic_name=topic_name)
        print(topic_definition)
        for row in topic_definition:
            topic_definition = row['description']
            break

        return {"topic_definition" : topic_definition, "topic_name": topic_name}

    @staticmethod
    def get_out_of_scope_string():
        return "What your referring to is outside my scope. Ask for the taxonomy."

    # retrieve children nodes (subtopics) of a main topic
    @staticmethod
    def _get_children_nodes(tx, topic_name):
        topic_name = Neo4jApp._get_fuzzy_topic(topic_name)
        if topic_name == "":
            return Neo4jApp.get_out_of_scope_string()
        query = (
            "MATCH (n:FieldOfStudy {label: $topic_name }) -[]->(m:FieldOfStudy) "
            "WHERE n.level < m.level "
            "RETURN m.label "
            "LIMIT 100 "
        )
        children_nodes = tx.run(query, topic_name=topic_name)
        children_nodes = [row[0] for row in children_nodes]
        if len(children_nodes) == 0:
            return f"{topic_name} apparently doesn't have subtopics."
        children_nodes = ('\n- ').join(children_nodes)
        if children_nodes == "":
            return Neo4jApp.get_out_of_scope_string()
        result_str = f"""Subtopics of {topic_name} are:\
        \n- {children_nodes}\
        \nYou can ask me to define one of these topics or to provide the paper clusters within a subtopic."""
        
        return result_str
    
    # retrieve the parent node (main topic) of a subtopic
    @staticmethod
    def _get_parent_nodes(tx, topic_name):
        topic_name = Neo4jApp._get_fuzzy_topic(topic_name)
        # print(topic_name)
        if topic_name == "":
            return Neo4jApp.get_out_of_scope_string()

        query = (
            "MATCH (n:FieldOfStudy {label: $topic_name }) -[]->(m:FieldOfStudy) "
            "WHERE n.level > m.level "
            "RETURN m.label "
            "LIMIT 100 "
        )
        parent_nodes = tx.run(query, topic_name=topic_name)
        parent_nodes = [row[0] for row in parent_nodes]
        if len(parent_nodes) == 0:
            return f"{topic_name} apparently doesn't have parents."
        parent_nodes = ('\n  - ').join(parent_nodes)
        if parent_nodes == "":
            return Neo4jApp.get_out_of_scope_string()
        result_str = f"Parents of {topic_name} are \n- {parent_nodes}. \nYou can ask me for the definitions of these terms."
        
        return result_str
    
    # Just for back up if we want to be more dynamic
    # create main topics level under NLP 
    @staticmethod
    def _get_nlp_taxonomy(tx):
        bullet_point0 = '\n   -- '
        bullet_point1 = '\n-- '
        level0 = 'Natural Language Processing'
        level1 = tx.run("match (n:FieldOfStudy {label: 'Natural Language Processing'})-[]->(m:FieldOfStudy) where m.level = [1] return m.label limit 100")
        level1 = [row[0] for row in level1]
        for i, topic in enumerate(level1):
            subtopics_level_2 = tx.run("match (n:FieldOfStudy {label: $topic})-[]->(m:FieldOfStudy) where m.level = [2] return m.label limit 100", topic=topic)
            level1[i] += (bullet_point0 + '\n   -- '.join([row[0] for row in subtopics_level_2]))
        taxonomy = ('- ' + level0 + '\n '+ bullet_point1 + ('\n-- ').join(level1))
        return taxonomy[0:1000]
    
    # change names of attributes in node Publication
    @staticmethod
    def _get_paper_details(tx, paper_id):
        
        query = (
            "MATCH (p:Publication) "
            f"WHERE (id(p) = {paper_id}) "
            "RETURN p.publicationAbstract AS abstract, " 
            + "p.tldr AS tldr, "
            + "p.numberOfCitations AS citations, "
            + "p.objective AS objective, "
            + "p.background AS background, "
            + "p.methods AS methods, "
            + "p.results AS results, "
            + "p.conclusions AS conclusions, "
            + "p.publicationDate AS date, "
            + "p.publishedIn AS publication, "
            + "p.authorList AS authors, "
            + "p.publicationTitle AS title, "
            + "p.arxivUrl AS arxivurl, "
            + "p.aclUrl AS aclurl "
        )
        
        paper_details = tx.run(query, paper_id=paper_id).single()
        paper_details_dict = {
            "abstract": paper_details["abstract"],
            "tldr": paper_details["tldr"],
            "citations": paper_details["citations"],
            "objective": paper_details["objective"],
            "background": paper_details["background"],
            "methods": paper_details["methods"],
            "results": paper_details["results"],
            "conclusions": paper_details["conclusions"],
            "date": str(paper_details["date"]),
            "publication": paper_details["publication"],
            "authors": paper_details["authors"],
            "title": paper_details["title"],
            "arxivurl": paper_details["arxivurl"],
            "aclurl": paper_details["aclurl"]
        }

        return paper_details_dict
