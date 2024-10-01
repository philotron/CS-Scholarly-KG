# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions



from typing import Any, Text, Dict, List
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.events import FollowupAction, Restarted

import os
import requests
import json
import pandas as pd
import ast
import re


main_topic_list = ["Reasoning", "Text Generation", "Sentiment Analysis", "Multilinguality", "Syntactic Text Processing", "Linguistics & Cognitive NLP", "Information Extraction & Text Mining", "Information Retrieval", "Responsible NLP", "Multimodality", "Natural Language Interfaces", "Semantic Text Processing"]


#
#
#LISSA 2.
class ActionFollowUp(Action):

    def name(self) -> Text:
        return "action_follow_up"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        topic_is_set = tracker.get_slot('topic_is_set')
        #topic_is_set = True
        #topic = "NLP"
        #user_query = "What is NLP?"

        slot_events = []


        # Check if topic_is_set is True
        if topic_is_set:
            #Get message history
            events = tracker.events
            data =[]
            try:
                for event in events.reverse():
                    if event['event'] == 'user':
                        #user_events.append(event['text'])
                        data.append = event['text']
            except:
                pass
            user_query = tracker.get_slot('query')
            # Get topic
            topic = tracker.get_slot('predicted_topic_name')
            # Make the request to the external service

            headers = {"accept": "application/json", "Content-Type": "application/json", }
            #user_query="What is BERT?"
            #topic = "BERT"
            #data = ["message 1", "message 2"]
            data_json =json.dumps(data)
            url = f"http://follow_up:80/followup?user_query={user_query}&topic={topic}"
            response = requests.post(url, headers=headers, data=data_json)
            result = response.json()

            # Extract the question and options
            question = result.get('question')
            options = result.get('options')
            text="Which of the following options would you like to explore?\n---\n"
            # Return the question and options as buttons
            #dispatcher.utter_message(text=question)
            num = 1
            for option in options:
                #set slot
                slot_events.append(SlotSet('option_'+str(num), option))
                text+= "-"  + option + "\n"
                num += 1
            #dispatcher.utter_message(text=buttons)


            dispatcher.utter_message(text=text)
            
        else:
            dispatcher.utter_message(text="The topic is not set.")

        return slot_events

class ActionGetSubclusters(Action):
    """
    This action retrieves and processes subclusters related to a given topic for a Rasa chatbot.

    The action interacts with external services to get subclusters based on a specified topic 
    and user-selected clusters. It then presents the user with options to explore further clusters or 
    provides a list of relevant papers if clustering is complete.
    """

    def __init__(self):
        self.cluster_naming_url = os.getenv("cluster_naming_url", "http://localhost:6000") + "/generate_cluster_name"
        self.clusters_by_topic = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_clusters_by_topic"
        self.subclusters_by_cluster = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_subclusters_by_cluster"
        self.publications_by_cluster_and_subclusters = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_publications_by_cluster_and_subclusters"

    def name(self) -> Text:
        return "action_get_subclusters"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        topic = tracker.get_slot('topic_name')
        
        # uppercase the first alphabet of the topic
        topic = re.sub(r'\b(\w)', lambda match: match.group(1).upper(), topic)
        print('topic_name: ', topic)

        cluster_level = tracker.get_slot('cluster_level')

        last_cluster_index = tracker.get_slot('last_cluster_index')

        cluster_level_change = tracker.get_slot('cluster_level_change')

        selected_clusters = tracker.get_slot('selected_clusters')

        print('Cluster level: ', cluster_level)

        if cluster_level_change:
            cluster_level += 1
            last_cluster_index = 0

        stop_clustering_and_show_papers = False    

        if cluster_level == 0 and (selected_clusters == None or len(selected_clusters) == 0):
            first_level_cluster_request = requests.get(self.clusters_by_topic, data={'topic_name': topic})
            if (first_level_cluster_request.status_code == 200):
                cluster_data = first_level_cluster_request.json()
                print("Cluster data: ", cluster_data)
            else:
                dispatcher.utter_message(text="Failed to get subclusters")
                return []    
        
        if selected_clusters != None and len(selected_clusters) > 0:
            print("cluster_level: ", int(cluster_level))

            last_cluster = selected_clusters[-1]

            subcluster_request = requests.get(self.subclusters_by_cluster, data={'cluster_id': last_cluster['ClusterID']})
            if (subcluster_request.status_code == 200):
                cluster_data = subcluster_request.json()
                print("Cluster data: ", cluster_data)  


            get_papers_request = requests.get(self.publications_by_cluster_and_subclusters, data={'cluster_id': last_cluster['ClusterID']})
            if (get_papers_request.status_code == 200):
                paper_data = get_papers_request.json()
            else: 
                paper_data = []
                
            if len(paper_data) <= 10:
                stop_clustering_and_show_papers = True

            if stop_clustering_and_show_papers:
                paper_ids = []

                for paper in paper_data:
                    paper_ids.append(paper['PublicationID'])

                print("paper ids: ", paper_ids)

                return [SlotSet('final_cluster_paper_list', paper_ids), FollowupAction("action_show_paper_list_for_comparison")]        


        # Count the occurrences of each cluster in 'cluster_level_0'
                
        cluster_counts = len(cluster_data)

        print("Cluster counts", cluster_counts)

        if last_cluster_index + 4 > cluster_counts:
            last_cluster_index = cluster_counts - 1
        else: 
            last_cluster_index += 4    

        if last_cluster_index - 4 >= 0:
            first_element_index = last_cluster_index - 4
        else:
            first_element_index = 0


        cluster_dict = {}

        for i in range(len(cluster_data)):
            # Add to the dictionary
            cluster_dict[i] = cluster_data[i]

        # Now cluster_dict contains each cluster level and its corresponding tags
        print(cluster_dict)

        if len(cluster_data) == 1:
            return [SlotSet("selected_clusters", [cluster_data[0]]), SlotSet("cluster_level_change", True), FollowupAction("action_get_subclusters")]

        options_text = f"Great, you want to explore the topic {topic} further. Here are the clusters that I think may be relevant to you. Please type in the number of cluster, e.g. '1' to explore it further\n"
        for index in range(len(cluster_data)):
            if cluster_data[index]['ZephyrTag'] == None:
                tag = cluster_data[index]['Tag']
            else:
                tag = cluster_data[index]['ZephyrTag']    

            options_text += f"{index+1}. {tag}\n"

        # Add an option for selecting none of the clusters
        options_text += f"""{len(cluster_data) + 1}. None of the clusters listed above.\
        \nPlease select a cluster by entering its corresponding number. For example, type '1' to select the first cluster."""       

        dispatcher.utter_message(text=options_text)

        return [SlotSet("cluster_level", cluster_level), SlotSet("last_cluster_index", last_cluster_index), SlotSet("cluster_dict", cluster_dict)]


class ActionCaptureClusterSelection(Action):
    """
    This action captures and processes the user's cluster selection in a Rasa chatbot.

    The action retrieves the user's input, validates it against the available clusters, and updates the 
    session state accordingly. If the user selects a valid cluster, the selected cluster is stored and 
    the conversation progresses to explore subclusters. If the user indicates no selection or provides 
    an invalid input, appropriate messages are sent back to the user.
    """

    def name(self) -> Text:
        return "action_capture_cluster_selection"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        
        user_input = tracker.get_slot('selected_clusters')
        cluster_dict = tracker.get_slot('cluster_dict')

        print(cluster_dict)

        print("USER INPUT: ", user_input)

        try:
            user_selection = int(user_input)
            if user_selection == len(cluster_dict) + 1:
                dispatcher.utter_message(text="No cluster selected.")
                # Handle the 'none selected' case here
                return [SlotSet("cluster_level_change", False), FollowupAction("action_get_subclusters")]
            
            elif user_selection < 1 or user_selection > len(cluster_dict) + 1:
                dispatcher.utter_message(text="Please select a valid cluster number.")
                return []
            else:
                selected_clusters = []
                selected_clusters.append(cluster_dict[str(user_selection - 1)])

                print("SELECTED CLUSTERS: ", selected_clusters)
        except ValueError:
            dispatcher.utter_message(text="Please select a cluster with a valid number.")
            return []

        # Proceed with the logic for the selected cluster
        return [SlotSet("selected_clusters", selected_clusters), SlotSet("cluster_level_change", True), FollowupAction("action_get_subclusters")]


class ActionShowPaperListForComparison(Action):
    """
    Action to display a list of papers of the final cluster in a Rasa chatbot.

    This action retrieves details of papers in the final cluster and presents them to the user, providing
    options to summarize or compare the papers.
    """

    def __init__(self):
        self.neo4j_url = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_paper_details"

    def name(self) -> Text:
        return "action_show_paper_list_for_comparison"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:    
        
        paper_ids = tracker.get_slot('final_cluster_paper_list')

        paper_list_text = """You are now at the final cluster level which includes some papers.\
                           Here is a list of papers matching your selection:\
                           \n📎 I can provide a TL;DR summary of any paper in the list, or if you're interested, I can compare two papers for you. \
                           \n   If you want to compare two papers, use the selection boxes below, or type, e.g. 'Compare Paper 1 and Paper 2'.\
                           \n   If you want to see a summary of one paper, type, e.g. 'Summarize Paper 3'.""" 

        for i, paper_id in enumerate(paper_ids):
            paper_data_request = requests.post(self.neo4j_url, data={'paper_id': paper_id})
            if (paper_data_request.status_code == 200):
                paper_data = paper_data_request.json()

                paper_list_text += f"""\n- Paper {i+1}: {paper_data['title']}"""

        dispatcher.utter_message(text=paper_list_text)
        return [SlotSet("cluster_level", 0), SlotSet("last_cluster_index", 0), SlotSet("selected_clusters", None), SlotSet('cluster_level_change', False)]


class ActionGetPaperSummarization(Action):
    """
    Action to retrieve and display the TL;DR summary from the neo4j database of a selected paper in a Rasa chatbot.

    This action fetches the summary of a paper from the neo4j service and presents it to the user.
    """

    def __init__(self):
        self.neo4j_url = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_paper_details"
    
    def name(self) -> Text:
        return "action_get_paper_summarization"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        paper_id = tracker.get_slot('summarized_paper')
        paper_list = tracker.get_slot('final_cluster_paper_list')

        paper_data_request = requests.post(self.neo4j_url, data={'paper_id': paper_list[int(paper_id) - 1]})
        if (paper_data_request.status_code == 200):
            paper_data = paper_data_request.json()
            paper_summarization = paper_data['tldr']
            paper_title = paper_data['title']

            summarization_text = f"Here is the TL;DR summary of Paper {paper_id}: {paper_title}.\
            \n📌 {paper_summarization}\
            \nIf you are interested in this paper, I can provide you a link to it."
            dispatcher.utter_message(text=summarization_text)

        return []


class ActionPaperComparison(Action):
    """
    Action to retrieve and display the comparison of two selected papers in a Rasa chatbot.

    This action compares the selected papers with the comparison option which the user chose.
    The comparison results come from the LLM response
    """

    def __init__(self):
        self.neo4j_url = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_paper_details"

        self.paper_comparison_objective_url = os.getenv("paper_comparison_uri", "http://localhost:6000") + "/generate_comparison/objective"
        self.paper_comparison_results_url = os.getenv("paper_comparison_uri", "http://localhost:6000") + "/generate_comparison/results"
        self.paper_comparison_tldr_url = os.getenv("paper_comparison_uri", "http://localhost:6000") + "/generate_comparison/tldr"

    def name(self) -> Text:
        return "action_paper_comparison"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        print("INTENT: ", tracker.latest_message['intent'].get('name'))

        if (tracker.latest_message['intent'].get('name') not in ['compare_objective_papers_intent', 'compare_results_papers_intent', 'compare_overview_papers_intent']):

            # paper_ids = tracker.get_slot('final_cluster_paper_list')

            paper_a_id = tracker.get_slot('paper_a')
            paper_b_id = tracker.get_slot('paper_b')            

            return([SlotSet('paper_a', paper_a_id), SlotSet('paper_b', paper_b_id)])
        else:

            paper_list = tracker.get_slot('final_cluster_paper_list')
            
            paper_a_id = tracker.get_slot('paper_a')
            paper_b_id = tracker.get_slot('paper_b')

            paper_a_data_request = requests.post(self.neo4j_url, data={'paper_id': paper_list[int(paper_a_id) - 1]})
            paper_b_data_request = requests.post(self.neo4j_url, data={'paper_id': paper_list[int(paper_b_id) - 1]})

            if (paper_a_data_request.status_code == 200) and (paper_b_data_request.status_code == 200):
                paper_a_data = paper_a_data_request.json()
                paper_b_data = paper_b_data_request.json()

                paper_a_title = paper_a_data['title']
                paper_b_title = paper_b_data['title']

                paper_a_objective = paper_a_data['objective']
                paper_b_objective = paper_b_data['objective']

                paper_a_results = paper_a_data['results']
                paper_b_results = paper_b_data['results']

                paper_a_tldr = paper_a_data['tldr']
                paper_b_tldr = paper_b_data['tldr']

            if (tracker.latest_message['intent'].get('name') == 'compare_objective_papers_intent'):
                    comparison_request = requests.post(self.paper_comparison_objective_url, timeout=600, json={
                        'paper_1_obj': paper_a_objective,
                        'paper_2_obj': paper_b_objective,
                        'id_a': paper_a_id,
                        'id_b': paper_b_id,})
                    
                    if (comparison_request.status_code == 200):
                        comparison_data = comparison_request.json()

                        comparison_text = (f"Comparison between Paper {paper_a_id}: **{paper_a_title}** and Paper {paper_b_id}: **{paper_b_title}:**\n"
                                            + f"\n**Objective:**\n{comparison_data['comparison']}\n")

                        dispatcher.utter_message(text=comparison_text)
                        return []
                    
                    else:
                        print("Comparison request failed")
                        dispatcher.utter_message(text="Comparison request failed, please try again later")
                        return []
                    
            elif(tracker.latest_message['intent'].get('name') == 'compare_results_papers_intent'):
                    comparison_request = requests.post(self.paper_comparison_results_url, timeout=600, json={
                        'paper_1_res': paper_a_results,
                        'paper_2_res': paper_b_results,
                        'id_a': paper_a_id,
                        'id_b': paper_b_id,})
                    
                    if (comparison_request.status_code == 200):
                        comparison_data = comparison_request.json()

                        comparison_text = (f"Comparison between Paper {paper_a_id}: **{paper_a_title}** and Paper {paper_b_id}: **{paper_b_title}**:\n"
                                            + f"\n**Results:**\n\n{comparison_data['comparison']}\n")

                        dispatcher.utter_message(text=comparison_text)
                        return []
                    
                    else:
                        print("Comparison request failed")
                        dispatcher.utter_message(text="Comparison request failed, please try again later")
                        return []      
                    
            elif(tracker.latest_message['intent'].get('name') == 'compare_overview_papers_intent'):
                    comparison_request = requests.post(self.paper_comparison_tldr_url, timeout=600, json={
                        'paper_1_tldr': paper_a_tldr,
                        'paper_2_tldr': paper_b_tldr,
                        'id_a': paper_a_id,
                        'id_b': paper_b_id})
                    
                    if (comparison_request.status_code == 200):
                        comparison_data = comparison_request.json()

                        comparison_text = (f"Comparison between Paper {paper_a_id}: **{paper_a_title}** and Paper {paper_b_id}: **{paper_b_title}**:\n"
                                            + f"\n**General overviews:**\n{comparison_data['comparison']}\n")

                        dispatcher.utter_message(text=comparison_text)
                        return []
                    
                    else:
                        print("Comparison request failed")
                        dispatcher.utter_message(text="Comparison request failed, please try again later")
                        return []

        return []


class ActionGetTopicDefinition(Action):
    """
    Action to get the definition of a topic from the neo4j service

    This action obtains the definiton of the topic which the user asked
    """

    def __init__(self):
        self.neo4j_url = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_topic_definition"

    def name(self) -> Text:
        return "action_get_topic_definition"

    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        topic = tracker.get_slot('topic_name')

        r = requests.post(self.neo4j_url, data={'topic_name': topic})
        if r.status_code == 200:
            res = r.json()

            topic_definition = res["topic_definition"]
            topic_name = res["topic_name"]
            
            if topic_definition == "":
                result_str = f"We don't have an available definition for {topic_name}. Try to search for topics related to Natural Language Processing. You could also ask for the taxonomy."
                dispatcher.utter_message(text=result_str)
            else:
                # if the topic is the main topic of NLP
                if topic_name in main_topic_list:
                    result_str = f"""Sure! You'd like to know about the {topic_name}. It's a fascinating field of study in NLP domain, and I've provided you with a definition of it:\
                    \n📌 {topic_definition}\
                    \nIf you are interested in this topic of NLP, feel free to ask me about its subtopics."""

                # if the user asked about the subtopic
                else:
                    result_str = f"""Sure! You'd like to know about the {topic_name}. It's a fascinating field of study in NLP domain, and I've provided you with a definition of it:\
                    \n📌 {topic_definition}\
                    \nIf you are interested in this subtopic, feel free to ask me about its clusters."""
       
                dispatcher.utter_message(text=result_str)
                return [SlotSet('topic_name', topic_name), SlotSet('cluster_level', 0)]
        
        dispatcher.utter_message(text=f"The communication to the database seems to be unstable, please try it again")
        return []


class ActionGetTopicChildren(Action):
    """
    Action to get the subtopics of a topic from the neo4j service

    This action shows the subtopics when the user asked about it
    """

    def __init__(self):
        self.neo4j_url = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_children_nodes"

    def name(self) -> Text:
        return "action_get_topic_children"

    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        topic = tracker.get_slot('topic_name')
        
        r = requests.post(self.neo4j_url, data={'topic_name': topic})
        if r.status_code == 200:
            resp = r.json()
            dispatcher.utter_message(text=resp)
            return []
        
        dispatcher.utter_message(text=f"The communication to the database seems to be unstable, please try it again")
        return []


class ActionGetTopicParent(Action):
    """
    Action to get the parent topic of a topic from the neo4j service
    """

    def __init__(self):
        self.neo4j_url = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_parent_nodes"

    def name(self) -> Text:
        return "action_get_topic_parent"

    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        topic = tracker.get_slot('topic_name')
        
        r = requests.post(self.neo4j_url, data={'topic_name': topic})
        if r.status_code == 200:
            resp = r.json()
            dispatcher.utter_message(text=resp)
            return []

        dispatcher.utter_message(text=f"The communication to the database seems to be unstable, please try it again")
        return []

    # Just for back up if we want to be more dynamic


class ActionGetNLPTaxonomy(Action):
    """
    Action to get the topics overview in NLP from the neo4j service
    """

    def __init__(self):
        self.neo4j_url = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_nlp_taxonomy"

    def name(self) -> Text:
        return "action_get_nlp_taxonomy"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        r = requests.post(self.neo4j_url)
        if r.status_code == 200:
            resp = r.json()
            dispatcher.utter_message(text=resp)
            return []
        
        dispatcher.utter_message(text=f"The communication to the database seems to be unstable, please try it again")
        return []


class ActionPredictTopic(Action):
    """
    Action to predict the topic of a given text
    """
    
    def __init__(self):
        self.predictor_url = os.getenv("topic_recognition_uri", "http://localhost:5500/predict")
        self.neo4j_url = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_topic_definition"

    def name(self) -> Text:
        return "action_predict_topic"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        query = tracker.latest_message.get('text')

        r = requests.post(self.predictor_url, data={'query': query})
        if r.status_code == 200:
            topic, subtopic = r.json()['topic'], r.json()['subtopic']

            if subtopic == "random":
                dispatcher.utter_message(text=f"Sorry I couldn't find a topic for you. Please try paraphrasing your question.")
                return []

            r_def = requests.post(self.neo4j_url, data={'topic_name': subtopic})
            if r_def.status_code == 200:
                res = r_def.json()

                topic_definition = res["topic_definition"]

                # User didnt set a topic yet
                if not tracker.get_slot('topic_is_set'):
                    if topic == "random":
                        subtopic = "random"
                        random = True
                    else:
                        random = False
                predict_response = f"""From the prompts you've entered, I predict that you might be interested in the {subtopic}.\
                I have provided the definition of this subtopic for you right here:\
                \n📌 {topic_definition}\
                \nIs this the topic you were looking to explore? If not, could you please provide more details about your specific interest? This will help me assist you better"""

                dispatcher.utter_message(text=predict_response)
                return [SlotSet('topic_predicted_as_random', random), SlotSet('predicted_topic_name', topic), SlotSet('predicted_subtopic_name', subtopic), SlotSet('topic_name', subtopic), SlotSet('cluster_level', 0)]

        else:
            dispatcher.utter_message(text=f"Models are not running, please try again later")
        return []


class ActionResetTopic(Action):
    """
    Action to reset the topic, it can be used when the user want to initialized the conversation.
    """

    def __init__(self):
        pass

    def name(self) -> Text:
        return "action_reset_topic"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        # if tracker.get_slot('predicted_topic_name') != "random":
        reset_prompt = "Ok, let's start again. Feel free to ask me any question you have regarding NLP and I can guide you to the correct topic to look for."
        dispatcher.utter_message(text=reset_prompt)

        #return [SlotSet('topic_is_set', False), SlotSet('predicted_topic_name', "random"), SlotSet('predicted_subtopic_name', "random")]
        return [Restarted()]


class ActionGetPaperURL(Action):
    """
    Action to get the paper URL which the user is interested in, it can be used to read the full text of the paper.
    """

    def __init__(self):
        self.neo4j_url = os.getenv("neo4j_service_uri", "http://localhost:5001") + "/get_paper_details"

    def name(self) -> Text:
        return "action_get_paper_url"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # find the last user intent out
        events = tracker.events
        user_events = [event for event in events if event['event'] == 'user']
        last_event = user_events[-2]
        intent_name = last_event['parse_data']['intent']['name']
        paper_list = tracker.get_slot('final_cluster_paper_list')
    
        if (intent_name == "paper_selection_for_summarization_intent"):
            paper_id = tracker.get_slot('summarized_paper')

            summarized_paper_data_request = requests.post(self.neo4j_url, data={'paper_id': paper_list[int(paper_id) - 1]})
            if (summarized_paper_data_request.status_code == 200):
                summarized_paper_data = summarized_paper_data_request.json()

                paper_title = summarized_paper_data["title"]
                paper_aclurl = summarized_paper_data["aclurl"]
                paper_arxivurl = summarized_paper_data["arxivurl"]

                if paper_aclurl:
                    paper_url = paper_aclurl
                elif paper_arxivurl:
                    paper_url = paper_arxivurl
                else:
                    paper_url = "No URL found"
                
                paper_url_text = f"""Sure! You can access the full text of Paper {paper_id}: {paper_title} with:\
                    \n {paper_url}\
                    \nHappy reading!"""

        else:
            paper_a_id = tracker.get_slot('paper_a')
            paper_b_id = tracker.get_slot('paper_b') 

            paper_url_text = ""

            paper_a_data_request = requests.post(self.neo4j_url, data={'paper_id': paper_list[int(paper_a_id) - 1]})
            paper_b_data_request = requests.post(self.neo4j_url, data={'paper_id': paper_list[int(paper_b_id) - 1]})
            if (paper_a_data_request.status_code == 200) and (paper_b_data_request.status_code == 200):
                paper_a_data = paper_a_data_request.json()
                paper_b_data = paper_b_data_request.json()

                paper_a_description = f"Paper {paper_a_id}"
                paper_b_description = f"Paper {paper_b_id}"
                paper_a_title = paper_a_data["title"]
                paper_b_title = paper_b_data["title"]
                paper_a_aclurl = paper_a_data["aclurl"]
                paper_b_aclurl = paper_b_data["aclurl"]
                paper_a_arxivurl = paper_a_data["arxivurl"]
                paper_b_arxivurl = paper_b_data["arxivurl"]

                if paper_a_aclurl:
                    paper_a_url = paper_a_aclurl
                else:
                    paper_a_url = paper_a_arxivurl

                if paper_b_aclurl:
                    paper_b_url = paper_b_aclurl
                else:
                    paper_b_url = paper_b_arxivurl

                paper_url_text = f"""Sure! You can access the full text of {paper_a_description}: {paper_a_title} with:\
                    \n{paper_a_url};\
                    \nand {paper_b_description}: {paper_b_title} with:\
                    \n{paper_b_url};\
                    \nHappy reading!"""
                
            
        dispatcher.utter_message(text=paper_url_text)
        return []
    