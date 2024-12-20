import re
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from data import PostGreDatabaseHandler, MilvusDatabaseHandler

class QueryProcessor:
    def __init__(self):
        # Initialize LLM model and tokenizer
        self.model_name = "facebook/bart-large-cnn"  # Replace with your chosen model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Define intent patterns
        self.intent_patterns = {
            "DOC_SUMMARY": r"(?:give|show|get|what is the)?\s*summary\s+(?:of|for)?\s*document\s*(\d+\.jpg)",
            "DOC_ENTITIES": r"(?:give|show|get|what are the)?\s*(?:entities|pos|named entities|parts of speech)\s+(?:in|of|for|present in)?\s*document\s*(\d+\.jpg)",
            "ENTITY_DOCS": r"(?:give|show|get|what are the)?\s*documents\s+containing\s+entities\s+(?:like|such as)?\s*(\w+)",
            "TOPIC_DOCS": r"(?:give|show|get|what are the)?\s*documents\s+containing\s+information\s+about\s+(.*)",
            "SIMILAR_IMAGE_DOCS": r"(?:given this image|given image|with this image|using this image)?\s*(?:return|find|get|show)?\s*(?:documents containing similar images|similar images|documents with similar images|images like this)"
        }
        
        self.sqlconnection=PostGreDatabaseHandler("postgresql://postgres:12072024@localhost:5432/postgres")
        self.milvusconnection=MilvusDatabaseHandler("/milvus.db")

    def classify_intent(self, query: str) -> Tuple[str, Optional[Dict[str, str]]]:
        """Classify the intent of the user query and extract parameters."""
        query = query.lower().strip()
        
        for intent, pattern in self.intent_patterns.items():
            match = re.search(pattern, query)
            if match:
                params = self._extract_parameters(intent, match)
                return intent, params
                
        return "UNKNOWN", None

    def _extract_parameters(self, intent: str, match) -> Dict[str, str]:
        """Extract parameters based on the intent and regex match."""
        params = {}
        
        if intent == "DOC_SUMMARY":
            # Extract the complete filename (numbers.jpg)
            params["doc_id"] = match.group(1)
            
        elif intent == "DOC_ENTITIES":
            # Extract the complete filename (numbers.jpg)
            params["doc_id"] = match.group(1)
            params["entity_type"] = "entities" if "entities" in match.group(0) else "pos"
            
        elif intent == "ENTITY_DOCS":
            params["entity_type"] = match.group(1).upper()
            
        elif intent == "TOPIC_DOCS":
            params["topic"] = match.group(1).strip()
            
        elif intent == "SIMILAR_IMAGE_DOCS":
            params["path"] = ""
            
        return params

    async def execute_db_query(self, intent: str, params: Dict[str, str]) -> Dict[str, Any]:
        """Placeholder for database query execution."""
        if intent == "DOC_SUMMARY":
            result=self.sqlconnection.postgres_file_content(params['doc_id'])
            if not result:
                return {'content':[]}
            return {"content":[result['content']]}
            
        elif intent == "DOC_ENTITIES":
            result=self.sqlconnection.postgres_file_entities(params['doc_id'])
            if not result:
               return {'entities': [], 'label': []}
            x = {'entities':[result[i]['entity']for i in range(5)],
               'label':[result[i]['label']for i in range(5)]}
            return x
              
        elif intent == "ENTITY_DOCS":
            result=self.sqlconnection.postgres_entities_file(params['entity_type'])
            if not result:
                return {'file name':[]}
            return {'file name':[result[i]['file_name']for i in range(len(result))]}

        elif intent == "TOPIC_DOCS":
            result=self.milvusconnection.milvuskeyword_search(params['topic'])
            if not result:
              return {'file name':[]}
            return {'file name':[x[i]['file_name']for i in range(len(x))]}

        elif intent == "SIMILAR_IMAGE_DOCS":
            result=self.milvusconnection.milvusimage_search('1002403141_otsu_region_2421.png')
            if not result:
                return {'similar_docs':[]}
            return {'document name':[result[i]['document_name'] for i in range(min(2,len(result)))]}

            
        return {"error": "Unknown intent"}

    def generate_llm_prompt(self, intent: str, db_results: Dict[str, Any], original_query: str) -> str:
        """Generate appropriate prompt for LLM based on intent and results."""
        match intent:

            case "DOC_SUMMARY":
                return f"""
                You are a skilled document analyst tasked with creating a comprehensive yet concise summary. Please analyze the following document content:
                {db_results['content']}
                Provide a summary that:
                1. Captures the main ideas and key points
                2. Maintains the original meaning and intent
                3. Preserves important details and supporting evidence
                4. Uses clear, professional language
                5. Follows a logical flow

                Format the summary in well-organized paragraphs. Include:
                - A brief overview of the document's main topic/purpose
                - The key arguments or findings
                - Important supporting details or examples
                - Any significant conclusions or implications

                If the document contains technical terms, numbers, or data, incorporate them accurately.
                Aim for a length that balances completeness with conciseness - typically 15-25% of the original length.
                If you're unsure about any part of the content, maintain factual accuracy by focusing on the clearly stated information.
                """
            
            case "DOC_ENTITIES":

                entity_label_pairs = [
                    f"- Entity: {entity}\n  Label: {label}"
                    for entity, label in zip(db_results['entities'], db_results['label'])
                    ]
                formatted_entities = "\n".join(entity_label_pairs)
                return f"""
                You are a skilled entity analyzer tasked with explaining the named entities found in the text. Here are the entities and their labels:
                {formatted_entities}
                
                Please provide:
                1. A clear explanation of each entity and its classification
                2. Any patterns or relationships between the entities
                3. Context about why these entities might be important
                4. Group similar entities together (e.g., all organizations, all works of art)

                Format your response as follows:
                - Start with a brief overview of the types of entities found
                - Group entities by their labels
                - For each entity, explain its significance and why it was classified as its given label
                - Note any interesting patterns or relationships between entities

                If there are multiple mentions of the same entity with slight variations (e.g., "ABC" and "ABC Family"), explain the relationship between these variations.

                Keep your response clear and professional, focusing on accuracy and meaningful insights about the entities and their roles.
                """
            
            case "ENTITY_DOCS":
                # Format the list of filenames
                files = "\n".join(f"- {filename}" for filename in db_results['file name'])
                return f"""
                You are a document retrieval specialist tasked with explaining the search results for entity-related documents. The following files contain the requested entity:
                {files}

                Please provide:
                1. A clear overview of the search results
                2. The number of documents found

                Format your response as follows:
                - Start with a summary of the search results (e.g., "Found X documents containing the requested entity")
                - List the documents in a clear, organized manner

                Keep your response:
                - Clear and professional
                - Focused on helping the user understand what documents are available
                - Organized in a way that makes the search results easy to understand
                """
            case "TOPIC_DOCS":
                # Format the list of document names
                topic_docs = "\n".join(f"- {doc}" for doc in db_results['documents'])
                return f"""
                You are a topic search specialist tasked with explaining document search results. The following analysis is based on:
                Original Query: {original_query}

                Documents found containing relevant information:
                {topic_docs}

                Please provide:
                1. A clear explanation of how these documents relate to the query topic
                2. The total number of relevant documents found
                3. Any patterns in the type of information available
                4. Recommendations for which documents might be most relevant to the query

                Format your response as follows:
                - Start with a restatement of the search query and what was found
                - List the relevant documents in order of likely importance
                - Explain why each document might contain relevant information
                - Suggest a reading order that would best address the original query

                Important considerations:
                - Focus on the connection between the documents and the query topic
                - Note if there appear to be different aspects of the topic covered
                - Highlight documents that seem most directly relevant to the query
                - Consider how the documents might complement each other in answering the query

                Keep your response:
                - Focused on helping the user understand why these documents were found
                - Clear about the relationship between the documents and the query
                - Organized to help the user efficiently find the information they need
                If you notice any patterns in how the topic is covered across documents, include these insights to help guide the user's reading.
                """
                
            case "SIMILAR_IMAGE_DOCS":
                # Format the list of filenames
                similar_files = "\n".join(f"- {filename}" for filename in db_results['file name'])
                return f"""
                You are an image search specialist tasked with explaining the results of a visual similarity search. The following images were found to be visually similar to the query image:
                {similar_files}

                Please provide:
                1. A clear overview of the search results
                2. The total number of similar images found
                3. Any patterns in the types of images discovered
                4. Recommendations for which images might be most relevant

                Format your response as follows:
                - Begin with a summary of the visual search results (e.g., "Found X visually similar images")
                - List the similar images in a clear, organized manner
                - If there are patterns in the image names or types, explain their significance
                - Provide context about why these images might be similar to the query image

                Important considerations:
                - Focus on explaining the potential visual similarities
                - Note if there appear to be groups or clusters of similar images
                - Highlight any particularly strong matches based on the file organization
                - Suggest which images might be most worth reviewing first

                Keep your response clear and professional, helping the user understand the visual similarity search results and their potential relevance.
                """

    async def generate_llm_response(self, prompt: str) -> str:
        """Generate response using the LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    async def process_query(self, query: str, image_path: Optional[str] = None) -> str:
        """Main method to process user query and generate response."""
        # 1. Classify intent and extract parameters
        intent, params = self.classify_intent(query)
        print("intent",intent)
        print("params",params)
        
        if intent == "UNKNOWN":
            return "I'm sorry, I couldn't understand your query. Please rephrase it."
            
        # 2. Check if image is required but not provided
        if intent == "SIMILAR_IMAGE_DOCS":
            params['path']=image_path
        # 3. Execute database query
        db_results = await self.execute_db_query(intent, params)
        print("db result",db_results)
        # 4. Generate LLM prompt
        prompt = self.generate_llm_prompt(intent, db_results, query)
        print("prompt",prompt)
        # 5. Generate final response
        response = await self.generate_llm_response(prompt)
        
        return response
