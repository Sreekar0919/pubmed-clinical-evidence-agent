#%%
from openai import OpenAI
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
from typing import List, Dict
import xml.etree.ElementTree as ET
import re

# key now read from the environment

client = OpenAI()

# Helper function to safely parse JSON from LLM function calls
def safe_json_loads(json_string: str) -> dict:
    """Safely parse JSON string, handling control characters and common formatting issues"""
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Attempting to clean control characters...")
        # Remove control characters that cause JSON parsing issues
        cleaned = re.sub(r'[\x00-\x1f\x7f]', '', json_string)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e2:
            print(f"Still failed after cleaning: {e2}")
            # Try to fix common JSON issues
            try:
                # Fix missing quotes around keys
                cleaned = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
                # Fix missing quotes around string values
                cleaned = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9\s]*[a-zA-Z0-9])\s*([,}])', r': "\1"\2', cleaned)
                # Fix trailing commas
                cleaned = re.sub(r',\s*}', '}', cleaned)
                cleaned = re.sub(r',\s*]', ']', cleaned)
                # Fix unescaped quotes in strings
                cleaned = re.sub(r'([^\\])"([^"]*)"([^"]*)"', r'\1"\2\\"\3"', cleaned)
                # Fix newlines in strings
                cleaned = re.sub(r'\n', '\\n', cleaned)
                # Fix carriage returns in strings
                cleaned = re.sub(r'\r', '\\r', cleaned)
                return json.loads(cleaned)
            except json.JSONDecodeError as e3:
                print(f"All JSON parsing attempts failed: {e3}")
                print(f"Problematic JSON string: {json_string[:200]}...")
                # Try to extract what we can from the JSON
                try:
                    # Look for key-value pairs and build a dict manually
                    result = {}
                    # Find all key-value pairs
                    pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]*)"', json_string)
                    for key, value in pairs:
                        result[key] = value
                    if result:
                        print(f"Extracted partial JSON: {result}")
                        return result
                except:
                    pass
                # Return empty dict as last resort
                return {}

#%%
# Step 1: Generate PubMed search prompt from user query
def generate_pubmed_prompt(query: str) -> str:
    """
    Transform natural language user query into optimized PubMed search string.
    
    Args:
        query (str): Natural language query from user
        
    Returns:
        str: Optimized PubMed search string with Boolean operators and field tags
    """
    
    # Define the function that will be called by the LLM
    def create_pubmed_search(search_string: str, explanation: str) -> dict:
        """Function that creates a PubMed search string - called by the LLM"""
        print(f"LLM called create_pubmed_search with: {search_string}")
        print(f"Explanation: {explanation}")
        return {"status": "success", "search_string": search_string}
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "create_pubmed_search",
            "description": "Create an optimized PubMed search string from a natural language query",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_string": {
                        "type": "string", 
                        "description": "The optimized PubMed search string with Boolean operators, field tags, and filters"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of how the search string was constructed"
                    }
                },
                "required": ["search_string", "explanation"]
            }
        }
    }]
    
    # Create the prompt for the LLM
    system_prompt = """You are a medical research expert who specializes in creating optimized PubMed search queries. 
    
    Your task is to transform a natural language query into a PubMed search string that will return the most relevant research.
    
    CRITICAL REQUIREMENTS:
    1. Start with BROAD, GENERAL terms to ensure results are found
    2. Use the most common, widely-used medical terminology
    3. Avoid overly specific or rare phrases that might return 0 results
    4. Focus on 2-3 core concepts maximum to avoid over-restriction
    5. Use field tags like [tiab] (title/abstract), [mesh] (MeSH terms)
    6. Add Boolean operators (AND, OR, NOT) appropriately
    
    SEARCH STRATEGY:
    - Begin with the most fundamental concept (e.g., "semantic variant PPA" or "svPPA")
    - Add 1-2 additional key concepts with OR operators for synonyms
    - Avoid complex multi-concept searches that are too restrictive
    - If the query is very specific, start broader and let the user refine
    
    Example transformations:
    - "latest research on svPPA" → "(svPPA OR "semantic variant primary progressive aphasia")[tiab]"
    - "Alzheimer treatment" → "(Alzheimer[tiab] OR "Alzheimer disease"[mesh]) AND (treatment[tiab] OR therapy[tiab])"
    - Complex query → Start with main disease/condition, then add 1-2 key concepts
    
    You MUST call the create_pubmed_search function with your optimized search string and explanation."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transform this query into a PubMed search string: {query}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_pubmed_search"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = create_pubmed_search(
                search_string=function_args["search_string"],
                explanation=function_args["explanation"]
            )
            
            return result["search_string"]
        else:
            print("LLM did not make a function call, using fallback")
            return f'"{query}"[tiab]'
        
    except Exception as e:
        print(f"Error generating PubMed prompt: {e}")
        return f'"{query}"[tiab]'

#%%
# Step 2: Search PubMed and return PMIDs
def search_pubmed(search_str: str, max_results: int = 10) -> List[str]:
    """
    Search PubMed using Entrez API and return list of PMIDs.
    
    Args:
        search_str (str): PubMed search string
        max_results (int): Maximum number of results to return (default: 10)
        
    Returns:
        List[str]: List of PubMed IDs (PMIDs)
    """
    
    # Define the function that will be called by the LLM
    def execute_pubmed_search(search_query: str, max_count: int, sort_by: str = "relevance", use_mesh: bool = True) -> dict:
        """Function that executes PubMed search - called by the LLM"""
        print(f"LLM called execute_pubmed_search with query: {search_query}, max: {max_count}, sort: {sort_by}, use_mesh: {use_mesh}")
        
        # NCBI E-utilities base URL
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        try:
            # Use esearch to get PMIDs with enhanced relevance parameters
            search_url = f"{base_url}esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": search_query,
                "retmax": max_count * 2,  # Get more results initially for better relevance filtering
                "retmode": "json",
                "sort": sort_by,
                "field": "relevance" if sort_by == "relevance" else None,  # Emphasize relevance
                "reldate": None,  # Don't limit by date to get most relevant regardless of age
                "datetype": None
            }
            
            # Remove None values
            search_params = {k: v for k, v in search_params.items() if v is not None}
            
            response = requests.get(search_url, params=search_params)
            response.raise_for_status()
            
            search_data = response.json()
            
            # Extract PMIDs from the response
            if "esearchresult" in search_data and "idlist" in search_data["esearchresult"]:
                pmids = search_data["esearchresult"]["idlist"]
                count = search_data["esearchresult"].get("count", "0")
                
                # Take only the requested number of results (most relevant)
                pmids = pmids[:max_count]
                
                print(f"Found {count} total results, returning top {len(pmids)} most relevant PMIDs")
                return {
                    "status": "success",
                    "pmids": pmids,
                    "total_count": count,
                    "returned_count": len(pmids),
                    "sort_method": sort_by
                }
            else:
                return {
                    "status": "error",
                    "message": "No results found or unexpected response format",
                    "pmids": []
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Search error: {str(e)}",
                "pmids": []
            }
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "execute_pubmed_search",
            "description": "Execute a PubMed search and return PMIDs, prioritizing relevance over recency",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The PubMed search query to execute"
                    },
                    "max_count": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 10)"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "date"],
                        "description": "Sort results by relevance (recommended) or date. Use relevance to get the most relevant results regardless of publication date."
                    },
                    "use_mesh": {
                        "type": "boolean",
                        "description": "Whether to consider MeSH terms for relevance (default: true)"
                    }
                },
                "required": ["search_query", "max_count"]
            }
        }
    }]
    
    # Create the prompt for the LLM
    system_prompt = """You are a PubMed search expert. Your task is to execute a PubMed search using the provided search string.
    
    CRITICAL: Always prioritize RELEVANCE over recency. The goal is to find the most relevant research for the query, not necessarily the most recent.
    
    Guidelines:
    1. Use "relevance" as the sort method to get the most relevant results
    2. Don't limit by date - relevant research can be from any time period
    3. Focus on finding papers that directly address the research question
    4. Consider that seminal or highly cited papers may be older but more relevant
    
    You should call the execute_pubmed_search function with the search query and appropriate parameters.
    For research queries, always use sort_by="relevance" to get the most relevant results."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Execute this PubMed search: {search_str} (max {max_results} results, prioritize relevance)"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "execute_pubmed_search"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = execute_pubmed_search(
                search_query=function_args["search_query"],
                max_count=function_args.get("max_count", max_results),
                sort_by=function_args.get("sort_by", "relevance"),  # Default to relevance
                use_mesh=function_args.get("use_mesh", True)
            )
            
            return result["pmids"]
        else:
            print("LLM did not make a function call, using fallback")
            return []
        
    except Exception as e:
        print(f"Error in PubMed search: {e}")
        return []

#%%
# Step 3: Fetch PubMed abstracts and metadata
def fetch_pubmed_abstracts(pmids: List[str]) -> List[Dict]:
    """
    Fetch abstracts and metadata for given PMIDs using Entrez API.
    
    Args:
        pmids (List[str]): List of PubMed IDs (PMIDs)
        
    Returns:
        List[Dict]: List of dictionaries containing article metadata
    """
    
    # Define the function that will be called by the LLM
    def retrieve_abstracts(pmid_list: List[str], include_authors: bool = True, include_journal: bool = True) -> dict:
        """Function that retrieves abstracts - called by the LLM"""
        print(f"LLM called retrieve_abstracts for {len(pmid_list)} PMIDs")
        
        if not pmid_list:
            return {"status": "error", "message": "No PMIDs provided", "abstracts": []}
        
        # NCBI E-utilities base URL
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        try:
            # Use efetch to get article details
            fetch_url = f"{base_url}efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmid_list),
                "retmode": "xml",
                "rettype": "abstract"
            }
            
            response = requests.get(fetch_url, params=fetch_params)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            abstracts = []
            
            # Extract article information from XML
            for article in root.findall(".//PubmedArticle"):
                try:
                    # Extract PMID
                    pmid = article.find(".//PMID")
                    pmid_text = pmid.text if pmid is not None else "Unknown"
                    
                    # Extract title
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No title available"
                    
                    # Extract abstract
                    abstract_elem = article.find(".//AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else "No abstract available"
                    
                    # Extract authors (if requested)
                    authors = []
                    if include_authors:
                        author_list = article.find(".//AuthorList")
                        if author_list is not None:
                            for author in author_list.findall(".//Author"):
                                last_name = author.find("LastName")
                                first_name = author.find("ForeName")
                                if last_name is not None and first_name is not None:
                                    authors.append(f"{first_name.text} {last_name.text}")
                                elif last_name is not None:
                                    authors.append(last_name.text)
                    
                    # Extract journal information (if requested)
                    journal = "Unknown journal"
                    if include_journal:
                        journal_elem = article.find(".//Journal/Title")
                        journal = journal_elem.text if journal_elem is not None else "Unknown journal"
                    
                    # Extract publication date
                    pub_date = article.find(".//PubDate")
                    year = "Unknown"
                    if pub_date is not None:
                        year_elem = pub_date.find("Year")
                        if year_elem is not None:
                            year = year_elem.text
                    
                    # Create article dictionary
                    article_data = {
                        "pmid": pmid_text,
                        "title": title,
                        "abstract": abstract,
                        "authors": authors,
                        "journal": journal,
                        "year": year
                    }
                    
                    abstracts.append(article_data)
                    
                except Exception as e:
                    print(f"Error parsing article: {e}")
                    continue
            
            print(f"Successfully retrieved {len(abstracts)} abstracts")
            return {
                "status": "success",
                "abstracts": abstracts,
                "retrieved_count": len(abstracts),
                "requested_count": len(pmid_list)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Retrieval error: {str(e)}",
                "abstracts": []
            }
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "retrieve_abstracts",
            "description": "Retrieve abstracts and metadata for given PMIDs",
            "parameters": {
                "type": "object",
                "properties": {
                    "pmid_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of PubMed IDs to retrieve abstracts for"
                    },
                    "include_authors": {
                        "type": "boolean",
                        "description": "Whether to include author information (default: true)"
                    },
                    "include_journal": {
                        "type": "boolean",
                        "description": "Whether to include journal information (default: true)"
                    }
                },
                "required": ["pmid_list"]
            }
        }
    }]
    
    # Create the prompt for the LLM
    system_prompt = """You are a PubMed data retrieval expert. Your task is to retrieve abstracts and metadata for the provided PMIDs.
    
    You should call the retrieve_abstracts function with the PMID list and appropriate parameters.
    Consider whether to include authors and journal information based on the retrieval needs."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Retrieve abstracts for these PMIDs: {pmids}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "retrieve_abstracts"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = retrieve_abstracts(
                pmid_list=function_args["pmid_list"],
                include_authors=function_args.get("include_authors", True),
                include_journal=function_args.get("include_journal", True)
            )
            
            return result["abstracts"]
        else:
            print("LLM did not make a function call, using fallback")
            return []
        
    except Exception as e:
        print(f"Error fetching abstracts: {e}")
        return []

#%%
# Step 4: Summarize abstracts based on original query
def summarize_abstracts(abstracts: List[Dict], original_query: str) -> str:
    """
    Summarize abstracts using LLM based on original query with inline citations and quotes.
    
    Args:
        abstracts (List[Dict]): List of article dictionaries
        original_query (str): Original user query
        
    Returns:
        str: Summary of the abstracts with inline citations and quotes
    """
    
    # Check if no abstracts were found
    if not abstracts:
        return "No research articles were found for the given query. This could be due to:\n1. The search terms being too specific or restrictive\n2. Limited research on this particular topic\n3. The need to use broader or alternative search terms\n\nConsider trying a more general search or rephrasing the query with more common medical terminology."
    
    # Define the function that will be called by the LLM
    def create_summary_with_citations(summary_text: str, key_findings: List[Dict], confidence_level: str) -> dict:
        """Function that creates a summary with citations and quotes - called by the LLM"""
        print(f"LLM called create_summary_with_citations with confidence: {confidence_level}")
        print(f"Key findings with citations: {len(key_findings)} items identified")
        
        return {
            "status": "success",
            "summary": summary_text,
            "key_findings": key_findings,
            "confidence": confidence_level
        }
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "create_summary_with_citations",
            "description": "Create a comprehensive summary of research abstracts with inline citations and direct quotes",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_text": {
                        "type": "string",
                        "description": "The main summary text with inline citations [1], [2], etc. and direct quotes from abstracts"
                    },
                    "key_findings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "finding": {"type": "string"},
                                "citation": {"type": "string"},
                                "quote": {"type": "string"},
                                "pmid": {"type": "string"}
                            },
                            "required": ["finding", "citation", "quote", "pmid"]
                        },
                        "description": "List of key findings with their citations and supporting quotes"
                    },
                    "confidence_level": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Confidence level in the summary based on available evidence"
                    }
                },
                "required": ["summary_text", "key_findings", "confidence_level"]
            }
        }
    }]
    
    # Prepare abstracts for the LLM with numbered references
    abstracts_text = ""
    for i, article in enumerate(abstracts, 1):
        abstracts_text += f"\n[{i}] Article {i}:\n"
        abstracts_text += f"PMID: {article['pmid']}\n"
        abstracts_text += f"Title: {article['title']}\n"
        abstracts_text += f"Authors: {', '.join(article['authors'][:3])}\n"
        abstracts_text += f"Journal: {article['journal']} ({article['year']})\n"
        abstracts_text += f"Abstract: {article['abstract']}\n"
        abstracts_text += "-" * 50 + "\n"
    
    # Create the prompt for the LLM
    system_prompt = """You are a medical research expert who specializes in summarizing scientific literature with proper citations.

    Your task is to analyze the provided research abstracts and create a comprehensive summary that directly answers the original user query.
    
    CRITICAL REQUIREMENTS:
    1. Use inline citations [1], [2], [3], etc. for EVERY fact or claim
    2. Include DIRECT QUOTES from the abstracts to support key findings - use quotation marks
    3. Focus on answering the specific question asked
    4. Identify key findings and trends across the studies
    5. Note any conflicting evidence or limitations
    6. Provide evidence-backed statements with specific citations
    7. Assess the confidence level based on the quality and quantity of evidence
    8. Create COMPREHENSIVE summaries that cover multiple aspects of the research
    9. Aim for substantial length (300-800 words) to provide thorough coverage
    10. Include both supporting and contradictory findings when present
    
    FORMATTING REQUIREMENTS:
    - Use [1], [2], [3], etc. for inline citations
    - Include direct quotes in quotation marks with citations, e.g., "quote here" [1]
    - Structure the summary to flow logically with multiple paragraphs
    - Make sure every claim is supported by at least one citation
    - Include at least 3-5 direct quotes from the abstracts
    - Each major finding should have a supporting quote
    - Cover multiple aspects: mechanisms, clinical features, diagnostic criteria, outcomes, etc.
    - Provide context and background when relevant
    
    CONTENT REQUIREMENTS:
    - Start with a broad overview of the topic
    - Include specific mechanisms or pathophysiology when discussed
    - Cover clinical manifestations and diagnostic features
    - Address treatment approaches if mentioned
    - Discuss prognosis and outcomes
    - Note any controversies or conflicting evidence
    - Include relevant anatomical or neuroimaging findings
    - Mention study limitations and future research directions
    
    EXAMPLE FORMAT (like the provided example):
    "Frontotemporal dementia patients, including those with the semantic variant of primary progressive aphasia (svPPA), exhibit degradation in semantic memory due to atrophy in the anterior temporal lobe (ATL), which interferes with hierarchical semantic categorization" [2][4][5]. The ATL is proposed to be a functionally unitary 'semantic hub' that supports both social and non-social semantic knowledge, indicating that semantic degradation in svPPA might reflect a disintegration of complex conceptual networks within this region [4].
    
    You MUST call the create_summary_with_citations function with your comprehensive analysis."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Original query: {original_query}\n\nResearch abstracts:\n{abstracts_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_summary_with_citations"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = create_summary_with_citations(
                summary_text=function_args["summary_text"],
                key_findings=function_args["key_findings"],
                confidence_level=function_args["confidence_level"]
            )
            
            return result["summary"]
        else:
            print("LLM did not make a function call, using fallback")
            return "Unable to generate summary due to technical issues."
        
    except Exception as e:
        print(f"Error summarizing abstracts: {e}")
        return "Error occurred during summarization."

#%%
# Step 5: Format references for the articles
def format_references(abstracts: List[Dict]) -> str:
    """
    Format references for the articles in a standardized format.
    
    Args:
        abstracts (List[Dict]): List of article dictionaries
        
    Returns:
        str: Formatted reference list
    """
    
    # Check if no abstracts were found
    if not abstracts:
        return "No references available - no articles were found for the given query."
    
    # Define the function that will be called by the LLM
    def format_reference_list(references: List[str], format_style: str = "APA") -> dict:
        """Function that formats references - called by the LLM"""
        print(f"LLM called format_reference_list with style: {format_style}")
        print(f"Formatted {len(references)} references")
        
        return {
            "status": "success",
            "formatted_references": references,
            "style": format_style,
            "count": len(references)
        }
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "format_reference_list",
            "description": "Format a list of references in a standardized citation style",
            "parameters": {
                "type": "object",
                "properties": {
                    "references": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of formatted references"
                    },
                    "format_style": {
                        "type": "string",
                        "enum": ["APA", "MLA", "Chicago", "Vancouver"],
                        "description": "Citation format style to use (default: APA)"
                    }
                },
                "required": ["references"]
            }
        }
    }]
    
    # Prepare article data for the LLM
    articles_data = []
    for i, article in enumerate(abstracts, 1):
        articles_data.append({
            "number": i,
            "pmid": article["pmid"],
            "title": article["title"],
            "authors": article["authors"],
            "journal": article["journal"],
            "year": article["year"]
        })
    
    # Create the prompt for the LLM
    system_prompt = """You are a bibliographic expert who specializes in formatting academic references.
    
    Your task is to format the provided article information into a standardized reference list.
    
    Guidelines:
    1. Use consistent formatting throughout
    2. Include all necessary bibliographic information
    3. Follow standard citation conventions
    4. Number the references sequentially [1], [2], [3], etc.
    5. Include PMID when available
    6. Make sure the numbering matches the inline citations in the summary
    
    You MUST call the format_reference_list function with your formatted references."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Format references for these articles: {json.dumps(articles_data, indent=2)}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "format_reference_list"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = format_reference_list(
                references=function_args["references"],
                format_style=function_args.get("format_style", "APA")
            )
            
            return "\n".join(result["formatted_references"])
        else:
            print("LLM did not make a function call, using fallback")
            return "Unable to format references due to technical issues."
        
    except Exception as e:
        print(f"Error formatting references: {e}")
        return "Error occurred during reference formatting."

#%%
# Main pipeline function that orchestrates all 5 LLM calls
def run_research_pipeline(user_query: str, max_results: int = 10) -> Dict:
    """
    Run the complete research pipeline with 5 LLM function calls.
    
    Args:
        user_query (str): Original user query
        max_results (int): Maximum number of results to retrieve
        
    Returns:
        Dict: Complete pipeline results
    """
    
    print("=== Research Pipeline with LLM Function Calling ===")
    print(f"Query: {user_query}")
    print(f"Max results: {max_results}")
    print()
    
    # Step 1: Generate PubMed search prompt (LLM Call #1)
    print("Step 1: LLM generating PubMed search prompt...")
    search_string = generate_pubmed_prompt(user_query)
    print(f"Generated search: {search_string}")
    print()
    
    # Step 2: Search PubMed (LLM Call #2)
    print("Step 2: LLM executing PubMed search...")
    pmids = search_pubmed(search_string, max_results)
    print(f"Found PMIDs: {pmids}")
    print()
    
    # Fallback: If no results found, try a broader search
    if not pmids:
        print("No results found with initial search. Trying broader fallback search...")
        # Extract key terms and create a simpler search
        fallback_search = create_fallback_search(user_query)
        print(f"Fallback search: {fallback_search}")
        pmids = search_pubmed(fallback_search, max_results)
        print(f"Fallback search found PMIDs: {pmids}")
        print()
    
    # Step 3: Fetch abstracts (LLM Call #3)
    print("Step 3: LLM retrieving abstracts...")
    abstracts = fetch_pubmed_abstracts(pmids)
    print(f"Retrieved {len(abstracts)} abstracts")
    print(abstracts)
    print()
    
    # Step 4: Summarize abstracts (LLM Call #4)
    print("Step 4: LLM summarizing abstracts...")
    summary = summarize_abstracts(abstracts, user_query)
    print(f"Generated summary ({len(summary)} characters)")
    print(summary)
    print()
    
    # Step 5: Format references (LLM Call #5)
    print("Step 5: LLM formatting references...")
    references = format_references(abstracts)
    print(f"Formatted {len(abstracts)} references")
    print()
    
    # Compile results
    results = {
        "original_query": user_query,
        "search_string": search_string,
        "pmids": pmids,
        "abstracts": abstracts,
        "summary": summary,
        "references": references,
        "llm_calls": 5
    }
    
    print("=== Pipeline Complete ===")
    print(f"Total LLM function calls: {results['llm_calls']}")
    print(f"Articles processed: {len(abstracts)}")
    print()
    
    return results

def create_fallback_search(query: str) -> str:
    """
    Create a broader fallback search when the initial search returns no results.
    
    Args:
        query (str): Original user query
        
    Returns:
        str: Broader search string
    """
    
    # Define the function that will be called by the LLM
    def create_broad_search(search_string: str, explanation: str) -> dict:
        """Function that creates a broad search string - called by the LLM"""
        print(f"LLM called create_broad_search with: {search_string}")
        print(f"Explanation: {explanation}")
        return {"status": "success", "search_string": search_string}
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "create_broad_search",
            "description": "Create a broad PubMed search string when the initial search returns no results",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_string": {
                        "type": "string", 
                        "description": "A broad PubMed search string using only the most fundamental concepts"
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Brief explanation of how the broad search string was constructed"
                    }
                },
                "required": ["search_string", "explanation"]
            }
        }
    }]
    
    # Create the prompt for the LLM
    system_prompt = """You are a medical research expert who needs to create a BROAD search when the initial search returned no results.
    
    Your task is to extract the most fundamental, widely-used medical terms from the query and create a simple search that will definitely return results.
    
    CRITICAL REQUIREMENTS:
    1. Use ONLY the most basic, fundamental medical terms
    2. Focus on 1-2 core concepts maximum
    3. Use the most common synonyms and variations
    4. Avoid complex or specific terminology
    5. Use broad field tags like [tiab] or [mesh]
    6. Use OR operators liberally for synonyms
    
    EXAMPLES:
    - Complex query about "progressive lexical-semantic degradation in semantic variant PPA" → "(svPPA OR "semantic variant primary progressive aphasia")[tiab]"
    - Query about "heteromodal conceptual hubs" → "(conceptual OR semantic)[tiab] AND (temporal lobe OR brain)[tiab]"
    - Very specific query → Start with the main disease/condition name only
    
    You MUST call the create_broad_search function with your broad search string and explanation."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Create a broad search for this query that returned no results: {query}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_broad_search"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = create_broad_search(
                search_string=function_args["search_string"],
                explanation=function_args["explanation"]
            )
            
            return result["search_string"]
        else:
            print("LLM did not make a function call, using fallback")
            # Extract basic terms manually
            basic_terms = extract_basic_terms(query)
            return f'"{basic_terms}"[tiab]'
        
    except Exception as e:
        print(f"Error creating fallback search: {e}")
        # Extract basic terms manually
        basic_terms = extract_basic_terms(query)
        return f'"{basic_terms}"[tiab]'

def extract_basic_terms(query: str) -> str:
    """
    Extract basic medical terms from a query as a fallback.
    
    Args:
        query (str): Original user query
        
    Returns:
        str: Basic search terms
    """
    # Common medical terms to look for
    common_terms = [
        "PPA", "svPPA", "semantic variant", "primary progressive aphasia",
        "Alzheimer", "dementia", "temporal lobe", "semantic",
        "cognitive", "neurological", "brain", "neurodegenerative"
    ]
    
    query_lower = query.lower()
    found_terms = []
    
    for term in common_terms:
        if term.lower() in query_lower:
            found_terms.append(term)
    
    if found_terms:
        return " OR ".join(f'"{term}"' for term in found_terms[:2])  # Use max 2 terms
    else:
        # If no common terms found, use the first few words
        words = query.split()[:3]
        return " ".join(words)

#%%
# Test the complete pipeline
if __name__ == "__main__":
    # Test with a sample query
    test_query = "To what extent does the progressive lexical-semantic degradation in semantic variant PPA reflect a disintegration of heteromodal conceptual hubs within the anterior temporal lobe, and how might this selectively compromise hierarchical semantic categorization while sparing syntactic scaffolding and non-verbal cognition?"
    
    print("=== DIRECT LLM RESPONSE (for comparison) ===")
    print(f"Query: {test_query}")
    print()
    
    # Direct LLM prompt without any structured pipeline
    direct_prompt = f"""You are a medical research expert. Please answer this question based on your knowledge. Write it as a summary.

{test_query}"""
    
    try:
        direct_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical research expert specializing in neurology and cognitive disorders."},
                {"role": "user", "content": direct_prompt}
            ]
        )
        
        print("Direct LLM Response:")
        print(direct_response.choices[0].message.content)
        print()
        print("=" * 80)
        print()
        
        # Now run the structured pipeline for comparison
        print("=== STRUCTURED PIPELINE RESPONSE ===")
        results = run_research_pipeline(test_query, max_results=5)
        
        # Display results
        print("=== Final Results ===")
        print(f"Summary: {results['summary']}")
        print()
        print("References:")
        print(results['references'])
        
    except Exception as e:
        print(f"Error in direct LLM response: {e}")
        # Still run the pipeline
        print("=== STRUCTURED PIPELINE RESPONSE ===")
        results = run_research_pipeline(test_query, max_results=5)
        
        # Display results
        print("=== Final Results ===")
        print(f"Summary: {results['summary']}")
        print()
        print("References:")
        print(results['references'])

# %%

