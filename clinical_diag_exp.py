#%%
from openai import OpenAI
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET
import re

# Import functions from simple_pubmed.py
from simple_pubmed import (
    safe_json_loads,
    generate_pubmed_prompt,
    search_pubmed,
    fetch_pubmed_abstracts,
    summarize_abstracts,
    format_references
)

# Remove the hardcoded API key and environment variable setting
# os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"] # Set up API

# Remove the global client - will be created per function call
# client = OpenAI() # Set up client

#%%
# Step 1: Break down clinical diagnosis into individual claims
def extract_claims_from_diagnosis(diagnosis_text: str, api_key: str) -> List[Dict]:
    """
    Extract individual claims from a clinical diagnosis and reasoning.
    
    Args:
        diagnosis_text (str): Clinical diagnosis with reasoning
        api_key (str): OpenAI API key
        
    Returns:
        List[Dict]: List of claims with their details
    """
    
    # Create client with provided API key
    client = OpenAI(api_key=api_key)
    
    # Define the function that will be called by the LLM
    def create_claims_list(claims: List[Dict], diagnosis_summary: str) -> dict:
        """Function that creates a list of claims - called by the LLM"""
        print(f"LLM called create_claims_list with {len(claims)} claims")
        print(f"Diagnosis summary: {diagnosis_summary}")
        
        return {
            "status": "success",
            "claims": claims,
            "diagnosis_summary": diagnosis_summary,
            "total_claims": len(claims)
        }
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "create_claims_list",
            "description": "Extract individual claims from a clinical diagnosis and create a structured list",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id": {"type": "string"},
                                "claim_text": {"type": "string"},
                                "claim_type": {
                                    "type": "string",
                                    "enum": ["symptom", "sign", "diagnosis", "etiology", "prognosis", "treatment", "mechanism", "risk_factor", "comorbidity"]
                                },
                                "confidence": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                    "description": "Confidence level in this claim based on the clinical evidence"
                                },
                                "clinical_evidence": {"type": "string"},
                                "research_question": {"type": "string"}
                            },
                            "required": ["claim_id", "claim_text", "claim_type", "confidence", "clinical_evidence", "research_question"]
                        },
                        "description": "List of individual claims extracted from the diagnosis"
                    },
                    "diagnosis_summary": {
                        "type": "string",
                        "description": "Brief summary of the overall diagnosis"
                    }
                },
                "required": ["claims", "diagnosis_summary"]
            }
        }
    }]
    
    # Create the prompt for the LLM
    system_prompt = """You are a clinical research expert who specializes in breaking down clinical diagnoses into testable claims.

    Your task is to analyze a clinical diagnosis and extract individual claims that can be validated against medical literature.
    
    CRITICAL REQUIREMENTS:
    1. Break down the diagnosis into 5-12 specific, testable claims (aim for comprehensive coverage)
    2. Each claim should be a single, clear statement that can be researched
    3. Identify the type of each claim (symptom, sign, diagnosis, etiology, prognosis, treatment, mechanism, risk_factor, comorbidity)
    4. Assess confidence level based on clinical evidence provided
    5. Generate a specific research question for each claim
    6. Focus on claims that would benefit from literature validation
    7. Include both major and minor claims for thorough validation
    
    CLAIM TYPES:
    - symptom: Patient-reported symptoms
    - sign: Observable clinical signs
    - diagnosis: Diagnostic conclusions
    - etiology: Underlying causes
    - prognosis: Expected outcomes
    - treatment: Therapeutic interventions
    - mechanism: Pathophysiological processes
    - risk_factor: Risk factors or predispositions
    - comorbidity: Associated conditions
    
    EXAMPLE CLAIMS:
    - "Patient exhibits semantic variant PPA" → diagnosis claim
    - "Anterior temporal lobe atrophy is present" → sign claim  
    - "Progressive semantic memory decline" → symptom claim
    - "TDP-43 pathology underlies the condition" → mechanism claim
    - "Left hemisphere predominance of atrophy" → sign claim
    - "Circumlocutory speech pattern" → symptom claim
    - "Preserved syntax and phonology" → sign claim
    - "Impaired semantic knowledge" → symptom claim
    
    You MUST call the create_claims_list function with your extracted claims."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract claims from this clinical diagnosis:\n\n{diagnosis_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_claims_list"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = create_claims_list(
                claims=function_args["claims"],
                diagnosis_summary=function_args["diagnosis_summary"]
            )
            
            return result["claims"]
        else:
            print("LLM did not make a function call, using fallback")
            return []
        
    except Exception as e:
        print(f"Error extracting claims: {e}")
        return []

#%%
# Step 2: Generate PubMed queries for each claim
def generate_claim_queries(claims: List[Dict], api_key: str) -> List[Dict]:
    """
    Generate PubMed search queries for each claim.
    
    Args:
        claims (List[Dict]): List of claims extracted from diagnosis
        api_key (str): OpenAI API key
        
    Returns:
        List[Dict]: Claims with PubMed search queries added
    """
    
    # Create client with provided API key
    client = OpenAI(api_key=api_key)
    
    # Define the function that will be called by the LLM
    def create_claim_queries(claims_with_queries: List[Dict]) -> dict:
        """Function that creates PubMed queries for claims - called by the LLM"""
        print(f"LLM called create_claim_queries for {len(claims_with_queries)} claims")
        
        return {
            "status": "success",
            "claims_with_queries": claims_with_queries,
            "total_queries": len(claims_with_queries)
        }
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "create_claim_queries",
            "description": "Generate PubMed search queries for each clinical claim",
            "parameters": {
                "type": "object",
                "properties": {
                    "claims_with_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id": {"type": "string"},
                                "claim_text": {"type": "string"},
                                "claim_type": {"type": "string"},
                                "confidence": {"type": "string"},
                                "clinical_evidence": {"type": "string"},
                                "research_question": {"type": "string"},
                                "pubmed_query": {"type": "string"},
                                "query_strategy": {"type": "string"}
                            },
                            "required": ["claim_id", "claim_text", "claim_type", "confidence", "clinical_evidence", "research_question", "pubmed_query", "query_strategy"]
                        },
                        "description": "Claims with PubMed search queries added"
                    }
                },
                "required": ["claims_with_queries"]
            }
        }
    }]
    
    # Prepare claims data for the LLM
    claims_data = []
    for claim in claims:
        claims_data.append({
            "claim_id": claim["claim_id"],
            "claim_text": claim["claim_text"],
            "claim_type": claim["claim_type"],
            "research_question": claim["research_question"]
        })
    
    # Create the prompt for the LLM
    system_prompt = """You are a medical research expert who specializes in creating PubMed search queries for clinical claims.

    Your task is to generate PubMed search queries that will find EVIDENCE FOR OR AGAINST each clinical claim, not just the claim itself.
    
    CRITICAL REQUIREMENTS:
    1. Focus on finding EVIDENCE that supports or refutes the claim, not just the claim itself
    2. Start with BROAD, FUNDAMENTAL concepts to ensure results are found
    3. Use the most common, widely-used medical terminology
    4. Avoid overly specific phrases that might return 0 results
    5. Focus on 2-3 core concepts maximum to avoid over-restriction
    6. Use field tags like [tiab] (title/abstract) and [mesh] (MeSH terms)
    7. Add Boolean operators (AND, OR, NOT) appropriately
    
    EVIDENCE-FOCUSED QUERY STRATEGIES:
    - For diagnostic claims: Search for the condition + "diagnosis" OR "diagnostic criteria" OR "clinical features"
    - For symptom claims: Search for the symptom + condition + "assessment" OR "evaluation" OR "characteristics"
    - For mechanism claims: Search for the condition + "pathophysiology" OR "mechanism" OR "pathology"
    - For treatment claims: Search for the condition + "treatment" OR "therapy" OR "intervention"
    - For prognosis claims: Search for the condition + "prognosis" OR "outcome" OR "progression"
    - For anatomical claims: Search for the anatomical structure + condition + "imaging" OR "MRI" OR "atrophy"
    
    BROADENING STRATEGIES:
    - If a claim is very specific, start with the broader condition/disease
    - Use OR operators for synonyms and alternative terms
    - Focus on finding ANY relevant evidence, not just perfect matches
    - Consider related conditions or symptoms that might provide evidence
    
    EXAMPLE EVIDENCE-FOCUSED QUERIES:
    - Claim: "Patient has semantic variant PPA" → Query: "(svPPA OR "semantic variant primary progressive aphasia")[tiab] AND (diagnosis OR "clinical features" OR characteristics)[tiab]"
    - Claim: "Anterior temporal lobe atrophy present" → Query: "("anterior temporal lobe"[tiab] OR ATL[tiab]) AND atrophy[tiab] AND (imaging OR MRI OR "brain imaging")[tiab]"
    - Claim: "Severe anomia" → Query: "(anomia OR "word finding")[tiab] AND (svPPA OR "primary progressive aphasia")[tiab]"
    - Claim: "TDP-43 proteinopathy" → Query: "(TDP-43 OR "TARDBP")[tiab] AND (svPPA OR "primary progressive aphasia")[tiab]"
    
    You MUST call the create_claim_queries function with your evidence-focused queries."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Generate evidence-focused PubMed queries for these claims:\n\n{json.dumps(claims_data, indent=2)}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_claim_queries"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = create_claim_queries(
                claims_with_queries=function_args["claims_with_queries"]
            )
            
            return result["claims_with_queries"]
        else:
            print("LLM did not make a function call, using fallback")
            # Generate basic queries using the imported function
            claims_with_queries = []
            for claim in claims:
                basic_query = generate_pubmed_prompt(claim["research_question"])
                claims_with_queries.append({
                    **claim,
                    "pubmed_query": basic_query,
                    "query_strategy": "Basic query generation"
                })
            return claims_with_queries
        
    except Exception as e:
        print(f"Error generating claim queries: {e}")
        # Generate basic queries using the imported function
        claims_with_queries = []
        for claim in claims:
            basic_query = generate_pubmed_prompt(claim["research_question"])
            claims_with_queries.append({
                **claim,
                "pubmed_query": basic_query,
                "query_strategy": "Basic query generation"
            })
        return claims_with_queries

#%%
# Fallback function for broader queries when no evidence is found
def try_broader_query(claim: Dict) -> Dict:
    """
    Try a broader, more general query when the original query returns no evidence.
    
    Args:
        claim (Dict): Original claim that had no evidence
        
    Returns:
        Dict: Claim with broader research results, or None if still no evidence
    """
    
    print(f"  Trying broader query for: {claim['claim_text']}")
    
    # Generate broader query based on claim type
    broader_query = generate_broader_query(claim)
    
    if not broader_query:
        return None
    
    print(f"  Broader query: {broader_query}")
    
    # Try the broader query with more results
    pmids = search_pubmed(broader_query, max_results=20)
    
    if not pmids:
        print(f"  Still no results with broader query")
        return None
    
    # Fetch abstracts
    abstracts = fetch_pubmed_abstracts(pmids)
    
    if not abstracts:
        print(f"  No abstracts retrieved with broader query")
        return None
    
    # Generate evidence summary
    evidence_summary = summarize_abstracts(abstracts, claim["research_question"])
    
    # Generate references for this specific claim
    claim_references = format_references(abstracts)
    
    # Generate brief evidence summary for analysis
    brief_evidence_summary = generate_brief_evidence_summary(abstracts, claim["research_question"])
    
    # Analyze evidence support
    evidence_analysis = analyze_evidence_support(claim, abstracts, brief_evidence_summary, claim['api_key'])
    
    return {
        **claim,
        "pmids": pmids,
        "abstracts": abstracts,
        "evidence_summary": brief_evidence_summary,
        "detailed_summary": evidence_summary,
        "claim_references": claim_references,
        "evidence_level": evidence_analysis["evidence_level"],
        "supporting_evidence": evidence_analysis["supporting_evidence"],
        "contradicting_evidence": evidence_analysis["contradicting_evidence"],
        "research_status": evidence_analysis["research_status"],
        "query_strategy": f"Broader fallback: {broader_query}"
    }

def generate_broader_query(claim: Dict) -> str:
    """
    Generate a broader, more general query for a claim that had no evidence.
    
    Args:
        claim (Dict): Original claim
        
    Returns:
        str: Broader PubMed query
    """
    
    claim_text = claim['claim_text'].lower()
    claim_type = claim['claim_type']
    
    # Extract key terms from the claim
    key_terms = []
    
    # Extract disorder-specific terms from the claim text
    # This will be dynamically identified based on the clinical diagnosis
    disorder_terms = []
    
    # Look for common disorder patterns in the claim text
    if 'diagnosis' in claim_text or 'diagnosed' in claim_text:
        # Extract the disorder name from diagnosis statements
        import re
        diagnosis_patterns = [
            r'diagnosis is (\w+(?:\s+\w+)*)',
            r'diagnosed with (\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*) diagnosis',
            r'condition (\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*) syndrome',
            r'(\w+(?:\s+\w+)*) disease'
        ]
        
        for pattern in diagnosis_patterns:
            matches = re.findall(pattern, claim_text, re.IGNORECASE)
            if matches:
                disorder_terms.extend(matches)
    
    # Also look for common medical abbreviations and full forms
    if not disorder_terms:
        # Extract any capitalized medical terms that might be disorders
        potential_disorders = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim_text)
        disorder_terms = [term.lower() for term in potential_disorders if len(term.split()) >= 2]
    
    # Use the identified disorder terms or fall back to general medical terms
    medical_terms = disorder_terms if disorder_terms else [
        'diagnosis', 'symptoms', 'clinical', 'medical', 'patient', 'treatment'
    ]
    
    for term in medical_terms:
        if term in claim_text:
            key_terms.append(term)
    
    # If no specific terms found, use the most general terms
    if not key_terms:
        if 'semantic' in claim_text or 'ppa' in claim_text:
            key_terms = ['primary progressive aphasia', 'semantic']
        elif 'atrophy' in claim_text or 'temporal' in claim_text:
            key_terms = ['atrophy', 'temporal lobe']
        elif 'anomia' in claim_text or 'word' in claim_text:
            key_terms = ['anomia', 'word finding']
        else:
            # Fallback to very general terms
            key_terms = ['primary progressive aphasia']
    
    # Create broader query based on claim type
    if claim_type == 'diagnosis':
        # Focus on the condition itself
        if len(key_terms) >= 2:
            return f"({key_terms[0]}[tiab] OR {key_terms[1]}[tiab])"
        else:
            return f"{key_terms[0]}[tiab]"
    
    elif claim_type == 'symptom':
        # Focus on the symptom and condition
        if 'anomia' in key_terms or 'word' in key_terms:
            return f"(anomia[tiab] OR 'word finding'[tiab]) AND (primary progressive aphasia[tiab] OR PPA[tiab])"
        else:
            return f"{key_terms[0]}[tiab]"
    
    elif claim_type == 'sign':
        # Focus on the anatomical finding
        if 'atrophy' in key_terms:
            return f"atrophy[tiab] AND (temporal[tiab] OR brain[tiab])"
        else:
            return f"{key_terms[0]}[tiab]"
    
    elif claim_type == 'mechanism':
        # Focus on the pathological mechanism
        if 'tdp-43' in key_terms:
            return f"(TDP-43[tiab] OR TARDBP[tiab]) AND (primary progressive aphasia[tiab] OR PPA[tiab])"
        else:
            return f"{key_terms[0]}[tiab]"
    
    elif claim_type == 'prognosis':
        # Focus on outcomes and progression
        return f"(prognosis[tiab] OR progression[tiab] OR outcome[tiab]) AND (primary progressive aphasia[tiab] OR PPA[tiab])"
    
    else:
        # Default to the most general term
        return f"{key_terms[0]}[tiab]"

#%%
# Step 3: Research each claim and generate evidence summary
def research_claim(claim: Dict, max_results: int = 15, api_key: str = None) -> Dict:
    """
    Research a single claim using PubMed and generate evidence summary.
    
    Args:
        claim (Dict): Claim with PubMed query
        max_results (int): Maximum number of results to retrieve (default: 15)
        api_key (str): OpenAI API key
        
    Returns:
        Dict: Claim with research results and evidence summary
    """
    
    print(f"Researching claim: {claim['claim_text']}")
    
    # Step 1: Search PubMed using the claim's query
    pmids = search_pubmed(claim["pubmed_query"], max_results)
    
    if not pmids:
        print(f"No results found for claim: {claim['claim_text']}")
        return {
            **claim,
            "pmids": [],
            "abstracts": [],
            "evidence_summary": "No research evidence found for this claim.",
            "evidence_level": "none",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "research_status": "no_evidence",
            "detailed_summary": "No research articles were found for this claim. This could be due to:\n1. The search terms being too specific or restrictive\n2. Limited research on this particular topic\n3. The need to use broader or alternative search terms\n\nConsider trying a more general search or rephrasing the query with more common medical terminology.",
            "claim_references": "No references available - no articles were found for this claim."
        }
    
    # Step 2: Fetch abstracts
    abstracts = fetch_pubmed_abstracts(pmids)
    
    if not abstracts:
        print(f"No abstracts retrieved for claim: {claim['claim_text']}")
        return {
            **claim,
            "pmids": pmids,
            "abstracts": [],
            "evidence_summary": "No abstracts could be retrieved for this claim.",
            "evidence_level": "none",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "research_status": "no_abstracts",
            "detailed_summary": "No abstracts could be retrieved for this claim despite finding PMIDs. This may indicate access issues or incomplete data.",
            "claim_references": "No references available - abstracts could not be retrieved."
        }
    
    # Step 3: Generate detailed evidence summary with citations (like in simple_pubmed.py)
    detailed_summary = summarize_abstracts(abstracts, claim["research_question"])
    
    # Step 4: Generate references for this specific claim
    claim_references = format_references(abstracts)
    
    # Step 5: Generate brief evidence summary for analysis
    evidence_summary = generate_brief_evidence_summary(abstracts, claim["research_question"])
    
    # Step 6: Analyze evidence support/contradiction
    evidence_analysis = analyze_evidence_support(claim, abstracts, evidence_summary, api_key)
    
    return {
        **claim,
        "pmids": pmids,
        "abstracts": abstracts,
        "evidence_summary": evidence_summary,
        "detailed_summary": detailed_summary,
        "claim_references": claim_references,
        "evidence_level": evidence_analysis["evidence_level"],
        "supporting_evidence": evidence_analysis["supporting_evidence"],
        "contradicting_evidence": evidence_analysis["contradicting_evidence"],
        "research_status": evidence_analysis["research_status"]
    }

def generate_brief_evidence_summary(abstracts: List[Dict], research_question: str) -> str:
    """
    Generate a brief evidence summary for analysis purposes.
    
    Args:
        abstracts (List[Dict]): List of article dictionaries
        research_question (str): The research question being addressed
        
    Returns:
        str: Brief summary for analysis
    """
    
    if not abstracts:
        return "No research evidence found."
    
    # Create a brief summary focusing on key findings
    summary_parts = []
    
    for i, article in enumerate(abstracts[:3], 1):  # Focus on first 3 articles
        summary_parts.append(f"Article {i}: {article['title']}")
    
    return f"Found {len(abstracts)} relevant articles. Key studies include: {'; '.join(summary_parts)}"

#%%
# Step 4: Analyze evidence support for claims
def analyze_evidence_support(claim: Dict, abstracts: List[Dict], summary: str, api_key: str) -> Dict:
    """
    Analyze whether the research evidence supports or contradicts the claim.
    
    Args:
        claim (Dict): The clinical claim
        abstracts (List[Dict]): Research abstracts
        summary (str): Evidence summary
        api_key (str): OpenAI API key
        
    Returns:
        Dict: Analysis of evidence support
    """
    
    # Create client with provided API key
    client = OpenAI(api_key=api_key)
    
    # Define the function that will be called by the LLM
    def create_evidence_analysis(evidence_level: str, supporting_evidence: List[str], contradicting_evidence: List[str], research_status: str, confidence_adjustment: str) -> dict:
        """Function that analyzes evidence support - called by the LLM"""
        print(f"LLM called create_evidence_analysis with evidence level: {evidence_level}")
        print(f"Supporting evidence: {len(supporting_evidence)} items")
        print(f"Contradicting evidence: {len(contradicting_evidence)} items")
        
        return {
            "status": "success",
            "evidence_level": evidence_level,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "research_status": research_status,
            "confidence_adjustment": confidence_adjustment
        }
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "create_evidence_analysis",
            "description": "Analyze whether research evidence supports or contradicts a clinical claim",
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence_level": {
                        "type": "string",
                        "enum": ["strong", "moderate", "weak", "none", "mixed"],
                        "description": "Level of evidence found for the claim"
                    },
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific findings that support the claim"
                    },
                    "contradicting_evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of specific findings that contradict the claim"
                    },
                    "research_status": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "mixed", "insufficient", "no_evidence"],
                        "description": "Overall research status of the claim"
                    },
                    "confidence_adjustment": {
                        "type": "string",
                        "enum": ["increase", "decrease", "maintain", "uncertain"],
                        "description": "How the research evidence should adjust the original confidence level"
                    }
                },
                "required": ["evidence_level", "supporting_evidence", "contradicting_evidence", "research_status", "confidence_adjustment"]
            }
        }
    }]
    
    # Prepare data for the LLM
    abstracts_text = ""
    for i, abstract in enumerate(abstracts, 1):
        abstracts_text += f"\n[{i}] {abstract['title']}\n"
        abstracts_text += f"Abstract: {abstract['abstract']}\n"
    
    # Create the prompt for the LLM
    system_prompt = """You are a clinical research expert who analyzes whether medical literature supports or contradicts clinical claims.

    Your task is to evaluate whether the provided research evidence supports, contradicts, or is insufficient for the given clinical claim.
    
    CRITICAL REQUIREMENTS:
    1. Carefully compare the claim against the research evidence
    2. Identify specific findings that support the claim
    3. Identify specific findings that contradict the claim
    4. Assess the overall evidence level (strong, moderate, weak, none, mixed)
    5. Determine if confidence in the claim should increase, decrease, or remain the same
    6. Be objective and evidence-based in your analysis
    
    EVIDENCE LEVELS:
    - strong: Multiple high-quality studies with consistent findings
    - moderate: Some supporting evidence with minor inconsistencies
    - weak: Limited or low-quality evidence
    - none: No relevant evidence found
    - mixed: Conflicting evidence present
    
    RESEARCH STATUS:
    - supported: Evidence generally supports the claim
    - contradicted: Evidence generally contradicts the claim
    - mixed: Conflicting evidence present
    - insufficient: Not enough evidence to determine
    - no_evidence: No relevant evidence found
    
    You MUST call the create_evidence_analysis function with your analysis."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Claim: {claim['claim_text']}\nClaim Type: {claim['claim_type']}\nOriginal Confidence: {claim['confidence']}\n\nResearch Summary:\n{summary}\n\nAbstracts:\n{abstracts_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_evidence_analysis"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = create_evidence_analysis(
                evidence_level=function_args["evidence_level"],
                supporting_evidence=function_args["supporting_evidence"],
                contradicting_evidence=function_args["contradicting_evidence"],
                research_status=function_args["research_status"],
                confidence_adjustment=function_args["confidence_adjustment"]
            )
            
            return result
        else:
            print("LLM did not make a function call, using fallback")
            return {
                "evidence_level": "weak",
                "supporting_evidence": ["Limited evidence available"],
                "contradicting_evidence": [],
                "research_status": "insufficient",
                "confidence_adjustment": "maintain"
            }
        
    except Exception as e:
        print(f"Error analyzing evidence support: {e}")
        return {
            "evidence_level": "weak",
            "supporting_evidence": ["Analysis error occurred"],
            "contradicting_evidence": [],
            "research_status": "insufficient",
            "confidence_adjustment": "maintain"
        }

#%%
# Step 5: Generate final clinical assessment report
def generate_clinical_assessment_report(original_diagnosis: str, researched_claims: List[Dict], api_key: str) -> str:
    """
    Generate a comprehensive clinical assessment report based on research findings.
    
    Args:
        original_diagnosis (str): Original clinical diagnosis
        researched_claims (List[Dict]): Claims with research results
        api_key (str): OpenAI API key
        
    Returns:
        str: Comprehensive clinical assessment report
    """
    
    # Create client with provided API key
    client = OpenAI(api_key=api_key)
    
    # Define the function that will be called by the LLM
    def create_assessment_report(report_text: str, key_recommendations: List[str], confidence_summary: str) -> dict:
        """Function that creates the assessment report - called by the LLM"""
        print(f"LLM called create_assessment_report with {len(key_recommendations)} recommendations")
        print(f"Confidence summary: {confidence_summary}")
        
        return {
            "status": "success",
            "report": report_text,
            "recommendations": key_recommendations,
            "confidence_summary": confidence_summary
        }
    
    # Define the tools for the LLM
    tools = [{
        "type": "function",
        "function": {
            "name": "create_assessment_report",
            "description": "Create a comprehensive clinical assessment report based on research findings",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_text": {
                        "type": "string",
                        "description": "Comprehensive clinical assessment report"
                    },
                    "key_recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key clinical recommendations based on research findings"
                    },
                    "confidence_summary": {
                        "type": "string",
                        "description": "Summary of confidence levels across all claims"
                    }
                },
                "required": ["report_text", "key_recommendations", "confidence_summary"]
            }
        }
    }]
    
    # Prepare research data for the LLM
    claims_summary = ""
    for i, claim in enumerate(researched_claims, 1):
        claims_summary += f"\nClaim {i}: {claim['claim_text']}\n"
        claims_summary += f"Type: {claim['claim_type']}\n"
        claims_summary += f"Original Confidence: {claim['confidence']}\n"
        claims_summary += f"Evidence Level: {claim['evidence_level']}\n"
        claims_summary += f"Research Status: {claim['research_status']}\n"
        claims_summary += f"Articles Found: {len(claim['abstracts'])}\n"
        claims_summary += f"Evidence Summary: {claim['evidence_summary']}\n"
        claims_summary += f"Detailed Summary Available: {len(claim['detailed_summary']) > 100}\n"
        claims_summary += f"References Available: {len(claim['claim_references']) > 50}\n"
        claims_summary += "-" * 50 + "\n"
    
    # Create the prompt for the LLM
    system_prompt = """You are a senior clinical researcher who creates comprehensive assessment reports based on literature validation of clinical claims.

    Your task is to synthesize the research findings for each claim and create a comprehensive clinical assessment report.
    
    CRITICAL REQUIREMENTS:
    1. Provide an executive summary of the overall assessment
    2. Address each claim with its research validation status
    3. Identify areas where the diagnosis is well-supported vs. uncertain
    4. Provide specific clinical recommendations based on findings
    5. Suggest additional diagnostic tests or assessments if needed
    6. Highlight any discrepancies between clinical presentation and research evidence
    7. Provide a confidence assessment for the overall diagnosis
    8. Note that detailed research summaries with citations and references are available for each claim
    
    REPORT STRUCTURE:
    1. Executive Summary
    2. Claim-by-Claim Analysis
    3. Overall Assessment
    4. Clinical Recommendations
    5. Areas for Further Investigation
    
    IMPORTANT: Each claim has been researched with detailed summaries including inline citations [1], [2], etc. and direct quotes from research articles. These detailed summaries help identify any potential hallucinations and provide comprehensive research backing for clinical decisions.
    
    You MUST call the create_assessment_report function with your comprehensive report."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Original Diagnosis:\n{original_diagnosis}\n\nResearch Findings:\n{claims_summary}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_assessment_report"}}
        )
        
        # Check if the LLM made a function call
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            # Call the function that the LLM requested
            result = create_assessment_report(
                report_text=function_args["report_text"],
                key_recommendations=function_args["key_recommendations"],
                confidence_summary=function_args["confidence_summary"]
            )
            
            return result["report"]
        else:
            print("LLM did not make a function call, using fallback")
            return "Unable to generate assessment report due to technical issues."
        
    except Exception as e:
        print(f"Error generating assessment report: {e}")
        return "Error occurred during report generation."

#%%
# Main pipeline function for clinical diagnosis exploration
def run_clinical_diagnosis_exploration(diagnosis_text: str, max_results_per_claim: int = 15, max_claims: int = 12, api_key: str = None) -> Dict:
    """
    Run the complete clinical diagnosis exploration pipeline.
    
    Args:
        diagnosis_text (str): Clinical diagnosis with reasoning
        max_results_per_claim (int): Maximum PubMed results per claim (default: 15)
        max_claims (int): Maximum number of claims to extract (default: 12)
        api_key (str): OpenAI API key
        
    Returns:
        Dict: Complete exploration results
    """
    
    if not api_key:
        raise ValueError("OpenAI API key is required")
    
    print("=== Clinical Diagnosis Exploration Pipeline ===")
    print(f"Diagnosis: {diagnosis_text}")
    print(f"Max results per claim: {max_results_per_claim}")
    print(f"Max claims: {max_claims}")
    print()
    
    # Step 1: Extract claims from diagnosis (LLM Call #1)
    print("Step 1: Extracting claims from diagnosis...")
    claims = extract_claims_from_diagnosis(diagnosis_text, api_key)
    
    # Limit the number of claims if too many
    if len(claims) > max_claims:
        print(f"Limiting claims from {len(claims)} to {max_claims}")
        claims = claims[:max_claims]
    
    print(f"Extracted {len(claims)} claims")
    for i, claim in enumerate(claims, 1):
        print(f"  {i}. {claim['claim_text']} ({claim['claim_type']})")
    print()
    
    # Step 2: Generate PubMed queries for each claim (LLM Call #2)
    print("Step 2: Generating PubMed queries for claims...")
    claims_with_queries = generate_claim_queries(claims, api_key)
    print(f"Generated queries for {len(claims_with_queries)} claims")
    for i, claim in enumerate(claims_with_queries, 1):
        print(f"  {i}. Query: {claim['pubmed_query']}")
    print()
    
    # Step 3: Research each claim (Multiple LLM calls for each claim)
    print("Step 3: Researching each claim...")
    researched_claims = []
    total_llm_calls = 2  # Initial extraction and query generation
    
    for i, claim in enumerate(claims_with_queries, 1):
        print(f"Researching claim {i}/{len(claims_with_queries)}: {claim['claim_text']}")
        
        # This will make multiple LLM calls per claim (search, fetch, summarize, analyze)
        researched_claim = research_claim(claim, max_results_per_claim, api_key)
        
        # If no evidence found, try a broader fallback query
        if researched_claim['research_status'] == 'no_evidence':
            print(f"  No evidence found, trying broader fallback query...")
            fallback_claim = try_broader_query(claim)
            if fallback_claim:
                researched_claim = fallback_claim
                total_llm_calls += 4  # Additional LLM calls for fallback
        
        researched_claims.append(researched_claim)
        
        # Count LLM calls (approximately 4 per claim: search, fetch, summarize, analyze)
        total_llm_calls += 4
        
        print(f"  Evidence level: {researched_claim['evidence_level']}")
        print(f"  Research status: {researched_claim['research_status']}")
        
        # Display detailed summary for this claim
        print(f"  Detailed Summary: {researched_claim['detailed_summary']}")
        print(f"  Articles found: {len(researched_claim['abstracts'])}")
        print()
        
        # Display claim-specific references
        print(f"  Claim References:")
        print(researched_claim['claim_references'])
        print("-" * 80)
        print()
    
    # Step 4: Generate final assessment report (LLM Call #N)
    print("Step 4: Generating clinical assessment report...")
    assessment_report = generate_clinical_assessment_report(diagnosis_text, researched_claims, api_key)
    total_llm_calls += 1
    print()
    
    # Step 5: Format references for all articles
    print("Step 5: Formatting references...")
    all_abstracts = []
    for claim in researched_claims:
        all_abstracts.extend(claim['abstracts'])
    
    # Remove duplicates based on PMID
    unique_abstracts = []
    seen_pmids = set()
    for abstract in all_abstracts:
        if abstract['pmid'] not in seen_pmids:
            unique_abstracts.append(abstract)
            seen_pmids.add(abstract['pmid'])
    
    references = format_references(unique_abstracts)
    print(f"Formatted {len(unique_abstracts)} unique references")
    print()
    
    # Compile results
    results = {
        "original_diagnosis": diagnosis_text,
        "claims": researched_claims,
        "assessment_report": assessment_report,
        "references": references,
        "total_llm_calls": total_llm_calls,
        "total_articles": len(unique_abstracts),
        "claims_researched": len(researched_claims)
    }
    
    print("=== Pipeline Complete ===")
    print(f"Total LLM function calls: {results['total_llm_calls']}")
    print(f"Claims researched: {results['claims_researched']}")
    print(f"Total articles: {results['total_articles']}")
    print()
    
    return results

#%%
# Test the clinical diagnosis exploration pipeline
if __name__ == "__main__":
    # Example clinical diagnosis for testing
    test_diagnosis = """
    Patient presents with progressive language difficulties characterized by fluent but circumlocutory speech, 
    severe anomia, and word-finding difficulties. MRI shows anterior temporal lobe atrophy, particularly 
    in the left hemisphere. The patient demonstrates preserved syntax and phonology but impaired semantic 
    knowledge. Based on these findings, the diagnosis is semantic variant primary progressive aphasia (svPPA). 
    The underlying pathology is likely TDP-43 proteinopathy, and the prognosis includes eventual progression 
    to affect other cognitive domains beyond language.
    """
    
    print("=== Clinical Diagnosis Exploration Test ===")
    print(f"Test diagnosis: {test_diagnosis}")
    print()
    
    # Run the complete pipeline with higher limits for better evidence coverage
    results = run_clinical_diagnosis_exploration(
        test_diagnosis, 
        max_results_per_claim=5,  # Increased from 3 to 15
        max_claims=7,  # Allow up to 10 claims
        api_key=os.environ["OPENAI_API_KEY"]
    )
    
    # Display results
    print("=== Final Results ===")
    print("Clinical Assessment Report:")
    print(results['assessment_report'])
    print()
    print("Overall References:")
    print(results['references'])
    print()
    print("=== DETAILED CLAIM-BY-CLAIM ANALYSIS ===")
    for i, claim in enumerate(results['claims'], 1):
        print(f"CLAIM {i}: {claim['claim_text']}")
        print(f"Type: {claim['claim_type']}")
        print(f"Original Confidence: {claim['confidence']}")
        print(f"Evidence Level: {claim['evidence_level']}")
        print(f"Research Status: {claim['research_status']}")
        print(f"Query Strategy: {claim.get('query_strategy', 'Original query')}")
        print(f"PubMed Query: {claim['pubmed_query']}")
        print(f"Articles Found: {len(claim['abstracts'])}")
        print()
        
        print("DETAILED RESEARCH SUMMARY:")
        print(claim['detailed_summary'])
        print()
        
        print("CLAIM-SPECIFIC REFERENCES:")
        print(claim['claim_references'])
        print()
        
        if claim['supporting_evidence']:
            print("SUPPORTING EVIDENCE:")
            for evidence in claim['supporting_evidence']:
                print(f"  • {evidence}")
            print()
        
        if claim['contradicting_evidence']:
            print("CONTRADICTING EVIDENCE:")
            for evidence in claim['contradicting_evidence']:
                print(f"  • {evidence}")
            print()
        
        print("=" * 100)
        print()

#%%

def test_single_claim():
    """
    Test function to verify that a single claim gets detailed summary and references.
    """
    print("=== Testing Single Claim with Detailed Summary ===")
    
    # Create a test claim
    test_claim = {
        "claim_id": "test_1",
        "claim_text": "Patient has semantic variant primary progressive aphasia",
        "claim_type": "diagnosis",
        "confidence": "high",
        "clinical_evidence": "Clinical presentation and MRI findings",
        "research_question": "What is the evidence for semantic variant PPA diagnosis?",
        "pubmed_query": "(svPPA OR \"semantic variant primary progressive aphasia\")[tiab] AND (diagnosis OR \"clinical features\")[tiab]"
    }
    
    # Research the claim
    result = research_claim(test_claim, max_results=5, api_key=os.environ["OPENAI_API_KEY"])
    
    print(f"Claim: {result['claim_text']}")
    print(f"Evidence Level: {result['evidence_level']}")
    print(f"Research Status: {result['research_status']}")
    print(f"Articles Found: {len(result['abstracts'])}")
    print()
    
    print("DETAILED SUMMARY:")
    print(result['detailed_summary'])
    print()
    
    print("REFERENCES:")
    print(result['claim_references'])
    print()
    
    return result

# Uncomment to test single claim
#test_single_claim()

#%%

def test_enhanced_summaries():
    """
    Test function to verify that the enhanced pipeline produces longer, more comprehensive summaries.
    """
    print("=== Testing Enhanced Summaries ===")
    
    # Create a test claim
    test_claim = {
        "claim_id": "test_1",
        "claim_text": "Patient has semantic variant primary progressive aphasia with anterior temporal lobe atrophy",
        "claim_type": "diagnosis",
        "confidence": "high",
        "clinical_evidence": "Clinical presentation and MRI findings",
        "research_question": "What is the evidence for semantic variant PPA diagnosis and anterior temporal lobe atrophy?",
        "pubmed_query": "(svPPA OR \"semantic variant primary progressive aphasia\")[tiab] AND (\"anterior temporal lobe\" OR ATL)[tiab] AND atrophy[tiab]"
    }
    
    print(f"Testing claim: {test_claim['claim_text']}")
    print(f"Research question: {test_claim['research_question']}")
    print(f"PubMed query: {test_claim['pubmed_query']}")
    print()
    
    # Research the claim
    result = research_claim(test_claim, max_results=8, api_key=os.environ["OPENAI_API_KEY"])
    
    print(f"Evidence Level: {result['evidence_level']}")
    print(f"Research Status: {result['research_status']}")
    print(f"Articles Found: {len(result['abstracts'])}")
    print()
    
    print("DETAILED SUMMARY LENGTH:", len(result['detailed_summary']), "characters")
    print("DETAILED SUMMARY:")
    print(result['detailed_summary'])
    print()
    
    print("REFERENCES:")
    print(result['claim_references'])
    print()
    
    # Check if the summary is comprehensive
    if len(result['detailed_summary']) > 500:
        print("✓ SUMMARY IS COMPREHENSIVE (longer than 500 characters)")
    else:
        print("⚠ SUMMARY MAY BE TOO SHORT (less than 500 characters)")
    
    # Check for citations
    citation_count = result['detailed_summary'].count('[')
    if citation_count >= 3:
        print(f"✓ GOOD CITATION COVERAGE ({citation_count} citations found)")
    else:
        print(f"⚠ LIMITED CITATIONS ({citation_count} citations found)")
    
    # Check for quotes
    quote_count = result['detailed_summary'].count('"')
    if quote_count >= 4:
        print(f"✓ GOOD QUOTE COVERAGE ({quote_count} quotes found)")
    else:
        print(f"⚠ LIMITED QUOTES ({quote_count} quotes found)")
    
    return result

# Uncomment to test enhanced summaries
#test_enhanced_summaries()

# %%
