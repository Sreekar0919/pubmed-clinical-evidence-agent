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
    format_references,
    create_fallback_search
)

# key now read from the environment

client = OpenAI()

#%%
# ENHANCED: Diagnosis-first claim extraction with fuzzy search
def extract_clinical_claims_and_gram(diagnosis_text: str, max_claims: int = 12) -> Dict:
    """
    ENHANCED: Extract diagnosis first, then find clinical claims that relate to and support that diagnosis.
    Uses GPT-4o for better claim identification and includes fuzzy search capabilities.
    
    Args:
        diagnosis_text (str): Clinical note with reasoning
        max_claims (int): Maximum number of claims to extract
        
    Returns:
        Dict: Diagnosis with supporting claims and PubMed queries
    """
    
    def create_diagnosis_and_claims(diagnosis: str, claims: List[Dict], 
                                  missing_information: List[str]) -> dict:
        """Function that creates diagnosis and supporting claims - called by the LLM"""
        return {
            "status": "success",
            "diagnosis": diagnosis,
            "claims": claims,
            "missing_information": missing_information
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "create_diagnosis_and_claims",
            "description": "Extract diagnosis and claims that relate to that diagnosis with PubMed queries",
            "parameters": {
                "type": "object",
                "properties": {
                    "diagnosis": {"type": "string"},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id": {"type": "string"},
                                "claim_text": {"type": "string"},
                                "claim_type": {
                                    "type": "string",
                                    "enum": ["symptom", "sign", "imaging", "exam_finding", "history", "exclusion", "diagnosis", "etiology", "prognosis", "mechanism"]
                                },
                                "clinical_evidence": {"type": "string"},
                                "pubmed_query": {"type": "string"},
                                "query_strategy": {"type": "string"}
                            },
                            "required": ["claim_id", "claim_text", "claim_type", "clinical_evidence", "pubmed_query", "query_strategy"]
                        }
                    },
                    "missing_information": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["diagnosis", "claims", "missing_information"]
            }
        }
    }]
    
    system_prompt = """You are a clinical research expert with expertise in evidence-based medicine. 
    Extract the primary diagnosis from the clinical note, then identify ALL clinical claims that relate to and support that diagnosis.

    CRITICAL REQUIREMENTS:
    1. **Extract the diagnosis first** - Identify the primary clinical diagnosis mentioned in the note
    2. **Find ALL related claims** - Extract any clinical statement, observation, or finding that relates to the diagnosis
    3. **Don't filter for correctness** - Include claims even if they seem wrong or contradictory to the diagnosis
    4. **Focus on diagnosis-specific queries** - All PubMed queries should include the diagnosis
    5. **Be comprehensive** - Include symptoms, signs, imaging, exam findings, etc. that relate to the diagnosis
    
    CLAIM TYPES:
    - symptom: Patient-reported symptoms related to the diagnosis
    - sign: Observable clinical signs on examination related to the diagnosis
    - imaging: Radiological or imaging findings related to the diagnosis
    - exam_finding: Specific examination results related to the diagnosis
    - history: Historical information or temporal patterns related to the diagnosis
    - exclusion: Negative findings that help rule out other conditions
    - diagnosis: The primary diagnosis itself
    - etiology: Underlying causes or mechanisms related to the diagnosis
    - prognosis: Expected outcomes related to the diagnosis
    - mechanism: Pathophysiological processes related to the diagnosis
    
    QUERY STRATEGIES:
    - **ALWAYS include the diagnosis** in every PubMed query
    - Use the diagnosis name and common abbreviations/synonyms
    - Combine the diagnosis with the specific claim
    - Use field tags like [tiab] for title/abstract searches
    - Avoid overly specific phrases that might return 0 results
    
    EXAMPLE QUERIES (all include the diagnosis):
    - Diagnosis: "Semantic Variant Primary Progressive Aphasia (svPPA)"
    - Claim: "progressive word comprehension deficits" → "\"semantic variant primary progressive aphasia\" AND \"word comprehension deficits\""
    - Claim: "surface dyslexia" → "\"semantic variant primary progressive aphasia\" AND \"surface dyslexia\""
    - Claim: "anterior temporal lobe atrophy" → "\"semantic variant primary progressive aphasia\" AND \"anterior temporal lobe atrophy\""
    - Claim: "fluent speech" → "\"semantic variant primary progressive aphasia\" AND \"fluent speech\""
    
    DIAGNOSIS EXTRACTION:
    - Look for explicit diagnosis statements (e.g., "Diagnosis:", "Impression:", "Conclusion:")
    - Identify the primary disorder/condition being discussed
    - Use the most specific diagnosis mentioned
    - Include common abbreviations (e.g., svPPA, PPA, etc.)
    
    You MUST extract the diagnosis first, then find ALL claims that relate to it.
    You MUST call create_diagnosis_and_claims with your results."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract diagnosis and related claims from this clinical note:\n\n{diagnosis_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # ENHANCED: Use GPT-4o for better analysis
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_diagnosis_and_claims"}}
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            claims = function_args["claims"]
            
            # Limit claims if too many
            if len(claims) > max_claims:
                claims = claims[:max_claims]
            
            return {
                "diagnosis": function_args["diagnosis"],
                "claims": claims,
                "missing_information": function_args.get("missing_information", [])
            }
        else:
            print("LLM did not make function call, using fallback")
            return {
                "diagnosis": "Diagnosis extraction failed",
                "claims": [],
                "missing_information": ["Extraction failed"]
            }
        
    except Exception as e:
        print(f"Error in diagnosis and claims extraction: {e}")
        return {
            "diagnosis": "Error occurred",
            "claims": [],
            "missing_information": [f"Error: {e}"]
        }

#%%
# ENHANCED: More rigorous research function with fuzzy search
def research_claim_enhanced(claim: Dict, max_results: int = 12) -> Dict:
    """
    ENHANCED: Research claim with more rigorous evidence evaluation and fuzzy search.
    Uses GPT-4o for better analysis and is willing to say "no" when evidence doesn't support claims.
    
    Args:
        claim (Dict): Claim with PubMed query
        max_results (int): Maximum results to retrieve
        
    Returns:
        Dict: Claim with enhanced research results
    """
    
    print(f"Researching claim: {claim['claim_text']}")
    print(f"Query strategy: {claim.get('query_strategy', 'Standard query')}")
    print(f"PubMed query: {claim['pubmed_query']}")
    
    # Step 1: Search PubMed with fuzzy search to ensure results
    pmids = search_pubmed_with_fallback(claim["pubmed_query"], max_results)
    
    if not pmids:
        return {
            **claim,
            "pmids": [],
            "abstracts": [],
            "evidence_summary": "No research evidence found despite multiple search attempts.",
            "evidence_level": "none",
            "research_status": "no_evidence",
            "detailed_summary": "No research articles were found for this claim despite using multiple search strategies including fallback searches. This could indicate:\n1. The claim may be too specific or novel\n2. Limited research on this particular topic\n3. The claim may not be well-established in the literature\n4. Alternative terminology may be needed\n\nConsider whether this claim is evidence-based or speculative.",
            "claim_references": "No references available - no articles were found for this claim.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "evidence_quality": "no_evidence",
            "critical_assessment": "No evidence found to support or refute this claim."
        }
    
    # Step 2: Fetch abstracts
    abstracts = fetch_pubmed_abstracts_direct(pmids)
    
    if not abstracts:
        return {
            **claim,
            "pmids": pmids,
            "abstracts": [],
            "evidence_summary": "No abstracts retrieved.",
            "evidence_level": "none",
            "research_status": "no_abstracts",
            "detailed_summary": "No abstracts could be retrieved for this claim despite finding PMIDs. This may indicate access issues or incomplete data.",
            "claim_references": "No references available - abstracts could not be retrieved.",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "evidence_quality": "no_evidence",
            "critical_assessment": "Unable to assess evidence due to retrieval issues."
        }
    
    # Step 3: Enhanced LLM call for comprehensive analysis
    summary_and_analysis = generate_enhanced_summary_and_analysis(claim, abstracts)
    
    # Step 4: Generate claim-specific references
    claim_references = format_references_simple(abstracts)
    
    return {
        **claim,
        "pmids": pmids,
        "abstracts": abstracts,
        "evidence_summary": summary_and_analysis["brief_summary"],
        "detailed_summary": summary_and_analysis["detailed_summary"],
        "evidence_level": summary_and_analysis["evidence_level"],
        "research_status": summary_and_analysis["research_status"],
        "supporting_evidence": summary_and_analysis["supporting_evidence"],
        "contradicting_evidence": summary_and_analysis["contradicting_evidence"],
        "evidence_quality": summary_and_analysis["evidence_quality"],
        "critical_assessment": summary_and_analysis["critical_assessment"],
        "claim_references": claim_references
    }

def search_pubmed_with_fallback(search_str: str, max_results: int = 12) -> List[str]:
    """
    Search PubMed with fuzzy search fallback to ensure results are found.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    try:
        # First attempt: Original search
        search_url = f"{base_url}esearch.fcgi"
        search_params = {
            'db': 'pubmed',
            'term': search_str,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        response = requests.get(search_url, params=search_params)
        response.raise_for_status()
        
        data = response.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        
        # If no results, try fallback search
        if not pmids:
            print(f"No results for: {search_str}")
            print("Trying fallback search...")
            
            # Use the fallback search function from simple_pubmed
            fallback_search = create_fallback_search(search_str)
            print(f"Fallback search: {fallback_search}")
            
            # Try the fallback search
            search_params['term'] = fallback_search
            response = requests.get(search_url, params=search_params)
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            
            if not pmids:
                print("Fallback search also failed, trying basic terms...")
                # Extract basic terms and try again
                basic_terms = extract_basic_terms(search_str)
                search_params['term'] = basic_terms
                response = requests.get(search_url, params=search_params)
                response.raise_for_status()
                
                data = response.json()
                pmids = data.get('esearchresult', {}).get('idlist', [])
        
        return pmids
        
    except Exception as e:
        print(f"Error in PubMed search with fallback: {e}")
        return []

def extract_basic_terms(query: str) -> str:
    """
    Extract basic medical terms from a query as a fallback.
    """
    # Common medical terms to look for
    common_terms = [
        "diagnosis", "symptoms", "treatment", "atrophy", "imaging", "MRI",
        "cognitive", "neurological", "brain", "clinical", "assessment",
        "progressive", "aphasia", "dementia", "temporal", "lobe"
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

def fetch_pubmed_abstracts_direct(pmids: List[str]) -> List[Dict]:
    """
    Direct abstract fetching without LLM call.
    """
    if not pmids:
        return []
    
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    abstracts = []
    
    try:
        # Fetch abstracts
        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }
        
        response = requests.get(fetch_url, params=fetch_params)
        response.raise_for_status()
        
        # Parse XML
        root = ET.fromstring(response.content)
        
        for article in root.findall('.//PubmedArticle'):
            pmid = article.find('.//PMID')
            title = article.find('.//ArticleTitle')
            abstract = article.find('.//AbstractText')
            journal = article.find('.//Journal/Title')
            authors = article.findall('.//Author/LastName')
            
            if pmid is not None:
                abstract_data = {
                    'pmid': pmid.text,
                    'title': title.text if title is not None else 'No title',
                    'abstract': abstract.text if abstract is not None else 'No abstract',
                    'journal': journal.text if journal is not None else 'Unknown journal',
                    'authors': [author.text for author in authors if author.text]
                }
                abstracts.append(abstract_data)
        
        return abstracts
        
    except Exception as e:
        print(f"Error in direct abstract fetching: {e}")
        return []

def generate_enhanced_summary_and_analysis(claim: Dict, abstracts: List[Dict]) -> Dict:
    """
    ENHANCED: Comprehensive analysis with GPT-4o for better rigor.
    Willing to say "no" when evidence doesn't support claims.
    """
    
    def create_enhanced_analysis(detailed_summary: str, brief_summary: str, evidence_level: str, 
                               research_status: str, supporting_evidence: List[str], 
                               contradicting_evidence: List[str], evidence_quality: str,
                               critical_assessment: str, methodological_issues: List[str]) -> dict:
        """Function that creates enhanced analysis - called by the LLM"""
        return {
            "status": "success",
            "detailed_summary": detailed_summary,
            "brief_summary": brief_summary,
            "evidence_level": evidence_level,
            "research_status": research_status,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence,
            "evidence_quality": evidence_quality,
            "critical_assessment": critical_assessment,
            "methodological_issues": methodological_issues
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "create_enhanced_analysis",
            "description": "Create enhanced summary and analysis for a clinical claim",
            "parameters": {
                "type": "object",
                "properties": {
                    "detailed_summary": {"type": "string"},
                    "brief_summary": {"type": "string"},
                    "evidence_level": {
                        "type": "string",
                        "enum": ["strong", "moderate", "weak", "none", "mixed", "contradictory"]
                    },
                    "research_status": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "mixed", "insufficient", "no_evidence", "refuted"]
                    },
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "contradicting_evidence": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "evidence_quality": {
                        "type": "string",
                        "enum": ["high", "moderate", "low", "very_low", "inadequate"]
                    },
                    "critical_assessment": {"type": "string"},
                    "methodological_issues": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["detailed_summary", "brief_summary", "evidence_level", "research_status", "supporting_evidence", "contradicting_evidence", "evidence_quality", "critical_assessment", "methodological_issues"]
            }
        }
    }]
    
    # Use more abstracts for better analysis
    limited_abstracts = abstracts[:8]  # Increased from 5
    
    abstracts_text = ""
    for i, abstract in enumerate(limited_abstracts, 1):
        # Use full abstracts for better analysis
        abstracts_text += f"\n[{i}] {abstract['title']}\n{abstract['abstract']}\n"
    
    system_prompt = """You are a senior clinical researcher with expertise in evidence-based medicine and critical appraisal. 
    Analyze the evidence for a clinical claim with RIGOROUS standards.

    CRITICAL ANALYSIS REQUIREMENTS:
    1. **Be willing to say "NO"** - If the evidence doesn't support the claim, say so clearly
    2. **Assess evidence quality** - Consider study design, sample size, methodology
    3. **Identify methodological issues** - Small samples, bias, confounding factors
    4. **Look for contradictory evidence** - Don't just confirm, actively seek refutation
    5. **Consider alternative explanations** - What else could explain these findings?
    6. **Assess clinical relevance** - Are the findings applicable to real patients?
    
    EVIDENCE EVALUATION FRAMEWORK:
    - Study design and methodology quality
    - Sample size and statistical power
    - Potential biases and confounding factors
    - Consistency across studies
    - Clinical vs. statistical significance
    - Generalizability of findings
    
    DETAILED SUMMARY REQUIREMENTS:
    1. Write a comprehensive paragraph that critically evaluates the evidence
    2. Use inline citations [1], [2], [3], etc. to reference specific studies
    3. Include both supporting AND contradicting evidence
    4. Highlight methodological limitations and quality issues
    5. Assess the overall strength and reliability of the evidence
    6. Be honest about what the evidence does and doesn't show
    
    CRITICAL ASSESSMENT:
    - What are the major limitations of the evidence?
    - How confident can we be in these findings?
    - What additional research is needed?
    - Are there alternative explanations?
    - Is the claim supported, refuted, or inconclusive?
    
    You MUST be rigorous and honest. If the evidence is weak or contradictory, say so clearly.
    You MUST call create_enhanced_analysis with your results."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Claim: {claim['claim_text']}\nClaim Type: {claim['claim_type']}\nQuery Strategy: {claim.get('query_strategy', 'Standard query')}\nPubMed Query: {claim['pubmed_query']}\n\nAbstracts:\n{abstracts_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # ENHANCED: Use GPT-4o for better analysis
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_enhanced_analysis"}}
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            return {
                "detailed_summary": function_args["detailed_summary"],
                "brief_summary": function_args["brief_summary"],
                "evidence_level": function_args["evidence_level"],
                "research_status": function_args["research_status"],
                "supporting_evidence": function_args["supporting_evidence"],
                "contradicting_evidence": function_args["contradicting_evidence"],
                "evidence_quality": function_args["evidence_quality"],
                "critical_assessment": function_args["critical_assessment"],
                "methodological_issues": function_args.get("methodological_issues", [])
            }
        else:
            return {
                "detailed_summary": "Analysis failed",
                "brief_summary": "Limited evidence available",
                "evidence_level": "weak",
                "research_status": "insufficient",
                "supporting_evidence": [],
                "contradicting_evidence": [],
                "evidence_quality": "very_low",
                "critical_assessment": "Analysis failed",
                "methodological_issues": ["Analysis failed"]
            }
        
    except Exception as e:
        print(f"Error in enhanced analysis: {e}")
        return {
            "detailed_summary": "Error occurred",
            "brief_summary": "Analysis error",
            "evidence_level": "weak",
            "research_status": "insufficient",
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "evidence_quality": "very_low",
            "critical_assessment": "Error occurred during analysis",
            "methodological_issues": ["Analysis error"]
        }

#%%
# ENHANCED: Generate revised clinical note with citations and evidence
def generate_revised_clinical_note(original_note: str, researched_claims: List[Dict]) -> str:
    """
    ENHANCED: Generate a revised clinical note that keeps original text but adds citations, evidence, and removes false claims.
    """
    
    def create_revised_note(revised_note: str, removed_claims: List[str], added_citations: List[str], evidence_integration: List[str]) -> dict:
        return {
            "status": "success",
            "revised_note": revised_note,
            "removed_claims": removed_claims,
            "added_citations": added_citations,
            "evidence_integration": evidence_integration
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "create_revised_note",
            "description": "Create revised clinical note with citations and evidence integration",
            "parameters": {
                "type": "object",
                "properties": {
                    "revised_note": {"type": "string"},
                    "removed_claims": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "added_citations": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "evidence_integration": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["revised_note", "removed_claims", "added_citations", "evidence_integration"]
            }
        }
    }]
    
    # Create comprehensive summary of research findings for LLM
    claims_summary = ""
    for i, claim in enumerate(researched_claims, 1):
        claims_summary += f"\nClaim {i}: {claim['claim_text']}\n"
        claims_summary += f"Type: {claim['claim_type']}\n"
        claims_summary += f"PubMed Query: {claim['pubmed_query']}\n"
        claims_summary += f"Evidence Level: {claim['evidence_level']}\n"
        claims_summary += f"Research Status: {claim['research_status']}\n"
        claims_summary += f"Articles Found: {len(claim['abstracts'])}\n"
        claims_summary += f"Detailed Summary: {claim['detailed_summary'][:300]}...\n"
        
        # Include supporting evidence
        if claim.get('supporting_evidence'):
            claims_summary += f"Supporting Evidence:\n"
            for evidence in claim['supporting_evidence']:
                claims_summary += f"  • {evidence}\n"
        
        # Include contradicting evidence
        if claim.get('contradicting_evidence'):
            claims_summary += f"Contradicting Evidence:\n"
            for evidence in claim['contradicting_evidence']:
                claims_summary += f"  • {evidence}\n"
        
        claims_summary += "\n"
    
    system_prompt = """You are a senior clinical researcher. Create a revised clinical note that maintains the original text structure but incorporates research findings, evidence, and citations.

    REQUIREMENTS:
    1. **Keep original text structure** - Maintain the same flow and style as the original note
    2. **Add inline citations** - Insert [1], [2], [3], etc. after statements that have research support
    3. **Integrate supporting evidence** - Add brief supporting evidence directly into the note where relevant
    4. **Address contradicting evidence** - Include contradicting evidence in the note to show complexity
    5. **Remove false claims** - Remove or modify statements that are clearly contradicted by strong evidence
    6. **Add qualifying language** - Use phrases like "may be", "suggests", or "evidence indicates" for uncertain claims
    7. **Maintain clinical tone** - Keep the professional medical writing style
    8. **Be comprehensive** - Include both supporting and contradicting evidence in the narrative
    
    EVIDENCE INTEGRATION GUIDELINES:
    - **Supporting evidence**: Add brief supporting findings directly after relevant statements
    - **Contradicting evidence**: Include contradicting evidence to show the complexity of the case
    - **Mixed evidence**: Acknowledge when evidence is mixed or inconclusive
    - **Evidence gaps**: Note when there's insufficient evidence for certain claims
    
    CITATION GUIDELINES:
    - Add citations after statements that have research support
    - Use sequential numbering [1], [2], [3], etc.
    - Only cite claims that have moderate or strong evidence
    - Don't cite claims with weak or contradictory evidence
    
    REMOVAL GUIDELINES:
    - Only remove claims that are clearly contradicted by strong evidence
    - Replace removed claims with "No clear evidence for [specific claim]"
    - Be conservative - when in doubt, keep the claim but add qualifying language
    
    EXAMPLE INTEGRATION:
    Original: "MRI reveals bilateral anterior temporal lobe atrophy"
    Revised: "MRI reveals bilateral anterior temporal lobe atrophy [1], which is strongly associated with semantic variant PPA. However, some studies suggest this pattern may also occur in other neurodegenerative conditions [2]."
    
    You MUST call create_revised_note with your results."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Original Clinical Note:\n{original_note}\n\nResearch Findings:\n{claims_summary}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # ENHANCED: Use GPT-4o for better note revision
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_revised_note"}}
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            return function_args["revised_note"]
        else:
            return "Unable to generate revised note."
        
    except Exception as e:
        print(f"Error generating revised note: {e}")
        return "Error occurred during note revision."

#%%
# ENHANCED MAIN PIPELINE
def run_clinical_diagnosis_exploration_enhanced(diagnosis_text: str, max_results_per_claim: int = 12, max_claims: int = 12) -> Dict:
    """
    ENHANCED VERSION: Run clinical diagnosis exploration with diagnosis-first approach and fuzzy search.
    
    ENHANCEMENTS:
    1. Diagnosis-first claim extraction approach
    2. Fuzzy search with multiple fallback strategies
    3. Enhanced evidence analysis with GPT-4o
    4. Willingness to say "no" when evidence doesn't support claims
    5. Revised clinical note output with citations and evidence
    6. Better cost-quality balance (increased cost for better results)
    
    Expected cost: ~$2-4 (vs $0.30-0.60 in optimized version, but much better quality)
    """
    
    print("=== ENHANCED Clinical Diagnosis Exploration ===")
    print(f"Max claims: {max_claims}")
    print(f"Max results per claim: {max_results_per_claim}")
    print()
    
    # Step 1: Extract diagnosis and related claims
    print("Step 1: Extracting diagnosis and related claims...")
    claims_analysis = extract_clinical_claims_and_gram(diagnosis_text, max_claims)
    print(f"Primary diagnosis: {claims_analysis['diagnosis']}")
    print(f"Claims extracted: {len(claims_analysis['claims'])}")
    print(f"Missing information: {len(claims_analysis['missing_information'])} identified")
    print()
    
    # Step 2: Research each claim with enhanced rigor and fuzzy search
    print("Step 2: Researching claims with enhanced rigor and fuzzy search...")
    researched_claims = []
    total_llm_calls = 1  # Initial diagnosis and claims extraction
    
    for i, claim in enumerate(claims_analysis['claims'], 1):
        print(f"Researching claim {i}/{len(claims_analysis['claims'])}: {claim['claim_text']}")
        print(f"  Type: {claim['claim_type']}")
        print(f"  Query strategy: {claim.get('query_strategy', 'Standard query')}")
        print(f"  PubMed query: {claim['pubmed_query']}")
        
        # Enhanced LLM call per claim with fuzzy search
        researched_claim = research_claim_enhanced(claim, max_results_per_claim)
        researched_claims.append(researched_claim)
        total_llm_calls += 1
        
        print(f"  Evidence level: {researched_claim['evidence_level']}")
        print(f"  Research status: {researched_claim['research_status']}")
        print(f"  Articles found: {len(researched_claim['abstracts'])}")
        print()
        
        # Display claim-specific references
        print(f"  Claim References:")
        print(researched_claim['claim_references'])
        print("-" * 80)
        print()
    
    # Step 3: Generate revised clinical note
    print("Step 3: Generating revised clinical note...")
    revised_note = generate_revised_clinical_note(diagnosis_text, researched_claims)
    total_llm_calls += 1
    print()
    
    # Step 4: Format references
    print("Step 4: Formatting references...")
    all_abstracts = []
    for claim in researched_claims:
        all_abstracts.extend(claim['abstracts'])
    
    # Remove duplicates
    unique_abstracts = []
    seen_pmids = set()
    for abstract in all_abstracts:
        if abstract['pmid'] not in seen_pmids:
            unique_abstracts.append(abstract)
            seen_pmids.add(abstract['pmid'])
    
    references = format_references_simple(unique_abstracts)
    print(f"Formatted {len(unique_abstracts)} references")
    print()
    
    # Compile results
    results = {
        "original_note": diagnosis_text,
        "diagnosis": claims_analysis['diagnosis'],
        "revised_note": revised_note,
        "claims": researched_claims,
        "references": references,
        "total_llm_calls": total_llm_calls,
        "total_articles": len(unique_abstracts),
        "claims_researched": len(researched_claims),
        "missing_information": claims_analysis['missing_information'],
        "enhancement": "GPT-4o analysis, diagnosis-first approach, fuzzy search, revised note output"
    }
    
    print("=== ENHANCED PIPELINE COMPLETE ===")
    print(f"Total LLM calls: {results['total_llm_calls']}")
    print(f"Claims researched: {results['claims_researched']}")
    print(f"Total articles: {results['total_articles']}")
    print(f"Missing information identified: {len(results['missing_information'])}")
    print(f"Enhanced features: Diagnosis-first approach, fuzzy search, revised note output, GPT-4o analysis")
    print()
    
    return results

def format_references_simple(abstracts: List[Dict]) -> str:
    """
    Simple reference formatting without LLM call.
    """
    if not abstracts:
        return "No references available."
    
    references = []
    for i, abstract in enumerate(abstracts, 1):
        authors = ", ".join(abstract['authors'][:3]) if abstract['authors'] else "Unknown"
        if len(abstract['authors']) > 3:
            authors += " et al."
        
        ref = f"{i}. {authors}. {abstract['title']}. {abstract['journal']}. PMID: {abstract['pmid']}"
        references.append(ref)
    
    return "\n".join(references)

#%%
# Test the enhanced pipeline
if __name__ == "__main__":
    test_diagnosis = """
    28-year-old right-handed female with an 18-month history of progressive word comprehension deficits, 
    surface dyslexia, and difficulty naming common objects (e.g., calling a zebra a dog), while speech 
    remains fluent and grammatically intact. She also reports trouble recognizing familiar faces and 
    increasing social withdrawal. Neurological exam is notable for fluent but semantically empty speech, 
    impaired single-word comprehension, and poor performance on object naming tasks. No motor abnormalities 
    or cognitive fluctuations noted. MRI reveals bilateral anterior temporal lobe atrophy, more prominent 
    on the left. No basal ganglia changes. Currently taking sertraline 50 mg daily for anxiety; no other 
    medications. Diagnosis: Semantic Variant Primary Progressive Aphasia (svPPA).
    """
    
    print("=== ENHANCED TEST ===")
    
    # Run the enhanced pipeline
    results = run_clinical_diagnosis_exploration_enhanced(
        test_diagnosis, 
        max_results_per_claim=12,  # Increased for better analysis
        max_claims=12
    )
    
    # Display results
    print("=== Final Results ===")
    print(f"DIAGNOSIS: {results['diagnosis']}")
    print()
    
    print("REVISED CLINICAL NOTE:")
    print(results['revised_note'])
    print()
    
    print("REFERENCES:")
    print(results['references'])
    print()
    
    print("=== DETAILED RESEARCH FOR EACH CLAIM ===")
    for i, claim in enumerate(results['claims'], 1):
        print(f"CLAIM {i}: {claim['claim_text']}")
        print(f"Type: {claim['claim_type']}")
        print(f"PubMed Query: {claim['pubmed_query']}")
        print(f"Evidence Level: {claim['evidence_level']}")
        print(f"Research Status: {claim['research_status']}")
        print(f"Articles Found: {len(claim['abstracts'])}")
        print()
        
        print("DETAILED PARAGRAPH:")
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
        
        if claim.get('methodological_issues'):
            print("METHODOLOGICAL ISSUES:")
            for issue in claim['methodological_issues']:
                print(f"  • {issue}")
            print()
        
        print("=" * 100)
        print()
    
    print(f"Total LLM calls: {results['total_llm_calls']}")
    print(f"Enhanced features: {results['enhancement']}")