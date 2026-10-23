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

# key now read from the environment

client = OpenAI()

#%%
# COST OPTIMIZATION: Batch claim extraction and query generation
def extract_claims_and_queries_batch(diagnosis_text: str, max_claims: int = 8) -> List[Dict]:
    """
    COST OPTIMIZATION: Extract claims AND generate queries in a single LLM call.
    This reduces 2 separate LLM calls to 1.
    
    Args:
        diagnosis_text (str): Clinical diagnosis with reasoning
        max_claims (int): Maximum number of claims to extract
        
    Returns:
        List[Dict]: Claims with PubMed queries already added
    """
    
    def create_claims_with_queries(claims_with_queries: List[Dict], diagnosis_summary: str) -> dict:
        """Function that creates claims with queries - called by the LLM"""
        print(f"LLM called create_claims_with_queries for {len(claims_with_queries)} claims")
        return {
            "status": "success",
            "claims_with_queries": claims_with_queries,
            "diagnosis_summary": diagnosis_summary
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "create_claims_with_queries",
            "description": "Extract claims from diagnosis and generate PubMed queries in one step",
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
                                "claim_type": {
                                    "type": "string",
                                    "enum": ["symptom", "sign", "diagnosis", "etiology", "prognosis", "treatment", "mechanism", "risk_factor", "comorbidity"]
                                },
                                "confidence": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"]
                                },
                                "clinical_evidence": {"type": "string"},
                                "research_question": {"type": "string"},
                                "pubmed_query": {"type": "string"}
                            },
                            "required": ["claim_id", "claim_text", "claim_type", "confidence", "clinical_evidence", "research_question", "pubmed_query"]
                        }
                    },
                    "diagnosis_summary": {"type": "string"}
                },
                "required": ["claims_with_queries", "diagnosis_summary"]
            }
        }
    }]
    
    system_prompt = """You are a clinical research expert. Extract 5-8 key claims from the diagnosis and generate PubMed queries for each.

    REQUIREMENTS:
    1. Extract 5-8 most important, testable claims
    2. Generate evidence-focused PubMed queries for each claim
    3. Focus on finding EVIDENCE for/against claims, not just the claims themselves
    4. Use broad, fundamental medical terminology
    5. Keep queries simple (2-3 core concepts max)
    
    QUERY STRATEGIES:
    - Diagnosis: condition + "diagnosis" OR "clinical features"
    - Symptoms: symptom + condition + "assessment"
    - Signs: anatomical + "imaging" OR "MRI" OR "atrophy"
    - Mechanisms: condition + "pathophysiology" OR "mechanism"
    
    EXAMPLE QUERIES:
    - "svPPA diagnosis" → "(svPPA OR "semantic variant primary progressive aphasia")[tiab] AND (diagnosis OR "clinical features")[tiab]"
    - "anterior temporal lobe atrophy" → "("anterior temporal lobe"[tiab] OR ATL[tiab]) AND atrophy[tiab]"
    - "severe anomia" → "(anomia OR "word finding")[tiab] AND (svPPA OR "primary progressive aphasia")[tiab]"
    
    You MUST call create_claims_with_queries with your results."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract claims and generate queries for:\n\n{diagnosis_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # COST OPTIMIZATION: Use cheaper model
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_claims_with_queries"}}
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            claims = function_args["claims_with_queries"]
            
            # Limit claims if too many
            if len(claims) > max_claims:
                claims = claims[:max_claims]
            
            return claims
        else:
            print("LLM did not make function call, using fallback")
            return []
        
    except Exception as e:
        print(f"Error in batch extraction: {e}")
        return []

#%%
# COST OPTIMIZATION: Simplified research function with fewer LLM calls
def research_claim_optimized(claim: Dict, max_results: int = 8) -> Dict:
    """
    COST OPTIMIZATION: Research claim with minimal LLM calls.
    Reduces from 4-5 LLM calls to 1-2 per claim.
    
    Args:
        claim (Dict): Claim with PubMed query
        max_results (int): Maximum results to retrieve
        
    Returns:
        Dict: Claim with research results
    """
    
    print(f"Researching claim: {claim['claim_text']}")
    
    # Step 1: Search PubMed (no LLM call - direct API)
    pmids = search_pubmed_direct(claim["pubmed_query"], max_results)
    
    if not pmids:
        return {
            **claim,
            "pmids": [],
            "abstracts": [],
            "evidence_summary": "No research evidence found.",
            "evidence_level": "none",
            "research_status": "no_evidence",
            "detailed_summary": "No research articles were found for this claim. This could be due to:\n1. The search terms being too specific or restrictive\n2. Limited research on this particular topic\n3. The need to use broader or alternative search terms\n\nConsider trying a more general search or rephrasing the query with more common medical terminology.",
            "claim_references": "No references available - no articles were found for this claim.",
            "supporting_evidence": [],
            "contradicting_evidence": []
        }
    
    # Step 2: Fetch abstracts (no LLM call - direct API)
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
            "contradicting_evidence": []
        }
    
    # Step 3: Single LLM call for summary AND analysis (COST OPTIMIZATION)
    summary_and_analysis = generate_summary_and_analysis(claim, abstracts)
    
    # Step 4: Generate claim-specific references (no LLM call - direct formatting)
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
        "claim_references": claim_references
    }

def search_pubmed_direct(search_str: str, max_results: int = 8) -> List[str]:
    """
    COST OPTIMIZATION: Direct PubMed search without LLM call.
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    try:
        # Search for PMIDs
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
        
        return pmids
        
    except Exception as e:
        print(f"Error in direct PubMed search: {e}")
        return []

def fetch_pubmed_abstracts_direct(pmids: List[str]) -> List[Dict]:
    """
    COST OPTIMIZATION: Direct abstract fetching without LLM call.
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

def generate_summary_and_analysis(claim: Dict, abstracts: List[Dict]) -> Dict:
    """
    COST OPTIMIZATION: Single LLM call for both summary and analysis.
    Reduces from 2-3 LLM calls to 1.
    """
    
    def create_summary_analysis(detailed_summary: str, brief_summary: str, evidence_level: str, research_status: str, supporting_evidence: List[str], contradicting_evidence: List[str]) -> dict:
        """Function that creates summary and analysis - called by the LLM"""
        return {
            "status": "success",
            "detailed_summary": detailed_summary,
            "brief_summary": brief_summary,
            "evidence_level": evidence_level,
            "research_status": research_status,
            "supporting_evidence": supporting_evidence,
            "contradicting_evidence": contradicting_evidence
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "create_summary_analysis",
            "description": "Create summary and analysis for a clinical claim",
            "parameters": {
                "type": "object",
                "properties": {
                    "detailed_summary": {"type": "string"},
                    "brief_summary": {"type": "string"},
                    "evidence_level": {
                        "type": "string",
                        "enum": ["strong", "moderate", "weak", "none", "mixed"]
                    },
                    "research_status": {
                        "type": "string",
                        "enum": ["supported", "contradicted", "mixed", "insufficient", "no_evidence"]
                    },
                    "supporting_evidence": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "contradicting_evidence": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["detailed_summary", "brief_summary", "evidence_level", "research_status", "supporting_evidence", "contradicting_evidence"]
            }
        }
    }]
    
    # COST OPTIMIZATION: Limit abstracts to reduce token usage
    limited_abstracts = abstracts[:5]  # Only use first 5 abstracts
    
    abstracts_text = ""
    for i, abstract in enumerate(limited_abstracts, 1):
        # COST OPTIMIZATION: Truncate abstracts to reduce tokens
        truncated_abstract = abstract['abstract'][:500] if len(abstract['abstract']) > 500 else abstract['abstract']
        abstracts_text += f"\n[{i}] {abstract['title']}\n{truncated_abstract}\n"
    
    system_prompt = """You are a clinical research expert. Analyze the evidence for a clinical claim.

    TASK: Create a comprehensive, detailed summary paragraph that integrates multiple sources with inline citations [1], [2], etc.
    
    REQUIREMENTS FOR DETAILED SUMMARY:
    1. Write a cohesive paragraph (3-5 sentences) that synthesizes findings from multiple sources
    2. Use inline citations [1], [2], [3], etc. to reference specific studies
    3. Include key findings, mechanisms, and clinical implications
    4. Connect different aspects of the evidence to tell a complete story
    5. Use academic writing style with proper transitions and logical flow
    6. Focus on the most relevant and high-quality evidence
    
    EXAMPLE FORMAT:
    "Frontotemporal dementia patients, including those with the semantic variant of primary progressive aphasia (svPPA), exhibit degradation in semantic memory due to atrophy in the anterior temporal lobe (ATL), which interferes with hierarchical semantic categorization [2][4][5]. The ATL is proposed to be a functionally unitary 'semantic hub' that supports both social and non-social semantic knowledge, indicating that semantic degradation in svPPA might reflect a disintegration of complex conceptual networks within this region [4]."
    
    REQUIREMENTS FOR BRIEF SUMMARY:
    - 2-3 sentences summarizing the main findings
    - Focus on key conclusions and clinical relevance
    
    ASSESSMENT:
    - Evidence level: strong/moderate/weak/none/mixed
    - Research status: supported/contradicted/mixed/insufficient/no_evidence
    - List specific supporting and contradicting evidence points
    
    You MUST call create_summary_analysis with your results."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Claim: {claim['claim_text']}\n\nAbstracts:\n{abstracts_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # COST OPTIMIZATION: Use cheaper model
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_summary_analysis"}}
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
                "contradicting_evidence": function_args["contradicting_evidence"]
            }
        else:
            return {
                "detailed_summary": "Analysis failed",
                "brief_summary": "Limited evidence available",
                "evidence_level": "weak",
                "research_status": "insufficient",
                "supporting_evidence": [],
                "contradicting_evidence": []
            }
        
    except Exception as e:
        print(f"Error in summary and analysis: {e}")
        return {
            "detailed_summary": "Error occurred",
            "brief_summary": "Analysis error",
            "evidence_level": "weak",
            "research_status": "insufficient",
            "supporting_evidence": [],
            "contradicting_evidence": []
        }

#%%
# COST OPTIMIZATION: Simplified assessment report
def generate_assessment_report_optimized(original_diagnosis: str, researched_claims: List[Dict]) -> str:
    """
    COST OPTIMIZATION: Generate assessment report with minimal token usage.
    """
    
    def create_optimized_report(report_text: str, key_recommendations: List[str]) -> dict:
        return {
            "status": "success",
            "report": report_text,
            "recommendations": key_recommendations
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "create_optimized_report",
            "description": "Create clinical assessment report",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_text": {"type": "string"},
                    "key_recommendations": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["report_text", "key_recommendations"]
            }
        }
    }]
    
    # COST OPTIMIZATION: Create concise summary for LLM
    claims_summary = ""
    for i, claim in enumerate(researched_claims, 1):
        claims_summary += f"\nClaim {i}: {claim['claim_text']}\n"
        claims_summary += f"Evidence: {claim['evidence_level']} - {claim['research_status']}\n"
        claims_summary += f"Summary: {claim['evidence_summary'][:100]}...\n"
    
    system_prompt = """You are a clinical researcher. Create a concise assessment report.

    REQUIREMENTS:
    1. Executive summary of findings
    2. Key clinical recommendations
    3. Areas needing further investigation
    4. Overall confidence assessment
    
    Keep the report focused and actionable."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Diagnosis: {original_diagnosis}\n\nFindings:\n{claims_summary}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # COST OPTIMIZATION: Use cheaper model
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_optimized_report"}}
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            return function_args["report_text"]
        else:
            return "Unable to generate report."
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return "Error occurred during report generation."

#%%
# COST OPTIMIZED MAIN PIPELINE
def run_clinical_diagnosis_exploration_optimized(diagnosis_text: str, max_results_per_claim: int = 8, max_claims: int = 8) -> Dict:
    """
    COST OPTIMIZED VERSION: Run clinical diagnosis exploration with minimal cost.
    
    COST REDUCTIONS:
    1. Batch claim extraction and query generation (2→1 LLM calls)
    2. Use GPT-3.5-turbo instead of GPT-4 (70% cost reduction)
    3. Direct PubMed API calls (no LLM calls)
    4. Single LLM call per claim for summary+analysis (4→1 LLM calls)
    5. Reduced token usage (truncated abstracts, concise prompts)
    6. Fewer claims processed (8 instead of 12)
    7. Fewer results per claim (8 instead of 15)
    
    Expected cost reduction: ~80-90% (from $3 to ~$0.30-0.60)
    """
    
    print("=== COST OPTIMIZED Clinical Diagnosis Exploration ===")
    print(f"Max claims: {max_claims}")
    print(f"Max results per claim: {max_results_per_claim}")
    print()
    
    # Step 1: Extract claims AND generate queries in one call (COST OPTIMIZATION)
    print("Step 1: Extracting claims and generating queries (batch)...")
    claims_with_queries = extract_claims_and_queries_batch(diagnosis_text, max_claims)
    print(f"Extracted {len(claims_with_queries)} claims with queries")
    print()
    
    # Step 2: Research each claim (1 LLM call per claim)
    print("Step 2: Researching claims...")
    researched_claims = []
    total_llm_calls = 1  # Initial batch extraction
    
    for i, claim in enumerate(claims_with_queries, 1):
        print(f"Researching claim {i}/{len(claims_with_queries)}: {claim['claim_text']}")
        
        # Single LLM call per claim (COST OPTIMIZATION)
        researched_claim = research_claim_optimized(claim, max_results_per_claim)
        researched_claims.append(researched_claim)
        total_llm_calls += 1
        
        print(f"  Evidence level: {researched_claim['evidence_level']}")
        print(f"  Research status: {researched_claim['research_status']}")
        
        # Display detailed summary for this claim (matching original format)
        print(f"  Detailed Summary: {researched_claim['detailed_summary'][:200]}...")
        print(f"  Articles found: {len(researched_claim['abstracts'])}")
        print()
        
        # Display claim-specific references (matching original format)
        print(f"  Claim References:")
        print(researched_claim['claim_references'])
        print("-" * 80)
        print()
    
    # Step 3: Generate assessment report (1 LLM call)
    print("Step 3: Generating assessment report...")
    assessment_report = generate_assessment_report_optimized(diagnosis_text, researched_claims)
    total_llm_calls += 1
    print()
    
    # Step 4: Format references (no LLM call - direct formatting)
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
    
    # Simple reference formatting without LLM call
    references = format_references_simple(unique_abstracts)
    print(f"Formatted {len(unique_abstracts)} references")
    print()
    
    # Compile results
    results = {
        "original_diagnosis": diagnosis_text,
        "claims": researched_claims,
        "assessment_report": assessment_report,
        "references": references,
        "total_llm_calls": total_llm_calls,
        "total_articles": len(unique_abstracts),
        "claims_researched": len(researched_claims),
        "cost_optimization": "GPT-3.5-turbo, batched calls, direct APIs"
    }
    
    print("=== OPTIMIZED PIPELINE COMPLETE ===")
    print(f"Total LLM calls: {results['total_llm_calls']} (vs ~50+ in original)")
    print(f"Claims researched: {results['claims_researched']}")
    print(f"Total articles: {results['total_articles']}")
    print(f"Estimated cost reduction: 80-90%")
    print()
    
    return results

def format_references_simple(abstracts: List[Dict]) -> str:
    """
    COST OPTIMIZATION: Simple reference formatting without LLM call.
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
# Test the cost-optimized pipeline
if __name__ == "__main__":
    test_diagnosis = """
    Patient presents with progressive language difficulties characterized by fluent but circumlocutory speech, 
    severe anomia, and word-finding difficulties. MRI shows anterior temporal lobe atrophy, particularly 
    in the left hemisphere. The patient demonstrates preserved syntax and phonology but impaired semantic 
    knowledge. Based on these findings, the diagnosis is semantic variant primary progressive aphasia (svPPA). 
    The underlying pathology is likely TDP-43 proteinopathy, and the prognosis includes eventual progression 
    to affect other cognitive domains beyond language.
    """
    
    print("=== COST OPTIMIZED TEST ===")
    
    # Run the optimized pipeline
    results = run_clinical_diagnosis_exploration_optimized(
        test_diagnosis, 
        max_results_per_claim=8,  # Reduced from 15
        max_claims=8  # Reduced from 12
    )
    
    # Display results (matching original format)
    print("=== Final Results ===")
    print("Clinical Assessment Report:")
    print(results['assessment_report'])
    print()
    print("Overall References:")
    print(results['references'])
    print()
    print("=== DETAILED PARAGRAPHS FOR EACH CLAIM ===")
    for i, claim in enumerate(results['claims'], 1):
        print(f"CLAIM {i}: {claim['claim_text']}")
        print(f"Type: {claim['claim_type']}")
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
        
        print("=" * 100)
        print()
    
    print(f"Total LLM calls: {results['total_llm_calls']}")
    print(f"Cost optimization: {results['cost_optimization']}")

# %% 