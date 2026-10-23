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
# ENHANCED: Diagnosis revision and alternative suggestions
def analyze_diagnosis_alternatives(diagnosis_text: str) -> Dict:
    """
    ENHANCED: Analyze the diagnosis and suggest alternatives or revisions.
    Uses GPT-4 for critical analysis to identify potential misdiagnoses.
    
    Args:
        diagnosis_text (str): Clinical diagnosis with reasoning
        
    Returns:
        Dict: Analysis with alternative diagnoses and confidence assessment
    """
    
    def create_diagnosis_analysis(primary_diagnosis: str, confidence_level: str, 
                                alternative_diagnoses: List[Dict], 
                                red_flags: List[str], 
                                missing_information: List[str],
                                reasoning: str) -> dict:
        """Function that creates diagnosis analysis - called by the LLM"""
        return {
            "status": "success",
            "primary_diagnosis": primary_diagnosis,
            "confidence_level": confidence_level,
            "alternative_diagnoses": alternative_diagnoses,
            "red_flags": red_flags,
            "missing_information": missing_information,
            "reasoning": reasoning
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "create_diagnosis_analysis",
            "description": "Analyze diagnosis and suggest alternatives",
            "parameters": {
                "type": "object",
                "properties": {
                    "primary_diagnosis": {"type": "string"},
                    "confidence_level": {
                        "type": "string",
                        "enum": ["high", "moderate", "low", "very_low"]
                    },
                    "alternative_diagnoses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "diagnosis": {"type": "string"},
                                "probability": {"type": "string"},
                                "supporting_evidence": {"type": "string"},
                                "distinguishing_features": {"type": "string"}
                            },
                            "required": ["diagnosis", "probability", "supporting_evidence", "distinguishing_features"]
                        }
                    },
                    "red_flags": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "missing_information": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "reasoning": {"type": "string"}
                },
                "required": ["primary_diagnosis", "confidence_level", "alternative_diagnoses", "red_flags", "missing_information", "reasoning"]
            }
        }
    }]
    
    system_prompt = """You are a senior clinical neurologist with expertise in differential diagnosis. 
    Critically analyze the provided diagnosis and identify potential issues, alternatives, or missing information.

    CRITICAL ANALYSIS REQUIREMENTS:
    1. **Be willing to say "NO"** - If the diagnosis doesn't fit the symptoms, say so clearly
    2. **Identify red flags** - Symptoms that don't match the diagnosis or suggest other conditions
    3. **Consider age-appropriate diagnoses** - Some conditions are extremely rare in certain age groups
    4. **Look for missing information** - What key details are needed for a proper diagnosis?
    5. **Suggest alternatives** - What other conditions could explain these symptoms?
    6. **Assess confidence** - How certain can we be about this diagnosis?

    DIAGNOSIS EVALUATION FRAMEWORK:
    - Symptom consistency with diagnosis
    - Age and demographic appropriateness  
    - Temporal progression pattern
    - Imaging findings correlation
    - Missing clinical information
    - Alternative explanations

    You MUST be critical and honest - if the diagnosis is wrong, say so clearly.
    You MUST call create_diagnosis_analysis with your results."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this diagnosis critically:\n\n{diagnosis_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # ENHANCED: Use GPT-4 for critical analysis
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_diagnosis_analysis"}}
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            return {
                "primary_diagnosis": function_args["primary_diagnosis"],
                "confidence_level": function_args["confidence_level"],
                "alternative_diagnoses": function_args["alternative_diagnoses"],
                "red_flags": function_args["red_flags"],
                "missing_information": function_args["missing_information"],
                "reasoning": function_args["reasoning"]
            }
        else:
            return {
                "primary_diagnosis": "Analysis failed",
                "confidence_level": "very_low",
                "alternative_diagnoses": [],
                "red_flags": ["Unable to analyze diagnosis"],
                "missing_information": ["Analysis failed"],
                "reasoning": "Analysis failed"
            }
        
    except Exception as e:
        print(f"Error in diagnosis analysis: {e}")
        return {
            "primary_diagnosis": "Error occurred",
            "confidence_level": "very_low",
            "alternative_diagnoses": [],
            "red_flags": ["Analysis error"],
            "missing_information": ["Analysis error"],
            "reasoning": "Error occurred during analysis"
        }

#%%
# ENHANCED: More rigorous claim extraction with critical evaluation
def extract_claims_and_queries_enhanced(diagnosis_text: str, max_claims: int = 8) -> Dict:
    """
    ENHANCED: Extract claims with more rigorous evaluation and critical analysis.
    Uses GPT-4 for better claim identification and validation.
    
    Args:
        diagnosis_text (str): Clinical diagnosis with reasoning
        max_claims (int): Maximum number of claims to extract
        
    Returns:
        Dict: Claims with PubMed queries and critical assessment
    """
    
    def create_enhanced_claims(claims_with_queries: List[Dict], diagnosis_summary: str, 
                             critical_issues: List[str], claim_validation: List[Dict]) -> dict:
        """Function that creates enhanced claims - called by the LLM"""
        return {
            "status": "success",
            "claims_with_queries": claims_with_queries,
            "diagnosis_summary": diagnosis_summary,
            "critical_issues": critical_issues,
            "claim_validation": claim_validation
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "create_enhanced_claims",
            "description": "Extract claims with enhanced validation",
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
                                    "enum": ["high", "medium", "low", "very_low"]
                                },
                                "clinical_evidence": {"type": "string"},
                                "research_question": {"type": "string"},
                                "pubmed_query": {"type": "string"},
                                "potential_issues": {"type": "string"}
                            },
                            "required": ["claim_id", "claim_text", "claim_type", "confidence", "clinical_evidence", "research_question", "pubmed_query", "potential_issues"]
                        }
                    },
                    "diagnosis_summary": {"type": "string"},
                    "critical_issues": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "claim_validation": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "claim_id": {"type": "string"},
                                "is_valid": {"type": "boolean"},
                                "issues": {"type": "string"},
                                "suggestions": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["claims_with_queries", "diagnosis_summary", "critical_issues", "claim_validation"]
            }
        }
    }]
    
    system_prompt = """You are a clinical research expert with expertise in evidence-based medicine. 
    Extract claims from the diagnosis and generate PubMed queries, but be CRITICAL and RIGOROUS.

    ENHANCED REQUIREMENTS:
    1. **Be critical** - Don't accept claims at face value, question everything
    2. **Identify problematic claims** - Mark claims that seem dubious or lack evidence
    3. **Generate counter-arguments** - What evidence might contradict each claim?
    4. **Assess claim validity** - Rate confidence from high to very_low
    5. **Identify missing information** - What would make each claim more credible?
    6. **Generate rigorous queries** - Focus on finding BOTH supporting AND contradicting evidence
    
    CLAIM VALIDATION FRAMEWORK:
    - Is the claim testable?
    - Is there sufficient clinical detail?
    - Are there alternative explanations?
    - What evidence would contradict this?
    - What additional information is needed?
    
    QUERY STRATEGIES (ENHANCED):
    - Include both positive and negative search terms
    - Search for contradictory evidence explicitly
    - Use broader terms to capture edge cases
    - Include age-specific and demographic factors
    
    EXAMPLE ENHANCED QUERIES:
    - "diagnosis" → "(diagnosis OR "clinical features" OR "differential diagnosis")[tiab]"
    - "atrophy" → "(atrophy OR "brain imaging" OR MRI)[tiab]"
    - "symptoms" → "(symptoms OR "clinical presentation" OR assessment)[tiab]"
    
    You MUST be rigorous and critical. You MUST call create_enhanced_claims with your results."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract claims and generate queries for:\n\n{diagnosis_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # ENHANCED: Use GPT-4 for better claim extraction
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_enhanced_claims"}}
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            claims = function_args["claims_with_queries"]
            
            # Limit claims if too many
            if len(claims) > max_claims:
                claims = claims[:max_claims]
            
            return {
                "claims": claims,
                "critical_issues": function_args.get("critical_issues", []),
                "claim_validation": function_args.get("claim_validation", [])
            }
        else:
            print("LLM did not make function call, using fallback")
            return {"claims": [], "critical_issues": ["Extraction failed"], "claim_validation": []}
        
    except Exception as e:
        print(f"Error in enhanced claim extraction: {e}")
        return {"claims": [], "critical_issues": [f"Error: {e}"], "claim_validation": []}

#%%
# ENHANCED: More rigorous research function with fuzzy search
def research_claim_enhanced(claim: Dict, max_results: int = 12) -> Dict:
    """
    ENHANCED: Research claim with more rigorous evidence evaluation and fuzzy search.
    Uses GPT-4 for better analysis and is willing to say "no" when evidence doesn't support claims.
    
    Args:
        claim (Dict): Claim with PubMed query
        max_results (int): Maximum results to retrieve
        
    Returns:
        Dict: Claim with enhanced research results
    """
    
    print(f"Researching claim: {claim['claim_text']}")
    print(f"Potential issues: {claim.get('potential_issues', 'None identified')}")
    
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
        "cognitive", "neurological", "brain", "clinical", "assessment"
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
    ENHANCED: Comprehensive analysis with GPT-4 for better rigor.
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
        {"role": "user", "content": f"Claim: {claim['claim_text']}\nPotential Issues: {claim.get('potential_issues', 'None identified')}\n\nAbstracts:\n{abstracts_text}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # ENHANCED: Use GPT-4 for better analysis
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
# ENHANCED: Comprehensive assessment report with diagnosis revision
def generate_enhanced_assessment_report(original_diagnosis: str, diagnosis_analysis: Dict, 
                                      researched_claims: List[Dict]) -> str:
    """
    ENHANCED: Generate comprehensive assessment report with diagnosis revision suggestions.
    """
    
    def create_enhanced_report(report_text: str, key_recommendations: List[str], 
                             diagnosis_revision: str, confidence_statement: str,
                             next_steps: List[str]) -> dict:
        return {
            "status": "success",
            "report": report_text,
            "recommendations": key_recommendations,
            "diagnosis_revision": diagnosis_revision,
            "confidence_statement": confidence_statement,
            "next_steps": next_steps
        }
    
    tools = [{
        "type": "function",
        "function": {
            "name": "create_enhanced_report",
            "description": "Create enhanced clinical assessment report",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_text": {"type": "string"},
                    "key_recommendations": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "diagnosis_revision": {"type": "string"},
                    "confidence_statement": {"type": "string"},
                    "next_steps": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["report_text", "key_recommendations", "diagnosis_revision", "confidence_statement", "next_steps"]
            }
        }
    }]
    
    # Create comprehensive summary for LLM
    claims_summary = ""
    for i, claim in enumerate(researched_claims, 1):
        claims_summary += f"\nClaim {i}: {claim['claim_text']}\n"
        claims_summary += f"Evidence: {claim['evidence_level']} - {claim['research_status']}\n"
        claims_summary += f"Quality: {claim.get('evidence_quality', 'unknown')}\n"
        claims_summary += f"Critical Assessment: {claim.get('critical_assessment', 'No assessment')}\n"
        claims_summary += f"Summary: {claim['evidence_summary'][:150]}...\n"
    
    diagnosis_summary = f"""
    Original Diagnosis: {original_diagnosis}
    
    Diagnosis Analysis:
    - Primary Diagnosis: {diagnosis_analysis.get('primary_diagnosis', 'Unknown')}
    - Confidence Level: {diagnosis_analysis.get('confidence_level', 'unknown')}
    - Red Flags: {', '.join(diagnosis_analysis.get('red_flags', []))}
    - Missing Information: {', '.join(diagnosis_analysis.get('missing_information', []))}
    - Reasoning: {diagnosis_analysis.get('reasoning', 'No reasoning provided')}
    
    Alternative Diagnoses:
    """
    
    for alt in diagnosis_analysis.get('alternative_diagnoses', []):
        diagnosis_summary += f"- {alt.get('diagnosis', 'Unknown')} (Probability: {alt.get('probability', 'Unknown')})\n"
    
    system_prompt = """You are a senior clinical neurologist. Create a comprehensive assessment report that includes diagnosis revision suggestions.

    REQUIREMENTS:
    1. **Executive Summary** - Key findings and overall assessment
    2. **Diagnosis Evaluation** - Critical assessment of the original diagnosis
    3. **Evidence Review** - Summary of research findings for each claim
    4. **Diagnosis Revision** - Suggest better diagnoses if the original is problematic
    5. **Confidence Assessment** - How certain can we be about the diagnosis?
    6. **Key Recommendations** - Specific clinical actions needed
    7. **Next Steps** - What additional information or testing is needed?
    
    DIAGNOSIS REVISION GUIDELINES:
    - If the original diagnosis is clearly wrong, suggest better alternatives
    - Consider age-appropriate diagnoses
    - Look for more common conditions that could explain the symptoms
    - Consider whether symptoms might be due to multiple conditions
    - Assess whether the diagnosis is premature or needs more evidence
    
    Be honest and critical - if the diagnosis is wrong, say so clearly and suggest better alternatives.
    You MUST call create_enhanced_report with your results."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Diagnosis Information:\n{diagnosis_summary}\n\nResearch Findings:\n{claims_summary}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # ENHANCED: Use GPT-4 for better report generation
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "create_enhanced_report"}}
        )
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            function_args = safe_json_loads(tool_call.function.arguments)
            
            return function_args["report_text"]
        else:
            return "Unable to generate enhanced report."
        
    except Exception as e:
        print(f"Error generating enhanced report: {e}")
        return "Error occurred during enhanced report generation."

#%%
# ENHANCED MAIN PIPELINE
def run_clinical_diagnosis_exploration_enhanced(diagnosis_text: str, max_results_per_claim: int = 12, max_claims: int = 8) -> Dict:
    """
    ENHANCED VERSION: Run clinical diagnosis exploration with diagnosis revision and improved rigor.
    
    ENHANCEMENTS:
    1. Diagnosis analysis and alternative suggestions
    2. More rigorous claim extraction with critical evaluation
    3. Enhanced evidence analysis with GPT-4
    4. Willingness to say "no" when evidence doesn't support claims
    5. Comprehensive assessment with diagnosis revision
    6. Fuzzy search to ensure results are found
    7. Better cost-quality balance (increased cost for better results)
    
    Expected cost: ~$2-4 (vs $0.30-0.60 in optimized version, but much better quality)
    """
    
    print("=== ENHANCED Clinical Diagnosis Exploration ===")
    print(f"Max claims: {max_claims}")
    print(f"Max results per claim: {max_results_per_claim}")
    print()
    
    # Step 1: Analyze diagnosis and suggest alternatives
    print("Step 1: Analyzing diagnosis and suggesting alternatives...")
    diagnosis_analysis = analyze_diagnosis_alternatives(diagnosis_text)
    print(f"Primary diagnosis: {diagnosis_analysis['primary_diagnosis']}")
    print(f"Confidence level: {diagnosis_analysis['confidence_level']}")
    print(f"Red flags: {len(diagnosis_analysis['red_flags'])} identified")
    print(f"Alternative diagnoses: {len(diagnosis_analysis['alternative_diagnoses'])} suggested")
    print()
    
    # Step 2: Extract claims with enhanced validation
    print("Step 2: Extracting claims with enhanced validation...")
    claims_extraction = extract_claims_and_queries_enhanced(diagnosis_text, max_claims)
    claims_with_queries = claims_extraction["claims"]
    critical_issues = claims_extraction["critical_issues"]
    claim_validation = claims_extraction["claim_validation"]
    
    print(f"Extracted {len(claims_with_queries)} claims")
    print(f"Critical issues: {len(critical_issues)} identified")
    print(f"Claims requiring validation: {len(claim_validation)}")
    print()
    
    # Step 3: Research each claim with enhanced rigor and fuzzy search
    print("Step 3: Researching claims with enhanced rigor and fuzzy search...")
    researched_claims = []
    total_llm_calls = 2  # Initial diagnosis analysis + claim extraction
    
    for i, claim in enumerate(claims_with_queries, 1):
        print(f"Researching claim {i}/{len(claims_with_queries)}: {claim['claim_text']}")
        print(f"  Confidence: {claim['confidence']}")
        print(f"  Potential issues: {claim.get('potential_issues', 'None')}")
        
        # Enhanced LLM call per claim with fuzzy search
        researched_claim = research_claim_enhanced(claim, max_results_per_claim)
        researched_claims.append(researched_claim)
        total_llm_calls += 1
        
        print(f"  Evidence level: {researched_claim['evidence_level']}")
        print(f"  Research status: {researched_claim['research_status']}")
        print(f"  Evidence quality: {researched_claim.get('evidence_quality', 'unknown')}")
        print(f"  Critical assessment: {researched_claim.get('critical_assessment', 'No assessment')[:100]}...")
        print(f"  Articles found: {len(researched_claim['abstracts'])}")
        print()
        
        # Display claim-specific references
        print(f"  Claim References:")
        print(researched_claim['claim_references'])
        print("-" * 80)
        print()
    
    # Step 4: Generate enhanced assessment report
    print("Step 4: Generating enhanced assessment report...")
    assessment_report = generate_enhanced_assessment_report(diagnosis_text, diagnosis_analysis, researched_claims)
    total_llm_calls += 1
    print()
    
    # Step 5: Format references
    print("Step 5: Formatting references...")
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
        "original_diagnosis": diagnosis_text,
        "diagnosis_analysis": diagnosis_analysis,
        "claims": researched_claims,
        "assessment_report": assessment_report,
        "references": references,
        "total_llm_calls": total_llm_calls,
        "total_articles": len(unique_abstracts),
        "claims_researched": len(researched_claims),
        "critical_issues": critical_issues,
        "claim_validation": claim_validation,
        "enhancement": "GPT-4 analysis, diagnosis revision, enhanced rigor, fuzzy search"
    }
    
    print("=== ENHANCED PIPELINE COMPLETE ===")
    print(f"Total LLM calls: {results['total_llm_calls']}")
    print(f"Claims researched: {results['claims_researched']}")
    print(f"Total articles: {results['total_articles']}")
    print(f"Critical issues identified: {len(results['critical_issues'])}")
    print(f"Enhanced features: Diagnosis revision, improved rigor, GPT-4 analysis, fuzzy search")
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