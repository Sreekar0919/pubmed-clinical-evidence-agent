import gradio as gr
import os
import sys
from typing import Dict, Any
import json

# Import the ENHANCED clinical diagnosis exploration functions
from clinical_diag_exp_enhanced import run_clinical_diagnosis_exploration_enhanced

# Global variable to store the API key
stored_api_key = None

def validate_and_store_api_key(api_key: str) -> str:
    """
    Validate the user-provided OpenAI API key by making a test request.
    """
    global stored_api_key
    
    if not api_key or api_key.strip() == "":
        return "❌ Please enter your OpenAI API key."

    try:
        # Test the API key by making a simple request
        from openai import OpenAI
        # Use the provided API key directly in the client
        client = OpenAI(api_key=api_key.strip())

        # Make a simple test call with GPT-4o (enhanced version)
        response = client.chat.completions.create(
            model="gpt-4o",  # ENHANCED: Use GPT-4o for better quality
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )

        # Store the API key for later use
        stored_api_key = api_key.strip()
        
        return "✅ OpenAI API key validated and stored successfully! You can now use the ENHANCED clinical diagnosis exploration tool with diagnosis-first approach, fuzzy search, and GPT-4o analysis."
    
    except Exception as e:
        return f"❌ Error validating API key: {str(e)}"

def run_clinical_exploration_enhanced(diagnosis_text: str, max_results: int, max_claims: int) -> Dict[str, Any]:
    """
    Run the ENHANCED clinical diagnosis exploration pipeline.
    """
    global stored_api_key
    
    if not stored_api_key:
        return {
            "error": "❌ Please validate your OpenAI API key first before running the clinical exploration."
        }
    
    if not diagnosis_text or diagnosis_text.strip() == "":
        return {
            "error": "❌ Please enter a clinical diagnosis to analyze."
        }
    
    try:
        # Run the ENHANCED clinical diagnosis exploration
        results = run_clinical_diagnosis_exploration_enhanced(
            diagnosis_text=diagnosis_text.strip(),
            max_results_per_claim=max_results,
            max_claims=max_claims
        )
        
        # Format the results for display
        formatted_results = {
            "diagnosis": results.get("diagnosis", "No diagnosis extracted"),
            "revised_note": results.get("revised_note", "No revised note generated"),
            "total_claims": results.get("claims_researched", 0),
            "total_articles": results.get("total_articles", 0),
            "total_llm_calls": results.get("total_llm_calls", 0),
            "claims_summary": "",
            "references": results.get("references", "No references available."),
            "missing_information": results.get("missing_information", []),
            "enhancement": results.get("enhancement", "Enhanced features"),
        }
        
        # Create a detailed summary of claims with full research summaries and references
        claims = results.get("claims", [])
        claims_summary = "## Claims Analyzed:\n\n"
        for i, claim in enumerate(claims, 1):
            claims_summary += f"### Claim {i}: {claim['claim_text']}\n"
            claims_summary += f"- **Type:** {claim['claim_type']}\n"
            claims_summary += f"- **Evidence Level:** {claim['evidence_level']}\n"
            claims_summary += f"- **Research Status:** {claim['research_status']}\n"
            claims_summary += f"- **Evidence Quality:** {claim.get('evidence_quality', 'unknown')}\n"
            claims_summary += f"- **Articles Found:** {len(claim['abstracts'])}\n"
            claims_summary += f"- **PubMed Query:** `{claim['pubmed_query']}`\n"
            claims_summary += f"- **Query Strategy:** {claim.get('query_strategy', 'Standard query')}\n\n"
            
            # Add the detailed summary with inline citations
            claims_summary += f"**Detailed Summary:** {claim['detailed_summary']}\n\n"
            claims_summary += f"**Brief Summary:** {claim['evidence_summary']}\n\n"
            
            # Add critical assessment
            if claim.get('critical_assessment'):
                claims_summary += f"**Critical Assessment:** {claim['critical_assessment']}\n\n"
            
            # Add supporting/contradicting evidence if available
            if claim.get('supporting_evidence'):
                claims_summary += "**Supporting Evidence:**\n"
                for evidence in claim['supporting_evidence']:
                    claims_summary += f"- {evidence}\n"
                claims_summary += "\n"
            
            if claim.get('contradicting_evidence'):
                claims_summary += "**Contradicting Evidence:**\n"
                for evidence in claim['contradicting_evidence']:
                    claims_summary += f"- {evidence}\n"
                claims_summary += "\n"
            
            # Add methodological issues if available
            if claim.get('methodological_issues'):
                claims_summary += "**Methodological Issues:**\n"
                for issue in claim['methodological_issues']:
                    claims_summary += f"- {issue}\n"
                claims_summary += "\n"
            
            # Add the claim-specific references
            claims_summary += f"**Claim-Specific References:**\n{claim['claim_references']}\n\n"
            
            claims_summary += "---\n\n"
        
        formatted_results["claims_summary"] = claims_summary
        
        return formatted_results
        
    except Exception as e:
        return {
            "error": f"❌ Error running clinical exploration: {str(e)}"
        }

def create_interface():
    """
    Create the ENHANCED Gradio interface with multiple tabs.
    """
    
    with gr.Blocks(title="Enhanced Clinical Diagnosis Explorer", theme=gr.themes.Soft()) as iface:
        
        gr.Markdown("# Enhanced Clinical Diagnosis Explorer")
        gr.Markdown("**Features:** Diagnosis-first approach, fuzzy search, GPT-4o analysis, revised clinical note")
        
        # API Key Setup    
        with gr.Tab("🔑 API Key Setup"):
            gr.Markdown("## Step 1: Enter Your OpenAI API Key")
            gr.Markdown("Enter your OpenAI API key to get started. This will be used for all clinical analysis.")
            gr.Markdown("**Note:** This enhanced version uses GPT-4o for better quality analysis.")
            
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-proj-...",
                type="password",
                lines=1
            )
            
            validate_btn = gr.Button("🔑 Validate API Key", variant="primary")
            api_status = gr.Textbox(label="API Key Status", interactive=False)
            
            validate_btn.click(
                fn=validate_and_store_api_key,
                inputs=api_key_input,
                outputs=api_status
            )
        
        with gr.Tab("🏥 Enhanced Clinical Analysis"):
            gr.Markdown("## Step 2: Analyze Clinical Diagnosis")
            gr.Markdown("Enter a clinical diagnosis to extract claims and validate each against medical literature.")
            gr.Markdown("**Enhanced Features:**")
            gr.Markdown("- 🔍 **Diagnosis-First Approach:** Extracts primary diagnosis, then finds related claims")
            gr.Markdown("- 🔎 **Fuzzy Search:** Multiple fallback strategies to ensure PubMed results")
            gr.Markdown("- 📝 **Revised Clinical Note:** Outputs evidence-based note with citations")
            gr.Markdown("- 🎯 **GPT-4o Analysis:** Enhanced rigor and willingness to say 'no' when evidence doesn't support claims")
            gr.Markdown("- 📊 **PubMed Queries:** Transparent display of search queries for each claim")
            
            diagnosis_input = gr.Textbox(
                label="Clinical Diagnosis",
                placeholder="Enter the clinical diagnosis and reasoning here...",
                lines=10
            )
            
            with gr.Row():
                max_results = gr.Slider(
                    minimum=5,
                    maximum=20,
                    value=12,
                    step=1,
                    label="Max Results per Claim",
                    info="Number of PubMed articles to analyze per claim (increased for better analysis)"
                )
                
                max_claims = gr.Slider(
                    minimum=3,
                    maximum=12,
                    value=8,
                    step=1,
                    label="Max Claims to Extract",
                    info="Maximum number of claims to extract from diagnosis"
                )
            
            analyze_btn = gr.Button("🔍 Analyze Diagnosis (Enhanced)", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Results")
                    results_output = gr.JSON(label="Analysis Results", visible=False)
                    error_output = gr.Textbox(label="Error", visible=False)
                
                with gr.Column():
                    gr.Markdown("### Analysis Summary")
                    summary_output = gr.Markdown(label="Analysis Summary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Primary Diagnosis")
                    diagnosis_output = gr.Markdown(label="Extracted Diagnosis")
                
                with gr.Column():
                    gr.Markdown("### Missing Information")
                    missing_info_output = gr.Markdown(label="Missing Information")
            
            with gr.Row():
                gr.Markdown("### Revised Clinical Note")
                revised_note_output = gr.Markdown(label="Evidence-Based Clinical Note")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Claims Analysis")
                    claims_output = gr.Markdown(label="Detailed Claims Analysis")
                
                with gr.Column():
                    gr.Markdown("### References")
                    references_output = gr.Markdown(label="Research References")
            
            def process_enhanced_analysis(diagnosis, max_res, max_clms):
                results = run_clinical_exploration_enhanced(diagnosis, max_res, max_clms)
                
                if "error" in results:
                    return {
                        results_output: gr.update(value=None, visible=False),
                        error_output: gr.update(value=results["error"], visible=True),
                        summary_output: gr.update(value=""),
                        diagnosis_output: gr.update(value=""),
                        missing_info_output: gr.update(value=""),
                        revised_note_output: gr.update(value=""),
                        claims_output: gr.update(value=""),
                        references_output: gr.update(value=""),
                    }
                else:
                    # Format analysis summary
                    summary_text = f"""
## Analysis Summary

**Total Claims Researched:** {results.get("total_claims", 0)}
**Total Articles Found:** {results.get("total_articles", 0)}
**Total LLM Calls:** {results.get("total_llm_calls", 0)}
**Enhanced Features:** {results.get("enhancement", "Enhanced analysis")}

### Key Features:
- 🔍 **Diagnosis-First Approach:** Extracted primary diagnosis before analyzing claims
- 🔎 **Fuzzy Search:** Used multiple fallback strategies to ensure PubMed results
- 📝 **Revised Clinical Note:** Generated evidence-based note with citations
- 🎯 **GPT-4o Analysis:** Enhanced rigor and critical assessment
- 📊 **Transparent Queries:** Displayed PubMed queries for each claim
"""
                    
                    # Format diagnosis
                    diagnosis_text = f"""
## Primary Diagnosis

**Extracted Diagnosis:** {results.get("diagnosis", "No diagnosis extracted")}

This diagnosis was identified as the primary clinical condition and used as the foundation for claim extraction and PubMed query generation.
"""
                    
                    # Format missing information
                    missing_info = results.get("missing_information", [])
                    missing_text = "## Missing Information\n\n"
                    if missing_info:
                        for i, info in enumerate(missing_info, 1):
                            missing_text += f"{i}. ❓ {info}\n"
                    else:
                        missing_text += "No missing information identified."
                    
                    return {
                        results_output: gr.update(value=results, visible=True),
                        error_output: gr.update(value="", visible=False),
                        summary_output: gr.update(value=summary_text),
                        diagnosis_output: gr.update(value=diagnosis_text),
                        missing_info_output: gr.update(value=missing_text),
                        revised_note_output: gr.update(value=results["revised_note"]),
                        claims_output: gr.update(value=results["claims_summary"]),
                        references_output: gr.update(value=results["references"]),
                    }
            
            analyze_btn.click(
                fn=process_enhanced_analysis,
                inputs=[diagnosis_input, max_results, max_claims],
                outputs=[results_output, error_output, summary_output, diagnosis_output, missing_info_output, revised_note_output, claims_output, references_output]
            )
            
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()