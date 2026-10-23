import gradio as gr
import os
import sys
from typing import Dict, Any
import json

# Import the COST OPTIMIZED clinical diagnosis exploration functions
from clinical_diag_exp_cost_optimized import run_clinical_diagnosis_exploration_optimized

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

        # Make a simple test call with GPT-3.5-turbo (cost optimization)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # COST OPTIMIZATION: Use cheaper model
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )

        # Store the API key for later use
        stored_api_key = api_key.strip()
        
        return "✅ OpenAI API key validated and stored successfully! You can now use the COST OPTIMIZED clinical diagnosis exploration tool."
    
    except Exception as e:
        return f"❌ Error validating API key: {str(e)}"

def run_clinical_exploration_optimized(diagnosis_text: str, max_results: int, max_claims: int) -> Dict[str, Any]:
    """
    Run the COST OPTIMIZED clinical diagnosis exploration pipeline.
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
        # Run the COST OPTIMIZED clinical diagnosis exploration
        results = run_clinical_diagnosis_exploration_optimized(
            diagnosis_text=diagnosis_text.strip(),
            max_results_per_claim=max_results,
            max_claims=max_claims
        )
        
        # Format the results for display
        formatted_results = {
            "assessment_report": results.get("assessment_report", "No assessment report generated."),
            "total_claims": results.get("claims_researched", 0),
            "total_articles": results.get("total_articles", 0),
            "total_llm_calls": results.get("total_llm_calls", 0),
            "claims_summary": "",
            "references": results.get("references", "No references available."),
        }
        
        # Create a detailed summary of claims with full research summaries and references
        claims = results.get("claims", [])
        claims_summary = "## Claims Analyzed:\n\n"
        for i, claim in enumerate(claims, 1):
            claims_summary += f"### Claim {i}: {claim['claim_text']}\n"
            claims_summary += f"- **Type:** {claim['claim_type']}\n"
            claims_summary += f"- **Evidence Level:** {claim['evidence_level']}\n"
            claims_summary += f"- **Research Status:** {claim['research_status']}\n"
            claims_summary += f"- **Articles Found:** {len(claim['abstracts'])}\n"
            claims_summary += f"- **PubMed Query:** {claim['pubmed_query']}\n\n"
            
            # Add the detailed summary with inline citations
            claims_summary += f"**Summary:** {claim['detailed_summary']}\n\n {claim['evidence_summary']}\n\n "
            
            # Add the claim-specific references
            claims_summary += f"**References:**\n{claim['claim_references']}\n\n"
            
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
            
            claims_summary += "---\n\n"
        
        formatted_results["claims_summary"] = claims_summary
        
        return formatted_results
        
    except Exception as e:
        return {
            "error": f"❌ Error running clinical exploration: {str(e)}"
        }

def create_interface():
    """
    Create the COST OPTIMIZED Gradio interface with multiple tabs.
    """
    
    with gr.Blocks(title="Clinical Diagnosis Explorer", theme=gr.themes.Soft()) as iface:
        
        gr.Markdown("Validate your OpenAI API key and explore clinical diagnoses with evidence-based research.")
        
        # Cost optimization info    
        with gr.Tab("🔑 API Key Setup"):
            gr.Markdown("## Step 1: Enter Your OpenAI API Key")
            gr.Markdown("Enter your OpenAI API key to get started. This will be used for all clinical analysis.")
            
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
        
        with gr.Tab("🏥 Clinical Analysis"):
            gr.Markdown("## Step 2: Analyze Clinical Diagnosis")
            gr.Markdown("Enter a clinical diagnosis to break it down into claims and validate each against medical literature.")
            
            
            diagnosis_input = gr.Textbox(
                label="Clinical Diagnosis",
                placeholder="Enter the clinical diagnosis and reasoning here...",
                lines=8
            )
            
            with gr.Row():
                max_results = gr.Slider(
                    minimum=3,
                    maximum=15,
                    value=8,
                    step=1,
                    label="Max Results per Claim",
                    info="Number of PubMed articles to analyze per claim"
                )
                
                max_claims = gr.Slider(
                    minimum=3,
                    maximum=12,
                    value=8,
                    step=1,
                    label="Max Claims to Extract",
                    info="Maximum number of claims to extract from diagnosis"
                )
            
            analyze_btn = gr.Button("🔍 Analyze Diagnosis", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Results")
                    results_output = gr.JSON(label="Analysis Results", visible=False)
                    error_output = gr.Textbox(label="Error", visible=False)
                
                with gr.Column():
                    gr.Markdown("### Clinical Assessment Report")
                    assessment_output = gr.Markdown(label="Assessment Report")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Claims Summary")
                    claims_output = gr.Markdown(label="Claims Analysis")
                
                with gr.Column():
                    gr.Markdown("### References")
                    references_output = gr.Markdown(label="Research References")
            
            def process_analysis(diagnosis, max_res, max_clms):
                results = run_clinical_exploration_optimized(diagnosis, max_res, max_clms)
                
                if "error" in results:
                    return {
                        results_output: gr.update(value=None, visible=False),
                        error_output: gr.update(value=results["error"], visible=True),
                        assessment_output: gr.update(value=""),
                        claims_output: gr.update(value=""),
                        references_output: gr.update(value=""),
                    }
                else:
                    # Update cost info with actual LLM call count                    
                    return {
                        results_output: gr.update(value=results, visible=True),
                        error_output: gr.update(value="", visible=False),
                        assessment_output: gr.update(value=results["assessment_report"]),
                        claims_output: gr.update(value=results["claims_summary"]),
                        references_output: gr.update(value=results["references"]),
                    }
            
            analyze_btn.click(
                fn=process_analysis,
                inputs=[diagnosis_input, max_results, max_claims],
                outputs=[results_output, error_output, assessment_output, claims_output, references_output]
            )
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()