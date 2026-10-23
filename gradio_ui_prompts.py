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

        # Make a simple test call with GPT-4 (enhanced version)
        response = client.chat.completions.create(
            model="gpt-4",  # ENHANCED: Use GPT-4 for better quality
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )

        # Store the API key for later use
        stored_api_key = api_key.strip()
        
        return "✅ OpenAI API key validated and stored successfully! You can now use the ENHANCED clinical diagnosis exploration tool with diagnosis revision and improved rigor."
    
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
            "diagnosis_analysis": results.get("diagnosis_analysis", {}),
            "assessment_report": results.get("assessment_report", "No assessment report generated."),
            "total_claims": results.get("claims_researched", 0),
            "total_articles": results.get("total_articles", 0),
            "total_llm_calls": results.get("total_llm_calls", 0),
            "claims_summary": "",
            "references": results.get("references", "No references available."),
            "critical_issues": results.get("critical_issues", []),
            "claim_validation": results.get("claim_validation", []),
        }
        
        # Create a detailed summary of claims with full research summaries and references
        claims = results.get("claims", [])
        claims_summary = "## Claims Analyzed:\n\n"
        for i, claim in enumerate(claims, 1):
            claims_summary += f"### Claim {i}: {claim['claim_text']}\n"
            claims_summary += f"- **Type:** {claim['claim_type']}\n"
            claims_summary += f"- **Confidence:** {claim['confidence']}\n"
            claims_summary += f"- **Evidence Level:** {claim['evidence_level']}\n"
            claims_summary += f"- **Research Status:** {claim['research_status']}\n"
            claims_summary += f"- **Evidence Quality:** {claim.get('evidence_quality', 'unknown')}\n"
            claims_summary += f"- **Articles Found:** {len(claim['abstracts'])}\n"
            claims_summary += f"- **PubMed Query:** {claim['pubmed_query']}\n"
            claims_summary += f"- **Potential Issues:** {claim.get('potential_issues', 'None identified')}\n\n"
            
            # Add the detailed summary with inline citations
            claims_summary += f"**Summary:** {claim['detailed_summary']}\n\n"
            claims_summary += f"**Evidence Summary:** {claim['evidence_summary']}\n\n"
            
            # Add critical assessment
            if claim.get('critical_assessment'):
                claims_summary += f"**Critical Assessment:** {claim['critical_assessment']}\n\n"
            
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
            
            # Add methodological issues if available
            if claim.get('methodological_issues'):
                claims_summary += "**Methodological Issues:**\n"
                for issue in claim['methodological_issues']:
                    claims_summary += f"- {issue}\n"
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
    Create the ENHANCED Gradio interface with multiple tabs.
    """
    
    with gr.Blocks(title="Enhanced Clinical Diagnosis Explorer", theme=gr.themes.Soft()) as iface:
        
        gr.Markdown("# Enhanced Clinical Diagnosis Explorer")
        gr.Markdown("**Features:** Diagnosis revision, improved rigor, GPT-4 analysis, critical assessment")
        
        # API Key Setup    
        with gr.Tab("🔑 API Key Setup"):
            gr.Markdown("## Step 1: Enter Your OpenAI API Key")
            gr.Markdown("Enter your OpenAI API key to get started. This will be used for all clinical analysis.")
            gr.Markdown("**Note:** This enhanced version uses GPT-4 for better quality analysis.")
            
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                placeholder="sk-proj-...",
                type="password",
                lines=1
            )
            
            validate_btn = gr.Button("�� Validate API Key", variant="primary")
            api_status = gr.Textbox(label="API Key Status", interactive=False)
            
            validate_btn.click(
                fn=validate_and_store_api_key,
                inputs=api_key_input,
                outputs=api_status
            )
        
        with gr.Tab("🏥 Enhanced Clinical Analysis"):
            gr.Markdown("## Step 2: Analyze Clinical Diagnosis")
            gr.Markdown("Enter a clinical diagnosis to break it down into claims and validate each against medical literature.")
            gr.Markdown("**Enhanced Features:**")
            gr.Markdown("- 🔍 **Diagnosis Revision:** Suggests better diagnoses if the original is problematic")
            gr.Markdown("- ⚠️ **Critical Assessment:** Identifies red flags and missing information")
            gr.Markdown("- 🎯 **Improved Rigor:** Willing to say 'no' when evidence doesn't support claims")
            gr.Markdown("- 📊 **Evidence Quality:** Assesses methodological quality of research")
            
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
                    gr.Markdown("### Clinical Assessment Report")
                    assessment_output = gr.Markdown(label="Assessment Report")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Diagnosis Analysis")
                    diagnosis_analysis_output = gr.Markdown(label="Diagnosis Analysis")
                
                with gr.Column():
                    gr.Markdown("### Critical Issues")
                    critical_issues_output = gr.Markdown(label="Critical Issues")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Claims Summary")
                    claims_output = gr.Markdown(label="Claims Analysis")
                
                with gr.Column():
                    gr.Markdown("### References")
                    references_output = gr.Markdown(label="Research References")
            
            def process_enhanced_analysis(diagnosis, max_res, max_clms):
                results = run_clinical_exploration_enhanced(diagnosis, max_res, max_clms)
                
                if "error" in results:
                    return {
                        results_output: gr.update(value=None, visible=False),
                        error_output: gr.update(value=results["error"], visible=True),
                        assessment_output: gr.update(value=""),
                        diagnosis_analysis_output: gr.update(value=""),
                        critical_issues_output: gr.update(value=""),
                        claims_output: gr.update(value=""),
                        references_output: gr.update(value=""),
                    }
                else:
                    # Format diagnosis analysis
                    diagnosis_analysis = results.get("diagnosis_analysis", {})
                    diagnosis_text = f"""
## Diagnosis Analysis

**Primary Diagnosis:** {diagnosis_analysis.get('primary_diagnosis', 'Unknown')}
**Confidence Level:** {diagnosis_analysis.get('confidence_level', 'unknown')}

### Red Flags Identified:
"""
                    for flag in diagnosis_analysis.get('red_flags', []):
                        diagnosis_text += f"- ⚠️ {flag}\n"
                    
                    diagnosis_text += "\n### Missing Information:\n"
                    for info in diagnosis_analysis.get('missing_information', []):
                        diagnosis_text += f"- ❓ {info}\n"
                    
                    diagnosis_text += "\n### Alternative Diagnoses:\n"
                    for alt in diagnosis_analysis.get('alternative_diagnoses', []):
                        diagnosis_text += f"- **{alt.get('diagnosis', 'Unknown')}** (Probability: {alt.get('probability', 'Unknown')})\n"
                        diagnosis_text += f"  - Supporting Evidence: {alt.get('supporting_evidence', 'None')}\n"
                        diagnosis_text += f"  - Distinguishing Features: {alt.get('distinguishing_features', 'None')}\n\n"
                    
                    diagnosis_text += f"\n### Reasoning:\n{diagnosis_analysis.get('reasoning', 'No reasoning provided')}"
                    
                    # Format critical issues
                    critical_issues = results.get("critical_issues", [])
                    critical_text = "## Critical Issues Identified:\n\n"
                    if critical_issues:
                        for i, issue in enumerate(critical_issues, 1):
                            critical_text += f"{i}. ⚠️ {issue}\n"
                    else:
                        critical_text += "No critical issues identified."
                    
                    # Format claim validation
                    claim_validation = results.get("claim_validation", [])
                    if claim_validation:
                        critical_text += "\n\n## Claim Validation:\n\n"
                        for validation in claim_validation:
                            status = "✅" if validation.get('is_valid', False) else "❌"
                            critical_text += f"{status} **Claim {validation.get('claim_id', 'Unknown')}:** {validation.get('issues', 'No issues')}\n"
                            if validation.get('suggestions'):
                                critical_text += f"   💡 Suggestions: {validation.get('suggestions')}\n"
                    
                    return {
                        results_output: gr.update(value=results, visible=True),
                        error_output: gr.update(value="", visible=False),
                        assessment_output: gr.update(value=results["assessment_report"]),
                        diagnosis_analysis_output: gr.update(value=diagnosis_text),
                        critical_issues_output: gr.update(value=critical_text),
                        claims_output: gr.update(value=results["claims_summary"]),
                        references_output: gr.update(value=results["references"]),
                    }
            
            analyze_btn.click(
                fn=process_enhanced_analysis,
                inputs=[diagnosis_input, max_results, max_claims],
                outputs=[results_output, error_output, assessment_output, diagnosis_analysis_output, critical_issues_output, claims_output, references_output]
            )
        
        with gr.Tab("📊 Analysis Summary"):
            gr.Markdown("## Analysis Summary")
            gr.Markdown("This tab will show a summary of the analysis results.")
            
            summary_output = gr.Markdown(label="Summary")
            
            def generate_summary(results):
                if not results or "error" in results:
                    return "No analysis results available."
                
                summary = f"""
## Analysis Summary

### Overview
- **Total Claims Analyzed:** {results.get('total_claims', 0)}
- **Total Articles Reviewed:** {results.get('total_articles', 0)}
- **Total LLM Calls:** {results.get('total_llm_calls', 0)}
- **Critical Issues Identified:** {len(results.get('critical_issues', []))}

### Diagnosis Analysis
- **Primary Diagnosis:** {results.get('diagnosis_analysis', {}).get('primary_diagnosis', 'Unknown')}
- **Confidence Level:** {results.get('diagnosis_analysis', {}).get('confidence_level', 'unknown')}
- **Red Flags:** {len(results.get('diagnosis_analysis', {}).get('red_flags', []))}
- **Alternative Diagnoses:** {len(results.get('diagnosis_analysis', {}).get('alternative_diagnoses', []))}

### Key Findings
- **Claims with Strong Evidence:** {sum(1 for claim in results.get('claims', []) if claim.get('evidence_level') == 'strong')}
- **Claims with Weak Evidence:** {sum(1 for claim in results.get('claims', []) if claim.get('evidence_level') == 'weak')}
- **Claims Refuted:** {sum(1 for claim in results.get('claims', []) if claim.get('research_status') == 'contradicted')}

### Recommendations
1. Review the diagnosis analysis for potential misdiagnosis
2. Address critical issues identified
3. Consider alternative diagnoses suggested
4. Follow up on claims with weak or contradictory evidence
"""
                return summary
            
            # This would need to be connected to the analysis results
            # For now, it's a placeholder
            
    return iface

if __name__ == "__main__":
    iface = create_interface()
    iface.launch()